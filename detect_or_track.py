import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import sys
import os
import av  # добавьте в начало файла

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, increment_path, set_logging, scale_coords, non_max_suppression
from utils.torch_utils import select_device, time_synchronized

from sort import Sort


def draw_boxes(img, bbox, identities=None, categories=None, names=None, colors=None):
    """
    Рисует bounding box и ID объекта на изображении.
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = 2  # Толщина линии

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        color = colors[cat]

        # Отрисовка bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        label = f"ID {id}: {names[cat]}"
        tf = max(tl - 1, 1)  # Толщина шрифта
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, tf, cv2.LINE_AA)

    return img


def apply_datamosh(input_path, output_path):
    """Применяет эффект datamosh к видео"""
    container = av.open(input_path)
    
    # Получаем параметры входного видео
    in_stream = container.streams.video[0]
    codec_name = in_stream.codec_context.name
    
    # Создаем выходной контейнер с нужным кодеком
    output = av.open(output_path, mode='w')
    output_stream = output.add_stream(codec_name)
    
    # Копируем параметры и конвертируем в целые числа где нужно
    output_stream.width = int(in_stream.width)
    output_stream.height = int(in_stream.height)
    output_stream.pix_fmt = in_stream.pix_fmt
    
    # Устанавливаем битрейт и другие параметры
    output_stream.bit_rate = 2000000  # 2 Mbps
    output_stream.options = {'crf': '23'}  # Качество сжатия
    
    try:
        # Сохраняем первый I-frame
        first_frame = True
        skip_frames = 2  # Пропускаем каждый второй I-frame для усиления эффекта
        frame_count = 0
        
        for frame in container.decode(video=0):
            frame_count += 1
            
            # Сохраняем первый кадр как есть
            if first_frame:
                packet = output_stream.encode(frame)
                if packet:
                    output.mux(packet)
                first_frame = False
                continue
                
            # Пропускаем некоторые I-frames для создания эффекта
            if frame_count % skip_frames != 0:
                # Изменяем тип кадра на P-frame
                frame.pict_type = 'P'
                
            packet = output_stream.encode(frame)
            if packet:
                output.mux(packet)
        
        # Записываем оставшиеся кадры
        packet = output_stream.encode(None)
        if packet:
            output.mux(packet)
            
    finally:
        # Закрываем файлы
        container.close()
        output.close()


def detect():
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Загрузка модели YOLO
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if half:
        model.half()

    # Загрузка датасета
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Получение имен классов и цветов
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Инициализация трекера
    sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

    # Инициализация видеописателей
    vid_writer, track_writer = None, None
    vid_path, track_path = None, None

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Предсказание YOLO
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes)

        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s.copy()

            # Создаем черное изображение для второго видео
            black_img = np.zeros_like(im0)

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                dets_to_sort = np.empty((0, 6))
                for x1, y1, x2, y2, conf, detclass in det.cpu().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                if opt.track:
                    tracked_dets = sort_tracker.update(dets_to_sort)

                    if len(tracked_dets) > 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]

                        # Рисуем bbox и ID на обоих изображениях
                        im0 = draw_boxes(im0, bbox_xyxy, identities, categories, names, colors)
                        black_img = draw_boxes(black_img, bbox_xyxy, identities, categories, names, colors)

                        if opt.show_track:
                            # Копируем объекты на черный фон
                            for i, box in enumerate(bbox_xyxy):
                                x1, y1, x2, y2 = map(int, box)
                                # Добавляем небольшой отступ вокруг объекта
                                padding = 10
                                y1 = max(0, y1 - padding)
                                y2 = min(im0.shape[0], y2 + padding)
                                x1 = max(0, x1 - padding)
                                x2 = min(im0.shape[1], x2 + padding)
                                
                                # Копируем область с объектом из основного видео на черный фон
                                black_img[y1:y2, x1:x2] = im0[y1:y2, x1:x2]

            # Запись в видеофайл
            if vid_path != p:
                vid_path = p
                track_path = str(save_dir / ('tracks_' + Path(p).name))

                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                if isinstance(track_writer, cv2.VideoWriter):
                    track_writer.release()

                if vid_cap:
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    fps, w, h = 30, im0.shape[1], im0.shape[0]

                vid_writer = cv2.VideoWriter(str(save_dir / Path(p).name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                track_writer = cv2.VideoWriter(track_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            if vid_writer is not None:
                vid_writer.write(im0)
            if track_writer is not None:
                track_writer.write(black_img)

    # После записи обычного видео, применяем datamosh если включена опция
    if isinstance(track_writer, cv2.VideoWriter):
        track_writer.release()
        track_writer = None  # Освобождаем ресурс
        
        if opt.datamosh:
            try:
                # Путь к временному файлу
                temp_path = str(save_dir / 'temp_tracks.mp4')
                final_path = str(save_dir / ('tracks_' + Path(p).name))
                
                # Даем время на освобождение файла
                time.sleep(1)
                
                # Переименовываем оригинальный файл
                os.rename(track_path, temp_path)
                
                # Применяем datamosh
                apply_datamosh(temp_path, final_path)
                
                # Даем время на освобождение файла
                time.sleep(1)
                
                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"Ошибка при создании datamosh эффекта: {e}")
                # В случае ошибки возвращаем оригинальный файл
                if os.path.exists(temp_path):
                    if not os.path.exists(final_path):
                        os.rename(temp_path, final_path)

    # Освобождение ресурсов
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
        vid_writer = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--device', default='', help='cuda device or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help="don't increment name if exists")
    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font thickness')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--unique-track-color', action='store_true', help='use unique color for each track')
    parser.add_argument('--datamosh', action='store_true', help='apply datamosh effect to tracking video')

    opt = parser.parse_args()
    random.seed(1)

    with torch.no_grad():
        detect()
