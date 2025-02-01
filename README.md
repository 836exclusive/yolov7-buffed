##

Visit [speesh.ru/tracking](https://speesh.ru/tracking).
sample: python detect_or_track.py --weights yolov7.pt --source street.mp4 --track --show-track --datamosh

## 🤝 Support

For questions and support:

- Telegram: [t.me/speesh](https://t.me/speesh)
- Website: [speesh.ru](https://speesh.ru)

## 📝 Citation

```bibtex
В 2023 году я открыл этот способ, пытался его пропушить и применять где только можно. В следующем году успешно защитил диплом бакалавра на эту тему. Спустя время начал находить неестественные попытки это повторить. Ребята делали это с помощью AE и собственноручно. Такой способ реализации нарушает всю мою идеологию, ибо я считаю что настоящий вебпанк не должен быть фейковым.

Не претендую на уникальность, но спустя год раскатываю это в сеть чтобы как можно больше людей знали как делать нужно.
}
```

## 📜 License

This project is released under the same license as the original YOLOv7.

## 🙏 Acknowledgements

• [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
• [haroonshakeel/yolov7-object-tracking
](https://github.com/haroonshakeel/yolov7-object-tracking)

## 🛠️ Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/yolov7-tracking
cd yolov7-tracking
```

2. Install required packages

```bash
# Base requirements
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install av-python
pip install filterpy
pip install scikit-image

# For CUDA support (optional, replace XX with your CUDA version)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu11X
```

3. Or install all requirements at once

```bash
pip install -r requirements.txt
```

4. Download weights

```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```
