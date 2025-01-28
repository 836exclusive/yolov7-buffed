import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                            QVBoxLayout, QWidget, QLabel, QMessageBox)
from PyQt5.QtCore import Qt
import subprocess

class YOLOv7GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Tracker")
        self.setGeometry(100, 100, 400, 200)

        # Получаем путь к директории с исполняемым файлом
        if getattr(sys, 'frozen', False):
            self.application_path = os.path.dirname(sys.executable)
        else:
            self.application_path = os.path.dirname(os.path.abspath(__file__))

        # Создаем центральный виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Выбор видео
        self.source_label = QLabel("Выберите видео для обработки")
        self.source_label.setAlignment(Qt.AlignCenter)
        self.source_path = QLabel("Файл не выбран")
        self.source_path.setAlignment(Qt.AlignCenter)
        self.source_btn = QPushButton("Выбрать видео")
        self.source_btn.clicked.connect(self.select_source)
        
        # Кнопка запуска
        self.run_btn = QPushButton("Начать отслеживание")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 15px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_btn.clicked.connect(self.run_detection)

        # Добавляем виджеты
        layout.addWidget(self.source_label)
        layout.addWidget(self.source_path)
        layout.addWidget(self.source_btn)
        layout.addSpacing(20)
        layout.addWidget(self.run_btn)

    def show_error(self, message):
        QMessageBox.critical(self, "Ошибка", message)

    def select_source(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите видео", 
            "", 
            "Видео файлы (*.mp4 *.avi *.mov);;Все файлы (*.*)"
        )
        if filename:
            self.source_path.setText(filename)

    def run_detection(self):
        if self.source_path.text() == "Файл не выбран":
            self.show_error("Пожалуйста, выберите видео файл!")
            return

        # Проверяем наличие необходимых файлов
        required_files = {
            'detect_or_track.exe': 'Программа обнаружения',
            'yolov7.pt': 'Файл модели',
            'models': 'Директория models',
            'utils': 'Директория utils',
            'sort.py': 'Модуль трекинга'
        }

        for file, description in required_files.items():
            path = os.path.join(self.application_path, file)
            if not os.path.exists(path):
                self.show_error(f"Не найден {description}: {path}")
                return

        # Формируем команду
        command = [
            os.path.join(self.application_path, 'detect_or_track.exe'),
            "--weights", os.path.join(self.application_path, "yolov7.pt"),
            "--source", self.source_path.text(),
            "--conf-thres", "0.25",
            "--iou-thres", "0.45",
            "--view-img",
            "--track",
            "--show-track",
            "--show-fps",
            "--save-img"
        ]

        try:
            subprocess.Popen(command)
        except Exception as e:
            self.show_error(f"Ошибка при запуске: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLOv7GUI()
    window.show()
    sys.exit(app.exec_()) 