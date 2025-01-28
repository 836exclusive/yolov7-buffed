import PyInstaller.__main__
import shutil
import os
import time

# Создаем директорию для сборки если её нет
if not os.path.exists('dist'):
    os.makedirs('dist')

# Собираем detect_or_track.py первым
print("Сборка detect_or_track.py...")
PyInstaller.__main__.run([
    'detect_or_track.py',
    '--onefile',
    '--name', 'detect_or_track',
    '--add-data', 'yolov7.pt;.' if os.name == 'nt' else 'yolov7.pt:.',
    '--add-data', 'models;models' if os.name == 'nt' else 'models:models',
    '--add-data', 'utils;utils' if os.name == 'nt' else 'utils:utils',
    '--hidden-import', 'models',
    '--hidden-import', 'utils',
    '--hidden-import', 'sort',
    '--hidden-import', 'torch',
    '--hidden-import', 'torchvision',
    '--hidden-import', 'cv2',
    '--hidden-import', 'numpy',
    '--hidden-import', 'PIL',
    '--hidden-import', 'tqdm',
    '--hidden-import', 'pandas',
    '--hidden-import', 'seaborn',
    '--hidden-import', 'yaml'
])

# Собираем GUI
print("Сборка gui.py...")
PyInstaller.__main__.run([
    'gui.py',
    '--onefile',
    '--windowed',
    '--name', 'object_tracker',
    '--add-data', 'yolov7.pt;.' if os.name == 'nt' else 'yolov7.pt:.',
])

# Ждем завершения сборки
time.sleep(2)

# Создаем необходимые директории
os.makedirs(os.path.join('dist', 'models'), exist_ok=True)
os.makedirs(os.path.join('dist', 'utils'), exist_ok=True)

# Копируем необходимые директории
def copy_directory(src, dst):
    if os.path.exists(src):
        print(f"Копирование {src}...")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        print(f"Ошибка: директория {src} не найдена!")
        exit(1)

# Копируем директории
copy_directory('models', os.path.join('dist', 'models'))
copy_directory('utils', os.path.join('dist', 'utils'))

# Проверяем наличие файлов в dist
detect_track_exe = os.path.join('dist', 'detect_or_track.exe' if os.name == 'nt' else 'detect_or_track')
if not os.path.exists(detect_track_exe):
    print(f"Ошибка: {detect_track_exe} не найден!")
    exit(1)

# Копируем yolov7.pt если он существует
if os.path.exists('yolov7.pt'):
    print("Копирование yolov7.pt...")
    shutil.copy2('yolov7.pt', os.path.join('dist', 'yolov7.pt'))
else:
    print("Ошибка: yolov7.pt не найден!")
    exit(1)

# Копируем sort.py если он существует
if os.path.exists('sort.py'):
    print("Копирование sort.py...")
    shutil.copy2('sort.py', os.path.join('dist', 'sort.py'))
else:
    print("Ошибка: sort.py не найден!")
    exit(1)

print("Сборка завершена успешно!") 