# Импортируем модуль для работы с путями
import os 
import sys

# Добавляем папку проекта в путь поиска Python, чтобы он мог найти 'src'
sys.path.append('/content/ml_project')

# Открываем файл config.py
with open('src/config.py', 'r') as file:
    config_content = file.read()

# Заменяем старый путь на новый
new_config_content = config_content.replace(
    'DATA_PATH = "data/sentiment_dataset.csv"',
    'DATA_PATH = "/content/drive/MyDrive/sentiment_dataset.csv"'
)

# Сохраняем файл с новым путем
with open('src/config.py', 'w') as file:
    file.write(new_config_content)

print("Путь к данным в файле config.py успешно обновлен! 🎉")

# Теперь запускаем скрипт подготовки данных
!python src/data.py
