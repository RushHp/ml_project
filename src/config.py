import os  # импортируем модуль для работы с переменными окружения

"""# Telegram токен (можно хранить в переменной окружения или прямо прописать)
BOT_TOKEN = os.getenv("BOT_TOKEN", "ВАШ_ТОКЕН_ЗДЕСЬ")

# Пути к данным и модели
DATA_PATH = "data/sentiment_dataset.csv"  # путь к CSV с дата сетом
MODEL_PATH = "src/sentiment_model.pkl"    # путь для сохранения модели"""

# Открываем файл config.py
with open('src/config.py', 'r') as file:
    config_content = file.read()

# Заменяем старый путь на новый
new_config_content = config_content.replace(
    'DATA_PATH = "data/sentiment_dataset.csv"',
    'DATA_PATH = "/content/drive/MyDrive/data/sentiment_dataset.csv"'
)

# Сохраняем файл с новым путем
with open('src/config.py', 'w') as file:
    file.write(new_config_content)

print("Путь к данным в файле config.py успешно обновлен! 🎉")
