import os  # импортируем модуль для работы с переменными окружения

"""# Telegram токен (можно хранить в переменной окружения или прямо прописать)
BOT_TOKEN = os.getenv("BOT_TOKEN", "ВАШ_ТОКЕН_ЗДЕСЬ")

# Пути к данным и модели
DATA_PATH = "data/sentiment_dataset.csv"  # путь к CSV с дата сетом
MODEL_PATH = "src/sentiment_model.pkl"    # путь для сохранения модели"""

import os

# Путь к данным и модели
DATA_PATH = "data/sentiment_dataset.csv"
MODEL_PATH = "artifacts/pipeline_rf.joblib"