import os

# Получаем абсолютный путь к корню проекта (ml_project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Определяем пути
BOT_TOKEN = os.getenv("BOT_TOKEN", "ВАШ_ТОКЕН_ЗДЕСЬ")

DATA_PATH = os.path.join(BASE_DIR, "data", "sentiment_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "sentiment_model.pkl")
