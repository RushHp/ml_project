# src/model.py — готовый к запуску на ПК с сохранением в artifacts
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.config import DATA_PATH  # пути к CSV берём из конфига

# Путь для сохранения модели в папку artifacts
MODEL_PATH = "artifacts/sentiment_model.pkl"

# -----------------------------
# Загружаем данные
# -----------------------------
df = pd.read_csv(DATA_PATH)
print("Первые 5 строк датасета:")
print(df.head())

# Выделяем признаки и метки
X = df['text']
y = df['label']

# -----------------------------
# Делим на обучение и тест
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Размер обучающей выборки:", len(X_train))
print("Размер тестовой выборки:", len(X_test))

# -----------------------------
# Создаём пайплайн
# -----------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# -----------------------------
# Обучаем модель
# -----------------------------
print("Обучение модели...")
pipeline.fit(X_train, y_train)

# -----------------------------
# Проверяем качество на тесте
# -----------------------------
y_pred = pipeline.predict(X_test)
print("\nОтчёт о классификации:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Сохраняем модель
# -----------------------------
joblib.dump(pipeline, MODEL_PATH)
print(f"\nМодель успешно сохранена в {MODEL_PATH}!")
