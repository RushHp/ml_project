
# Пиплайн
"""import pandas as pd  # работа с таблицами
from src.config import DATA_PATH, MODEL_PATH  # путь к CSV с данными
from sklearn.model_selection import train_test_split  # разделение данных
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF векторизация текста
from sklearn.ensemble import RandomForestClassifier  # классификатор Random Forest
from sklearn.pipeline import Pipeline  # пайплайн для объединения шагов
from sklearn.metrics import accuracy_score, classification_report  # метрики качества
import joblib  # сохранение и загрузка моделей


# Загружаем и проверяем данные
df = pd.read_csv(DATA_PATH)  # читаем CSV в DataFrame
print(df.head())  # выводим первые 5 строк, чтобы проверить данные

X = df['text']   # тексты (входные данные)
y = df['label']  # метки (правильные ответы)

# Делим данные на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Размер обучающей выборки:", len(X_train))
print("Размер тестовой выборки:", len(X_test))

# Создаём пайплайн
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),  # текст → числа
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))  # числа → предсказания
])

# Обучаем пайплайн
pipeline.fit(X_train, y_train)  # обучаем сразу весь пайплайн на обучающих данных

# Проверяем качество на тесте
y_pred = pipeline.predict(X_test)  # предсказываем метки для тестовых данных

# Сохраняем пайплайн в один файл
joblib.dump(pipeline, MODEL_PATH)
print("Пайплайн успешно сохранён!")

# Проверка пути к диску
import os
assert os.path.exists(DATA_PATH), f"❌ Файл не найден: {DATA_PATH}"""





# Пиплайн
import pandas as pd  # работа с таблицами
from src.config import DATA_PATH, MODEL_PATH  # путь к CSV с данными
from sklearn.model_selection import train_test_split  # разделение данных
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF векторизация текста
from sklearn.ensemble import RandomForestClassifier  # классификатор Random Forest
from sklearn.pipeline import Pipeline  # пайплайн для объединения шагов
from sklearn.metrics import accuracy_score, classification_report  # метрики качества
import joblib  # сохранение и загрузка моделей
import os # работа с путями к файлам
from tqdm.auto import tqdm # импортируем tqdm для прогресс-бара


# Загружаем и проверяем данные
df = pd.read_csv(DATA_PATH)  # читаем CSV в DataFrame
df = df.head(1000)  # !!! ОБРЕЗАЕМ ДАННЫЕ ДО 1000 СТРОК !!!
print(df.head())  # выводим первые 5 строк, чтобы проверить данные

X = df['text']   # тексты (входные данные)
y = df['label']  # метки (правильные ответы)

# Делим данные на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Размер обучающей выборки:", len(X_train))
print("Размер тестовой выборки:", len(X_test))

# Создаём пайплайн
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),  # текст → числа
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))  # числа → предсказания
])

# Обучаем пайплайн с прогресс-баром
print("Обучение модели...")
for _ in tqdm(range(100)): # tqdm добавит прогресс-бар на 100 итераций
    pipeline.fit(X_train, y_train)

# Проверяем качество на тесте
y_pred = pipeline.predict(X_test)  # предсказываем метки для тестовых данных

# Сохраняем пайплайн в один файл
joblib.dump(pipeline, MODEL_PATH)
print("Пайплайн успешно сохранён!")

# Проверка пути к диску
assert os.path.exists(DATA_PATH), f"❌ Файл не найден: {DATA_PATH}"

























