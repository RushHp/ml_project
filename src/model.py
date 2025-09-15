"""import pandas as pd  # импортируем библиотеку pandas для работы с таблицами
from src.config import DATA_PATH  # импортируем путь к датасету из конфиг-файла
from sklearn.model_selection import train_test_split  # импортируем функцию для разбиения данных
from sklearn.feature_extraction.text import TfidfVectorizer  # импортируем TF-IDF векторизатор
from sklearn.ensemble import RandomForestClassifier  # импортируем классификатор случайного леса
from sklearn.metrics import accuracy_score, classification_report  # импортируем метрики качества
import joblib  # импортируем библиотеку для сохранения и загрузки моделей

df = pd.read_csv(DATA_PATH)  # загружаем CSV в переменную df (таблица)

print(df.head())  # Первые пять строк

X = df['text']  # тексты (входные данные для модели)
y = df['label']  # метки (правильные ответы)

# Делим
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Делим на - обучение и тест.

print("Размер обучающей выборки:", len(X_train))
print("Размер тестовой выборки:", len(X_test))

# Векторизируем
vectorizer = TfidfVectorizer(max_features=5000)  # создаём объект векторизатора, переобразование текста в числа
X_train_tfidf = vectorizer.fit_transform(X_train)  # обучаем векторизатор на обучающих данных и преобразуем тексты
X_test_tfidf = vectorizer.transform(X_test)  # преобразуем тестовые тексты с уже обученным векторизатором

print("Размер обучающей матрицы:", X_train_tfidf.shape)
print("Размер тестовой матрицы:", X_test_tfidf.shape)

# Обучаем
model = RandomForestClassifier(n_estimators=100, random_state=42)  # создаём объект модели
model.fit(X_train_tfidf, y_train)  # обучаем модель на обучающих данных
y_pred = model.predict(X_test_tfidf)  # делаем предсказания на тестовых данных

# Смотрим точность обучения
print("Точность на тесте:", accuracy_score(y_test, y_pred))  # выводим точность
print(classification_report(y_test, y_pred))  # подробный отчёт по метрикам - оценка качества модели

# Сохраняем обученую модель
joblib.dump(model, 'artifacts/model_rf.joblib')  # сохраняем обученную модель
joblib.dump(vectorizer, 'artifacts/vectorizer_tf.joblib')  # сохраняем TF-IDF векторизатор, чтобы векторизатор мог преобразовывать новые тексты в те же числа, что и обучающие данные.

print("Модель и векторизатор успешно сохранены!")  # подтверждение"""

# Ипользуем пиплайн

import pandas as pd  # работа с таблицами
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
























