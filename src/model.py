# src/model.py — код для обучения модели, которая угадывает, токсичный комментарий или нет
import pandas as pd  # для работы с датасетом (таблицами)
from sklearn.model_selection import train_test_split  # для деления данных на обучение и тест
from sklearn.feature_extraction.text import TfidfVectorizer  # для превращения текстов в числа
from sklearn.ensemble import RandomForestClassifier  # модель, которая как лес из деревьев угадывает классы
from sklearn.pipeline import Pipeline  # чтобы объединить шаги обработки данных и обучения
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # для проверки, как хорошо модель работает
import joblib  # для сохранения модели в файл

# Импортируем пути из нашего config-файла
from config import DATA_PATH, MODEL_PATH

# Для балансировки данных
from sklearn.utils import resample

# --- Шаг 1: Загружаем и балансируем датасет ---
df = pd.read_csv(DATA_PATH)
print("Первые 5 строк датасета:")  # показываем, как выглядят данные
print(df.head())

# Отделяем токсичные и нетоксичные сообщения
df_toxic = df[df['toxic'] == 1]
df_non_toxic = df[df['toxic'] == 0]

# "Клонируем" токсичные сообщения, чтобы их стало столько же, сколько нетоксичных
df_toxic_upsampled = resample(df_toxic,
                             replace=True,  # разрешаем выбирать одни и те же строки несколько раз
                             n_samples=len(df_non_toxic),  # количество копий должно быть равно количеству нетоксичных
                             random_state=42)  # фиксируем случайность, чтобы результат был одинаковым при каждом запуске

# Объединяем теперь уже "сбалансированные" данные в новый датафрейм
df_balanced = pd.concat([df_non_toxic, df_toxic_upsampled])

# Перемешиваем строки, чтобы модель не училась на блоках одинаковых данных
df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

# Проверяем, что получилось
print("\nКоличество сообщений после балансировки:")
print(df_balanced['toxic'].value_counts())

# --- Шаг 2: Выбираем данные и делим их на обучение и тест ---
# Используем сбалансированный датасет
X = df_balanced['comment']
y = df_balanced['toxic']

# Делим данные на обучение и тест
# 80% данных — для обучения, 20% — для проверки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nРазмер обучающей выборки:", len(X_train))  # сколько комментариев для обучения
print("Размер тестовой выборки:", len(X_test))  # сколько для теста

# --- Шаг 3: Создаём пайплайн и обучаем модель ---
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, max_df=0.8)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)) # Вот так!
])

print("\nОбучаю модель...")
pipeline.fit(X_train, y_train)

# --- Шаг 4: Проверяем, как модель угадывает на тестовых данных ---
y_pred = pipeline.predict(X_test)  # предсказываем класс (токсичный или нет)

print("\n--- Отчёт о классификации ---")
print(classification_report(y_test, y_pred))  # подробный отчёт: точность, полнота, F1
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))  # показывает, где модель ошиблась
print("Точность модели:", accuracy_score(y_test, y_pred))  # сколько процентов угадано правильно

# --- Шаг 5: Сохраняем модель и визуализируем результаты ---
joblib.dump(pipeline, MODEL_PATH)
print(f"\nМодель сохранена в файл: {MODEL_PATH}")

# Получаем вероятности для ROC-кривой
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Запускаем визуализацию (убедись, что visualization.py находится в той же папке)
from visualization import visualize_all
visualize_all(df_balanced, pipeline, y_test, y_pred, y_prob)