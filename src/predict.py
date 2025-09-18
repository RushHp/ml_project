# src/predict.py
import joblib # для загрузки нашей модели
import pandas as pd # для работы с данными
from config import MODEL_PATH # путь к сохранённой модели


# Функция, которая предсказывает токсичность
def predict_toxicity(text: str) -> str:
    """
    Предсказывает, является ли текст токсичным.
    Возвращает "Токсичное" или "Нетоксичное".
    """
    # Загружаем нашу обученную модель из файла
    # Модель "помнит" все шаги, включая TF-IDF
    model = joblib.load(MODEL_PATH)
    
    # Модель принимает данные в формате Series, поэтому оборачиваем текст
    # Если бы мы не обернули, модель бы не поняла, что делать
    text_series = pd.Series([text])
    
    # Предсказываем класс (0 или 1) для нашего текста
    prediction = model.predict(text_series)
    
    # Конвертируем число в понятный для человека текст
    if prediction[0] == 1:
        return "Токсичное"
    else:
        return "Нетоксичное"

# Пример использования:
if __name__ == "__main__":
    # Текст, который мы хотим проверить
#    Попробуй эти примеры
    comment1 = "иди на хуй"
    comment2 = "спасибо"
    
    # Используем нашу функцию
    result1 = predict_toxicity(comment1)
    result2 = predict_toxicity(comment2)
    
    # Печатаем результат
    print(f'Комментарий "{comment1}" классифицирован как: {result1}')
    print(f'Комментарий "{comment2}" классифицирован как: {result2}')
 