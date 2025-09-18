import pandas as pd  # для работы с таблицами (датасетом)
import numpy as np  # для подсчёта чисел (среднее, медиана и т.д.)
from config import DATA_PATH

# Загружаем датасет
df = pd.read_csv(DATA_PATH)

#----Основные статистические показатели

print(df.describe())
# Находим медиану токсичности, смотрим вероятность отклонения
print(df['toxic'].median())
print('---------------')


#----Креляция

# Создаем новый признак 'message_length'
df['message_length'] = df['comment'].apply(len)

# Теперь смотрим статистику по длине предложения
print(df['message_length'].describe())

    