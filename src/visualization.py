# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.pipeline import Pipeline
from typing import Tuple
import numpy as np  # <-- ЭТО Я ДОБАВИЛ


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Визуализация распределения классов"""
    plt.figure(figsize=(8, 4))
    sns.countplot(x='toxic', data=df)
    plt.title('Распределение классов')
    plt.show()


def plot_feature_importances(pipeline: Pipeline) -> None:
    """Визуализация важности признаков"""
    feature_importances = pipeline.named_steps['rf'].feature_importances_
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20))
    plt.title('Топ-20 важных признаков')
    plt.show()


def plot_roc_curve(y_test: pd.Series, y_prob: np.ndarray) -> None:
    """ROC-кривая"""
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (площадь = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Матрица ошибок"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Нетоксичный', 'Токсичный'],
                yticklabels=['Нетоксичный', 'Токсичный'])
    plt.xlabel('Предсказанные')
    plt.ylabel('Реальные')
    plt.title('Матрица ошибок')
    plt.show()


def visualize_all(df: pd.DataFrame, pipeline: Pipeline,
                  y_test: pd.Series, y_pred: np.ndarray,
                  y_prob: np.ndarray) -> None:
    """Запуск всех визуализаций"""
    plot_class_distribution(df)
    plot_feature_importances(pipeline)
    plot_roc_curve(y_test, y_prob)
    plot_confusion_matrix(y_test, y_pred)