# -*- coding: utf-8 -*-
"""Tarea 2: PCA and SVM used for wine fraud detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('wine_fraud.csv')

data.describe()

data.columns

data.head(10)

print(data["type"].unique())

print(data["quality"].unique())

quality_counts = data["quality"].value_counts()

plt.figure(figsize=(6,4))
sns.barplot(x=quality_counts.index, y=quality_counts.values, palette="pastel")

plt.xlabel("Quality")
plt.ylabel("Cantidad de muestras")
plt.title("Distribución de la variable dependiente")

plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data=data, x="type", hue="quality", palette="pastel")

plt.xlabel("Tipo de vino")
plt.ylabel("Cantidad de muestras")
plt.title("Comparación entre Tipo de Vino y Calidad")

plt.show()

vino_tinto = data[data["type"] == "red"]["quality"].value_counts(normalize=True)["Fraud"] * 100

vino_blanco = data[data["type"] == "white"]["quality"].value_counts(normalize=True)["Fraud"] * 100

print(f"Vinos tintos que son fraude: {vino_tinto:.2f}%")
print(f"Vinos blancos que son fraude: {vino_blanco:.2f}%")
