"""
Tarea 2: PCA and SVM used for wine fraud detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(filepath)
    if data is None:
        print("Error cargando el archivo.")
    else:
        print("\nArchivo cargado con éxito.")
    return data

def eda(data):
    """Perform exploratory data analysis (EDA)."""
    print("Unique wine types:", data["type"].unique())
    print("Unique quality values:", data["quality"].unique())

    # Data visualizations
    plt.figure(figsize=(12, 6))
    
    # Quality distribution
    quality_counts = data["quality"].value_counts()
    plt.subplot(1, 2, 1)
    sns.barplot(x=quality_counts.index, y=quality_counts.values, hue=quality_counts.index, palette="pastel")
    plt.xlabel("Quality")
    plt.ylabel("Cantidad de muestras")
    plt.title("Distribución de la variable dependiente")

    # Comparison between wine type and quality
    plt.subplot(1, 2, 2)
    sns.countplot(data=data, x="type", hue="quality", palette="pastel")
    plt.xlabel("Tipo de vino")
    plt.ylabel("Cantidad de muestras")
    plt.title("Comparación entre Tipo de Vino y Calidad")

    plt.tight_layout()
    plt.show()

    # Fraud percentage for red and white wines
    calculate_fraud_percentage(data)

    print("\nResumen de valores faltantesn:")
    print(data.isnull().sum())

def show_correlated_values(data):
    #  

    # Correlation matrix
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlación")
    plt.show()


def calculate_fraud_percentage(data):
    """Calculate and print the percentage of fraudulent wines."""
    try:
        vino_tinto = data[data["type"] == "red"]["quality"].value_counts(normalize=True)["Fraud"] * 100
        vino_blanco = data[data["type"] == "white"]["quality"].value_counts(normalize=True)["Fraud"] * 100
        print(f"Vinos tintos que son fraude: {vino_tinto:.2f}%")
        print(f"Vinos blancos que son fraude: {vino_blanco:.2f}%")
    except KeyError:
        print("The 'Fraud' category is not present in the dataset.")

def preprocess_data(data):
    """Preprocess the data (e.g., handle missing values, encode labels and normalization)."""
    # Encoding
    # label encoding
    data["quality"] = data["quality"].map({"Legit": 1, "Fraud": 0})
    # one-hot encoding
    normalized_data = pd.get_dummies(data, columns=["type"], dtype=int)
    column_names = normalized_data.columns.tolist()
    # No missing values to handle

    # Normalization
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(normalized_data)
    normalized_data = pd.DataFrame(normalized_data, columns=column_names) 
    print("\nDatos normalizados:")
    print(normalized_data.head())
    return normalized_data

def apply_pca(data):
    """Apply PCA to reduce dimensionality."""
    # Placeholder for PCA implementation
    print("PCA function not implemented yet.")
    return data

def split_data(data):
    """Split the data into training and testing sets."""
    # Placeholder for splitting data
    print("Data splitting function not implemented yet.")
    return data

def train_model(data):
    """Train a model (e.g., SVM) on the dataset."""
    # Placeholder for training logic
    print("Model training function not implemented yet.")

def main():
    filepath = 'Data/wine_fraud.csv'
    data = load_data(filepath)
    eda(data)
    preprocess_data(data)
    #show_correlated_values(data)
    data = apply_pca(data)
    data = split_data(data)
    train_model(data)

if __name__ == "__main__":
    main()