import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def load_data(filepath):
    data = pd.read_csv(filepath)
    if data is None:
        print("Error cargando el archivo.")
    else:
        print("\nArchivo cargado con éxito.")
    return data

def eda(data):
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
    # Correlation between the dependent variable and the independent variables
    quality_corr = data.corr()["quality"].sort_values(ascending=True)
    print("\nCorrelación con la variable dependiente:")
    print(quality_corr)

    # Correlation matrix
    corr = data.corr()
    plt.figure(figsize=(14, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlación")
    plt.show()


def calculate_fraud_percentage(data):
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
    X = data.drop(columns=["quality"])  # Variables predictoras
    y = data["quality"]  # Variable objetivo

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101, stratify=y)

    return X_train, X_test, y_train, y_test

def train_model(data):

    X_train, X_test, y_train, y_test = split_data(data)

    # Estandarizar los datos (SVM es sensible a la escala)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------------------------------
    param_grid = {
        "C": [0.001, 0.01, 0.1, 0.5, 1],
        "gamma": ["scale", "auto"],
        "kernel": ["linear", "poly", "rbf", "sigmoid"]
    }

    svc = SVC(class_weight="balanced")

    # Grid search
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring="accuracy", verbose=2, n_jobs=-1)

    # entrenamiento del modelo
    grid_search.fit(X_train_scaled, y_train)

    # hiperparámetros
    print("Hiperparametros:", grid_search.best_params_)

    # Guardar modelo en joblib
    joblib.dump(grid_search.best_estimator_, "modelo_svc.pkl")

    print("modelo_svc.pkl guardado")

    return grid_search.best_estimator_


def main():
    filepath = 'Data/wine_fraud.csv'
    data = load_data(filepath)
    eda(data)
    preprocessed_data = preprocess_data(data)
    show_correlated_values(preprocessed_data)
    preprocessed_data = apply_pca(preprocessed_data)
    preprocessed_data = split_data(preprocessed_data)
    train_model(preprocessed_data)

if __name__ == "__main__":
    main()