"""
Tarea 2: PCA and SVM used for wine fraud detection
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

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
    # Correlation between the dependent variable and the independent variables
    quality_corr = data.corr()["quality"].sort_values(ascending=False)
    print("\nCorrelación con la variable dependiente:")
    print(quality_corr)

    # Correlation matrix
    corr = data.corr()
    plt.figure(figsize=(14, 8))
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

def apply_pca(x):
    # Apply PCA with 2 and 3 dimensions to reduce dimensionality
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(x)
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(x)
    return [X_pca_2d, X_pca_3d]

def visualize_pca(X_pca_2d, X_pca_3d, y):
    # Scatter plot for 2D PCA
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=y, palette="pastel")
    plt.title("PCA with 2 Components")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    # Scatter plot for 3D PCA
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap="viridis")
    ax.set_title("PCA with 3 Components")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()

def split_data(X, y):
    return train_test_split(X, y, test_size=0.1, random_state=101)

def train_model(X_train, y_train, label, param_grid, svc):
    print(f"Training model for: {label}")
    # Perform grid search
    model = GridSearchCV(svc, param_grid, cv=5, verbose=2, n_jobs=-1)
    model.fit(X_train, y_train)
    # Save the best model for this Label
    print(f"Best parameters for {label}: {model.best_params_}")
    print(f"Best score for {label}: {model.best_score_}")
    return model

def split_train_and_save_models(X_scaled, X_pca_2d, X_pca_3d, y):
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    X_train_pca2, X_test_pca2, _, _ = split_data(X_pca_2d, y)
    X_train_pca3, X_test_pca3, _, _ = split_data(X_pca_3d, y)

    # Define grid search parameters
    param_grid = {
        "C": [0.001, 0.01, 0.1, 0.5, 1],
        "gamma": ["scale", "auto"],
        "kernel": ["linear", "poly", "rbf", "sigmoid"]
    }

    # Initialize the SVM model
    svc = SVC(class_weight="balanced")
    model_original_x = train_model(X_train, y_train, "original_x", param_grid, svc)
    model_pca2 = train_model(X_train_pca2, y_train, "pca2", param_grid, svc)
    model_pca3 = train_model(X_train_pca3, y_train, "pca3", param_grid, svc)

    # Store models and scores
    models = {
        "original_x": model_original_x,
        "pca2": model_pca2,
        "pca3": model_pca3
    }

    # Find the best model based on Grid Search score
    best_model_label = max(models, key=lambda label: models[label].best_score_)
    best_model = models[best_model_label]

    # Save the best model
    joblib.dump(best_model.best_estimator_, f"best_svc_model.pkl")
    print(f"\n🔥 Best Model: {best_model_label}")
    print(f"🏆 Best Parameters: {best_model.best_params_}")
    print(f"📊 Best Score: {best_model.best_score_}")
    print(f"✅ Best model saved as 'best_svc_model_{best_model_label}.pkl'")
    
    if best_model_label == "original_x":
        X_test = X_test
    elif best_model_label == "pca2":
        X_test = X_test_pca2
    else:
        X_test = X_test_pca3

    # Save x_test and y_test
    joblib.dump(X_test, "X_test.pkl")
    joblib.dump(y_test, "y_test.pkl")


def main():
    filepath = 'Data/wine_fraud.csv'
    data = load_data(filepath)
    eda(data)
    preprocessed_data = preprocess_data(data)
    show_correlated_values(preprocessed_data)
    X_pca = apply_pca(preprocessed_data)
    X_pca_2d = X_pca[0]
    X_pca_3d = X_pca[1]
    X_original = preprocessed_data.drop("quality", axis=1)
    y = preprocessed_data["quality"]
    visualize_pca(X_pca_2d, X_pca_3d, y)
    split_train_and_save_models(X_original, X_pca_2d, X_pca_3d, y)

if __name__ == "__main__":
    main()