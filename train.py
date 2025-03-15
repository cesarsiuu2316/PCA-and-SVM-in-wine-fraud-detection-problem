"""
Tarea 2: PCA and SVM used for wine fraud detection
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


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
    print("Tipos de vino únicos:", data["type"].unique())
    print("Tipos de calidad únicos:", data["quality"].unique())
    print("Shape:", data.shape)

    # Data visualizations
    plt.figure(figsize=(12, 6))
    
    # Quality distribution
    quality_counts = data["quality"].value_counts()
    print("\nValue counts", quality_counts)
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
    new_data = pd.get_dummies(data, columns=["type"], dtype=int)
    features_to_scale = new_data.drop(["quality", "type_red", "type_white"], axis=1)
    # No missing values to handle
    # Normalization
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(features_to_scale)
    normalized_data = pd.DataFrame(normalized_data, columns=features_to_scale.columns) 

    # Concatenate the normalized data with the encoded labels
    processed_data = pd.concat([normalized_data, new_data[["quality", "type_red", "type_white"]]], axis=1)
    print("\nDatos normalizados:")
    print(processed_data.head())
    return processed_data

def apply_pca(x_train, x_test, n):
    # Apply PCA  to reduce dimensionality
    pca = PCA(n_components=n)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_test_pca

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

def apply_undersampling_and_oversampling(X_train, y_train):
    # Define the pipeline
    over_sampler = SMOTE(sampling_strategy=0.1) # add synthetic samples, to have exactly 20% of the majority class / 0.2*6251 = 1250
    #under_sampler = RandomUnderSampler(sampling_strategy=1) # reduce the majority class to 150% of the minority class / 1.5*1250 = 1875
    # Apply the pipeline
    X_resampled, y_resampled = over_sampler.fit_resample(X_train, y_train)
    quality_counts = y_train.value_counts()
    print("\nValue counts y_train", quality_counts)
    quality_counts = y_resampled.value_counts()
    print("\nValue counts oversample", quality_counts)
    """
    X_resampled, y_resampled = under_sampler.fit_resample(X_resampled, y_resampled)
    quality_counts = y_resampled.value_counts()
    print("\nValue counts undersample", quality_counts)    
    """
    return X_resampled, y_resampled

def split_train_and_save_models(X_train_resampled, X_pca_train_2d, X_pca_train_3d, y_train_resampled, X_test, X_pca_test_2d, X_pca_test_3d, y_test):
    # Define grid search parameters
    param_grid = {
        "C": [0.001, 0.01, 0.1, 0.5, 1],
        "gamma": ["scale", "auto"],
        "kernel": ["linear", "poly", "rbf", "sigmoid"]
    }
    """
    # parameters for polynomial kernel, degree 4 takes too long to train
    param_grid = {
        "C": [0.001, 0.01, 0.1, 0.5, 1],
        "gamma": ["scale", "auto"],
        "kernel": ["poly"],
        "degree": [2, 3, 4]
    }
    """
    # Initialize the SVM model without over-sampling and under-sampling
    svc = SVC(class_weight="balanced")
    # Initialize the SVM model with over-sampling and under-sampling
    #svc = SVC()
    model_original_x = train_model(X_train_resampled, y_train_resampled, "original_x", param_grid, svc)
    model_pca2 = train_model(X_pca_train_2d, y_train_resampled, "pca2", param_grid, svc)
    model_pca3 = train_model(X_pca_train_3d, y_train_resampled, "pca3", param_grid, svc)

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
        X_test = X_pca_test_2d
    else:
        X_test = X_pca_test_3d

    # Save x_test and y_test
    joblib.dump(X_test, "X_test.pkl")
    joblib.dump(y_test, "y_test.pkl")


def main():
    filepath = 'Data/wine_fraud.csv'
    data = load_data(filepath)
    eda(data)
    preprocessed_data = preprocess_data(data)
    show_correlated_values(preprocessed_data)

    # Split the data
    X_original = preprocessed_data.drop("quality", axis=1)
    y = preprocessed_data["quality"]
    X_train, X_test, y_train, y_test = split_data(X_original, y)

    # Undersampling and Oversampling
    #X_train_resampled, y_train_resampled = apply_undersampling_and_oversampling(X_train, y_train)
    # without undersampling and oversampling
    X_train_resampled, y_train_resampled = X_train, y_train

    # Apply PCA
    [X_pca_train_2d, X_pca_test_2d] = apply_pca(X_train_resampled, X_test, 2)
    [X_pca_train_3d, X_pca_test_3d] = apply_pca(X_train_resampled, X_test, 3)

    # Visualize PCA for 2D and 3D training data
    visualize_pca(X_pca_train_2d, X_pca_train_3d, y_train_resampled)

    # Split, train and save models
    split_train_and_save_models(X_train_resampled, X_pca_train_2d, X_pca_train_3d, y_train_resampled, X_test, X_pca_test_2d, X_pca_test_3d, y_test)

if __name__ == "__main__":
    main()