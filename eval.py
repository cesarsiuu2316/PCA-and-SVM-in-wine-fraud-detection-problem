import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def cargar_modelo_y_datos(ruta_modelo='best_svc_model.pkl', ruta_x_test='X_test.pkl', ruta_y_test='y_test.pkl'):
    """
    Carga el modelo entrenado y los datos de evaluación.
    """
    modelo = joblib.load(ruta_modelo)
    x_eval = joblib.load(ruta_x_test)
    y_eval = joblib.load(ruta_y_test)
    
    return modelo, x_eval, y_eval

def realizar_predicciones(modelo, x_eval):
    y_pred = modelo.predict(x_eval)
    return y_pred

def calcular_metricas_clasificacion(y_eval, y_pred):
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred, average='weighted')
    recall = recall_score(y_eval, y_pred, average='weighted')
    f1 = f1_score(y_eval, y_pred, average='weighted')
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def mostrar_metricas_clasificacion(metricas):
    print('EVALUACIÓN DEL MODELO DE CLASIFICACIÓN')
    print('Exactitud (Accuracy):', round(metricas['Accuracy'], 4))
    print('Precisión (Precision):', round(metricas['Precision'], 4))
    print('Sensibilidad (Recall):', round(metricas['Recall'], 4))
    print('Puntuación F1 (F1 Score):', round(metricas['F1 Score'], 4))
    print('-' * 50)

def mostrar_classification_report(y_eval, y_pred):
    print('INFORME DE CLASIFICACIÓN DETALLADO')
    print(classification_report(y_eval, y_pred))
    print('-' * 50)

def visualizar_matriz_confusion(y_eval, y_pred):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_eval, y_pred)
    # Crear heatmap con valores exactos
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.show()

def evaluar_modelo():
    # Cargar modelo y datos
    modelo, x_eval, y_eval = cargar_modelo_y_datos()
    # Realizar predicciones
    y_pred = realizar_predicciones(modelo, x_eval)
    # Calcular métricas de clasificación
    metricas = calcular_metricas_clasificacion(y_eval, y_pred)
    # Mostrar resultados
    mostrar_metricas_clasificacion(metricas)
    mostrar_classification_report(y_eval, y_pred)
    # Visualizacion
    visualizar_matriz_confusion(y_eval, y_pred)

if __name__ == "__main__":
    evaluar_modelo()