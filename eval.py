import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

modelo = joblib.load('modelo_final.joblib')

x_eval = pd.read_csv('x_eval.csv')
y_eval = pd.read_csv('y_eval.csv')
y_eval = y_eval.iloc[:, 1]  # Toma solo la primera columna

y_pred = modelo.predict(x_eval)

pmse = mean_squared_error(y_eval, y_pred)
pmae = mean_absolute_error(y_eval, y_pred)
prmse = mean_squared_error(y_eval, y_pred) ** 0.5

print('Prediccion')
print('Error Absoluto Medio: ', pmae)
print('Error Cuadrático Medio: ', pmse)
print('Raíz del Error Cuadrático Medio: ', prmse)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_eval, y=y_pred, alpha=0.5)
plt.plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()])
plt.xlabel('Entreamiento')
plt.ylabel('Predicción')
plt.title('Entrenamiento vs Predicción')
plt.show()