import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Número de muestras
N = 1000

# Pesos y parámetros para las fronteras de clase
W1 = 0.7
W2 = 0.3
B = -0.5
NOISE = 0.01

# Generar features aleatorios
X1 = np.random.rand(N)
X2 = np.random.rand(N)


# Inicializar matriz de etiquetas multilabel (N x 2)
Y = np.zeros((N, 2))

for i in range(N):
    noise = np.random.normal(0, NOISE)
    f = W1 * X1[i] + W2 * X2[i] + B + noise
    # Clase 0: f > 0
    if f > 0:
        Y[i, 0] = 1
    # Clase 1: x2 > 0.5
    if X2[i] + noise > 0.5:
        Y[i, 1] = 1


# Visualización para 2 clases
colors = []
for y in Y:
    if y.sum() > 1:
        colors.append('black')
    elif y[0]:
        colors.append('red')
    elif y[1]:
        colors.append('green')
    else:
        colors.append('gray')

plt.scatter(X1, X2, color=colors, alpha=0.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Multilabel dataset (2 clases)')
plt.show()

# Guardar el dataset (solo 2 clases)
data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y0': Y[:,0], 'Y1': Y[:,1]})
data.to_csv('dataset-lab2-multilabel.csv', index=False)