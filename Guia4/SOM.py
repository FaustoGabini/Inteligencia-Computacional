import time
import numpy as np
import matplotlib.pyplot as plt

class SOM:  

  def __init__ (self, map_size, w=None):
    self.map_size = map_size
    self.w = w

  def fit(self, inputs, max_epocas, tasa_aprendizaje, r_vecinos): 
    n = inputs.shape[1]
    # Si los pesos no fueron dados, inicializarlos aleatoriamente
    if self.w is None:
      self.w = np.random.uniform(-0.5, 0.5, size=(self.map_size, self.map_size, n))

    
    inicio = time.time()
    for epoca in range(max_epocas):
      for input in inputs:
        step = (tasa_aprendizaje[0] - tasa_aprendizaje[1]) / max_epocas
        eta = tasa_aprendizaje[0] - step * epoca

        step = (r_vecinos[0] - r_vecinos[1])/max_epocas
        radio = round(r_vecinos[0] - step * epoca)

        # Con ese axis = 2 le decimos que haga toda esa operacion sobre la 3er dimension
        distancias = np.linalg.norm(self.w - input, axis=2)     
        ganadora_index = np.unravel_index(np.argmin(distancias), distancias.shape)

        for i in range(ganadora_index[0] - radio, ganadora_index[0] + radio + 1): # Recorremos las filas
          for j in range(ganadora_index[1] - radio, ganadora_index[1] + radio + 1): # Recorremos las col
            if 0 <= i < self.w.shape[0] and 0 <= j < self.w.shape[1]:
              self.w[i, j] += eta * (input - self.w[i, j])
    
    fin = time.time()
    print('El entrenamiento finalizó en la época',epoca,'en',round(fin-inicio,2),'segundos.')
    

  # Devuelve la lista de posiciones de las neuronas ganadoras para cada uno de los patrones
  def obtener_neuronas_ganadoras(self, inputs):
    input_neurona = []
    for input in inputs:
      dist = np.linalg.norm(self.w - input, axis=2)
      indice_lineal = np.argmin(dist)
      input_neurona.append(indice_lineal)
    return input_neurona

  def graficar(self, X):
    ganadoras = []
    for x in X: 
      distancias = np.linalg.norm(self.w - x, axis=2) 
      ganadora_index = np.unravel_index(np.argmin(distancias), distancias.shape)
      ganadoras.append(ganadora_index)
    
    ganadoras = np.array(ganadoras)

    # Asignar un color a cada neurona ganadora
    colores = ganadoras[:,0] * self.map_size + ganadoras[:,1]
    plt.figure(figsize=(8,8))
    # Graficar los datos de entrenamiento coloreados por neurona ganadora
    plt.scatter(X[:,0], X[:,1], c=colores, cmap='tab20', marker='x', label='Datos')

    # Graficar los centroides
    for i in range(self.map_size):
      for j in range(self.map_size):
        plt.scatter(self.w[i,j,0], self.w[i,j,1], marker='o', color='black', s=100, edgecolor='white', label='Centroide' if (i==0 and j==0) else "")
        # Unir con vecinos (derecha y abajo)
        if i + 1 < self.map_size:
          plt.plot([self.w[i,j,0], self.w[i+1,j,0]], [self.w[i,j,1], self.w[i+1,j,1]], color='blue', linewidth=1)
        if j + 1 < self.map_size:
          plt.plot([self.w[i,j,0], self.w[i,j+1,0]], [self.w[i,j,1], self.w[i,j+1,1]], color='blue', linewidth=1)
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Scatter de los datos X')
    plt.legend()
    plt.show()





               
        




