import numpy as np
import pandas as pd

class PerceptronSimple: 
  def __init__ (self, tasa_aprendizaje=0.1, max_epocas = 100, tolerancia=0): 
    self.tasa = tasa_aprendizaje
    self.max_epocas = max_epocas
    self.tolerancia = tolerancia
    self.w = None # pesos
    self.errores = []
    
    # Guardamos el historial de pesos
    self.hist_w = []

  # El _ es porque es una funcion interna del metodo
  def _funcion_activacion(self, z): 
    # Funcion signo(z)
    return np.where(z >= 0, 1, -1)  # sgn(z)

  def entrenar(self, X, y): 
    
    # Agregamos bias como x0 = -1
    # Con el hstack concatenamos el bias en cada uno de los patrones
    X = np.hstack([ -np.ones((X.shape[0], 1)), X ]) 

    # Inicializamos los pesos aleatorios entre [-0.5, 0.5]
    self.w = np.random.uniform(-0.5, 0.5, X.shape[1])
    self.hist_w.append(self.w) 


    for epoca in range(self.max_epocas): 
      errores = 0
      for xi, objetivo in zip(X, y): 
        salida = self._funcion_activacion(np.dot(self.w, xi))

        if salida != objetivo: 
          errores += 1
          self.w += self.tasa * (objetivo - salida) * xi
          self.hist_w.append(self.w.copy()) 
        
      print(f"Época {epoca+1:3d} | Errores: {errores:3d}")
      self.errores.append(errores)
      # Criterio de finalizacion
      if errores <= self.tolerancia: 
        break
      
  def predecir(self, X): 
     X = np.hstack([ -np.ones((X.shape[0], 1)), X ])
     output = np.dot(X, self.w)
     return self._funcion_activacion(output)
  
  def evaluar(self, X, y): 
    pred = self.predecir(X)

    # nos devuelve el porcentaje de aciertos 
    return np.mean(pred == y)
  
  @staticmethod
  def cargar_csv(ruta): 
    datos = pd.read_csv(ruta, header = None)
    # Obtenemos todas las columnas menos la ultima
    # con el .values lo convertimos en un array de Numpy
    X = datos.iloc[:, :-1].values
    # Solo obtenemos la ultima
    y = datos.iloc[:, -1].values

    return X, y