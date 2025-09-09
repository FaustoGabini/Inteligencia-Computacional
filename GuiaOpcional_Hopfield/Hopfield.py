import numpy as np
from funciones import funcion_signo

class Hopfield: 
  def __init__(self): 
    self.W = None

  def entrenar(self, patrones): 
    N = patrones.shape[1]
    self.W = np.zeros((N, N))

    for patron in patrones:
      patron = np.reshape(patron, (1, N))
      self.W += np.dot(patron.T, patron)
    
    np.fill_diagonal(self.W, 0)
  
  def recuperar(self, x, max_iter=100): 
    if self.W is None:
      raise ValueError("La red no ha sido entrenada aún.")
     
    for _ in range(max_iter): 
      x_new = funcion_signo(np.dot(x, self.W))
      if np.array_equal(x, x_new):  
        break
      x = x_new
    return x
