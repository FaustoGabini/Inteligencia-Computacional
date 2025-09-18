import numpy as np
class KMeans: 
  def __init__ (self, K, max_iter): 
    self.K = K
    self.max_iter = max_iter
  


  def inicializar_centroides(self, X): 
    idx = np.random.choice(X.shape[0], self.K, replace = False)
    self.centroides = X[idx]
  
  def calcular_centroide(self, patron): 
    distancias = np.linalg.norm(self.centroides - patron, axis=1)
    idx_min = np.argmin(distancias)
    return idx_min



  def fit(self, X): 
    self.labels = np.empty(len(X), dtype=int)
    self.inicializar_centroides(X)
    for iter in range(self.max_iter): 
      flag = False

      # Asigno a cada patron un centroide
      for i in range(len(X)):
        centroide = self.calcular_centroide(X[i])
        if self.labels[i] != centroide:
          flag = True 
          self.labels[i] = centroide

      # Recalculo nuevamente los centroides
      for k in range(self.K):
          puntos_del_cluster = X[self.labels == k]
          if len(puntos_del_cluster) > 0: 
            self.centroides[k] = np.mean(puntos_del_cluster, axis=0)

      if flag == False: # No hubo reasignaciones
        print(f"No hubo reasignaciones, el entramiento finaliza en la epoca: {iter}")
        break; 
         

  

