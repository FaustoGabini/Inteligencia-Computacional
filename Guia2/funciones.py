import numpy as np

def sigmoidea(v):
    return ((2/(1+np.exp(-v)))-1)

def sigmoidea_derivada(y):
    return ((1+y)*(1-y))