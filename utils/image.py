import numpy as np

def crop(I):
    '''supprime les 20 px ajoutés '''

    return(I[10:-10,10:-10])

def norm(X):
    return((X-np.min(X))/(np.max(X)-np.min(X)))
