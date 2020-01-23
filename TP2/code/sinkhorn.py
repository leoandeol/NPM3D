import numpy as np
from tqdm import tqdm

class Sinkhorn(object):

    def __init__(self):
        pass
    
    def __call__(self, C, lbd, n_iter):
        return self.iterations(C, lbd, n_iter)
    
    def iterations(self, C, lbd, n_iter, verbose=False):
        """
        lbd = 1/eps, regularization
        """
        n,m = C.shape

        K = np.exp(-1*lbd*C)
        u = np.ones(n)
        v = np.zeros(m)

        iterator = range(n_iter)
        if verbose:
            iterator = tqdm(iterator)

        for k in iterator:
            v = (np.ones(m)/m) / (K.T@u)
            u = (np.ones(n)/n) / (K@v)

        # Compute Transport Matrix
        return np.diag(u)@K@np.diag(v)