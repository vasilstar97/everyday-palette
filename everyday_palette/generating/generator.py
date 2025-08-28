import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from everyday_palette import Palette

class Generator():

    def __init__(self, gmm : GaussianMixture | None = None):
        self._gmm = gmm

    @property
    def gmm(self) -> GaussianMixture:
        if self._gmm is None:
            raise ValueError('No GMM is set. Load or fit first.')
        return self._gmm
        
    def _fit(self, data : np.ndarray, k, params) -> tuple[GaussianMixture, float]:
        gmm = GaussianMixture(k, **params)
        gmm.fit(data)
        bic = gmm.bic(data)
        return gmm,bic

    def fit(self, data : np.ndarray, max_k : int = 20, params : dict | None = None):
        params = params or {'covariance_type':'full', 'n_init':5}
        best_gmm, best_bic = None, np.inf
        for k in range(1,max_k+1):
            gmm,bic = self._fit(data, k, params)
            if bic < best_bic:
                best_gmm, best_bic = gmm, bic
        self._gmm = best_gmm

    def sample(self, n : int = 10):
        samples, _ = self.gmm.sample(n)
        return samples.clip(0,1)

    @classmethod
    def load(cls, path : str):
        gmm = joblib.load(path)
        return cls(gmm)

    def save(self, path : str):
        joblib.dump(self.gmm, path)

    