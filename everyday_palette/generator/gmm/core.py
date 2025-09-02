import os
import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from ..base import BaseGenerator

MODEL_FILENAME = 'model.joblib'

class GMMGenerator(BaseGenerator):
    
    def __init__(self, model_params : dict | None = None):
        super().__init__(GaussianMixture, model_params or {})

    def train(self, x : np.ndarray, resume : bool = False):
        if resume:
            raise NotImplementedError('Resuming train is not available yet')
        else:
            model = self.model_cls(**self.model_params)
        self.model = model.fit(x)

    def sample(self, n : int = 1) -> np.ndarray:
        samples, _ = self.model.sample(n)
        return samples
    
    def _save_model(self, path : str):
        joblib.dump(self.model, os.path.join(path, MODEL_FILENAME))

    def _load_model(self, path : str):
        model = joblib.load(os.path.join(path, MODEL_FILENAME))
        self.model = model
