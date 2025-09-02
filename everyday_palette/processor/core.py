import joblib
from abc import ABC
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from .utils import palettes_to_numpy, numpy_to_palettes
from everyday_palette import Palette

class _Processor(ABC):

    def __init__(self, scaler : TransformerMixin):
        self.scaler = scaler
    
    def _fit(self, data : np.ndarray):
        self.scaler = self.scaler.fit(data)

    def _transform(self, data : np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)
    
    def _inverse_transform(self, data : np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    @classmethod
    def load(cls, path : str):
        scaler = joblib.load(path)
        return cls(scaler)

    def save(self, path : str):
        joblib.dump(self.scaler, path)


class Processor(_Processor):

    def fit(self, palettes : list[Palette], space : str):
        data = palettes_to_numpy(palettes, space)
        self._fit(data)

    def transform(self, palettes : list[Palette], space : str) -> np.ndarray:
        data = palettes_to_numpy(palettes, space)
        return self._transform(data)        

    def fit_transform(self, palettes : list[Palette], space : str) -> np.ndarray:
        data = palettes_to_numpy(palettes, space)
        self._fit(data)
        return self._transform(data)
    
    def inverse_transform(self, data : np.ndarray, space : str) -> list[Palette]:
        data = self._inverse_transform(data)
        return numpy_to_palettes(data, space)