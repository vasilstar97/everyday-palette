import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from everyday_palette import Palette
from .pre import palettes_to_numpy
from .post import numpy_to_palettes

class Processor():

    def __init__(self, scaler = None):
        self._scaler = scaler

    @property
    def scaler(self) -> MinMaxScaler:
        if self._scaler is None:
            raise ValueError('No scaler is set. Load or fit first.')
        return self._scaler
        
    def _fit(self, data : np.ndarray):
        scaler = MinMaxScaler()
        self._scaler = scaler.fit(data)

    def _transform(self, data : np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)
    
    def _inverse_transform(self, data : np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(data)

    def fit(self, palettes : list[Palette]):
        data = palettes_to_numpy(palettes)
        self._fit(data)

    def transform(self, palettes : list[Palette]):
        data = palettes_to_numpy(palettes)
        return self._transform(data)        

    def fit_transform(self, palettes : list[Palette]):
        data = palettes_to_numpy(palettes)
        self._fit(data)
        return self._transform(data)
    
    def inverse_transform(self, data : np.ndarray) -> list[Palette]:
        data = self._inverse_transform(data)
        return numpy_to_palettes(data)

    @classmethod
    def load(cls, path : str):
        scaler = joblib.load(path)
        return cls(scaler)

    def save(self, path : str):
        joblib.dump(self.scaler, path)

    