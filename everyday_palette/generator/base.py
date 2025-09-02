import os
import json
from abc import ABC, abstractmethod
import numpy as np

META_FILENAME = "meta.json"
GENERATOR_CLS_KEY = "generator_cls"
MODEL_CLS_KEY = "model_cls"
MODEL_PARAMS_KEY = "model_params"

class _BaseGenerator(ABC):

    def __init__(self, model_cls, model_params : dict):
        self._model = None
        self.model_cls = model_cls
        self.model_params = model_params

    @property
    def model(self):
        if self._model is None:
            raise ValueError("Model is not initialized")
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model

    def _get_meta(self) -> dict:
        return {
            GENERATOR_CLS_KEY: self.__class__.__name__,
            MODEL_CLS_KEY: self.model_cls.__name__,
            MODEL_PARAMS_KEY: self.model_params,
        }

    def _save_meta(self, path: str):
        meta = self._get_meta()
        with open(os.path.join(path, META_FILENAME), "w") as f:
            json.dump(meta, f)

    def _load_meta(self, path: str):
        meta_path = os.path.join(path, META_FILENAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found at {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta[GENERATOR_CLS_KEY] != self.__class__.__name__:
            raise ValueError(
                f"Generator class mismatch: expected {self.__class__.__name__}, got {meta[GENERATOR_CLS_KEY]}"
            )
        if meta[MODEL_CLS_KEY] != self.model_cls.__name__:
            raise ValueError(f"Model class mismatch: expected {self.model_cls.__name__}, got {meta[MODEL_CLS_KEY]}")
        self.model_params = meta.get(MODEL_PARAMS_KEY)


class BaseGenerator(_BaseGenerator, ABC):

    @abstractmethod
    def train(self, x : np.ndarray, resume : bool = False):
        pass

    @abstractmethod
    def sample(self, n : int = 1) -> np.ndarray:
        pass

    @abstractmethod
    def _save_model(self, path: str):
        pass

    @abstractmethod
    def _load_model(self, path: str):
        pass

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._save_meta(path)
        self._save_model(path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")
        self._load_meta(path)
        self._load_model(path)
