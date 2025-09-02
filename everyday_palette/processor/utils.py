import numpy as np
from everyday_palette import Palette

def palettes_to_numpy(palettes : list[Palette], space : str) -> np.ndarray:
    return np.array([p.to_numpy(space) for p in palettes])

def numpy_to_palettes(arr : np.ndarray, space : str) -> list[Palette]:
    return [Palette.from_numpy(a, space) for a in arr]