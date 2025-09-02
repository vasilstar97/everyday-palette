import numpy as np
from coloraide import Color

class Palette():

    sort_space = 'oklab'

    def __init__(self, colors : list[Color]):
        for c in colors:
            if not isinstance(c, Color):
                raise TypeError('Color must be an instance of coloraide.Color')
        self._colors = [c.convert('srgb') for c in colors]
    
    @property
    def colors(self) -> list[Color]:
        colors = [c.convert(self.sort_space) for c in self._colors]
        start = min(colors, key=lambda c : np.linalg.norm(c.coords()))
        order = [start]
        while len(order)<len(colors):
            prev = order[-1]
            left_colors = [c for c in colors if c not in order]
            next = min(left_colors, key=lambda c : prev.distance(c))
            order.append(next)
        return order
    
    def to_numpy(self, space : str) -> np.ndarray:
        arr = np.array([c.convert(space).coords() for c in self.colors], dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        return arr.flatten()
    
    @classmethod
    def from_numpy(cls, arr : np.ndarray, space : str) -> np.ndarray:
        arr = np.array(arr, dtype=float)
        arr = arr.reshape(-1, 3)
        colors = [Color(space, coords) for coords in arr]
        return cls(colors)
        
    @classmethod
    def from_hex_list(cls, hex_list) -> "Palette":
        colors = []

        for hex in hex_list:
            if not '#' in hex:
                hex = '#' + hex
            color = Color(hex)
            colors.append(color)

        return cls(colors)