import numpy as np
from colormath.color_objects import LCHabColor, sRGBColor
from colormath.color_conversions import convert_color
from everyday_palette import Palette

def _numpy_to_color(arr : np.ndarray) -> LCHabColor:
    L = float(arr[0])
    C = float(arr[1])
    H = float(arr[2])
    return LCHabColor(L, C, H)

def _numpy_to_palette(arr : np.ndarray) -> Palette:
    colors = [_numpy_to_color(arr[i:i+3]) for i in range(0, 12, 3)]
    colors = [convert_color(c,sRGBColor) for c in colors]
    return Palette.from_srgb_colors(colors)

def numpy_to_palettes(arr : np.ndarray) -> list[Palette]:
    return [_numpy_to_palette(row) for row in arr]