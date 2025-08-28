import numpy as np
from everyday_palette import Palette

def _color_to_numpy(lchab_color) -> np.ndarray:
    return np.array([
        lchab_color.lch_l,
        lchab_color.lch_c,
        lchab_color.lch_h
    ], dtype=np.float32)

def _palette_to_numpy(palette : Palette) -> np.ndarray:
    return np.concatenate([_color_to_numpy(palette.nodes[n]['lchab']) for n in palette.tsp()])

def palettes_to_numpy(palettes : list[Palette]) -> np.ndarray:
    return np.stack([_palette_to_numpy(p) for p in palettes], axis=0)