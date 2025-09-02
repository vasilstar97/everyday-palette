from .core import Palette
from PIL import Image, ImageDraw

def palette_to_png(palette : Palette, width : int = 1024, height : int = 1024) -> Image:
    
    colors = [c.convert('srgb') for c in palette.colors]
    n = len(colors)
    w = width // n
    
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    
    for i, color in enumerate(colors):
        x0 = i * w
        y0 = 0
        x1 = (i+1) * w
        y1 = height
        draw.rectangle([x0, y0, x1, y1], fill=color.to_string(hex=True))
    
    return img

def palettes_to_gif(palettes : list[Palette], width : int = 1024, height : int = 1024) -> Image:
    raise NotImplementedError('To be implemented in the future')