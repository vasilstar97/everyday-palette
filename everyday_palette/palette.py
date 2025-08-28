import itertools
import networkx as nx
from colormath.color_objects import LCHabColor, sRGBColor, LabColor
from colormath.color_conversions import convert_color 
from PIL import Image, ImageDraw
from .utils import delta_e_cie2000


TARGET_ILLIMINANT_KEY = 'target_illuminant'

class Palette(nx.Graph):

    def tsp(self, weight : str | None = 'cie2000') -> list[str]:
        nodes = list(self.nodes)
        
        if weight is None:
            return nodes

        best_path, best_len = None, float("inf")

        for perm in itertools.permutations(nodes):
            length = 0.0
            for i in range(len(perm) - 1):
                length += self[perm[i]][perm[i+1]][weight]
            if length < best_len:
                best_len = length
                best_path = perm
        
        return list(best_path)
    
    def is_above_threshold(self, threshold : float = 5.0) -> bool:
        for _,_,d in self.edges(data=True):
            weight = d['cie2000']
            if weight < threshold:
                return False
        return True

    def add_node(self, srgb_color : sRGBColor):
        target_illuminant = self.graph[TARGET_ILLIMINANT_KEY]
        node_a = srgb_color.get_rgb_hex()
        attr = {
            'srgb': srgb_color,
            'lab': convert_color(srgb_color, LabColor, target_illuminant=target_illuminant),
            'lchab': convert_color(srgb_color, LCHabColor, target_illuminant=target_illuminant)
        }
        super().add_node(node_a, **attr)
        for node_b in self.nodes:
            if node_a == node_b:
                continue
            self.add_edge(node_a, node_b)

    def add_edge(self, node_a, node_b):
        lab_a = self.nodes[node_a]['lab']
        lab_b = self.nodes[node_b]['lab']
        attr = {
            'cie2000': delta_e_cie2000(lab_a, lab_b)
        }
        super().add_edge(node_a, node_b, **attr)

    def to_img(self, size : int = 1024, margin : int = 0) -> Image:
        
        colors = self.tsp()
        n = len(colors)
        cell_w = size // n
        
        img = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(img)
        
        for i, color in enumerate(colors):
            x0 = i * cell_w + margin
            y0 = margin
            x1 = (i+1) * cell_w - margin
            y1 = size - margin
            draw.rectangle([x0, y0, x1, y1], fill=color)
        
        return img
        
    @classmethod
    def from_srgb_colors(cls, srgb_colors : list[sRGBColor], target_illuminant : str = 'd65') -> "Palette":
        graph = cls()
        graph.graph[TARGET_ILLIMINANT_KEY] = target_illuminant

        for srgb_color in srgb_colors:
            r = srgb_color.rgb_r
            g = srgb_color.rgb_g
            b = srgb_color.rgb_b
            srgb_color = sRGBColor(min(r, 1), min(g, 1), min(b, 1))
            graph.add_node(srgb_color)

        return graph