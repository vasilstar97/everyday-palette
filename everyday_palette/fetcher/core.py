import requests
from colormath.color_objects import sRGBColor

def _code_to_colors(code : str) -> list[sRGBColor]:
    return [sRGBColor.new_from_rgb_hex(code[i:i+6]) for i in range(0, len(code), 6)]

class Fetcher():

    def __init__(self, endpoint : str = 'https://colorhunt.co/php/feed.php'):
        self.endpoint = endpoint

    def _fetch_palettes(self, step : int, **data) -> list[list[sRGBColor]]:
        res = requests.post(self.endpoint, data={'step': step, **data})
        res_json = res.json()
        return [_code_to_colors(entry['code']) for entry in res_json]
    
    def fetch_palettes(self, sort : str = 'new', timeframe : int = 30, tags = None, **data) -> list[list[sRGBColor]]:
        result = []
        page = 0
        while True:
            palettes = self._fetch_palettes(page, sort=sort, timeframe=timeframe, tags=tags, **data)
            if len(palettes) == 0:
                break
            else:
                result.extend(palettes)
            page += 1
        return result

