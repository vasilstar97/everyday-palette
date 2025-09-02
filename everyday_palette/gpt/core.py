from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.auth import APIKeyAuth
from everyday_palette import Palette

class YandexGPT():

    def __init__(self, folder_id : str, api_token : str, instruction : list[str]):
        sdk = YCloudML(folder_id=folder_id, auth=APIKeyAuth(api_token))
        model = sdk.models.completions('yandexgpt')
        self.sdk = sdk
        self.model = model
        self._instruction = instruction.copy()
    
    @property
    def instruction(self) -> str:
        return str.join('\n', self._instruction)
    
    def prompt(self, palette : Palette, temperature : float = 1) -> str:
        model = self.model.configure(temperature=temperature)
        result = model.run([
            {'role': 'system', 'text': self.instruction},
            {'role': 'user', 'text': str.join(', ', [c.convert('srgb').to_string(hex=True) for c in palette.colors])}
        ])
        return result[0].text