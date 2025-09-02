from PIL import Image
from io import BytesIO
from aiogram.types import BufferedInputFile
from aiogram import Bot

class Telegram():

    def __init__(self, api_token : str, channel_id : str):
        self.bot = Bot(token=api_token)
        self.channel_id = channel_id

    async def post(self, image : Image, text : str):
        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        photo = BufferedInputFile(buf.read(), filename="palette.png")

        await self.bot.send_photo(
            chat_id=self.channel_id,
            photo=photo,
            caption=text
        )