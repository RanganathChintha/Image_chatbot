"""Vision model adapter."""

import base64
from io import BytesIO

from groq import Groq

from config import Config


class ScoutVisionModel:
    """Extracts readable content and visual properties from images."""

    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.VISION_MODEL

    def extract_text(self, image) -> str:
        """Extract text and visual properties from an image."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        message = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract all visible text and describe the important "
                                "objects, layout, colors, and properties in this image."
                            ),
                        },
                    ],
                }
            ],
        )
        return message.choices[0].message.content
