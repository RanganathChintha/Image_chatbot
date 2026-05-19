# vision_model.py
from groq import Groq
import base64
from io import BytesIO
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class ScoutVisionModel:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.VISION_MODEL
    
    def extract_text(self, image) -> str:
        """Extract text and properties from image"""
        try:
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
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            },
                            {
                                "type": "text",
                                "text": "Extract all text and describe properties from this image."
                            }
                        ]
                    }
                ]
            )
            logger.info("Text extracted successfully from image")
            return message.choices[0].message.content
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise