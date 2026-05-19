# image_loader.py
from PIL import Image
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class ImageLoader:
    def __init__(self):
        self.supported_extensions = Config.IMAGE_EXTENSIONS
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load image from path"""
        try:
            image = Image.open(image_path)
            logger.info(f"Image loaded successfully: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise