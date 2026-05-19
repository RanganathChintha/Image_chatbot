"""Image loading utilities."""

import logging

from PIL import Image

from config import Config

logger = logging.getLogger(__name__)


class ImageLoader:
    """Loads image files from disk."""

    def __init__(self):
        self.supported_extensions = Config.IMAGE_EXTENSIONS

    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from a local path."""
        image = Image.open(image_path)
        logger.info("Image loaded successfully: %s", image_path)
        return image
