"""Image loading utilities."""

from PIL import Image

from config import Config


class ImageLoader:
    """Loads image files from disk."""

    def __init__(self):
        self.supported_extensions = Config.IMAGE_EXTENSIONS

    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from a local path."""
        return Image.open(image_path)
