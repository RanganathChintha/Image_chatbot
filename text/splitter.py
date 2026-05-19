"""Text splitting utilities."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config


class TextSplitter:
    """Splits extracted image content into retrievable chunks."""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

    def split(self, text: str) -> list[str]:
        """Split text into chunks."""
        return self.splitter.split_text(text)
