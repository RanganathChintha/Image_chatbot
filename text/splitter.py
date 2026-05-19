"""Text splitting utilities."""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config

logger = logging.getLogger(__name__)


class TextSplitter:
    """Splits extracted image content into retrievable chunks."""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

    def split(self, text: str) -> list[str]:
        """Split text into chunks."""
        chunks = self.splitter.split_text(text)
        logger.info("Text split into %s chunks", len(chunks))
        return chunks
