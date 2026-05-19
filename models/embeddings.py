"""Embedding model adapter."""

import logging

from langchain_huggingface import HuggingFaceEmbeddings

from config import Config

logger = logging.getLogger(__name__)


class Embedder:
    """Creates embeddings for text chunks."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDER_MODEL)

    def embed(self, texts: list[str]) -> list:
        """Embed text chunks."""
        embeddings = self.embeddings.embed_documents(texts)
        logger.info("Embedded %s text chunks", len(texts))
        return embeddings
