"""Chroma vector store adapter."""

import logging

from langchain_community.vectorstores import Chroma

from config import Config

logger = logging.getLogger(__name__)


class VectorStore:
    """Persists text chunks in Chroma."""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.persist_directory = Config.PERSIST_DIRECTORY
        self.db = None

    def add_texts(self, texts: list[str]) -> Chroma:
        """Add texts to the Chroma vector store."""
        self.db = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        logger.info("Added %s texts to Chroma vector store", len(texts))
        return self.db

    def get_db(self) -> Chroma:
        """Get or load the Chroma vector store."""
        if self.db is None:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        logger.info("Retrieved Chroma vector store")
        return self.db
