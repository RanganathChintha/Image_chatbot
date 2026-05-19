"""FAISS-based retrieval."""

import logging
import os

from langchain_community.vectorstores import FAISS

from config import Config

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant chunks for a user query."""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.faiss_index = None
        self.index_path = Config.FAISS_INDEX_PATH

    def create_index(self, texts: list[str]) -> None:
        """Create a FAISS index from text chunks."""
        self.faiss_index = FAISS.from_texts(texts=texts, embedding=self.embeddings)
        self.faiss_index.save_local(self.index_path)
        logger.info("FAISS index created with %s texts", len(texts))

    def load_index(self) -> None:
        """Load an existing FAISS index if present."""
        if os.path.exists(self.index_path):
            self.faiss_index = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded successfully")

    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        """Retrieve relevant documents."""
        if k is None:
            k = Config.RETRIEVAL_K

        if self.faiss_index is None:
            logger.warning("FAISS index is not initialized")
            return []

        docs = self.faiss_index.similarity_search(query, k=k)
        retrieved = [doc.page_content for doc in docs]
        logger.info("Retrieved %s documents", len(retrieved))
        return retrieved
