"""FAISS-based retrieval."""

import os

from langchain_community.vectorstores import FAISS

from config import Config


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

    def load_index(self) -> None:
        """Load an existing FAISS index if present."""
        if os.path.exists(self.index_path):
            self.faiss_index = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        """Retrieve relevant documents."""
        if k is None:
            k = Config.RETRIEVAL_K

        if self.faiss_index is None:
            return []

        docs = self.faiss_index.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
