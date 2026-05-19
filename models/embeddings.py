"""Embedding model adapter."""

from langchain_huggingface import HuggingFaceEmbeddings

from config import Config


class Embedder:
    """Creates embeddings for text chunks."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDER_MODEL)

    def embed(self, texts: list[str]) -> list:
        """Embed text chunks."""
        return self.embeddings.embed_documents(texts)
