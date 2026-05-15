# embedder.py
from langchain.embeddings import HuggingFaceEmbeddings
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDER_MODEL
        )
    
    def embed(self, texts: list) -> list:
        """Embed text chunks"""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Embedded {len(texts)} text chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise