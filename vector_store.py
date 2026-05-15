# vector_store.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDER_MODEL)
        self.persist_directory = Config.PERSIST_DIRECTORY
        self.db = None
    
    def add_texts(self, texts: list) -> Chroma:
        """Add texts to Chroma vector store"""
        try:
            self.db = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info(f"Added {len(texts)} texts to Chroma vector store")
            return self.db
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            raise
    
    def get_db(self) -> Chroma:
        """Get vector store"""
        try:
            if self.db is None:
                self.db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            logger.info("Retrieved Chroma vector store")
            return self.db
        except Exception as e:
            logger.error(f"Error retrieving vector store: {e}")
            raise