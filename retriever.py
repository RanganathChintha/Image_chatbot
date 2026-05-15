# retriever.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from config import Config
import logging
import os

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDER_MODEL)
        self.faiss_index = None
        self.index_path = Config.FAISS_INDEX_PATH
    
    def create_index(self, texts: list):
        """Create FAISS index from texts"""
        try:
            self.faiss_index = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings
            )
            self.faiss_index.save_local(self.index_path)
            logger.info(f"FAISS index created with {len(texts)} texts")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            raise
    
    def load_index(self):
        """Load existing FAISS index"""
        try:
            if os.path.exists(self.index_path):
                self.faiss_index = FAISS.load_local(
                    self.index_path,
                    self.embeddings
                )
                logger.info("FAISS index loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
    
    def retrieve(self, query: str, k: int = None) -> list:
        """Retrieve relevant documents"""
        try:
            if k is None:
                k = Config.RETRIEVAL_K
            
            if self.faiss_index is None:
                logger.warning("FAISS index not initialized")
                return []
            
            docs = self.faiss_index.similarity_search(query, k=k)
            retrieved = [doc.page_content for doc in docs]
            logger.info(f"Retrieved {len(retrieved)} documents")
            return retrieved
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise