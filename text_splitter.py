# text_splitter.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class TextSplitter:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
    
    def split(self, text: str) -> list:
        """Split text into chunks"""
        try:
            chunks = self.splitter.split_text(text)
            logger.info(f"Text split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            raise