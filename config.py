# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Model Names
    VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    EMBEDDER_MODEL = "all-minilm-l6-v2"
    LLM_MODEL = "openai/gpt-oss-120b"
    
    # Text Splitter Config
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Vector Store Config
    PERSIST_DIRECTORY = "./chroma_db"
    FAISS_INDEX_PATH = "./faiss_index"
    
    # Retriever Config
    RETRIEVAL_K = 4
    
    # Image Loader Config
    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
    
    # Logging
    LOG_LEVEL = "INFO"