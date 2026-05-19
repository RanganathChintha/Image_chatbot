"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for the image chatbot."""

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")

    VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    EMBEDDER_MODEL = "all-minilm-l6-v2"
    LLM_MODEL = "openai/gpt-oss-120b"

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    PERSIST_DIRECTORY = "./chroma_db"
    FAISS_INDEX_PATH = "./faiss_index"

    RETRIEVAL_K = 4

    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".bmp"]

    LOG_LEVEL = "INFO"
