# chain.py
from image_loader import ImageLoader
from vision_model import ScoutVisionModel
from text_splitter import TextSplitter
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from llm_model import LLMModel
import logging

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

class RAGChain:
    def __init__(self):
        self.image_loader = ImageLoader()
        self.vision_model = ScoutVisionModel()
        self.text_splitter = TextSplitter()
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.retriever = Retriever()
        self.llm = LLMModel()
        logger.info("RAG Chain initialized")
    
    def process(self, image_path: str, query: str) -> str:
        """Execute RAG chain: img_loader | scout_model | text_splitter | embedder | vector_store | retriever | llm"""
        
        logger.info("Starting RAG pipeline...")
        
        # Step 1: Load image
        logger.info("Step 1: Loading image")
        image = self.image_loader.load_image(image_path)
        
        # Step 2: Extract text with Scout model
        logger.info("Step 2: Extracting text with Scout model")
        extracted_text = self.vision_model.extract_text(image)
        
        # Step 3: Split text
        logger.info("Step 3: Splitting text")
        chunks = self.text_splitter.split(extracted_text)
        
        # Step 4: Embed chunks
        logger.info("Step 4: Embedding chunks")
        self.embedder.embed(chunks)
        
        # Step 5: Add to vector store
        logger.info("Step 5: Adding to vector store")
        self.vector_store.add_texts(chunks)
        
        # Step 6: Create retriever
        logger.info("Step 6: Creating FAISS retriever")
        self.retriever.create_index(chunks)
        
        # Step 7: Retrieve relevant documents
        logger.info("Step 7: Retrieving documents")
        retrieved_docs = self.retriever.retrieve(query)
        context = "\n".join(retrieved_docs)
        
        # Step 8: Generate with LLM
        logger.info("Step 8: Generating response with LLM")
        prompt = f"Context:\n{context}\n\nQuery: {query}"
        response = self.llm.generate(prompt)
        
        logger.info("RAG pipeline completed successfully")
        return response