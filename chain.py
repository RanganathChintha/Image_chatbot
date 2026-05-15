# chain.py
from image_loader import ImageLoader
from vision_model import ScoutVisionModel
from text_splitter import TextSplitter
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from llm_model import LLMModel
import logging
from pathlib import Path

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

class RAGChain:
    def __init__(self):
        self.image_loader = ImageLoader()
        self.vision_model = ScoutVisionModel()
        self.text_splitter = TextSplitter()
        self.embedder = Embedder()
        self.vector_store = VectorStore(self.embedder.embeddings)
        self.retriever = Retriever(self.embedder.embeddings)
        self.llm = LLMModel()
        logger.info("RAG Chain initialized")
    
    def index_images(self, image_paths):
        """Load images, extract their text, and build the vector indexes once."""
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        logger.info("Indexing images...")
        
        extracted_sections = []
        for image_path in image_paths:
            image_path = str(image_path)

            # Step 1: Load image
            logger.info(f"Step 1: Loading image: {image_path}")
            image = self.image_loader.load_image(image_path)
            
            # Step 2: Extract text with Scout model
            logger.info(f"Step 2: Extracting text with Scout model: {image_path}")
            extracted_text = self.vision_model.extract_text(image)
            extracted_sections.append(f"Image: {image_path}\n{extracted_text}")

        combined_text = "\n\n".join(extracted_sections)
        
        # Step 3: Split text
        logger.info("Step 3: Splitting text")
        chunks = self.text_splitter.split(combined_text)
        
        # Step 4: Embed chunks
        logger.info("Step 4: Embedding chunks")
        self.embedder.embed(chunks)
        
        # Step 5: Add to vector store
        logger.info("Step 5: Adding to vector store")
        self.vector_store.add_texts(chunks)
        
        # Step 6: Create retriever
        logger.info("Step 6: Creating FAISS retriever")
        self.retriever.create_index(chunks)

        logger.info("Image indexing completed successfully")
    
    def ask(self, query: str) -> str:
        """Answer a query using the already indexed image context."""
        logger.info("Retrieving documents")
        retrieved_docs = self.retriever.retrieve(query)
        context = "\n".join(retrieved_docs)
        
        logger.info("Generating response with LLM")
        prompt = f"Context:\n{context}\n\nQuery: {query}"
        response = self.llm.generate(prompt)
        
        logger.info("Query completed successfully")
        return response

    def process(self, image_paths, query: str) -> str:
        """Execute RAG chain for one or more images."""
        
        logger.info("Starting RAG pipeline...")
        self.index_images(image_paths)
        
        # Step 7: Retrieve relevant documents
        logger.info("Step 7: Retrieving documents")
        response = self.ask(query)
        
        logger.info("RAG pipeline completed successfully")
        return response
