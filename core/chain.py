"""RAG orchestration for image question answering."""

import logging
from pathlib import Path

from image_io import ImageLoader
from models import Embedder, LLMModel, ScoutVisionModel
from storage import Retriever, VectorStore
from text import TextSplitter

logger = logging.getLogger(__name__)


def log_workflow(step: str, message: str) -> None:
    """Log a workflow step in a terminal-friendly format."""
    workflow_message = f"[WORKFLOW] {step} | {message}"
    print(workflow_message, flush=True)
    logger.info(workflow_message)


class RAGChain:
    """Indexes uploaded images and answers questions against their extracted content."""

    def __init__(self):
        log_workflow("INIT", "Creating image chatbot pipeline")
        self.image_loader = ImageLoader()
        log_workflow("INIT", "Image loader ready")
        self.vision_model = ScoutVisionModel()
        log_workflow("INIT", "Vision model ready")
        self.text_splitter = TextSplitter()
        log_workflow("INIT", "Text splitter ready")
        self.embedder = Embedder()
        log_workflow("INIT", "Embedding model ready")
        self.vector_store = VectorStore(self.embedder.embeddings)
        log_workflow("INIT", "Chroma vector store ready")
        self.retriever = Retriever(self.embedder.embeddings)
        log_workflow("INIT", "FAISS retriever ready")
        self.llm = LLMModel()
        log_workflow("INIT", "LLM ready")
        log_workflow("INIT", "RAG chain initialized")

    def index_images(self, image_paths):
        """Load images, extract their text/properties, and build vector indexes."""
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        log_workflow("INDEX START", f"Received {len(image_paths)} image(s)")
        extracted_sections = []
        for image_number, image_path in enumerate(image_paths, start=1):
            image_path = str(image_path)

            log_workflow(
                "STEP 1",
                f"Loading image {image_number}/{len(image_paths)}: {image_path}",
            )
            image = self.image_loader.load_image(image_path)

            log_workflow(
                "STEP 2",
                f"Extracting text and visual details from image {image_number}/{len(image_paths)}",
            )
            extracted_text = self.vision_model.extract_text(image)
            log_workflow(
                "STEP 2 DONE",
                f"Extracted {len(extracted_text)} characters from image {image_number}",
            )
            extracted_sections.append(f"Image: {image_path}\n{extracted_text}")

        log_workflow("STEP 3", "Combining extracted image context")
        combined_text = "\n\n".join(extracted_sections)

        log_workflow("STEP 4", "Splitting extracted context into chunks")
        chunks = self.text_splitter.split(combined_text)
        log_workflow("STEP 4 DONE", f"Created {len(chunks)} chunk(s)")

        log_workflow("STEP 5", "Embedding text chunks")
        self.embedder.embed(chunks)

        log_workflow("STEP 6", "Saving chunks to Chroma vector store")
        self.vector_store.add_texts(chunks)

        log_workflow("STEP 7", "Creating FAISS retrieval index")
        self.retriever.create_index(chunks)

        log_workflow("INDEX DONE", "Image indexing completed successfully")
        return chunks

    def ask(self, query: str) -> str:
        """Answer a query using the currently indexed image context."""
        log_workflow("QUERY START", f"User query: {query}")
        log_workflow("STEP 8", "Retrieving relevant chunks from FAISS")
        retrieved_docs = self.retriever.retrieve(query)
        log_workflow("STEP 8 DONE", f"Retrieved {len(retrieved_docs)} document chunk(s)")
        context = "\n".join(retrieved_docs)

        log_workflow("STEP 9", "Building prompt with retrieved context")
        prompt = (
            "Answer the query using the image context below. "
            "If the context is missing the answer, say what is missing.\n\n"
            f"Context:\n{context}\n\nQuery: {query}"
        )
        log_workflow("STEP 10", "Generating answer with LLM")
        response = self.llm.generate(prompt)
        log_workflow("QUERY DONE", f"Generated {len(response)} response characters")
        return response

    def process(self, image_paths, query: str) -> str:
        """Index one or more images, then answer the query."""
        log_workflow("PROCESS START", "Running full image-to-answer workflow")
        self.index_images(image_paths)
        response = self.ask(query)
        log_workflow("PROCESS DONE", "Full workflow completed")
        return response
