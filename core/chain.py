"""RAG orchestration for image question answering."""

import logging
from pathlib import Path

from image_io import ImageLoader
from models import Embedder, LLMModel, ScoutVisionModel
from storage import Retriever, VectorStore
from text import TextSplitter

logger = logging.getLogger(__name__)


class RAGChain:
    """Indexes uploaded images and answers questions against their extracted content."""

    def __init__(self):
        self.image_loader = ImageLoader()
        self.vision_model = ScoutVisionModel()
        self.text_splitter = TextSplitter()
        self.embedder = Embedder()
        self.vector_store = VectorStore(self.embedder.embeddings)
        self.retriever = Retriever(self.embedder.embeddings)
        self.llm = LLMModel()
        logger.info("RAG chain initialized")

    def index_images(self, image_paths):
        """Load images, extract their text/properties, and build vector indexes."""
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        extracted_sections = []
        for image_path in image_paths:
            image_path = str(image_path)

            logger.info("Loading image: %s", image_path)
            image = self.image_loader.load_image(image_path)

            logger.info("Extracting image content: %s", image_path)
            extracted_text = self.vision_model.extract_text(image)
            extracted_sections.append(f"Image: {image_path}\n{extracted_text}")

        combined_text = "\n\n".join(extracted_sections)
        chunks = self.text_splitter.split(combined_text)

        self.embedder.embed(chunks)
        self.vector_store.add_texts(chunks)
        self.retriever.create_index(chunks)

        logger.info("Image indexing completed successfully")
        return chunks

    def ask(self, query: str) -> str:
        """Answer a query using the currently indexed image context."""
        retrieved_docs = self.retriever.retrieve(query)
        context = "\n".join(retrieved_docs)

        prompt = (
            "Answer the query using the image context below. "
            "If the context is missing the answer, say what is missing.\n\n"
            f"Context:\n{context}\n\nQuery: {query}"
        )
        return self.llm.generate(prompt)

    def process(self, image_paths, query: str) -> str:
        """Index one or more images, then answer the query."""
        self.index_images(image_paths)
        return self.ask(query)
