# main.py
import os
from chain import RAGChain
import logging

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

def main():
    try:
        rag_chain = RAGChain()
        
        image_path = "path/to/your/image.png"
        query = "What information is in this image?"
        
        result = rag_chain.process(image_path, query)
        print("\n" + "="*50)
        print("RAG Response:")
        print("="*50)
        print(result)
        print("="*50)
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()