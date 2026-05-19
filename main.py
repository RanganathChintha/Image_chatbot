import logging
from pathlib import Path

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def main():
    try:
        from config import Config

        image_paths = sorted(
            image_path
            for extension in Config.IMAGE_EXTENSIONS
            for image_path in Path(".").glob(f"*{extension}")
        )

        if not image_paths:
            logger.error("No image files found in the project folder")
            return 1

        from chain import RAGChain

        rag_chain = RAGChain()
        rag_chain.index_images(image_paths)

        print("Images indexed. Ask questions about them.")
        print("Type 'exit', 'quit', or 'q' to stop.")

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                break
            if not query:
                continue

            result = rag_chain.ask(query)
            print("\n" + "="*50)
            print("RAG Response:")
            print("="*50)
            print(result)
            print("="*50)

        return 0
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    raise SystemExit(main())
