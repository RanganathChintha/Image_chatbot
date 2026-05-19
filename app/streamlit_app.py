"""Streamlit interface for the image chatbot."""

import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config


logging.basicConfig(
    level=Config.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("streamlit").setLevel(logging.WARNING)


def save_uploaded_images(uploaded_files) -> list[Path]:
    """Persist uploaded images to temporary files for the RAG pipeline."""
    saved_paths = []
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower() or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            saved_paths.append(Path(temp_file.name))
        logger.info("Saved uploaded image '%s' to '%s'", uploaded_file.name, saved_paths[-1])
    return saved_paths


@st.cache_resource(show_spinner=False)
def get_rag_chain():
    """Create the RAG chain once per Streamlit session."""
    from core import RAGChain

    logger.info("Creating cached RAG chain")
    return RAGChain()


def initialize_session_state() -> None:
    """Initialize Streamlit session defaults."""
    st.session_state.setdefault("indexed", False)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("image_count", 0)


def render_sidebar() -> None:
    """Render app settings and status."""
    with st.sidebar:
        st.header("Settings")
        st.caption(f"Vision: {Config.VISION_MODEL}")
        st.caption(f"LLM: {Config.LLM_MODEL}")
        st.caption(f"Retriever K: {Config.RETRIEVAL_K}")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []

        if st.button("Reset indexed images", use_container_width=True):
            st.session_state.indexed = False
            st.session_state.image_count = 0
            st.session_state.messages = []


def render_chat_history() -> None:
    """Render existing chat messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    """Run the Streamlit app."""
    logger.info("Rendering Streamlit app")
    st.set_page_config(page_title="Image Chatbot", layout="wide")
    initialize_session_state()
    render_sidebar()

    st.title("Image Chatbot")
    st.write("Upload one or more images, index them, then ask questions about them.")

    uploaded_files = st.file_uploader(
        "Images",
        type=[extension.lstrip(".") for extension in Config.IMAGE_EXTENSIONS],
        accept_multiple_files=True,
    )

    if uploaded_files:
        preview_columns = st.columns(min(3, len(uploaded_files)))
        for index, uploaded_file in enumerate(uploaded_files):
            with preview_columns[index % len(preview_columns)]:
                st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    can_index = bool(uploaded_files)
    if st.button("Index images", type="primary", disabled=not can_index):
        try:
            logger.info("Index images clicked with %s uploaded file(s)", len(uploaded_files))
            with st.spinner("Reading images and building the retrieval index..."):
                image_paths = save_uploaded_images(uploaded_files)
                get_rag_chain().index_images(image_paths)

            st.session_state.indexed = True
            st.session_state.image_count = len(image_paths)
            logger.info("Indexed %s image(s) successfully", len(image_paths))
            st.success(f"Indexed {len(image_paths)} image(s).")
        except Exception as exc:
            st.session_state.indexed = False
            logger.exception("Could not index images")
            st.error(f"Could not index images: {exc}")

    if st.session_state.indexed:
        st.info(f"Ready to answer questions about {st.session_state.image_count} image(s).")
    else:
        st.warning("Upload and index images before asking a question.")

    render_chat_history()

    query = st.chat_input("Ask a question about the uploaded images")
    if query:
        logger.info("Received user query: %s", query)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        if not st.session_state.indexed:
            answer = "Please upload and index at least one image first."
        else:
            try:
                with st.spinner("Thinking through the image context..."):
                    answer = get_rag_chain().ask(query)
                logger.info("Generated answer for user query")
            except Exception as exc:
                logger.exception("Could not generate answer")
                answer = f"Could not generate an answer: {exc}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()
