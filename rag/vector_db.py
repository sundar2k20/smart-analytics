import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# =====================================================
# CONFIG
# =====================================================

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "electrical_manuals"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "chroma_db"
)

print("📁 Chroma DB path:", CHROMA_PATH)

# =====================================================
# EMBEDDINGS (Ollama)
# =====================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =====================================================
# LOAD / CREATE VECTOR DB
# =====================================================

def get_vector_db():
    """
    Load existing Chroma DB or create a new one.
    """

    return Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )