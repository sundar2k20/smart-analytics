from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import sys
from pathlib import Path


# Ensure the project root is on sys.path so sibling packages (e.g. `producer`)
# are importable when running this script directly: `python gateway/modbus_reader.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_pdf(path: str):
    """Read a PDF and return one Document per page."""
    reader = PdfReader(path)
    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        docs.append(Document(
            page_content=text,
            metadata={"source": path, "page": page_num},
        ))
    return docs


documents = []

pdfs = [
    "manuals/Breaker_Manual.pdf"
]

for pdf in pdfs:

    print(f"Processing {pdf}...")
    documents.extend(load_pdf(pdf))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(
    documents
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

print("Knowledge base created")