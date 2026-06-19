from pathlib import Path
import sys

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit import json
from vector_db import get_vector_db
import json

# Ensure the project root is on sys.path so sibling packages (database, ml, rca)
# are importable when running this script directly: `python consumer/telemetry_consumer.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def ingest(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print("Chunks created:", len(chunks))

    vector_db = get_vector_db()
    vector_db.add_documents(chunks)

    print("✔ Ingestion complete")
    print("DB count:", vector_db._collection.count())
    
# def ingest_pdf(pdf_path: str):
#     """
#     Load PDF → chunk → store in Chroma DB
#     """

#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )

#     chunks = splitter.split_documents(docs)

#       # 🔥 FIX: enforce metadata
#     for c in chunks:
#         if c.metadata is None:
#             c.metadata = {}

#         c.metadata.update({
#             "source": pdf_path
#         })

#     vector_db = get_vector_db()

#     print(f"✔ Ingesting {len(chunks)} chunks from {pdf_path} into Chroma DB...")



#     vector_db.add_documents(chunks)

#     print(f"✔ Ingested {len(chunks)} chunks from {pdf_path}")


# pdfs = [
#     "manuals/Breaker_Manual.pdf"
# ]

# for pdf in pdfs:
#     print(f"Processing {pdf}...")
#     ingest_pdf(pdf)

# client = chromadb.PersistentClient(path="./chroma_db")

# collections = client.list_collections()

# for c in collections:
#     print("Collection name:", c.name)

# client = chromadb.PersistentClient(path="./chroma_db")
# collection = client.get_collection("electrical_manuals")

# data = collection.get(limit=5)

# for i, meta in enumerate(data["metadatas"]):
#     print(f"\n--- Document {i} ---")
#     print(json.dumps(meta, indent=2))

ingest("manuals/Breaker_Manual.pdf")