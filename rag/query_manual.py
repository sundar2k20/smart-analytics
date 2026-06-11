from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

def search_manuals(question):

    docs = db.similarity_search(
        question,
        k=5
    )

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )