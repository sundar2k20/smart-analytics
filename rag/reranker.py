from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker


def create_reranker(base_retriever, top_n=5):
    """
    Create a cross-encoder reranking retriever.
    """

    model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base"

         #model_name=r"C:\Users\HCA2COB\.cache\huggingface\hub\models--BAAI--bge-reranker-base"
    )

    reranker = CrossEncoderReranker(
        model=model,
        top_n=top_n
    )

    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker
    )