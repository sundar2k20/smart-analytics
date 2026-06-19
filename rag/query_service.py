import json
import os
import re
import requests

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from rag.reranker import create_reranker
from rag.vector_db import get_vector_db


# =====================================================
# Configuration
# =====================================================

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = get_vector_db()


retriever = None


# =====================================================
# Retriever Initialization (LAZY LOADING)
# =====================================================

def initialize_retriever():
    global retriever

    print("📦 Collection count:", vector_db._collection.count())

    data = vector_db.get()
    texts = data.get("documents", [])
    metas = data.get("metadatas", [])

    if not texts:
        raise ValueError("❌ Chroma DB is empty. Run ingestion first.")

    documents = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(texts, metas)
    ]

    # BM25 (keyword search)
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 15

    # Vector (semantic search)
    vector_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 40}
    )

    # Hybrid (70% vector + 30% keyword)
    hybrid = EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[0.7, 0.3]
    )

    # Reranker
    retriever = create_reranker(
        base_retriever=hybrid,
        top_n=5
    )

    print("✅ Retriever initialized successfully")


# =====================================================
# Retrieval
# =====================================================

def retrieve_documents(query: str):
    global retriever

    if retriever is None:
        initialize_retriever()

    return retriever.invoke(query)


# =====================================================
# Context Builder
# =====================================================

def build_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =====================================================
# Ollama Call
# =====================================================

def call_ollama(prompt: str, model: str = "llama3"):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["response"]


# =====================================================
# JSON Parser (robust)
# =====================================================

def parse_response(response_text):
    def fallback():
        return {
            "question": "",
            "answer": response_text
        }

    if not response_text:
        return fallback()

    text = response_text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass

    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except Exception:
            pass

    return fallback()


# =====================================================
# RAG Q&A
# =====================================================

def generate_rag_qanda(text: str):
    docs = retrieve_documents(text)
    context = build_context(docs)

    prompt = f"""
You are an industrial knowledge assistant.

INPUT TEXT:
{text}

REFERENCE DOCUMENTS:
{context}

Generate:
1. A relevant question a user might ask
2. A concise answer using BOTH input text and reference docs

Return ONLY valid JSON:

{{
    "question": "",
    "answer": ""
}}
"""

    response_text = call_ollama(prompt)
    return parse_response(response_text)

def generate_rag_diagnostics(
    telemetry,
    profile,
    issues,
    anomaly_score
):

    query = build_search_query(
        telemetry,
        profile,
        issues
    )

    docs = retrieve_documents(query)

    context = build_context(docs)

    prompt = build_prompt(
        telemetry,
        profile,
        issues,
        anomaly_score,
        context
    )

    response_text = call_ollama(
        prompt,
        model="llama3"
    )

    result = parse_response(
        response_text
    )

    return result

def build_prompt(
    telemetry,
    profile,
    issues,
    anomaly_score,
    context
):

    return f"""
    You are an industrial electrical diagnostics expert.

    DEVICE PROFILE

    Device ID:
    {profile['device_id']}

    Device Type:
    {profile['device_type']}

    Manufacturer:
    {profile['manufacturer']}

    Model:
    {profile['model_number']}

    RATED VALUES

    Rated Voltage:
    {profile['rated_voltage']}

    Rated Current:
    {profile['rated_current']}

    REAL-TIME TELEMETRY

    Voltage:
    {telemetry.get('voltage')}

    Current:
    {telemetry.get('current')}

    Frequency:
    {telemetry.get('frequency')}

    Power Factor:
    {telemetry.get('power_factor')}

    Temperature:
    {telemetry.get('temperature')}

    Humidity:
    {telemetry.get('humidity')}

    RULE VIOLATIONS

    {format_issues(issues)}

    ANOMALY SCORE

    {anomaly_score}

    REFERENCE DOCUMENTS

    {context}

    RESPONSE INSTRUCTIONS (STRICT):
    - Respond with a SINGLE valid JSON object ONLY.
    - Do NOT include any prose, greetings, explanations, or commentary.
    - Do NOT wrap the JSON in markdown code fences (no ``` or ```json).
    - Do NOT add trailing commentary after the closing brace.
    - Use double quotes for all keys and string values.
    - All array fields must be JSON arrays of strings (use [] if empty).
    - "health_score" must be a number between 0 and 100.
    - "severity" must be one of: "OK", "INFO", "WARNING", "CRITICAL".
    - "maintenance_priority" must be one of: "LOW", "MEDIUM", "HIGH", "URGENT".
    - The response MUST start with '{{' and end with '}}'.

    Respond using exactly this schema and key order:

    {{
        "severity": "",
        "health_score": 0,
        "root_causes": [],
        "troubleshooting_steps": [],
        "corrective_actions": [],
        "preventive_actions": [],
        "maintenance_priority": "",
        "summary": ""
    }}
    """


def build_search_query(
    telemetry,
    profile,
    issues
):

    query = f"""
    Device Type: {profile['device_type']}
    Manufacturer: {profile['manufacturer']}
    Model: {profile['model_number']}

    Detected Issues:
    {format_issues(issues)}

    Voltage: {telemetry.get('voltage')}
    Current: {telemetry.get('current')}
    Frequency: {telemetry.get('frequency')}
    Power Factor: {telemetry.get('power_factor')}
    Temperature: {telemetry.get('temperature')}
    Humidity: {telemetry.get('humidity')}
    """

    return query

def format_issues(issues):
    """Render rule-engine issues (list of dicts or strings) as a comma-separated string."""
    if not issues:
        return "None"

    parts = []
    for issue in issues:
        if isinstance(issue, dict):
            code = issue.get("code", "UNKNOWN")
            severity = issue.get("severity")
            parts.append(f"{code} ({severity})" if severity else str(code))
        else:
            parts.append(str(issue))

    return ", ".join(parts)