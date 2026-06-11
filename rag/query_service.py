import json
import os
import re
from langchain_ollama import OllamaEmbeddings
import requests
from waitress import profile
from rag.query_manual import search_manuals
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


# read from env or config in production
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# embeddings = OllamaEmbeddings(
#     model="nomic-embed-text"
# )

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

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

def call_ollama(
    prompt,
    model="llama3"
):

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

def parse_response(response_text):

    def _fallback():
        return {
            "severity": "UNKNOWN",
            "health_score": 0,
            "root_causes": [],
            "troubleshooting_steps": [],
            "corrective_actions": [],
            "preventive_actions": [],
            "maintenance_priority": "UNKNOWN",
            "summary": response_text
        }

    if not response_text:
        return _fallback()

    text = response_text.strip()

    # 1. Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. Try to extract a fenced ```json ... ``` or ``` ... ``` block
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    # 3. Fallback: grab the first {...} balanced-looking substring
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except Exception:
            pass

    return _fallback()

def build_context(docs):

    context = []

    for doc in docs:
        context.append(doc.page_content)

    return "\n\n".join(context)

def retrieve_documents(query, k=5):

    docs = vector_db.similarity_search(
        query,
        k=k
    )

    return docs

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

# genearte new method named generate_rag_qanda to take the plain text as input and return the question and answer based on the input text using the same approach as generate_rag_diagnostics but with a different prompt and response schema. 
# update the method without device profile and issues as input, only take the plain text as input and return the question and answer based on the input text using the same approach as generate_rag_diagnostics but with a different prompt and response schema. The response schema should be {"question": "", "answer": ""}. The prompt should instruct the model to generate a relevant question and answer based on the input text and the retrieved documents. The question should be something that a user might ask after reading the input text, and the answer should be a concise response to that question based on the information in the input text and the retrieved documents.

def generate_rag_qanda(
    text   
):

    query = f"""
    Text:
    {text}
    Generate a relevant question and answer based on the above text and any relevant information from the reference documents. The question should be something that a user might ask after reading the text, and the answer should be a concise response to that question based on the information in the text and the retrieved documents.
    """
    response_text = call_ollama(
        query,
        model="llama3"
    )

    result = parse_response(response_text)
    return result


