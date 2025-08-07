"""
RAG Pipeline using LangGraph, LangChain, OpenAI Embeddings, and Qdrant.
This file is designed for clarity, modularity, and extensibility.
Brainstorm/TODO comments are included for key decision points.
"""

import os
from dotenv import load_dotenv
import requests
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, OpenAI  # Updated imports
from langgraph.graph import StateGraph, END, START
import pdfplumber
from pydantic import BaseModel
from collections import defaultdict
import operator

# Load environment variables from .env file
load_dotenv()

DOCUMENTS_DIR = "documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Get API keys and Qdrant config from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Ensure QDRANT_HOST uses http:// for local development
if not QDRANT_HOST.startswith("http://") and not QDRANT_HOST.startswith("https://"):
    QDRANT_HOST = f"http://{QDRANT_HOST}"

# --- State Schema for LangGraph ---
class RagIngestState(BaseModel):
    load: Optional[List[str]] = None  # document URLs
    chunk: Optional[List[str]] = None  # list of document texts
    embed: Optional[List[Dict[str, Any]]] = None  # list of chunk dicts

class RagQueryState(BaseModel):
    # Remove 'question' from the state to avoid concurrency issues
    translated_queries: Optional[List[str]] = None  # 5 diverse queries
    translation_contexts: Optional[List[Dict[str, Any]]] = None  # Chunks from translation queries
    hyde_context: Optional[List[Dict[str, Any]]] = None  # Chunks from HYDE
    merged_context: Optional[List[Dict[str, Any]]] = None  # Final context for generation
    answer: Optional[Any] = None  # answer from LLM

# --- 1. Document Downloading & Preprocessing ---
def download_document(url: str, save_dir: str = DOCUMENTS_DIR) -> str:
    local_filename = os.path.join(save_dir, url.split("?")[0].split("/")[-1])
    response = requests.get(url)
    with open(local_filename, "wb") as f:
        f.write(response.content)
    return local_filename

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def load_and_preprocess_documents(urls: List[str]) -> List[str]:
    texts = []
    for url in urls:
        file_path = download_document(url)
        if file_path.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            text = ""
        texts.append(text)
    return texts

def chunk_documents(texts: List[str]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for idx, text in enumerate(texts):
        for chunk in splitter.split_text(text):
            chunks.append({"text": chunk, "doc_index": idx})
    return chunks

def embed_and_store(chunks: List[Dict[str, Any]], collection_name: str = "docs") -> Qdrant:
    # Use OpenAI's text-embedding-3-large model for embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-large"
    )
    texts = [c["text"] for c in chunks]
    metadatas = [{"doc_index": c["doc_index"]} for c in chunks]
    vectordb = Qdrant.from_texts(
        texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        location=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
    )
    return vectordb

def retrieve_context(vectordb: Qdrant, query: str, k: int = 5) -> List[Dict[str, Any]]:
    docs = vectordb.similarity_search(query, k=k)
    return [{"text": doc.page_content, "metadata": doc.metadata, "score": getattr(doc, 'score', None)} for doc in docs]

def generate_answer(context: List[Dict[str, Any]], query: str) -> str:
    # Use OpenAI's GPT-4o-mini for answer generation
    llm = OpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o-mini"
    )
    context_str = "\n\n".join([c["text"] for c in context])
    prompt = (
        f"Answer the following question using ONLY the provided context. "
        f"Provide rationale and cite supporting text.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return llm.invoke(prompt)

def format_output(answer: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "answer": answer,
        "supporting_clauses": [
            {
                "text": c["text"],
                "doc_index": c["metadata"].get("doc_index"),
                "score": c.get("score"),
            }
            for c in context
        ],
        "rationale": "See supporting clauses and cited text for reasoning."
    }

# --- Reciprocal Rank Fusion (RRF) Algorithm ---
def reciprocal_rank_fusion(*list_of_list_ranks_system, K=60):
    """
    Fuse rank from multiple IR systems using Reciprocal Rank Fusion.
    Args:
    * list_of_list_ranks_system: Ranked results from different IR system.
    K (int): A constant used in the RRF formula (default is 60).
    Returns:
    Tuple of list of sorted documents by score and sorted documents
    """
    rrf_map = defaultdict(float)
    for rank_list in list_of_list_ranks_system:
        for rank, item in enumerate(rank_list, 1):
            # Use a tuple of (text, doc_index) as a unique key for deduplication
            key = (item['text'], str(item['metadata'].get('doc_index', '')))
            rrf_map[key] += 1 / (rank + K)
    # Map back to original chunk objects
    chunk_lookup = { (item['text'], str(item['metadata'].get('doc_index', ''))): item for rank_list in list_of_list_ranks_system for item in rank_list }
    sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
    sorted_chunks = [chunk_lookup[key] for key, score in sorted_items]
    return sorted_items, sorted_chunks

# --- Production-Grade Prompts ---
QUERY_TRANSLATION_PROMPT = (
    """
    You are an expert legal/insurance/HR assistant. Given the following user question, generate 5 diverse queries that:
    - Rephrase the original question in different ways
    - Widen the scope to cover related aspects
    - Break down compound or multi-part questions into focused sub-questions
    - Use different perspectives or approaches to maximize information retrieval
    Each query should be clear, information-seeking, and non-redundant. Return the 5 queries as a numbered list.
    
    User question: {question}
    """
)

HYDE_PROMPT = (
    """
    You are a world-class legal/insurance/HR expert. Given the following user question, generate a concise, fact-rich, and highly relevant hypothetical answer as if you had access to all authoritative documents. Your answer should be clear, actionable, and maximize the value for a professional user. Avoid speculation and focus on likely, useful information.
    
    User question: {question}
    """
)

# --- Ingestion Pipeline Nodes ---
def ingest_node_load(state: RagIngestState) -> RagIngestState:
    texts = load_and_preprocess_documents(state.load)
    return state.copy(update={"chunk": texts})

def ingest_node_chunk(state: RagIngestState) -> RagIngestState:
    chunks = chunk_documents(state.chunk)
    return state.copy(update={"embed": chunks})

def ingest_node_embed(state: RagIngestState, collection_name: str = "docs") -> RagIngestState:
    embed_and_store(state.embed, collection_name=collection_name)
    return state  # No update needed, as storage is side-effect

# --- Ingestion Pipeline Graph ---
def build_ingest_graph(collection_name: str = "docs"):
    graph = StateGraph(RagIngestState)
    graph.add_node("load", ingest_node_load)
    graph.add_node("chunk", ingest_node_chunk)
    # Use a lambda to pass collection_name
    graph.add_node("embed", lambda state: ingest_node_embed(state, collection_name=collection_name))
    graph.add_edge(START, "load")
    graph.add_edge("load", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", END)
    return graph.compile()

# --- Query Pipeline Nodes ---
def query_node_translate(state: RagQueryState, question, small_llm, vectordb, n: int = 5, k: int = 5) -> RagQueryState:
    prompt = QUERY_TRANSLATION_PROMPT.format(question=question)
    llm_output = small_llm.invoke(prompt)
    translated_queries = [q.strip(" .-") for q in llm_output.split("\n") if q.strip()][:n]
    all_chunks_per_query = []
    for tq in translated_queries:
        chunks = retrieve_context(vectordb, tq, k=k)
        for chunk in chunks:
            chunk['source_query'] = tq
        all_chunks_per_query.append(chunks)
    _, ranked_chunks = reciprocal_rank_fusion(*all_chunks_per_query, K=60)
    ranked_chunks = ranked_chunks[:k*n]
    return state.copy(update={
        "translated_queries": translated_queries,
        "translation_contexts": ranked_chunks
    })

def query_node_hyde(state: RagQueryState, question, gpt4_llm) -> RagQueryState:
    prompt = HYDE_PROMPT.format(question=question)
    hyde_answer = gpt4_llm.invoke(prompt)
    hyde_context = [{"text": hyde_answer, "metadata": {"source": "HYDE"}}]
    return state.copy(update={"hyde_context": hyde_context})

def query_node_merge(state: RagQueryState) -> RagQueryState:
    merged = (state.translation_contexts or []) + (state.hyde_context or [])
    return state.copy(update={"merged_context": merged})

def query_node_generate(state: RagQueryState, question) -> RagQueryState:
    answer = generate_answer(state.merged_context, question)
    return state.copy(update={"answer": answer})

# --- Query Pipeline Graph ---
def build_query_graph(vectordb, small_llm, gpt4_llm, n: int = 5, k: int = 5, question=None):
    def translate_node(state):
        return query_node_translate(state, question, small_llm, vectordb, n, k)
    def hyde_node(state):
        return query_node_hyde(state, question, gpt4_llm)
    def generate_node(state):
        return query_node_generate(state, question)
    graph = StateGraph(RagQueryState)
    graph.add_node("translate", translate_node)
    graph.add_node("hyde", hyde_node)
    graph.add_node("merge", query_node_merge)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "translate")
    graph.add_edge(START, "hyde")
    graph.add_edge("translate", "merge")
    graph.add_edge("hyde", "merge")
    graph.add_edge("merge", "generate")
    graph.add_edge("generate", END)
    return graph.compile()

# --- Entry Points ---
def ingest_documents(document_urls: List[str], collection_name: str = "docs"):
    graph = build_ingest_graph(collection_name)
    state = RagIngestState(load=document_urls)
    graph.invoke(state)
    # No return needed; data is stored in Qdrant

def answer_questions(questions: List[str], vectordb, k: int = 5) -> Dict[str, Any]:
    graph = build_query_graph(vectordb, k)
    answers = []
    for question in questions:
        state = RagQueryState()
        result = graph.invoke(state)
        answers.append(result["answer"])
    return {"answers": answers}

def answer_questions_advanced(questions: List[str], vectordb, small_llm, gpt4_llm, n: int = 5, k: int = 5) -> Dict[str, Any]:
    answers = []
    for question in questions:
        graph = build_query_graph(vectordb, small_llm, gpt4_llm, n, k, question=question)
        state = RagQueryState()
        result = graph.invoke(state)
        answers.append(result["answer"])
    return {"answers": answers}

# --- Example Usage ---
if __name__ == "__main__":
    document_urls = [
        "https://hackrx.blob.core.windows.net/assets/policy.pdf?..."
    ]
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
    # Step 1: Ingest documents (run only when documents change)
    ingest_documents(document_urls, collection_name="docs")
    # Step 2: Load vector DB for querying
    vectordb = Qdrant(
        collection_name="docs",
        location=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
    )
    # Step 3: Answer questions
    output = answer_questions(questions, vectordb)
    print(output)