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
from langchain_openai import OpenAIEmbeddings, OpenAI  # Updated imports
from langgraph.graph import StateGraph, END, START
import pdfplumber
from pydantic import BaseModel
from collections import defaultdict
import operator
import threading
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from langchain_qdrant import QdrantVectorStore  # Updated import
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()

DOCUMENTS_DIR = "documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Get API keys and Qdrant config from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "https://50f5ef9a-3e77-45a2-9e54-a62f9dd2af87.us-west-2-0.aws.cloud.qdrant.io")
QDRANT_PORT = os.getenv("QDRANT_PORT", "443")  # Qdrant Cloud uses HTTPS on port 443
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Ensure we have the full URL for Qdrant Cloud
if not QDRANT_HOST.startswith("http://") and not QDRANT_HOST.startswith("https://"):
    QDRANT_HOST = f"https://{QDRANT_HOST}"

# Configure logging for timing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

CACHE_FILE = "embedding_cache.json"

def load_embedding_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_embedding_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def download_document(url: str, save_dir: str = DOCUMENTS_DIR) -> str:
    # Use a hash of the URL for caching
    filename = hash_url(url) + "_" + url.split("?")[0].split("/")[-1]
    local_filename = os.path.join(save_dir, filename)
    if os.path.exists(local_filename):
        logger.info(f"Cache hit for {url}")
        return local_filename
    logger.info(f"Downloading {url}")
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

def get_max_threads(n_tasks: int) -> int:
    env_threads = os.getenv("MAX_THREADS")
    if env_threads is not None:
        try:
            env_threads = int(env_threads)
            logger.info(f"Using MAX_THREADS from environment: {env_threads}")
            return min(env_threads, n_tasks)
        except Exception:
            logger.warning(f"Invalid MAX_THREADS value: {env_threads}, falling back to auto-detect.")
    cpu_threads = os.cpu_count() or 8
    max_threads = min(cpu_threads, n_tasks, 16)  # Cap at 16 for safety
    logger.info(f"Using {max_threads} threads for parallel processing (cpu_count={os.cpu_count()})")
    return max_threads

def load_and_preprocess_documents(urls: List[str]) -> List[Dict[str, Any]]:
    start_time = time.time()
    texts = [None] * len(urls)
    file_paths = [None] * len(urls)
    max_threads = get_max_threads(len(urls))
    # Parallel download
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_idx = {executor.submit(download_document, url): idx for idx, url in enumerate(urls)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            file_paths[idx] = future.result()
    logger.info(f"Downloaded {len(urls)} documents in {time.time() - start_time:.2f}s")
    # Parallel parse and chunk
    def parse_and_chunk(idx, file_path):
        if file_path.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            text = ""
        # Chunk the text here
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = []
        for chunk in splitter.split_text(text):
            chunks.append({"text": chunk, "doc_index": idx})
        return chunks
    all_chunks = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_idx = {executor.submit(parse_and_chunk, idx, file_path): idx for idx, file_path in enumerate(file_paths)}
        for future in as_completed(future_to_idx):
            chunks = future.result()
            all_chunks.extend(chunks)
    logger.info(f"Parsed and chunked {len(file_paths)} documents into {len(all_chunks)} chunks in {time.time() - start_time:.2f}s (total)")
    return all_chunks

def chunk_documents(texts: List[str]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for idx, text in enumerate(texts):
        for chunk in splitter.split_text(text):
            chunks.append({"text": chunk, "doc_index": idx})
    return chunks

def embed_and_store(chunks: List[Dict[str, Any]], collection_name: str = "docs") -> QdrantVectorStore:
    import time
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-large"
    )
    texts = [c["text"] for c in chunks]
    metadatas = [{"doc_index": c["doc_index"]} for c in chunks]
    start_time = time.time()
    # Load cache
    cache = load_embedding_cache()
    chunk_hashes = [hash_text(text) for text in texts]
    to_embed = []
    to_embed_indices = []
    cached_embeddings = []
    for idx, (h, text) in enumerate(zip(chunk_hashes, texts)):
        if h in cache:
            logger.info(f"Embedding cache hit for chunk {idx}")
            cached_embeddings.append((idx, cache[h]))
        else:
            to_embed.append(text)
            to_embed_indices.append(idx)
    # Embed only new chunks
    new_embeddings = []
    if to_embed:
        logger.info(f"Embedding {len(to_embed)} new chunks in batch...")
        new_embeddings = embeddings_model.embed_documents(to_embed)
        for idx, emb, text in zip(to_embed_indices, new_embeddings, to_embed):
            h = hash_text(text)
            cache[h] = emb
        save_embedding_cache(cache)
    # Reconstruct embeddings in original order
    all_embeddings = [None] * len(texts)
    for idx, emb in cached_embeddings:
        all_embeddings[idx] = emb
    for idx, emb in zip(to_embed_indices, new_embeddings):
        all_embeddings[idx] = emb
    # QdrantClient approach (robust for all versions)
    try:
        client = QdrantClient(
            url=QDRANT_HOST,  # Should include protocol (http/https)
            api_key=QDRANT_API_KEY or None,
        )
        
        # Check if collection exists, create if it doesn't
        try:
            client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists")
        except Exception:
            logger.info(f"Creating collection '{collection_name}' with vector size 3072")
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": 3072,  # text-embedding-3-large dimension
                    "distance": "Cosine"
                }
            )
        
        vectordb = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings_model,
        )
        vectordb.add_texts(
            texts=texts,
            metadatas=metadatas,
            embeddings=all_embeddings
        )
        logger.info(f"Embedded and upserted {len(texts)} chunks to Qdrant in {time.time() - start_time:.2f}s (with cache)")
    except Exception as e:
        logger.error(f"Error during Qdrant upsert: {e}", exc_info=True)
        raise
    return vectordb

def retrieve_context(vectordb: QdrantVectorStore, query: str, k: int = 5) -> List[Dict[str, Any]]:
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
    # Now returns all chunks directly
    chunks = load_and_preprocess_documents(state.load)
    return state.copy(update={"embed": chunks})

def ingest_node_chunk(state: RagIngestState) -> RagIngestState:
    # No-op, as chunking is now done in load step
    return state

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
    print("Starting document ingestion...")
    ingest_documents(document_urls, collection_name="docs")
    print("Ingestion complete.")
    print("Initializing Qdrant client and vector store for querying...")
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-large"
    )
    client = QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY or None,
    )
    vectordb = QdrantVectorStore(
        client=client,
        collection_name="docs",
        embedding=embeddings_model,
    )
    print("Answering questions...")
    from langchain_openai import OpenAI
    small_llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    gpt4_llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    output = answer_questions_advanced(questions, vectordb, small_llm, gpt4_llm)
    print("\n--- Final Output ---")
    import json
    print(json.dumps(output, indent=2))