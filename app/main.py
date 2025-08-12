from fastapi import FastAPI
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Union
from app.core.rag_pipeline import (
    ingest_documents,
    answer_questions_advanced,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    OPENAI_API_KEY,
    get_qdrant_client,
)
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
import logging
import json
import re
print("FastAPI app is starting...")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("FastAPI app instance created.")

# --- Security ---
API_KEY = os.getenv("HACKRX_API_KEY", "testkey")
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    logger.info("Verifying API token.")
    if credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
        logger.warning("Invalid or missing API key.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")

# --- Request/Response Models ---
class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
def hackrx_run(
    req: HackRxRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None,
):
    logger.info("Received /hackrx/run request with documents: %s and questions: %s", req.documents, req.questions)
    verify_token(credentials)
    # Normalize documents to list
    document_urls = req.documents if isinstance(req.documents, list) else [req.documents]
    logger.info("Normalized document URLs: %s", document_urls)
    # Ingest documents (idempotent)
    try:
        # Pass optional overrides if provided
        # Current ingest_documents signature doesn't accept overrides; if needed, extend later.
        # Read dynamic overrides from headers (to preserve strict body schema)
        def _get_int_header(name: str, default: Union[int, None]) -> Union[int, None]:
            try:
                value = request.headers.get(name) if request else None
                return int(value) if value not in (None, "") else default
            except Exception:
                return default

        chunk_size = _get_int_header("X-Chunk-Size", None)
        chunk_overlap = _get_int_header("X-Chunk-Overlap", None)
        max_threads = _get_int_header("X-Max-Threads", None)

        ingest_documents(
            document_urls,
            collection_name="docs",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_threads=max_threads,
        )
        logger.info("Document ingestion completed.")
    except Exception as e:
        logger.error("Error during document ingestion: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {e}")
    # Prepare vector DB and LLMs
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-large"
        )
        
        # Create Qdrant client and vector store for querying
        qdrant_timeout_env = os.getenv("QDRANT_TIMEOUT", "30")
        try:
            qdrant_timeout = float(qdrant_timeout_env)
        except Exception:
            qdrant_timeout = 30.0
        # Use the same client acquisition helper with fallback
        try:
            client = get_qdrant_client()
        except Exception as e:
            # Final fallback: direct client with timeout (may still fail)
            client = QdrantClient(
                url=QDRANT_HOST,
                api_key=QDRANT_API_KEY or None,
                timeout=qdrant_timeout,
            )
        
        vectordb = QdrantVectorStore(
            client=client,
            collection_name="docs",
            embedding=embeddings,
        )
        
        small_llm = ChatOpenAI(
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4o-mini"
        )
        gpt4_llm = ChatOpenAI(
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4o"
        )
        logger.info("Vector DB and LLMs initialized.")
        # Env-driven knobs for query expansion and top-k
        def _get_int_header_default(name: str, default_env: str, env_key: str) -> int:
            h = None
            try:
                h = request.headers.get(name) if request else None
                if h not in (None, ""):
                    return int(h)
            except Exception:
                pass
            return int(os.getenv(env_key, default_env))

        n_translations = _get_int_header_default("X-N-Translations", "3", "N_TRANSLATIONS")
        top_k = _get_int_header_default("X-Top-K", "4", "TOP_K")
        # Run advanced pipeline
        output = answer_questions_advanced(req.questions, vectordb, small_llm, gpt4_llm, n=n_translations, k=top_k)

        # Normalize model JSON into plain strings per API contract
        def _strip_code_fences(text: str) -> str:
            s = text.strip()
            if s.startswith("```") and s.endswith("```"):
                s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE)
            return s.strip()

        def _extract_plain_answer(answer_blob: str) -> str:
            s = _strip_code_fences(answer_blob)
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "answer" in obj:
                    return str(obj["answer"]).strip()
            except Exception:
                pass
            # If not JSON or no 'answer' field, return as-is
            return s

        plain_answers = [_extract_plain_answer(a) for a in output.get("answers", [])]

        logger.info("Answer pipeline completed. Returning response.")
        return HackRxResponse(answers=plain_answers)
    except Exception as e:
        logger.error("Error during question answering: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Question answering failed: {e}")

@app.get("/api/v1/health")
def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "ok"}