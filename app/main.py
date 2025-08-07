from fastapi import FastAPI
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Union
from app.core.rag_pipeline import ingest_documents, answer_questions_advanced, Qdrant, QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, OPENAI_API_KEY
from langchain_openai import OpenAI
from qdrant_client import QdrantClient
from langchain_community.embeddings import OpenAIEmbeddings
import os
import logging
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
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    logger.info("Received /hackrx/run request with documents: %s and questions: %s", req.documents, req.questions)
    verify_token(credentials)
    # Normalize documents to list
    document_urls = req.documents if isinstance(req.documents, list) else [req.documents]
    logger.info("Normalized document URLs: %s", document_urls)
    # Ingest documents (idempotent)
    try:
        ingest_documents(document_urls, collection_name="docs")
        logger.info("Document ingestion completed.")
    except Exception as e:
        logger.error("Error during document ingestion: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {e}")
    # Prepare vector DB and LLMs
    try:
        client = QdrantClient(
            url=QDRANT_HOST if QDRANT_HOST.startswith("http") else None,
            host=None if QDRANT_HOST.startswith("http") else QDRANT_HOST,
            port=int(QDRANT_PORT),
            api_key=QDRANT_API_KEY,
        )
        logger.info("Qdrant client initialized.")
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-large"
        )
        vectordb = Qdrant(
            client=client,
            collection_name="docs",
            embeddings=embeddings
        )
        openai_llm = OpenAI(
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4o-mini"
        )
        logger.info("Vector DB and LLMs initialized.")
        # Run advanced pipeline
        output = answer_questions_advanced(req.questions, vectordb, openai_llm, openai_llm)
        logger.info("Answer pipeline completed. Returning response.")
        return HackRxResponse(answers=output["answers"])
    except Exception as e:
        logger.error("Error during question answering: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Question answering failed: {e}")

@app.get("/api/v1/health")
def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "ok"}