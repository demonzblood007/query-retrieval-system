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

app = FastAPI()

# --- Security ---
API_KEY = os.getenv("HACKRX_API_KEY", "testkey")
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
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
    verify_token(credentials)
    # Normalize documents to list
    document_urls = req.documents if isinstance(req.documents, list) else [req.documents]
    # Ingest documents (idempotent)
    ingest_documents(document_urls, collection_name="docs")
    # Prepare vector DB and LLMs
    client = QdrantClient(
        url=QDRANT_HOST if QDRANT_HOST.startswith("http") else None,
        host=None if QDRANT_HOST.startswith("http") else QDRANT_HOST,
        port=int(QDRANT_PORT),
        api_key=QDRANT_API_KEY,
    )
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
    # Run advanced pipeline
    output = answer_questions_advanced(req.questions, vectordb, openai_llm, openai_llm)
    return HackRxResponse(answers=output["answers"])

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}