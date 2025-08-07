import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.rag_pipeline import answer_questions_advanced, Qdrant, QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, OPENAI_API_KEY
from langchain_openai import OpenAI
from qdrant_client import QdrantClient
from langchain_community.embeddings import OpenAIEmbeddings

def test_hackrx_sample():
    document_urls = [
        "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    ]
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    # Load vector DB for querying
    # If QDRANT_HOST contains protocol, use 'url', else use 'host'
    if QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://"):
        client = QdrantClient(
            url=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
        )
    else:
        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
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
    # Use real OpenAI LLM for both translation and HYDE
    openai_llm = OpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o-mini"
    )
    output = answer_questions_advanced(questions, vectordb, openai_llm, openai_llm)
    print("\nTest Output:")
    for idx, answer in enumerate(output["answers"]):
        print(f"Q{idx+1}: {questions[idx]}")
        print(f"A{idx+1}: {answer}\n")

if __name__ == "__main__":
    test_hackrx_sample()