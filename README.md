# Legal Chatbot Hackathon Submission

## Tech Stack
- FastAPI (Backend)
- GPT-4o (LLM via OpenAI)
- Qdrant (Vector DB)
- Railway (Deployment)
- Python, LangChain, pdfplumber, etc.

## API Endpoint
- **POST** `/hackrx/run`
- **Authorization:** Bearer `<api_key>`
- **Request:**
  ```json
  {
    "documents": "https://...pdf",  // or list of URLs
    "questions": ["Question 1", "Question 2", ...]
  }
  ```
- **Response:**
  ```json
  {
    "answers": ["Answer 1", "Answer 2", ...]
  }
  ```

## Environment Variables
- `OPENAI_API_KEY` (required)
- `QDRANT_HOST` (required)
- `QDRANT_PORT` (required)
- `QDRANT_API_KEY` (if needed)
- `HACKRX_API_KEY` (required, for endpoint auth)

## Deployment (Railway)
1. **Push your code to GitHub.**
2. **Create a Railway account** at https://railway.app/
3. **New Project > Deploy from GitHub repo**
4. **Set environment variables** in Railway dashboard:
   - `OPENAI_API_KEY`, `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`, `HACKRX_API_KEY`
5. **Deploy!**
6. **Get your public HTTPS URL** (e.g., `https://your-app.up.railway.app/hackrx/run`)
7. **Test with curl/Postman:**
   ```sh
   curl -X POST https://your-app.up.railway.app/hackrx/run \
     -H "Authorization: Bearer <your_api_key>" \
     -H "Content-Type: application/json" \
     -d '{"documents": "https://...", "questions": ["..."]}'
   ```
8. **Submit your webhook URL and tech stack description to the hackathon platform.**

## Notes
- The ingestion pipeline is idempotent: documents are only downloaded/embedded if not already present.
- Response time is optimized for hackathon requirements (<30s typical).
- For any issues, check Railway logs or contact the team.
