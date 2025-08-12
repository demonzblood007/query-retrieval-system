# Environment Variables Documentation

## Required Variables

### OpenAI Configuration
- `OPENAI_API_KEY` - Your OpenAI API key (required)

### Qdrant Configuration
- `QDRANT_HOST` - Qdrant server URL (default: "http://localhost:6333")
- `QDRANT_PORT` - Qdrant server port (default: "6333")
- `QDRANT_API_KEY` - Qdrant API key (optional, default: None)
- `QDRANT_TIMEOUT` - Connection timeout in seconds (default: "30")
- `QDRANT_RETRIES` - Number of retry attempts (default: "3")

### API Configuration
- `HACKRX_API_KEY` - API key for accessing the hackRX endpoint (default: "testkey")
- `PORT` - Server port (default: 8080)

## Optional Performance Variables

### Document Processing
- `CHUNK_SIZE` - Text chunk size for document splitting (default: "800")
- `CHUNK_OVERLAP` - Overlap between text chunks (default: "100")
- `MAX_THREADS` - Maximum threads for parallel processing (default: auto-detect)

### RAG Pipeline
- `N_TRANSLATIONS` - Number of query translations for retrieval (default: "3")
- `TOP_K` - Number of top chunks to retrieve (default: "4")
- `EMBED_BATCH_SIZE` - Batch size for embedding (default: "64")

### Caching
- `ENABLE_QDRANT_CACHE` - Enable smart chunk caching to avoid re-embedding (default: "true")

## Docker-Specific Variables

### Container Configuration
- `START_LOCAL_QDRANT` - Whether to start Qdrant in the container (default: "true")

## Example .env File

```env
# Required
OPENAI_API_KEY=sk-your-openai-key-here

# Optional (these are the defaults)
QDRANT_HOST=http://localhost:6333
QDRANT_PORT=6333
QDRANT_API_KEY=
QDRANT_TIMEOUT=30
QDRANT_RETRIES=3

HACKRX_API_KEY=testkey
PORT=8080

CHUNK_SIZE=800
CHUNK_OVERLAP=100
N_TRANSLATIONS=3
TOP_K=4
EMBED_BATCH_SIZE=64

ENABLE_QDRANT_CACHE=true
START_LOCAL_QDRANT=true
```

## Performance Tuning

### For Hackathons (Fast Demo)
- `ENABLE_QDRANT_CACHE=true` (default)
- `N_TRANSLATIONS=2` (faster)
- `TOP_K=3` (faster)
- `MAX_THREADS=4` (controlled parallelism)

### For Production (Quality)
- `ENABLE_QDRANT_CACHE=true`
- `N_TRANSLATIONS=5` (more comprehensive)
- `TOP_K=6` (more context)
- `CHUNK_SIZE=1000` (larger chunks)

### For Testing/Development
- `ENABLE_QDRANT_CACHE=false` (force re-processing)
- `QDRANT_TIMEOUT=60` (longer timeout for debugging)
