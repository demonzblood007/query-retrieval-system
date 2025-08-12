#!/usr/bin/env python3
"""
Clear all caches: embedding cache, documents cache, and Qdrant collections
"""

import os
import json
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CACHE_FILE = "embedding_cache.json"

def clear_file_caches():
    """Clear local file-based caches"""
    cleared_items = []
    
    # Clear embedding cache
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        cleared_items.append(f"✅ Cleared embedding cache: {CACHE_FILE}")
    else:
        cleared_items.append(f"ℹ️  No embedding cache found: {CACHE_FILE}")
    
    # Clear documents cache
    documents_dir = "documents"
    if os.path.exists(documents_dir):
        import shutil
        shutil.rmtree(documents_dir)
        cleared_items.append(f"✅ Cleared documents cache: {documents_dir}")
    else:
        cleared_items.append(f"ℹ️  No documents cache found: {documents_dir}")
    
    return cleared_items

def clear_qdrant_collections():
    """Clear Qdrant collections"""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        # Get Qdrant config from environment
        QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
        QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        
        # Ensure we have the full URL
        if not QDRANT_HOST.startswith("http://") and not QDRANT_HOST.startswith("https://"):
            QDRANT_HOST = f"http://{QDRANT_HOST}"
        
        # Create client
        client = QdrantClient(
            url=QDRANT_HOST,
            api_key=QDRANT_API_KEY or None,
            timeout=30.0,
        )
        
        # Get all collections
        collections = client.get_collections()
        cleared_items = []
        
        if not collections.collections:
            cleared_items.append("ℹ️  No Qdrant collections found")
            return cleared_items
        
        # Delete each collection
        for collection in collections.collections:
            collection_name = collection.name
            try:
                client.delete_collection(collection_name=collection_name)
                cleared_items.append(f"✅ Cleared Qdrant collection: {collection_name}")
            except Exception as e:
                cleared_items.append(f"❌ Failed to clear collection {collection_name}: {e}")
        
        return cleared_items
        
    except ImportError:
        return ["⚠️  Qdrant client not available, skipping collection clearing"]
    except Exception as e:
        return [f"❌ Failed to connect to Qdrant: {e}"]

def clear_cache(include_qdrant=True):
    """Clear all caches"""
    print("🧹 Clearing all caches...\n")
    
    # Clear file-based caches
    print("📁 File Caches:")
    for item in clear_file_caches():
        print(f"   {item}")
    
    # Clear Qdrant collections if requested
    if include_qdrant:
        print("\n🗃️  Qdrant Collections:")
        for item in clear_qdrant_collections():
            print(f"   {item}")
    
    print(f"\n✨ Cache clearing completed!")

if __name__ == "__main__":
    # Check command line arguments
    include_qdrant = True
    if len(sys.argv) > 1 and sys.argv[1] == "--files-only":
        include_qdrant = False
        print("🔍 Clearing file caches only (--files-only flag detected)")
    
    clear_cache(include_qdrant=include_qdrant)
