#!/usr/bin/env python3
"""
Clear embedding cache if it exists
"""

import os
import json

CACHE_FILE = "embedding_cache.json"

def clear_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"✅ Cleared cache file: {CACHE_FILE}")
    else:
        print(f"ℹ️  No cache file found: {CACHE_FILE}")
    
    # Also clear documents cache
    documents_dir = "documents"
    if os.path.exists(documents_dir):
        import shutil
        shutil.rmtree(documents_dir)
        print(f"✅ Cleared documents cache: {documents_dir}")
    else:
        print(f"ℹ️  No documents cache found: {documents_dir}")

if __name__ == "__main__":
    clear_cache()
