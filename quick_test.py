#!/usr/bin/env python3
"""
Quick Qdrant Cloud Test - Just add your API key below
"""

from qdrant_client import QdrantClient, models

# CONFIGURATION - UPDATE WITH YOUR API KEY
QDRANT_HOST = "https://50f5ef9a-3e77-45a2-9e54-a62f9dd2af87.us-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "YOUR_API_KEY_HERE"  # <-- PUT YOUR API KEY HERE

def test_connection():
    print("🚀 Quick Qdrant Cloud Connection Test")
    print("=" * 50)
    
    if QDRANT_API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Please update QDRANT_API_KEY in this script with your actual API key")
        print("   Get it from: https://cloud.qdrant.io/ > Your Cluster > API Keys")
        return False
    
    try:
        client = QdrantClient(
            url=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
            timeout=30.0
        )
        
        # Test connection
        collections = client.get_collections()
        print(f"✅ Connection successful! Found {len(collections.collections)} collections")
        
        # Test create collection
        client.recreate_collection(
            collection_name="test_collection",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
        print("✅ Collection creation successful!")
        
        # Cleanup
        client.delete_collection("test_collection")
        print("✅ Cleanup successful!")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("Your Qdrant Cloud is ready to use!")
        
        print("\nNext steps:")
        print("1. Update your .env file with the API key")
        print("2. Update Railway environment variables")
        print("3. Deploy your app!")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
