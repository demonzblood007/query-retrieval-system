# 🚀 Qdrant Cloud Deployment Guide

## Current Status
✅ Project updated for Qdrant Cloud  
✅ Connection test shows 403 Forbidden (need API key)  
🔧 Need to update Railway environment variables  

## Step 1: Set Your Qdrant API Key

You need to get your API key from Qdrant Cloud and set it in your environment.

### Get API Key from Qdrant Cloud:
1. Go to https://cloud.qdrant.io/
2. Click on your cluster: `50f5ef9a-3e77-45a2-9e54-a62f9dd2af87`
3. Go to "API Keys" tab
4. Copy your API key

### Create .env file locally:
```bash
# Create .env file with your actual API key
OPENAI_API_KEY=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
QDRANT_HOST=https://50f5ef9a-3e77-45a2-9e54-a62f9dd2af87.us-west-2-0.aws.cloud.qdrant.io
QDRANT_PORT=443
QDRANT_API_KEY=your_actual_qdrant_api_key_here
HACKRX_API_KEY=a0f168f99eb28adabfc76cbfafa54eaa84a6688c21e6fb97d8c5984905f81d23
```

## Step 2: Test Locally

```bash
python test_qdrant_cloud.py
```

You should see:
```
🎉 ALL TESTS PASSED!
Your Qdrant Cloud is ready to use!
```

## Step 3: Update Railway Environment Variables

In your Railway dashboard:
1. Go to your FastAPI service
2. Click "Variables" tab
3. Update these variables:

```
QDRANT_HOST=https://50f5ef9a-3e77-45a2-9e54-a62f9dd2af87.us-west-2-0.aws.cloud.qdrant.io
QDRANT_PORT=443
QDRANT_API_KEY=your_actual_qdrant_api_key_here
```

## Step 4: Deploy

```bash
git add .
git commit -m "Update to Qdrant Cloud configuration"
git push origin main
```

## Step 5: Test Your Deployed App

Your webhook URL should be working:
```
https://web-production-e9e82.up.railway.app/hackrx/run
```

## What's Fixed:
- ✅ Updated Qdrant endpoint to Cloud
- ✅ Updated port to 443 (HTTPS)
- ✅ Fixed QdrantClient initialization
- ✅ Maintained all optimization features
- ✅ Ready for immediate deployment
