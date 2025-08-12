#!/bin/bash
set -e

echo "🚀 AWS Deployment Script for Legal Chatbot"

# Check required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY is required"
    exit 1
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t legal-chatbot:latest .

# Tag for AWS ECR (replace with your ECR URI)
echo "🏷️  Tagging for AWS..."
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-"123456789012"}
AWS_REGION=${AWS_REGION:-"us-east-1"}
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/legal-chatbot:latest"

docker tag legal-chatbot:latest $ECR_URI

# Login to ECR (uncomment when ready)
# echo "🔑 Logging into ECR..."
# aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Push to ECR (uncomment when ready)
# echo "⬆️  Pushing to ECR..."
# docker push $ECR_URI

echo "✅ Build complete! Image ready for AWS deployment."
echo "ECR URI: $ECR_URI"

# Test locally first
echo "🧪 Testing locally..."
docker run -d -p 8080:8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e HACKRX_API_KEY="${HACKRX_API_KEY:-testkey}" \
  --name legal-chatbot-test \
  legal-chatbot:latest

sleep 5

# Health check
echo "🔍 Health check..."
if curl -s http://localhost:8080/api/v1/health | grep -q "ok"; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    docker logs legal-chatbot-test
fi

echo "🎯 Ready for AWS deployment!"
echo "Container running on http://localhost:8080"
