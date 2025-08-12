@echo off
echo 🧹 Cleaning up for deployment...

:: Remove Python cache
if exist app\__pycache__ rmdir /s /q app\__pycache__
if exist app\core\__pycache__ rmdir /s /q app\core\__pycache__
if exist __pycache__ rmdir /s /q __pycache__

:: Remove virtual environment (don't commit this)
if exist venv rmdir /s /q venv

:: Remove local data
if exist documents rmdir /s /q documents
if exist qdrant_local rmdir /s /q qdrant_local
if exist embedding_cache.json del embedding_cache.json

echo ✅ Cleanup complete!
echo 🚀 Ready for git commit and deployment!

echo.
echo Next steps:
echo 1. git add .
echo 2. git commit -m "Hackathon submission ready"
echo 3. git push
echo 4. Deploy to AWS App Runner or Heroku
echo.

pause
