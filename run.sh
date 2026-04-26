#!/usr/bin/env bash
# run.sh — One-command startup for the AI Talent Scout demo
set -e

cd "$(dirname "$0")"

echo ""
echo "========================================"
echo "  AI Talent Scout — Starting Demo"
echo "========================================"
echo ""

# Activate venv if present
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Generate candidate data if missing
if [ ! -f "data/candidates.json" ]; then
    echo "📋 Generating mock candidate data..."
    python generate_candidates.py
fi

# Start FastAPI in background
echo "🚀 Starting FastAPI server on http://localhost:8000 ..."
PYTHONPATH="$(pwd)" python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "   API PID: $API_PID"

# Wait for API to be ready
echo "⏳ Waiting for API to be ready..."
for i in {1..15}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    sleep 1
done

# Start Streamlit UI
echo ""
echo "🎯 Starting Streamlit UI on http://localhost:8501 ..."
echo ""
echo "========================================"
echo "  Open http://localhost:8501 in browser"
echo "========================================"
echo ""
PYTHONPATH="$(pwd)" streamlit run ui/app.py --server.port 8501 --server.headless true &
UI_PID=$!

# Trap Ctrl+C to kill both processes
trap "echo ''; echo 'Shutting down...'; kill $API_PID $UI_PID 2>/dev/null; exit 0" INT

echo "Press Ctrl+C to stop both servers."
wait
