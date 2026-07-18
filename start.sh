#!/bin/bash
#
# Runs DocAI web application in a persistent screen session.
#

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating Python virtual environment..."
    source venv/bin/activate
fi

# Set the PYTHONPATH to include the project root, ensuring backend imports work
export PYTHONPATH=$(pwd)

# --- Configuration ---
SESSION_NAME="docai"
HOST="0.0.0.0"
PORT=8000

# If screen session already exists, just attach
if screen -list | grep -q "\.${SESSION_NAME}\s"; then
    echo "✅ DocAI already running in screen session '${SESSION_NAME}'"
    echo "   Attach: screen -r ${SESSION_NAME}"
    echo "   Access: http://localhost:${PORT}"
    exit 0
fi

echo "Starting DocAI server in screen session '${SESSION_NAME}'..."
echo "Access the application at http://localhost:${PORT}"
echo ""
echo "To detach: Ctrl+A, D"
echo "To reattach: screen -r ${SESSION_NAME}"

screen -dmS "${SESSION_NAME}" bash -c "cd $(pwd) && PYTHONPATH=$(pwd) python3 -c '
from backend.app import app
import uvicorn
uvicorn.run(app, host=\"${HOST}\", port=${PORT}, log_level=\"info\")
'"

if [ $? -eq 0 ]; then
    echo "✅ Server started successfully in screen session."
else
    echo "❌ Failed to start server."
    exit 1
fi
