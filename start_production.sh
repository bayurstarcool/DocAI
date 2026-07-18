#!/bin/bash
#
# DocAI Production Server — persistent via screen
#

cd /home/wahyu/docai

SESSION_NAME="docai_prod"
HOST="0.0.0.0"
PORT=8000

echo "=========================================="
echo "  DocAI - Document Processing AI Platform"
echo "  Production Server"
echo "=========================================="
echo ""
echo "Public URL: http://38.47.85.224:8000"
echo "Local URL:  http://localhost:8000"
echo ""

# If already running, just show status
if screen -list | grep -q "\.${SESSION_NAME}\s"; then
    echo "✅ Server already running in screen session '${SESSION_NAME}'"
    echo "   Reattach: screen -r ${SESSION_NAME}"
    exit 0
fi

echo "Starting server in screen session '${SESSION_NAME}'..."
echo "To detach: Ctrl+A, D"

screen -dmS "${SESSION_NAME}" bash -c "
cd /home/wahyu/docai
export PYTHONPATH=\$(pwd)
exec python3 -c '
from backend.app import app
import uvicorn
uvicorn.run(
    app,
    host=\"${HOST}\",
    port=${PORT},
    log_level=\"info\",
    access_log=True,
    loop=\"uvloop\",
    http=\"httptools\",
)
' 2>&1 | tee /tmp/docai_prod.log
"

if [ $? -eq 0 ]; then
    echo "✅ Production server started successfully."
else
    echo "❌ Failed to start production server."
    exit 1
fi
