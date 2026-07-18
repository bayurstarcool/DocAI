#!/bin/bash
echo "=== DocAI Server Status ==="
echo ""

# Check if server is running
if ss -tlnp | grep -q ":8000"; then
    echo "✅ Server is RUNNING on port 8000"
else
    echo "❌ Server is NOT running"
    exit 1
fi

# Check API
echo ""
echo "=== API Health Check ==="
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/health)
if [ "$response" = "200" ]; then
    echo "✅ API is responding (HTTP 200)"
else
    echo "❌ API is not responding properly (HTTP $response)"
fi

# Check GPU
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "⚠️ Cannot query GPU"

# Check disk usage
echo ""
echo "=== Disk Usage ==="
df -h / | tail -1

# Check memory
echo ""
echo "=== Memory Usage ==="
free -h | grep Mem

# Check output files
echo ""
echo "=== Output Files ==="
ls -lh /home/wahyu/docai/data/outputs/ 2>/dev/null | tail -5 || echo "No output files"

echo ""
echo "=== Access URLs ==="
echo "Public:  http://38.47.85.224:8000"
echo "Local:   http://localhost:8000"
echo ""
