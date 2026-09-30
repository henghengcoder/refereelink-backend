#!/usr/bin/env bash
# quick-verify.sh — 在当前环境验证 Web 数据大屏链路，不依赖模型权重
set -euo pipefail

cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo " RefereeLink Backend — Quick Verify Script"
echo "========================================"
echo ""

# 1. 检查环境
echo "[1/6] Checking environment..."
if [ ! -d ".venv" ]; then
    echo -e "${RED}ERROR: .venv not found. Run 'uv sync --dev' first.${NC}"
    exit 1
fi
source .venv/bin/activate
python -c "import torch, cv2, supervision, fastapi; print('  Python env OK')" || {
    echo -e "${RED}ERROR: Python dependencies missing.${NC}"
    exit 1
}

# 检查 Node.js
if ! command -v node &> /dev/null; then
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
fi
if ! command -v node &> /dev/null; then
    echo -e "${RED}ERROR: Node.js not found. Run the nvm install steps from SETUP_GUIDE.md.${NC}"
    exit 1
fi
echo "  Node.js $(node -v) OK"

# 2. 检查模型权重
echo ""
echo "[2/6] Checking model weights..."
MISSING=0
for f in football-player-detection.pt football-pitch-detection.pt; do
    if [ ! -f "assets/weights/$f" ]; then
        echo -e "  ${YELLOW}MISSING${NC}: assets/weights/$f"
        MISSING=1
    else
        echo -e "  ${GREEN}FOUND${NC}: assets/weights/$f"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}WARNING: Model weights missing. Full inference will fail.${NC}"
    echo "         But the web dashboard + API + WebSocket can still be verified."
    echo "         See SETUP_GUIDE.md for download instructions."
fi

# 3. 创建合成测试视频
echo ""
echo "[3/6] Creating synthetic test video..."
python3 << 'PYEOF'
import cv2, numpy as np, os
os.makedirs('assets/data', exist_ok=True)
path = 'assets/data/verify_test.mp4'
if os.path.exists(path) and os.path.getsize(path) > 10000:
    print(f"  Already exists: {path}")
else:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 10.0, (640, 480))
    for i in range(60):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 画一个绿色矩形模拟球场
        cv2.rectangle(frame, (50, 50), (590, 430), (0, 100, 0), -1)
        # 画一些圆点模拟球员
        for j in range(5):
            x = 100 + j * 100 + (i * 3) % 40
            y = 200 + (j % 2) * 100
            cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)
        # 画文字
        cv2.putText(frame, f"Test Frame {i}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    print(f"  Created: {path} ({os.path.getsize(path)} bytes)")
PYEOF

# 4. 启动后端
echo ""
echo "[4/6] Starting backend..."
# 检测 CUDA 是否可用，优先用 cuda
DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
echo "  Using device: $DEVICE"
python -m app.server.main --device "$DEVICE" --port 8000 &
BACKEND_PID=$!
sleep 4

# 检查后端是否启动
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${RED}ERROR: Backend failed to start.${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi
echo -e "  ${GREEN}Backend running${NC} on http://localhost:8000 (PID: $BACKEND_PID)"

# 5. 测试 API
echo ""
echo "[5/6] Testing API endpoints..."
HEALTH=$(curl -s http://localhost:8000/health)
echo "  /health -> $HEALTH"

STATUS=$(curl -s http://localhost:8000/api/status)
echo "  /api/status -> pipeline_running=$(echo $STATUS | python3 -c 'import sys,json; print(json.load(sys.stdin)["pipeline_running"])')"

CONFIG=$(curl -s http://localhost:8000/api/config)
echo "  /api/config -> device=$(echo $CONFIG | python3 -c 'import sys,json; print(json.load(sys.stdin)["device"])')"

# 启动 pipeline（带合成视频）
echo ""
echo "  Starting pipeline with synthetic video..."
START_RESULT=$(curl -s -X POST http://localhost:8000/api/pipeline/start \
    -H "Content-Type: application/json" \
    -d "{\"video_source\": \"assets/data/verify_test.mp4\", \"device\": \"$DEVICE\"}")
echo "  /api/pipeline/start -> $START_RESULT"

sleep 2

# 检查状态
STATUS2=$(curl -s http://localhost:8000/api/status)
echo "  /api/status after start -> source_status=$(echo $STATUS2 | python3 -c 'import sys,json; print(json.load(sys.stdin)["source_status"])')"

# 6. 前端构建检查
echo ""
echo "[6/6] Frontend build check..."
cd web
npm run build 2>&1 | tail -3
cd ..

echo ""
echo "========================================"
echo " Quick verify complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. If model weights are MISSING, download them (see SETUP_GUIDE.md)"
echo "  2. Start the frontend: cd web && npm run dev"
echo "  3. Open http://localhost:5173 in your browser"
echo "  4. In the dashboard, input a video source path and click START"
echo ""
echo "To stop the backend: kill $BACKEND_PID"
echo ""

# 保持后端运行
wait $BACKEND_PID
