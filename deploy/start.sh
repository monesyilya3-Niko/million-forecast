#!/bin/bash
set -e
echo "🚀 启动 百万竞猜 / Million Forecast Terminal ..."
docker compose up -d --build
echo "✅ 已启动: http://localhost:8502"
docker compose logs -f --tail=50
