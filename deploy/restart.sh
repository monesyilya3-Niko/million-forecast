#!/bin/bash
set -e
echo "🔄 重启 百万竞猜 ..."
docker compose down
docker compose up -d --build
echo "✅ 已重启: http://localhost:8502"
