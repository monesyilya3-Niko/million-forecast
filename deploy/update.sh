#!/bin/bash
set -e
echo "📥 更新 百万竞猜 ..."
git pull
docker compose up -d --build
echo "✅ 已更新并重启"
