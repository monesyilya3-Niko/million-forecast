# ── Million Forecast Terminal ──
# 使用 Python 3.12 slim（3.13 slim 镜像尚未稳定发布）
FROM python:3.12-slim

LABEL maintainer="Million Forecast Terminal"
LABEL description="足球竞彩 · 排列三 · 大乐透 概率分析系统"

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 Python 依赖（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建持久化目录
RUN mkdir -p /app/data /app/model_artifacts /app/uploads /app/logs

# Streamlit 配置
RUN mkdir -p /app/.streamlit

EXPOSE 8502

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8502/_stcore/health || exit 1

# 启动
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8502", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
