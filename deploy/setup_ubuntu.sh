#!/bin/bash
# ── 百万竞猜 / Million Forecast Terminal ──
# Oracle Cloud Always Free Ubuntu 初始化脚本
set -e

echo "=========================================="
echo " 百万竞猜 · 云服务器初始化"
echo " Million Forecast Terminal · Setup"
echo "=========================================="

# 1. 更新系统
echo "[1/7] 更新系统..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. 安装 Docker
echo "[2/7] 安装 Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker $USER
    echo "Docker 已安装。请重新登录以使 docker 组生效。"
else
    echo "Docker 已存在: $(docker --version)"
fi

# 3. 安装 Docker Compose
echo "[3/7] 安装 Docker Compose..."
if ! docker compose version &> /dev/null; then
    sudo apt-get install -y docker-compose-plugin
else
    echo "Docker Compose 已存在: $(docker compose version)"
fi

# 4. 安装 Git
echo "[4/7] 安装 Git..."
sudo apt-get install -y git curl

# 5. 创建项目目录
echo "[5/7] 创建项目目录..."
sudo mkdir -p /opt/million-forecast
sudo chown $USER:$USER /opt/million-forecast

# 6. 创建数据目录
echo "[6/7] 创建数据目录..."
mkdir -p /opt/million-forecast/data
mkdir -p /opt/million-forecast/uploads
mkdir -p /opt/million-forecast/model_artifacts
mkdir -p /opt/million-forecast/logs
mkdir -p /opt/million-forecast/backups

# 7. 防火墙
echo "[7/7] 配置防火墙..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 8502/tcp
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    echo "防火墙规则已添加"
fi

echo ""
echo "=========================================="
echo " 初始化完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. git clone <your-repo> /opt/million-forecast"
echo "  2. cd /opt/million-forecast"
echo "  3. cp .env.example .env"
echo "  4. bash deploy/start.sh"
echo ""
echo "Oracle Cloud 安全组："
echo "  请在 OCI Console > VCN > 子网 > 安全列表中"
echo "  添加入站规则：端口 8502/TCP，源 0.0.0.0/0"
echo ""
echo "访问地址：http://<公网IP>:8502"
echo ""
