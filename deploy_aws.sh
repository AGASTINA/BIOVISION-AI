#!/bin/bash
# =============================================================================
# BioVision AI — AWS EC2 Deployment Script
# Target: Ubuntu 24.04 LTS (t3.medium, 4GB RAM)
# Usage:  chmod +x deploy_aws.sh && sudo ./deploy_aws.sh
# =============================================================================

set -euo pipefail

APP_USER="ubuntu"
APP_DIR="/home/${APP_USER}/BIOVISION-AI"
REPO_URL="https://github.com/AGASTINA/BIOVISION-AI.git"
PYTHON_VERSION="3.12"
SERVICE_NAME="biovision"

echo "============================================="
echo "  BioVision AI — EC2 Deployment Starting"
echo "============================================="

# 1. System updates & Python
echo "[1/7] Installing system packages..."
apt-get update -y
apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# 2. Clone repository
echo "[2/7] Cloning repository..."
if [ -d "${APP_DIR}" ]; then
    echo "  -> Directory exists, pulling latest..."
    cd "${APP_DIR}"
    git pull origin main
else
    git clone "${REPO_URL}" "${APP_DIR}"
    cd "${APP_DIR}"
fi
chown -R ${APP_USER}:${APP_USER} "${APP_DIR}"

# 3. Create virtual environment
echo "[3/7] Setting up Python virtual environment..."
sudo -u ${APP_USER} python${PYTHON_VERSION} -m venv "${APP_DIR}/.venv"

# 4. Install dependencies
echo "[4/7] Installing Python dependencies (this takes 3-5 minutes)..."
sudo -u ${APP_USER} "${APP_DIR}/.venv/bin/pip" install --upgrade pip
sudo -u ${APP_USER} "${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

# 5. Seed database if needed
echo "[5/7] Initializing database..."
if [ ! -f "${APP_DIR}/biovision.db" ] || [ ! -s "${APP_DIR}/biovision.db" ]; then
    sudo -u ${APP_USER} "${APP_DIR}/.venv/bin/python" "${APP_DIR}/seed_data.py"
fi

# 6. Install systemd service
echo "[6/7] Configuring systemd service..."
cp "${APP_DIR}/biovision.service" /etc/systemd/system/${SERVICE_NAME}.service
systemctl daemon-reload
systemctl enable ${SERVICE_NAME}
systemctl restart ${SERVICE_NAME}

# 7. Verify
echo "[7/7] Verifying deployment..."
sleep 5
if systemctl is-active --quiet ${SERVICE_NAME}; then
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "UNKNOWN")
    echo ""
    echo "============================================="
    echo "  ✅ BioVision AI is LIVE!"
    echo "============================================="
    echo "  URL:    http://${PUBLIC_IP}:8501"
    echo "  Status: systemctl status ${SERVICE_NAME}"
    echo "  Logs:   journalctl -u ${SERVICE_NAME} -f"
    echo "  Stop:   sudo systemctl stop ${SERVICE_NAME}"
    echo "============================================="
else
    echo "❌ Service failed to start. Check logs:"
    echo "   journalctl -u ${SERVICE_NAME} -n 50"
    exit 1
fi
