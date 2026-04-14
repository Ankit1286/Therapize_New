#!/bin/bash
# =============================================================================
# Therapize — One-shot server setup script
# Run this once on a fresh Ubuntu 22.04 GCP VM.
# It installs everything and starts the app automatically.
# =============================================================================

set -e  # stop immediately if any command fails

echo "======================================"
echo " Therapize Server Setup"
echo "======================================"

# --- 1. Update the system and install required tools --------------------------
echo ""
echo "[1/7] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q python3.11 python3.11-venv python3-pip git nginx curl

# --- 2. Create a dedicated user for the app -----------------------------------
echo ""
echo "[2/7] Creating app user..."
sudo useradd --system --create-home --shell /bin/bash therapize 2>/dev/null || echo "User already exists, continuing."

# --- 3. Clone the repository --------------------------------------------------
echo ""
echo "[3/7] Cloning repository..."
sudo mkdir -p /opt/therapize
sudo chown therapize:therapize /opt/therapize
sudo -u therapize git clone https://github.com/Ankit1286/Therapize_New.git /opt/therapize || \
  (cd /opt/therapize && sudo -u therapize git pull)

# --- 4. Create a Python virtual environment and install dependencies ----------
echo ""
echo "[4/7] Installing Python dependencies (this takes a few minutes)..."
sudo -u therapize python3.11 -m venv /opt/therapize/venv
sudo -u therapize /opt/therapize/venv/bin/pip install --quiet --upgrade pip
sudo -u therapize /opt/therapize/venv/bin/pip install --quiet -r /opt/therapize/requirements.txt

# --- 5. Create the environment variables file ---------------------------------
echo ""
echo "[5/7] Creating environment file..."

if [ ! -f /etc/therapize.env ]; then
sudo tee /etc/therapize.env > /dev/null <<'ENV'
# Fill in your actual values below, then run:
#   sudo systemctl restart therapize
ANTHROPIC_API_KEY=REPLACE_ME
DATABASE_URL=REPLACE_ME
REDIS_URL=REPLACE_ME
ENVIRONMENT=production
LOG_LEVEL=WARNING
LLM_MODEL=claude-haiku-4-5-20251001
EMBEDDING_MODEL=all-MiniLM-L6-v2
ALLOWED_ORIGINS=REPLACE_ME_WITH_VERCEL_URL
ENV
  sudo chmod 600 /etc/therapize.env
  echo "  Created /etc/therapize.env — you need to fill in your API keys."
else
  echo "  /etc/therapize.env already exists, skipping."
fi

# --- 6. Install the systemd service (auto-start on boot) ----------------------
echo ""
echo "[6/7] Installing systemd service..."

sudo tee /etc/systemd/system/therapize.service > /dev/null <<'SERVICE'
[Unit]
Description=Therapize API
After=network.target

[Service]
User=therapize
WorkingDirectory=/opt/therapize
EnvironmentFile=/etc/therapize.env
ExecStart=/opt/therapize/venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable therapize

# --- 7. Configure nginx (the "front door" that forwards traffic to the app) ---
echo ""
echo "[7/7] Configuring nginx..."

sudo tee /etc/nginx/sites-available/therapize > /dev/null <<'NGINX'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 60s;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/therapize /etc/nginx/sites-enabled/therapize
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

echo ""
echo "======================================"
echo " Setup complete!"
echo "======================================"
echo ""
echo " NEXT STEP: Fill in your API keys:"
echo "   sudo nano /etc/therapize.env"
echo ""
echo " Then start the app:"
echo "   sudo systemctl start therapize"
echo ""
echo " Check it's running:"
echo "   sudo systemctl status therapize"
echo ""
