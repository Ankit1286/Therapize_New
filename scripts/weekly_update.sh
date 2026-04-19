#!/bin/bash
# =============================================================================
# Therapize — Weekly DB update
# Runs all maintenance tasks in order. Safe to re-run at any time.
#
# Cron setup (run as root or therapize user):
#   crontab -e
#   0 2 * * 0 /opt/therapize/scripts/weekly_update.sh >> /var/log/therapize-weekly.log 2>&1
# =============================================================================

set -e

PYTHON=/opt/therapize/venv/bin/python
DIR=/opt/therapize
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"

echo ""
echo "=============================================="
echo "$LOG_PREFIX Starting weekly Therapize update"
echo "=============================================="

cd "$DIR"

# Step 1: Pull latest code
echo ""
echo "$LOG_PREFIX [1/3] Pulling latest code..."
sudo -u therapize git pull

# Step 2: Scrape new therapists (gender/ethnicity + insurance extracted inline)
echo ""
echo "$LOG_PREFIX [2/3] Running ingestion..."
$PYTHON scripts/run_ingestion.py

# Step 3: Health check — deactivate stale/404 profiles
echo ""
echo "$LOG_PREFIX [3/3] Running profile health check..."
$PYTHON scripts/check_profiles.py

echo ""
echo "$LOG_PREFIX Weekly update complete."
echo "=============================================="
