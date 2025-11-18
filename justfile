# TolTECA Web V2 - Development Orchestration
# Simple wrapper around bash scripts in run/web_v2/

# Show available commands
default:
  @just --list

# =============================================================================
# SETUP
# =============================================================================

# Copy .env.example to .env if needed
init:
  #!/usr/bin/env bash
  cd ../run/web_v2
  if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created run/web_v2/.env"
    echo "Edit this file to configure your setup"
  else
    echo "✅ run/web_v2/.env already exists"
  fi

# Show current configuration
config:
  #!/usr/bin/env bash
  cd ../run/web_v2
  if [ -f .env ]; then
    echo "Current configuration (run/web_v2/.env):"
    echo "========================================="
    cat .env | grep -v "^#" | grep -v "^$"
  else
    echo "No .env file found. Run: just init"
  fi

# =============================================================================
# LEGACY MODE (direct toltecdb access)
# =============================================================================

# Start webapp in LEGACY mode (single service, no Dagster)
legacy:
  @echo "Starting TolTECA Web V2 in LEGACY mode..."
  @echo "This uses direct toltecdb access (no Dagster ingestion)"
  @bash ../run/tmux/tolteca_web_v2_legacy.sh

# Run webapp script directly (for debugging)
webapp:
  @bash ../run/web_v2/run_webapp.sh

# =============================================================================
# NEW MODE (tolteca_db with Dagster)
# =============================================================================

# Start both Dagster + webapp in NEW mode (two services)
new:
  @echo "Starting TolTECA Web V2 in NEW mode..."
  @echo "This uses tolteca_db with Dagster ingestion"
  @bash ../run/tmux/tolteca_web_v2_new.sh

# Run Dagster script directly (for debugging)
dagster:
  @bash ../run/web_v2/run_dagster.sh

# =============================================================================
# MONITORING
# =============================================================================

# Check if services are running
status:
  @echo "Checking TolTECA services..."
  @echo ""
  @echo "Tmux sessions:"
  @tmux list-sessions 2>/dev/null | grep tolteca_web_v2 || echo "No sessions running"
  @echo ""
  @echo "Ports:"
  @lsof -i :5000 2>/dev/null | grep LISTEN || echo "Port 5000 (webapp): not in use"
  @lsof -i :3000 2>/dev/null | grep LISTEN || echo "Port 3000 (Dagster): not in use"

# Attach to legacy mode session
attach-legacy:
  @tmux attach-session -t tolteca_web_v2_legacy 2>/dev/null || echo "Legacy session not running. Start with: just legacy"

# Attach to new mode session
attach-new:
  @tmux attach-session -t tolteca_web_v2_new 2>/dev/null || echo "New session not running. Start with: just new"

# Kill legacy mode session
kill-legacy:
  @tmux kill-session -t tolteca_web_v2_legacy 2>/dev/null && echo "✅ Killed legacy session" || echo "No legacy session to kill"

# Kill new mode session
kill-new:
  @tmux kill-session -t tolteca_web_v2_new 2>/dev/null && echo "✅ Killed new session" || echo "No new session to kill"

# Kill all TolTECA sessions
kill-all:
  @just kill-legacy
  @just kill-new

# =============================================================================
# DATABASE
# =============================================================================

# Check database files
check-db:
  #!/usr/bin/env bash
  cd ..
  echo "Database files:"
  echo ""
  echo "Legacy toltecdb:"
  if [ -f run/toltecdb_last_30days.sqlite ]; then
    size=$(du -h run/toltecdb_last_30days.sqlite | cut -f1)
    count=$(sqlite3 run/toltecdb_last_30days.sqlite "SELECT COUNT(*) FROM toltec;" 2>/dev/null || echo "?")
    echo "  ✅ run/toltecdb_last_30days.sqlite ($size, $count observations)"
  else
    echo "  ❌ run/toltecdb_last_30days.sqlite not found"
  fi
  echo ""
  echo "New metadata DB:"
  if [ -f run/scratch/tolteca_metadata.db ]; then
    size=$(du -h run/scratch/tolteca_metadata.db | cut -f1)
    count=$(sqlite3 run/scratch/tolteca_metadata.db "SELECT COUNT(*) FROM data_prod;" 2>/dev/null || echo "?")
    echo "  ✅ run/scratch/tolteca_metadata.db ($size, $count products)"
  else
    echo "  ℹ️  run/scratch/tolteca_metadata.db not created yet (will be created on first Dagster run)"
  fi

# Open sqlite3 shell for toltecdb
db-legacy:
  @sqlite3 ../run/toltecdb_last_30days.sqlite

# Open sqlite3 shell for tolteca_db
db-new:
  @sqlite3 ../run/scratch/tolteca_metadata.db

# =============================================================================
# HELPERS
# =============================================================================

# Show help
help:
  @echo "TolTECA Web V2 Development Commands"
  @echo "===================================="
  @echo ""
  @echo "Setup:"
  @echo "  just init           - Create .env file"
  @echo "  just config         - Show current configuration"
  @echo ""
  @echo "Run (LEGACY mode - direct toltecdb):"
  @echo "  just legacy         - Start webapp with legacy database"
  @echo "  just webapp         - Run webapp script directly"
  @echo ""
  @echo "Run (NEW mode - tolteca_db with Dagster):"
  @echo "  just new            - Start Dagster + webapp"
  @echo "  just dagster        - Run Dagster script directly"
  @echo ""
  @echo "Monitor:"
  @echo "  just status         - Check running services"
  @echo "  just attach-legacy  - Attach to legacy session"
  @echo "  just attach-new     - Attach to new session"
  @echo "  just kill-legacy    - Stop legacy session"
  @echo "  just kill-new       - Stop new session"
  @echo "  just kill-all       - Stop all sessions"
  @echo ""
  @echo "Database:"
  @echo "  just check-db       - Check database files"
  @echo "  just db-legacy      - Open sqlite3 shell (toltecdb)"
  @echo "  just db-new         - Open sqlite3 shell (tolteca_db)"
