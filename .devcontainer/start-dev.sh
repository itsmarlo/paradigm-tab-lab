#!/usr/bin/env bash
set -euo pipefail

# Reattaching to the same Codespace should keep the existing preview.
if curl --silent --fail http://127.0.0.1:5173/ > /dev/null; then
  exit 0
fi

nohup npm run dev > /tmp/paradigm-lab-vite.log 2>&1 < /dev/null &
