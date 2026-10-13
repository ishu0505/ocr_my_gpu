#!/usr/bin/env bash
set -euo pipefail
[[ "$(readlink CLAUDE.md)" == "AGENTS.md" ]] || { echo "CLAUDE.md symlink incorrect"; exit 1; }
echo "Shared agent instructions symlink OK"
