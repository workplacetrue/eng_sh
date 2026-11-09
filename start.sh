#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1 && [ -x /opt/homebrew/bin/uv ]; then
  export PATH="/opt/homebrew/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo 'Сначала выполните ./scripts/install-macos.sh' >&2
  exit 1
fi
# Installed environment is retained; no package/network sync during normal startup.
if [ ! -x .venv/bin/english-life ]; then
  echo 'Сначала выполните ./scripts/install-macos.sh' >&2
  exit 1
fi
exec .venv/bin/english-life "$@"
