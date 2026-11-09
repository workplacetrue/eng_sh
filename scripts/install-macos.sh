#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo 'Этот установщик предназначен для Apple Silicon macOS. См. README для ручного запуска.' >&2
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  else
    echo 'Установите Homebrew с https://brew.sh и повторите ./scripts/install-macos.sh' >&2
    exit 1
  fi
fi
command -v uv >/dev/null 2>&1 || brew install uv
command -v ollama >/dev/null 2>&1 || brew install ollama
uv python install 3.12
uv sync --frozen --python 3.12
# Homebrew manages a single local Ollama daemon. Never spawn a competing process.
if ! curl --noproxy '*' -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  brew services start ollama
fi
for attempt in {1..30}; do
  if curl --noproxy '*' -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    break
  fi
  if [ "$attempt" = 30 ]; then
    echo 'Ollama не запустилась. Проверьте brew services info ollama.' >&2
    exit 1
  fi
  sleep 1
done
# Explicit quantization: ~2.5 GB weights, additional RAM for context/runtime.
ollama pull qwen3:4b-q4_K_M
uv run --frozen english-life doctor
echo 'Установка завершена. Запуск: ./start.sh'
