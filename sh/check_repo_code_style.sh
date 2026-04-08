set -euo pipefail

cd "$(dirname "$0")/.."

echo ">> running repos files through ruff and mypy"

echo ">> ruff"
uv run ruff check .

echo ">> mypy"
uv run mypy .
