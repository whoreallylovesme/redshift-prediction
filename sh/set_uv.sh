set -euo pipefail

cd "$(dirname "$0")/.."

echo ">> setting uv dependecies"

uv sync && uv run pre-commit install
