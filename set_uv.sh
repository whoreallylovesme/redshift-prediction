set -euo pipefail

echo ">> setting uv dependecies"

uv sync && uv run pre-commit install
