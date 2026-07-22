#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
usage() {
  echo "usage: $0 [--artifacts]" >&2
}

ARTIFACTS=0
case "$#:${1:-}" in
  0:)
    ;;
  1:--artifacts)
    ARTIFACTS=1
    ;;
  *)
    usage
    exit 2
    ;;
esac
uv sync --frozen; uv run pytest -q; uv run ruff check .; bash -n scripts/*.sh; git diff --check
echo MILESTONE1_CPU_VALIDATION_OK=1
if (( ARTIFACTS )); then
  for f in reports/milestone1/baseline_generation.json reports/milestone1/traced_generation.json reports/milestone1/trace.jsonl reports/milestone1/source_map.json reports/milestone1/environment.json; do [[ -s "$f" ]] || { echo "missing artifact: $f" >&2; exit 1; }; done
  echo MILESTONE1_ARTIFACT_VALIDATION_OK=1
  echo MILESTONE1_OK=1
fi
