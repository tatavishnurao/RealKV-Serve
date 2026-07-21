#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${REALKV_RUNTIME_IMAGE:?Set REALKV_RUNTIME_IMAGE to an officially resolved digest-pinned NGC image}"
case "$REALKV_RUNTIME_IMAGE" in *'@sha256:'*) ;; *) echo "Refusing unpinned image: $REALKV_RUNTIME_IMAGE" >&2; exit 2;; esac
docker pull "$REALKV_RUNTIME_IMAGE" | tee "$ROOT/reports/milestone1/runtime_pull.txt"
docker image inspect "$REALKV_RUNTIME_IMAGE" > "$ROOT/reports/milestone1/runtime_image_inspect.json"
