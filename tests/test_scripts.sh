#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for script in scripts/*.sh; do
  bash -n "$script"
done

if REALKV_RUNTIME_IMAGE=nvcr.io/example/runtime:tag bash scripts/run_baseline.sh >/tmp/realkv-script-test.out 2>&1; then
  echo "unpinned runtime image was accepted" >&2
  exit 1
fi

if env -u REALKV_RUNTIME_IMAGE bash scripts/run_baseline.sh >/tmp/realkv-script-test.out 2>&1; then
  echo "missing runtime image was accepted" >&2
  exit 1
fi

if REALKV_RUNTIME_IMAGE='nvcr.io/example/runtime@sha256:deadbeef' bash scripts/run_baseline.sh >/tmp/realkv-script-test.out 2>&1; then
  echo "model download without consent was accepted" >&2
  exit 1
fi

set +e
bash scripts/validate_milestone1.sh --invalid >/tmp/realkv-script-test.out 2>&1
invalid_status=$?
set -e
if [[ "$invalid_status" -eq 0 ]]; then
  echo "invalid milestone argument was accepted" >&2
  exit 1
fi
test "$invalid_status" -eq 2

if REALKV_RUNTIME_IMAGE='nvcr.io/example/runtime@sha256:deadbeef' REALKV_ALLOW_MODEL_DOWNLOAD=1 \
  bash scripts/run_traced_request.sh >/tmp/realkv-script-test.out 2>&1; then
  echo "unconfigured traced execution was accepted" >&2
  exit 1
fi
grep -q 'Tracing requires' /tmp/realkv-script-test.out

echo SHELL_ENTRY_POINT_TESTS_OK=1
