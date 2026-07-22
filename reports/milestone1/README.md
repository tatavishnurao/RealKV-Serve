# Milestone-one report directory

Runtime-generated artifacts are intentionally separated by sensitivity and size.

The `raw/` directory is ignored. It may contain trace JSONL, complete stdout and
stderr, container inspection payloads, hostnames, and large runtime logs while the
target-machine run is being investigated. Raw files must never contain secrets.

The `published/` directory contains only small, sanitized evidence intended for
version control: environment, runtime, baseline, trace, and source-map summaries,
plus `FINDINGS.md`. Published files must omit model weights, cache listings with
private paths, credentials, hostnames, and full logs.

This directory must not contain model weights, caches, containers, credentials, or
fabricated traces.
