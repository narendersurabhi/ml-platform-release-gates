# AGENTS Log

This file tracks changes made to the repository for agent context.

## 2025-09-23
- Initialized repository structure under `src/mlprg` with API, pipelines, registry, storage, observability, contracts, and utils packages.
- Added FastAPI service, pipeline runner, evaluators, registry, artifact store implementations, and supporting configs.
- Added Docker Compose stack, Grafana/Prometheus configs, CI workflow, tests, Makefile, and developer tooling configs.
- Added README, Dockerfile, and environment examples for local usage plus a Grafana dashboard and Prometheus scrape configuration.

## 2025-09-24
- Fixed lint issues, formatted code, and added missing type annotations for mypy compliance.
- Updated mypy configuration to ignore missing imports and allow subclassing Any for third-party libraries.
