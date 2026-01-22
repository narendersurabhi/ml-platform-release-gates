# ML Platform Release Gates (MLPRG)

Production-grade, testable, deployable, observable, and maintainable reference system for **Model Governance & Release Gatekeeping**.

## Quickstart

```bash
make setup
make pipeline
make run
```

Or run everything with Docker Compose:

```bash
cp .env.example .env
make compose-up
```

## Architecture

```
                +----------------------+
                |   Training Pipeline  |
                |  (train/eval/promote)|
                +----------+-----------+
                           |
                           v
+-----------+     +--------+---------+     +----------------+
|  MinIO    |<--->| Artifact Store   |<--->| Registry (SQLite)
+-----------+     +------------------+     +----------------+
                           |
                           v
                    +-------------+
                    | FastAPI API |
                    | /predict    |
                    +-------------+
                           |
                           v
                 +---------------------+
                 | Prometheus/Grafana |
                 +---------------------+
```

## Release Gate Logic

Each training run produces a candidate model. Automated evaluation includes:

- **Functional metrics**: AUC + Accuracy thresholds.
- **Quality checks**: model size limit + p95 inference latency cap.
- **Safety checks**: regex-based PII scan + prompt-injection heuristic.

If all checks pass, the model is promoted **CANDIDATE → STAGING → PRODUCTION**. If any check fails, the model is **REJECTED**.

### Adjusting Thresholds

Update thresholds through environment variables:

- `MLPRG_MIN_AUC`
- `MLPRG_MIN_ACCURACY`
- `MLPRG_MAX_MODEL_SIZE_MB`
- `MLPRG_MAX_P95_INFERENCE_MS`
- `MLPRG_ENABLE_SAFETY_CHECKS`

## API Endpoints

- `GET /healthz`
- `GET /readyz`
- `POST /predict`
- `GET /model`
- `POST /admin/reload` (requires `x-api-key`)
- `GET /metrics`

### Curl Examples

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}'

curl -X POST http://localhost:8000/admin/reload \
  -H "x-api-key: dev-api-key"
```

## Running the Pipeline Locally

```bash
make pipeline
```

This executes training, evaluation, artifact storage, registry updates, and promotion, and writes a `pipeline_output.json` summary.

## Observability

- **Prometheus** scrapes `/metrics` from the API.
- **Grafana** provides a prebuilt dashboard.

Once `make compose-up` is running:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default credentials: `admin` / `admin`)

## Development

- Lint: `make lint`
- Test: `make test`

## Optional Future Work

- Add more advanced bias/fairness evaluation.
- Add model registry migrations and versioning.
