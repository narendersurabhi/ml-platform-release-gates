from __future__ import annotations

from prometheus_client import Counter, Histogram

request_count = Counter(
    "mlprg_request_count",
    "Total request count",
    ["endpoint", "status"],
)
request_latency_seconds = Histogram(
    "mlprg_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
model_reload_count = Counter(
    "mlprg_model_reload_count",
    "Model reload count",
)

eval_pass_total = Counter(
    "mlprg_eval_pass_total",
    "Total evaluation passes",
)

eval_fail_total = Counter(
    "mlprg_eval_fail_total",
    "Total evaluation failures",
)
