from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass

from logbatcher.util import count_message_tokens, count_prompt_tokens


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0


def _round_sec(value):
    return round(value, 3)


def _response_to_dict(response_obj):
    if response_obj is None:
        return {}
    if isinstance(response_obj, dict):
        return response_obj
    if hasattr(response_obj, "model_dump"):
        return response_obj.model_dump()
    if hasattr(response_obj, "dict"):
        return response_obj.dict()
    return {}


def _get_attr(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _coerce_int(value, default=0):
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip().replace(",", "")
            if stripped == "":
                return default
            return int(float(stripped))
    except (TypeError, ValueError):
        return default
    return default


def _coerce_float(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip().replace(",", "")
            if stripped == "":
                return default
            return float(stripped)
    except (TypeError, ValueError):
        return default
    return default


def _first_value(obj, names):
    for name in names:
        value = _get_attr(obj, name)
        if value is not None:
            return value
    return None


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    source: str = "missing"


def _normalise_usage(candidate, source):
    if candidate is None:
        return None

    prompt_tokens = _coerce_int(_first_value(candidate, ("prompt_tokens", "input_tokens")))
    completion_tokens = _coerce_int(
        _first_value(candidate, ("completion_tokens", "output_tokens"))
    )
    total_tokens = _coerce_int(_first_value(candidate, ("total_tokens", "tokens")))

    if total_tokens <= 0 and (prompt_tokens > 0 or completion_tokens > 0):
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens <= 0 and completion_tokens <= 0 and total_tokens <= 0:
        return None

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        source=source,
    )


def _candidate_containers(response_obj):
    response_dict = _response_to_dict(response_obj)
    containers = [response_obj, response_dict]

    for container in list(containers):
        if container is None:
            continue
        for key in ("model_extra", "extra_body", "metadata"):
            nested = _get_attr(container, key)
            if nested is not None:
                containers.append(nested)

    return containers


def _find_usage(response_obj, names):
    for container in _candidate_containers(response_obj):
        if container is None:
            continue
        for name in names:
            usage = _normalise_usage(_get_attr(container, name), name)
            if usage is not None:
                return usage
    return None


def _find_latency_sec(response_obj, names):
    for container in _candidate_containers(response_obj):
        if container is None:
            continue
        for name in names:
            value = _coerce_float(_get_attr(container, name))
            if value is None:
                continue
            if name.endswith("_ms"):
                value = value / 1000
            return value, name
    return None, "missing"


def _estimate_usage(messages, answer):
    try:
        prompt_tokens = count_message_tokens(messages, "gpt-4o-mini")
    except Exception:
        prompt_tokens = sum(len((message.get("content") or "").split()) for message in messages)

    try:
        completion_tokens = count_prompt_tokens(answer or "", "gpt-4o-mini")
    except Exception:
        completion_tokens = len((answer or "").split())

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        source="estimated_from_messages",
    )


class IndependentRuntimeMetrics:
    def __init__(self, dataset, total_logs):
        self.dataset = dataset
        self.total_logs = total_logs

        self.total_model_tokens = 0
        self.large_model_tokens = 0
        self.total_model_parse_time_sec = 0.0
        self.large_model_parse_time_sec = 0.0
        self.model_invocations = 0
        self.large_model_invocations = 0

        self.router_trigger_count = 0
        self.large_model_routed_tokens = 0
        self.routing_trace_tokens = 0

        self.cache_lookups = 0
        self.cache_matched_lookups = 0
        self.cache_trusted_hit_lookups = 0
        self.cache_hit_logs = 0
        self.pre_chunk_cache_hit_logs = 0
        self.pruning_cache_hit_logs = 0

        self.cache_template_checked_count = 0
        self.cache_template_error_count = 0

        self.total_token_sources = Counter()
        self.large_token_sources = Counter()
        self.large_time_sources = Counter()

    def record_cache_lookup(self, stage, matched, trusted):
        self.cache_lookups += 1
        if matched:
            self.cache_matched_lookups += 1
        if matched and trusted:
            self.cache_trusted_hit_lookups += 1

    def record_cache_hit_logs(self, stage, count):
        count = max(_coerce_int(count), 0)
        self.cache_hit_logs += count
        if stage == "pre_chunk":
            self.pre_chunk_cache_hit_logs += count
        elif stage == "pruning":
            self.pruning_cache_hit_logs += count

    def record_cache_template_check(
            self,
            matched_template,
            final_template,
            checked_by_model,
            error):
        if not checked_by_model or not matched_template:
            return

        self.cache_template_checked_count += 1
        normalised_match = str(matched_template).strip()
        normalised_final = str(final_template).strip()
        if error or normalised_match != normalised_final:
            self.cache_template_error_count += 1

    def record_model_call(
            self,
            response_obj,
            messages,
            answer,
            latency_sec,
            is_r2r,
            router_trigger_count=0,
            routed_token_count=0,
            token_trace_count=0):
        latency_sec = _coerce_float(latency_sec, 0.0) or 0.0
        routed_token_count = max(_coerce_int(routed_token_count), 0)
        token_trace_count = max(_coerce_int(token_trace_count), 0)
        router_trigger_count = max(_coerce_int(router_trigger_count), 0)

        total_usage = _find_usage(
            response_obj,
            (
                "usage",
                "total_usage",
                "aggregate_usage",
                "combined_usage",
                "endpoint_usage",
                "r2r_usage",
            ),
        )
        if total_usage is None:
            total_usage = _estimate_usage(messages, answer)

        large_usage = _find_usage(
            response_obj,
            (
                "reference_usage",
                "dashscope_usage",
                "large_model_usage",
                "large_usage",
                "llm_usage",
                "routed_usage",
            ),
        )
        if large_usage is None and not is_r2r:
            large_usage = TokenUsage(
                prompt_tokens=total_usage.prompt_tokens,
                completion_tokens=total_usage.completion_tokens,
                total_tokens=total_usage.total_tokens,
                source="single_model_total_usage",
            )
        elif large_usage is None and routed_token_count > 0:
            large_usage = TokenUsage(
                prompt_tokens=0,
                completion_tokens=routed_token_count,
                total_tokens=routed_token_count,
                source="routed_token_count_fallback",
            )
        elif large_usage is None:
            large_usage = TokenUsage(source="missing")

        aggregate_usage_sources = {
            "total_usage",
            "aggregate_usage",
            "combined_usage",
            "endpoint_usage",
            "r2r_usage",
        }
        if (
                is_r2r
                and large_usage.total_tokens > 0
                and total_usage.source not in aggregate_usage_sources):
            total_usage = TokenUsage(
                prompt_tokens=total_usage.prompt_tokens + large_usage.prompt_tokens,
                completion_tokens=(
                    total_usage.completion_tokens + large_usage.completion_tokens
                ),
                total_tokens=total_usage.total_tokens + large_usage.total_tokens,
                source=f"{total_usage.source}_plus_{large_usage.source}",
            )

        large_latency, large_latency_source = _find_latency_sec(
            response_obj,
            (
                "reference_latency_sec",
                "dashscope_latency_sec",
                "large_model_latency_sec",
                "large_latency_sec",
                "llm_latency_sec",
                "reference_latency_ms",
                "dashscope_latency_ms",
                "large_model_latency_ms",
                "large_latency_ms",
                "llm_latency_ms",
            ),
        )
        if large_latency is None:
            if not is_r2r:
                large_latency = latency_sec
                large_latency_source = "single_model_endpoint_latency"
            elif large_usage.total_tokens > 0 or routed_token_count > 0:
                large_latency = latency_sec
                large_latency_source = "endpoint_latency_fallback"
            else:
                large_latency = 0.0
                large_latency_source = "missing"

        self.model_invocations += 1
        self.total_model_tokens += total_usage.total_tokens
        self.total_model_parse_time_sec += latency_sec
        self.total_token_sources[total_usage.source] += 1

        if large_usage.total_tokens > 0 or large_latency > 0:
            self.large_model_invocations += 1
        self.large_model_tokens += large_usage.total_tokens
        self.large_model_parse_time_sec += large_latency
        self.large_token_sources[large_usage.source] += 1
        self.large_time_sources[large_latency_source] += 1

        self.router_trigger_count += router_trigger_count
        self.large_model_routed_tokens += routed_token_count
        self.routing_trace_tokens += token_trace_count

    def to_dict(self):
        required_metrics = {
            "total_token_consumption": self.total_model_tokens,
            "large_model_token_consumption": self.large_model_tokens,
            "total_model_parse_time_sec": _round_sec(self.total_model_parse_time_sec),
            "large_model_parse_time_sec": _round_sec(self.large_model_parse_time_sec),
            "large_model_token_routing_rate": _safe_div(
                self.large_model_routed_tokens,
                self.routing_trace_tokens,
            ),
            "cache_hit_rate": _safe_div(self.cache_hit_logs, self.total_logs),
            "cache_template_error_rate": _safe_div(
                self.cache_template_error_count,
                self.cache_template_checked_count,
            ),
        }

        return {
            "dataset": self.dataset,
            "total_logs": self.total_logs,
            **required_metrics,
            "model_invocations": self.model_invocations,
            "large_model_invocations": self.large_model_invocations,
            "router_trigger_count": self.router_trigger_count,
            "large_model_routed_tokens": self.large_model_routed_tokens,
            "routing_trace_tokens": self.routing_trace_tokens,
            "cache_lookups": self.cache_lookups,
            "cache_matched_lookups": self.cache_matched_lookups,
            "cache_trusted_hit_lookups": self.cache_trusted_hit_lookups,
            "cache_lookup_hit_rate": _safe_div(
                self.cache_trusted_hit_lookups,
                self.cache_lookups,
            ),
            "cache_hit_logs": self.cache_hit_logs,
            "pre_chunk_cache_hit_logs": self.pre_chunk_cache_hit_logs,
            "pruning_cache_hit_logs": self.pruning_cache_hit_logs,
            "cache_template_checked_count": self.cache_template_checked_count,
            "cache_template_error_count": self.cache_template_error_count,
            "total_token_sources": dict(self.total_token_sources),
            "large_token_sources": dict(self.large_token_sources),
            "large_time_sources": dict(self.large_time_sources),
            "required_metrics": required_metrics,
        }


def _aggregate_dataset_metrics(dataset_metrics):
    summary = {
        "dataset": "ALL",
        "total_logs": 0,
        "total_token_consumption": 0,
        "large_model_token_consumption": 0,
        "total_model_parse_time_sec": 0.0,
        "large_model_parse_time_sec": 0.0,
        "model_invocations": 0,
        "large_model_invocations": 0,
        "router_trigger_count": 0,
        "large_model_routed_tokens": 0,
        "routing_trace_tokens": 0,
        "cache_lookups": 0,
        "cache_matched_lookups": 0,
        "cache_trusted_hit_lookups": 0,
        "cache_hit_logs": 0,
        "pre_chunk_cache_hit_logs": 0,
        "pruning_cache_hit_logs": 0,
        "cache_template_checked_count": 0,
        "cache_template_error_count": 0,
    }

    for metrics in dataset_metrics:
        for key in list(summary.keys()):
            if key == "dataset":
                continue
            summary[key] += metrics.get(key, 0)

    summary["total_model_parse_time_sec"] = _round_sec(
        summary["total_model_parse_time_sec"]
    )
    summary["large_model_parse_time_sec"] = _round_sec(
        summary["large_model_parse_time_sec"]
    )
    summary["large_model_token_routing_rate"] = _safe_div(
        summary["large_model_routed_tokens"],
        summary["routing_trace_tokens"],
    )
    summary["cache_hit_rate"] = _safe_div(
        summary["cache_hit_logs"],
        summary["total_logs"],
    )
    summary["cache_lookup_hit_rate"] = _safe_div(
        summary["cache_trusted_hit_lookups"],
        summary["cache_lookups"],
    )
    summary["cache_template_error_rate"] = _safe_div(
        summary["cache_template_error_count"],
        summary["cache_template_checked_count"],
    )
    summary["required_metrics"] = {
        "total_token_consumption": summary["total_token_consumption"],
        "large_model_token_consumption": summary["large_model_token_consumption"],
        "total_model_parse_time_sec": summary["total_model_parse_time_sec"],
        "large_model_parse_time_sec": summary["large_model_parse_time_sec"],
        "large_model_token_routing_rate": summary["large_model_token_routing_rate"],
        "cache_hit_rate": summary["cache_hit_rate"],
        "cache_template_error_rate": summary["cache_template_error_rate"],
    }
    return summary


def write_independent_metrics(output_dir, metrics):
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "independent_runtime_metrics.json")
    csv_file = os.path.join(output_dir, "independent_runtime_metrics.csv")

    payload = {"datasets": {}}
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as file:
            existing = json.load(file)
        if isinstance(existing, dict):
            payload["datasets"] = existing.get("datasets", {})

    payload["datasets"][metrics.dataset] = metrics.to_dict()
    datasets = payload["datasets"]
    payload["summary"] = _aggregate_dataset_metrics(list(datasets.values()))

    with open(metrics_file, "w") as file:
        json.dump(payload, file, indent=2)

    csv_fields = [
        "dataset",
        "total_logs",
        "total_token_consumption",
        "large_model_token_consumption",
        "total_model_parse_time_sec",
        "large_model_parse_time_sec",
        "large_model_token_routing_rate",
        "cache_hit_rate",
        "cache_template_error_rate",
        "model_invocations",
        "large_model_invocations",
        "router_trigger_count",
        "large_model_routed_tokens",
        "routing_trace_tokens",
        "cache_lookups",
        "cache_matched_lookups",
        "cache_trusted_hit_lookups",
        "cache_lookup_hit_rate",
        "cache_hit_logs",
        "pre_chunk_cache_hit_logs",
        "pruning_cache_hit_logs",
        "cache_template_checked_count",
        "cache_template_error_count",
    ]
    rows = [payload["summary"]]
    rows.extend(datasets[name] for name in sorted(datasets))
    with open(csv_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, 0) for field in csv_fields})
