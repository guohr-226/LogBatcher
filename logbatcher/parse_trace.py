from __future__ import annotations

from dataclasses import asdict, dataclass, field


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


@dataclass
class TokenTrace:
    step: int
    prefix: list[str]
    slm_token: str
    accepted_token: str
    token_source: str
    router_action: str
    route_probability: float | None
    entropy: float | None = None
    margin: float | None = None
    top1_logprob: float | None = None
    top2_logprob: float | None = None
    cache_best_similarity: float | None = None
    cache_candidate_count: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class RoutedParseResult:
    final_template: str
    source: str
    router_trigger_count: int
    routed_token_count: int
    token_trace: list[TokenTrace]

    @classmethod
    def from_payload(cls, payload: dict):
        if "final_template" not in payload:
            raise ValueError("Routed parse payload must include final_template.")

        token_trace = []
        for item in payload.get("token_trace") or []:
            if isinstance(item, TokenTrace):
                token_trace.append(item)
            elif isinstance(item, dict):
                token_trace.append(TokenTrace(
                    step=item.get("step", 0),
                    prefix=item.get("prefix") or [],
                    slm_token=item.get("slm_token", ""),
                    accepted_token=item.get("accepted_token", ""),
                    token_source=item.get("token_source", ""),
                    router_action=item.get("router_action", ""),
                    route_probability=item.get("route_probability"),
                    entropy=item.get("entropy"),
                    margin=item.get("margin"),
                    top1_logprob=item.get("top1_logprob"),
                    top2_logprob=item.get("top2_logprob"),
                    cache_best_similarity=item.get("cache_best_similarity"),
                    cache_candidate_count=item.get("cache_candidate_count"),
                    warnings=item.get("warnings") or [],
                ))

        router_trigger_count = payload.get("router_trigger_count")
        if router_trigger_count is None:
            router_trigger_count = sum(
                1 for trace in token_trace
                if trace.router_action == "CALL_LLM_TOKEN"
            )

        routed_token_count = payload.get("routed_token_count")
        if routed_token_count is None:
            routed_token_count = sum(
                1 for trace in token_trace
                if trace.token_source == "llm"
            )

        return cls(
            final_template=payload["final_template"],
            source=payload.get("source", ""),
            router_trigger_count=_coerce_int(router_trigger_count),
            routed_token_count=_coerce_int(routed_token_count),
            token_trace=token_trace,
        )


@dataclass
class CorrectionSignal:
    cache_match_type: str
    matched_template_id: int | None
    best_similarity: float
    llm_used: bool
    final_template: str
    slm_template: str | None = None
    router_trigger_count: int = 0
    routed_token_count: int = 0
    token_trace: list[TokenTrace] = field(default_factory=list)
    template_changed: bool = False
    conflict: bool = False

    def to_dict(self):
        data = asdict(self)
        data["token_trace"] = [asdict(trace) for trace in self.token_trace]
        return data
