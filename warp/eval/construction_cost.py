"""Per-dimension graph-cost aggregation and token-efficiency metrics."""

from __future__ import annotations

from typing import Any

from warp.models import ConstructionCost


def consumed_tokens(payload: dict[str, Any] | None) -> int:
    """Sum tokens from a deployment / first-run / online cost dict."""
    if not payload:
        return 0
    if "input_tokens" in payload or "embedding_tokens" in payload:
        return (
            int(payload.get("input_tokens") or 0)
            + int(payload.get("output_tokens") or 0)
            + int(payload.get("embedding_tokens") or 0)
        )
    return int(payload.get("logical_input_tokens") or 0) + int(payload.get("logical_output_tokens") or 0)


def token_efficiency(quality: float, tokens: int) -> float | None:
    """Quality per consumed token; undefined when tokens == 0."""
    if tokens <= 0:
        return None
    return float(quality) / float(tokens)


def attach_token_efficiency(row: dict[str, Any], extra_online: int = 0) -> None:
    """Attach token efficiency with and without design cost."""
    deployed = consumed_tokens(row.get("deployment_cost") or row.get("actual_construction_cost"))
    first_payload = row.get("first_run_cost_including_probe")
    first_run = consumed_tokens(first_payload) if first_payload else deployed
    online = consumed_tokens(row.get("online_retrieval_cost")) + extra_online
    excluding = deployed + online
    including = first_run + online
    row["total_tokens_excluding_design"] = excluding
    row["total_tokens_including_design"] = including
    evidence = row.get("complete_evidence@10")
    if evidence is not None:
        row["token_efficiency_excluding_design"] = token_efficiency(float(evidence), excluding)
        row["token_efficiency_including_design"] = token_efficiency(float(evidence), including)
    answer_em = row.get("answer_em")
    if answer_em is not None:
        row["token_efficiency_em_excluding_design"] = token_efficiency(float(answer_em), excluding)
        row["token_efficiency_em_including_design"] = token_efficiency(float(answer_em), including)


def aggregate_costs(costs: list[ConstructionCost]) -> ConstructionCost:
    """Sum tokens, wall time, graph size, and storage."""
    return ConstructionCost(
        input_tokens=sum(cost.input_tokens for cost in costs),
        output_tokens=sum(cost.output_tokens for cost in costs),
        embedding_tokens=sum(cost.embedding_tokens for cost in costs),
        wall_seconds=sum(cost.wall_seconds for cost in costs),
        nodes=sum(cost.nodes for cost in costs),
        edges=sum(cost.edges for cost in costs),
        storage_bytes=sum(cost.storage_bytes for cost in costs),
        estimated_usd=sum(cost.estimated_usd for cost in costs),
    )
