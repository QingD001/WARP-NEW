"""WARP-G 的无预算区域选择器。"""

from __future__ import annotations

import random

from warp.models import RegionFeatures


class RegionSelector:
    """只替换区域排序公式；WARP 按 score>0 自然结束，controls 取同样多的区域。

    对照只保留 frequency / gain（以及随机打乱）；不再用构图成本排序或截断。
    """

    METHODS = {"warp", "random_region", "frequency_only", "gain_only"}

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def score(self, region_id: str, features: dict[str, RegionFeatures],
              costs: dict[str, float], estimated_gains: dict[str, float]) -> float:
        del costs
        return features[region_id].query_freq * max(estimated_gains[region_id], 0.0)

    def rank(
        self, method: str, features: dict[str, RegionFeatures], costs: dict[str, float],
        estimated_gains: dict[str, float],
    ) -> list[str]:
        if set(features) != set(costs) or any(cost <= 0 for cost in costs.values()):
            raise ValueError("Selection requires aligned regions and positive costs")
        method = method.lower()
        if method not in self.METHODS:
            raise ValueError(f"Unknown selection method: {method}")
        ids = sorted(features)
        if set(estimated_gains) != set(features):
            raise ValueError("Selection requires one estimated gain per region")
        if method == "random_region":
            ranked = ids.copy()
            random.Random(self.seed).shuffle(ranked)
            return ranked
        if method == "frequency_only":
            return sorted(ids, key=lambda key: (-features[key].query_freq, key))
        if method == "gain_only":
            return sorted(ids, key=lambda key: (-estimated_gains[key], key))
        return sorted(
            ids,
            key=lambda key: (-self.score(key, features, costs, estimated_gains), key),
        )

    def select(
        self, method: str, features: dict[str, RegionFeatures], costs: dict[str, float],
        estimated_gains: dict[str, float], limit: int | None = None,
    ) -> list[str]:
        ranked = self.rank(method, features, costs, estimated_gains)
        method = method.lower()
        if method == "warp":
            return [
                region_id for region_id in ranked
                if self.score(region_id, features, costs, estimated_gains) > 0
            ]
        if limit is None:
            raise ValueError("Control selectors must reuse WARP's selected region count")
        if limit < 0:
            raise ValueError("Control selector limit must be non-negative")
        return ranked[:limit]


BudgetSelector = RegionSelector
