"""Community detection that splits the co-access graph into stable regions."""

from __future__ import annotations

from collections import defaultdict

from warp.models import Region
from .coaccess_graph import CoaccessGraph


class RegionPartitioner:
    """Leiden partition of the co-access graph into stable regions."""

    def __init__(self, resolution: float = 1.0, seed: int = 42, min_region_size: int = 1) -> None:
        self.resolution = resolution
        self.seed = seed
        self.min_region_size = min_region_size

    def partition(self, graph: CoaccessGraph) -> list[Region]:
        """Run Leiden, merge tiny communities, and assign stable rXXXX IDs."""
        membership = self._leiden(graph)
        groups: dict[int, list[str]] = defaultdict(list)
        for node, community in zip(graph.nodes, membership):
            groups[int(community)].append(node)
        groups = self._merge_small(groups, graph)
        ordered = sorted((sorted(ids) for ids in groups.values()), key=lambda ids: (ids[0], len(ids)))
        return [Region(f"r{index:04d}", ids) for index, ids in enumerate(ordered)]

    def _leiden(self, graph: CoaccessGraph) -> list[int]:
        import igraph as ig
        import leidenalg as la
        index = {node: i for i, node in enumerate(graph.nodes)}
        edges = [(index[a], index[b]) for a, b in graph.edges]
        weights = list(graph.edges.values())
        value = ig.Graph(n=len(graph.nodes), edges=edges, directed=False)
        partition = la.find_partition(
            value, la.RBConfigurationVertexPartition, weights=weights,
            resolution_parameter=self.resolution, seed=self.seed,
        )
        return list(partition.membership)

    def _merge_small(self, groups: dict[int, list[str]], graph: CoaccessGraph) -> dict[int, list[str]]:
        """Merge tiny communities into the strongest eligible neighbor."""
        if self.min_region_size <= 1 or len(groups) <= 1:
            return groups
        node_group = {node: group for group, nodes in groups.items() for node in nodes}
        neighbors = graph.neighbors()
        for group in list(sorted(groups)):
            nodes = groups.get(group, [])
            if not nodes or len(nodes) >= self.min_region_size:
                continue
            scores: dict[int, float] = defaultdict(float)
            for node in nodes:
                for other, weight in neighbors[node].items():
                    target = node_group[other]
                    if target != group and len(groups.get(target, [])) >= self.min_region_size:
                        scores[target] += weight
            candidates = [key for key, value in groups.items() if key != group and value]
            if not candidates:
                continue
            target = min(scores, key=lambda key: (-scores[key], key)) if scores else min(candidates)
            groups[target].extend(nodes)
            groups[group] = []
            for node in nodes:
                node_group[node] = target
        return {key: value for key, value in groups.items() if value}
