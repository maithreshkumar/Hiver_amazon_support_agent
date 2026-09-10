from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(slots=True)
class UnionFind:
    parent: dict[str, str]

    def __init__(self) -> None:
        self.parent = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def component_map(ids: Iterable[str], edges: Iterable[tuple[str, str]]) -> dict[str, str]:
    uf = UnionFind()
    for item in ids:
        uf.add(str(item))
    for left, right in edges:
        if left in uf.parent and right in uf.parent:
            uf.union(left, right)
    members: dict[str, list[str]] = defaultdict(list)
    for item in uf.parent:
        members[uf.find(item)].append(item)
    mapping: dict[str, str] = {}
    for values in members.values():
        canonical = min(values, key=lambda value: (int(value) if value.isdigit() else float("inf"), value))
        for value in values:
            mapping[value] = canonical
    return mapping

