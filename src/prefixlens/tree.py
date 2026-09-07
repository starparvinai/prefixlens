from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RadixNode:
    block_hash: int
    parent: Optional["RadixNode"]
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    last_used_at: int = 0

    def depth(self) -> int:
        d = 0
        node = self.parent
        while node is not None and node.parent is not None:
            d += 1
            node = node.parent
        return d + (1 if self.parent is not None else 0)


class RadixTree:
    def __init__(self) -> None:
        self.root = RadixNode(block_hash=0, parent=None)
        self._index: dict[int, RadixNode] = {}

    def __len__(self) -> int:
        return len(self._index)

    def match_and_insert(self, chain: list[int], now: int) -> tuple[int, list[RadixNode]]:
        """Walk `chain` from the root, return (matched_depth, all_nodes_in_chain).

        Every node touched — matched or newly inserted — has last_used_at set to `now`.
        """
        node = self.root
        matched = 0
        touched: list[RadixNode] = []
        i = 0
        while i < len(chain) and chain[i] in node.children:
            node = node.children[chain[i]]
            node.last_used_at = now
            touched.append(node)
            matched += 1
            i += 1

        while i < len(chain):
            child = RadixNode(block_hash=chain[i], parent=node, last_used_at=now)
            node.children[chain[i]] = child
            self._index[chain[i]] = child
            node = child
            touched.append(node)
            i += 1

        return matched, touched
