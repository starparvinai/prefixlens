from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RadixNode:
    block_hash: int
    parent: Optional["RadixNode"]
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    last_used_at: int = 0

    def is_leaf(self) -> bool:
        return not self.children

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
        # Min-heap of (last_used_at, insertion_counter, node) for leaves.
        # Uses lazy deletion: entries whose node is no longer a leaf, has moved,
        # or whose timestamp doesn't match the node's current last_used_at are
        # discarded on pop.
        self._leaf_heap: list[tuple[int, int, "RadixNode"]] = []
        self._heap_counter = 0

    def __len__(self) -> int:
        return len(self._index)

    def _push_leaf(self, node: RadixNode) -> None:
        self._heap_counter += 1
        heapq.heappush(self._leaf_heap, (node.last_used_at, self._heap_counter, node))

    def match_and_insert(self, chain: list[int], now: int) -> tuple[int, list[RadixNode]]:
        """Walk `chain` from the root, return (matched_depth, all_nodes_in_chain).

        Every node touched — matched or newly inserted — has last_used_at set to `now`.
        The tail of any insertion (the new leaf) is registered with the LRU heap.
        Matched leaves have their timestamps bumped in place; the corresponding stale
        heap entries are cleaned up lazily by evict_lru_leaf.
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

        # If we inserted anything, `node` now points at the deepest new node — the
        # only genuine leaf we just created (intermediate inserts got a child
        # immediately). Register it. Any stale heap entry for the old parent —
        # which was a leaf before this insert — is discarded lazily.
        if matched < len(chain):
            self._push_leaf(node)

        return matched, touched

    def evict_lru_leaf(self) -> RadixNode | None:
        """Remove the least-recently-used leaf from the tree; return it, or None
        if the tree has nothing evictable.

        Skips stale heap entries: nodes already evicted, no-longer-leaves, or
        entries whose timestamp is older than the node's current last_used_at
        (which means the node was touched after the entry was pushed — re-push
        with the fresh timestamp and keep scanning).
        """
        while self._leaf_heap:
            ts, _cnt, node = heapq.heappop(self._leaf_heap)

            if node.parent is None:
                continue  # root or already-orphaned; defensive
            if self._index.get(node.block_hash) is not node:
                continue  # already evicted (or replaced)
            if node.children:
                continue  # no longer a leaf; a fresh child took over
            if node.last_used_at != ts:
                # Node was touched after this entry was pushed. Re-push with
                # its current recency and keep looking for the true LRU.
                self._push_leaf(node)
                continue

            parent = node.parent
            del parent.children[node.block_hash]
            del self._index[node.block_hash]

            # If evicting `node` left its parent without any other children,
            # the parent is now a leaf and becomes evictable itself. (The root
            # is never a candidate — it holds no block content.)
            if parent is not self.root and not parent.children:
                self._push_leaf(parent)

            return node

        return None
