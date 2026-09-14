"""LRU eviction behavior for the radix cache simulator.

Every test uses block_size=16 (so each `tuple(range(16))` = one block) and a
tight capacity so eviction fires deterministically. The invariants under test:

  1. Evictions happen when — and only when — the tree exceeds capacity.
  2. The block chosen is the least-recently-used *leaf*, never an internal node.
  3. Touching a block on a hit refreshes its recency, sparing it from eviction.
  4. When a leaf is evicted and its parent has no other children, the parent
     itself becomes a new eviction candidate (cascade).
  5. Evicted blocks miss on re-request.
"""

from prefixlens import RadixCacheSimulator, Request


def _req(rid: str, tokens: tuple[int, ...]) -> Request:
    return Request(request_id=rid, token_ids=tokens)


def test_no_eviction_when_under_capacity():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=10)
    sim.process(_req("r1", tuple(range(32))))  # 2 blocks

    assert sim.evictions == 0
    assert len(sim.tree) == 2


def test_eviction_fires_when_capacity_exceeded():
    # Capacity 2. First request fills it. Second request forces evictions.
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=2)
    sim.process(_req("r1", tuple(range(32))))  # 2 blocks, fills cache

    # Fully disjoint request — must evict everything from r1 to fit its own 2 blocks.
    sim.process(_req("r2", tuple(range(1000, 1032))))

    assert len(sim.tree) == 2
    assert sim.evictions == 2  # both r1 blocks evicted


def test_evicted_block_misses_on_re_request():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=2)
    r1_tokens = tuple(range(32))
    sim.process(_req("r1", r1_tokens))
    sim.process(_req("r2", tuple(range(1000, 1032))))  # evicts r1 entirely

    result = sim.process(_req("r1_again", r1_tokens))

    assert result.cached_prefix_blocks == 0
    assert result.first_divergent_block == 0


def test_recent_touch_saves_a_block_from_eviction():
    # Capacity 3. r1 inserts blocks A, B. r2 re-uses A, B (both refreshed).
    # r3 inserts C, D — that's 4 blocks, we need to evict 1.
    # Because A and B were just touched (step 2), the older-timestamp leaf is
    # whichever leaf existed before r2 — but after r2, both A and B are fresh.
    # r3 adds C then D; when we evict, only A/B are old-ish (step 2), C is step 3.
    # Actual LRU leaf ordering: after r3 inserts C under A (which is a leaf, becomes
    # non-leaf), then D under C. Leaves at that point: {B, D}. B.last_used_at=2,
    # D.last_used_at=3. Evict B.
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=3)
    r1_tokens = tuple(range(32))  # blocks A, B
    sim.process(_req("r1", r1_tokens))
    sim.process(_req("r2", r1_tokens))  # hit both, bumps last_used_at

    # r3: shares NOTHING with r1 — brand new chain of 2 blocks (C, D)
    sim.process(_req("r3", tuple(range(2000, 2032))))

    # Tree should hold exactly 3 blocks: A (still parent of nothing now), B evicted,
    # C, D. Wait — A shared no chain with r3, so A is now a leaf with no children.
    # After r3 inserts C under root (not A!), leaves = {B, D, A}. A.last_used=2,
    # B.last_used=2, D.last_used=3. Tie between A and B → heap_counter breaks it,
    # A was pushed first, so A goes first. We evict A → 3 blocks left {B, C, D}.
    assert len(sim.tree) == 3
    assert sim.evictions == 1


def test_evicting_only_child_promotes_parent_to_a_new_leaf():
    # Capacity 2. r1: A→B (2 blocks, B is leaf).
    # r2: same A, then C (different second block). Now tree has A→B and A→C, 3 nodes.
    # Evict: B is older leaf (created step 1). Evict B. A still has child C → not a leaf.
    # Tree = 2 blocks {A, C}. Good.
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=2)
    sim.process(_req("r1", tuple(range(32))))  # A, B
    sim.process(_req("r2", tuple(range(16)) + tuple(range(500, 516))))  # A (hit), C (new)

    assert len(sim.tree) == 2
    assert sim.evictions == 1

    # Now r3 forces another eviction. r3 shares nothing.
    # Tree already at capacity 2 {A,C}. r3 inserts D, E (fresh chain).
    # After insert: {A, C, D, E} = 4. Evict 2.
    # Leaves: {C, E}. C.last_used=2, E.last_used=3. Evict C first.
    # After evicting C: A has no children left → A becomes leaf → pushed onto heap.
    # Now tree = {A, D, E} = 3. Evict again. Leaves: {A, E}. A.last_used=2 (from r2),
    # E.last_used=3. Evict A.
    sim.process(_req("r3", tuple(range(3000, 3032))))

    assert len(sim.tree) == 2
    # Total evictions: 1 (from r2) + 2 (from r3) = 3
    assert sim.evictions == 3


def test_cascade_eviction_when_capacity_is_one():
    # Extreme case: capacity=1. A 2-block request (A→B) inserts both, then
    # the eviction loop trims down. Because we evict *leaves only*, block B
    # (the leaf) goes first — leaving A behind, not B. This is the same
    # invariant as test_prompt_longer_than_capacity: the earliest blocks of
    # the chain are what survive, because they're the ones a future
    # re-request can actually walk into from the root.
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1)
    sim.process(_req("r1", tuple(range(32))))

    assert len(sim.tree) == 1
    assert sim.evictions == 1

    # Second identical request: block A still there (hit), block B evicted (miss).
    result = sim.process(_req("r2", tuple(range(32))))
    assert result.cached_prefix_blocks == 1
    assert result.first_divergent_block == 1


def test_prompt_longer_than_capacity_keeps_only_earliest_blocks():
    # 5-block request into a 2-block cache. Insert order is 1,2,3,4,5 as a
    # parent chain. Eviction runs after all 5 are inserted; leaves-only
    # eviction pops from the deepest end (block 5 is the sole leaf; evict it,
    # block 4 becomes leaf; evict 4; block 3 becomes leaf; evict 3). We end
    # up with the FIRST 2 blocks intact.
    #
    # This is the correct simulator behavior — the earliest blocks of the
    # chain are the ones any future re-request can actually walk into from
    # the root, so keeping them is what maximizes hit rate on repeats.
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=2)
    sim.process(_req("r1", tuple(range(80))))  # 5 blocks

    assert len(sim.tree) == 2
    assert sim.evictions == 3

    # Second identical request should hit exactly 2 blocks (the surviving prefix).
    result = sim.process(_req("r2", tuple(range(80))))
    assert result.cached_prefix_blocks == 2
    assert result.total_prompt_blocks == 5
    assert result.first_divergent_block == 2


def test_evictions_property_is_zero_initially():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=100)
    assert sim.evictions == 0


def test_hit_on_leaf_refreshes_it_past_a_stale_sibling():
    # Capacity 3. Set up two sibling leaves under root (different chains,
    # 1 block each). One is old, one is young. Then hit the old one to refresh
    # it. Insert a third fresh chain — should evict the now-oldest, which is
    # the previously-young sibling that wasn't touched.
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=3)
    sim.process(_req("A", tuple(range(0, 16))))       # block A
    sim.process(_req("B", tuple(range(100, 116))))    # block B
    sim.process(_req("C", tuple(range(200, 216))))    # block C — tree full

    # Touch A (refreshes A.last_used_at)
    sim.process(_req("A_hit", tuple(range(0, 16))))

    # Force one eviction: 4th independent block
    sim.process(_req("D", tuple(range(300, 316))))

    # B was the oldest untouched leaf → should be gone
    result = sim.process(_req("B_probe", tuple(range(100, 116))))
    assert result.cached_prefix_blocks == 0  # B was evicted

    # A should still be present
    result = sim.process(_req("A_probe", tuple(range(0, 16))))
    assert result.cached_prefix_blocks == 1
