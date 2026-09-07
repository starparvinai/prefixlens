from prefixlens.hashing import block_hash_chain, fnv1a_64


def test_fnv1a_64_known_vectors():
    assert fnv1a_64(b"") == 0xCBF29CE484222325
    assert fnv1a_64(b"a") == 0xAF63DC4C8601EC8C
    assert fnv1a_64(b"foobar") == 0x85944171F73967E8


def test_block_hash_chain_produces_one_hash_per_complete_block():
    tokens = tuple(range(32))
    chain = block_hash_chain(tokens, block_size=16)
    assert len(chain) == 2


def test_block_hash_chain_ignores_trailing_partial_block():
    tokens = tuple(range(20))
    chain = block_hash_chain(tokens, block_size=16)
    assert len(chain) == 1


def test_block_hash_chain_same_prefix_same_hashes():
    a = tuple(range(32))
    b = tuple(range(16)) + tuple(range(1000, 1016))
    chain_a = block_hash_chain(a, block_size=16)
    chain_b = block_hash_chain(b, block_size=16)
    assert chain_a[0] == chain_b[0]  # first block identical
    assert chain_a[1] != chain_b[1]  # second block diverges


def test_block_hash_chain_parent_dependency():
    """Same tokens in a block get DIFFERENT hashes if their parents differ."""
    a = (1,) * 16 + (2,) * 16
    b = (9,) * 16 + (2,) * 16
    chain_a = block_hash_chain(a, block_size=16)
    chain_b = block_hash_chain(b, block_size=16)
    assert chain_a[0] != chain_b[0]
    assert chain_a[1] != chain_b[1]  # same block-2 content, different because parent differs
