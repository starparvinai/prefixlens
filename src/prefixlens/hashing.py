FNV_OFFSET_BASIS_64 = 0xCBF29CE484222325
FNV_PRIME_64 = 0x100000001B3
FNV_MASK_64 = 0xFFFFFFFFFFFFFFFF


def fnv1a_64(data: bytes) -> int:
    h = FNV_OFFSET_BASIS_64
    for b in data:
        h ^= b
        h = (h * FNV_PRIME_64) & FNV_MASK_64
    return h


def block_hash_chain(token_ids: tuple[int, ...], block_size: int) -> list[int]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    hashes: list[int] = []
    parent = 0
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        block = token_ids[start : start + block_size]
        payload = parent.to_bytes(8, "big") + b"".join(t.to_bytes(4, "big") for t in block)
        h = fnv1a_64(payload)
        hashes.append(h)
        parent = h
    return hashes
