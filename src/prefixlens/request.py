from dataclasses import dataclass, field


@dataclass(frozen=True)
class Request:
    request_id: str
    token_ids: tuple[int, ...]
    tags: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, tuple):
            object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if isinstance(self.tags, dict):
            object.__setattr__(self, "tags", tuple(sorted(self.tags.items())))
