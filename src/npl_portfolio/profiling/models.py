from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int


@dataclass(frozen=True)
class TableProfile:
    file_name: str
    file_size_mb: float
    row_count: int
    column_count: int
    duplicate_rows: int
    memory_usage_mb: float
    detected_keys: list[str] = field(default_factory=list)
    columns: list[ColumnProfile] = field(default_factory=list)
    encoding: str | None = None
    chunks_processed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
