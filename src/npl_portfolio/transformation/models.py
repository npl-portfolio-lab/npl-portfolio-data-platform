from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TransformationResult:
    source_file: str
    output_file: str

    source_size_mb: float
    output_size_mb: float

    row_count: int
    batch_count: int

    compression_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
