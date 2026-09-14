from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ValueQualityResult:
    table: str
    column: str
    rule: str

    rows_evaluated: int
    null_rows: int
    invalid_rows: int
    invalid_percentage: float

    invalid_samples: list[Any]

    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
