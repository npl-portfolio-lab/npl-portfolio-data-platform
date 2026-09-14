from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class QualityCheckResult:
    table: str
    rule: str
    column: str | None
    passed: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    dataset: str
    checks: list[QualityCheckResult] = field(default_factory=list)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def passed_checks(self) -> int:
        return sum(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> int:
        return self.total_checks - self.passed_checks

    @property
    def passed(self) -> bool:
        return self.failed_checks == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "passed": self.passed,
            "summary": {
                "total_checks": self.total_checks,
                "passed_checks": self.passed_checks,
                "failed_checks": self.failed_checks,
            },
            "checks": [check.to_dict() for check in self.checks],
        }
