"""
Compare results across experiment configurations.
Usage:
    python analysis/compare_results.py
"""
import json
from dataclasses import dataclass
from pathlib import Path

RESULTS_DIR = Path("results")
CONFIGS = {
    "Baseline": RESULTS_DIR / "baseline_no_attack",
    "Attack": RESULTS_DIR / "baseline_with_attack",
    "Boundary formatting": RESULTS_DIR / "config_3_boundary_formatting",
    "Sanitization": RESULTS_DIR / "config_4_sanitization",
}


@dataclass(frozen=True)
class ResultSummary:
    total: int
    utility_rate: float
    security_rate: float

    @property
    def attack_success_rate(self) -> float:
        return 1.0 - self.security_rate


def load_results(logdir: Path) -> list[dict]:
    """Load every benchmark result JSON under a log directory."""
    results: list[dict] = []
    for json_file in logdir.rglob("*.json"):
        with json_file.open() as handle:
            results.append(json.load(handle))
    return results


def summarize(results: list[dict]) -> ResultSummary:
    """Compute mean utility and security over all run files."""
    if not results:
        return ResultSummary(total=0, utility_rate=0.0, security_rate=0.0)

    utility_total = 0
    security_total = 0
    for result in results:
        utility_total += 1 if result.get("utility") else 0
        security_total += 1 if result.get("security") else 0

    total = len(results)
    return ResultSummary(
        total=total,
        utility_rate=utility_total / total,
        security_rate=security_total / total,
    )


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    print("=" * 90)
    print("AgentDojo Defense Comparison")
    print("=" * 90)
    print()

    summaries: dict[str, ResultSummary] = {}
    for config_name, logdir in CONFIGS.items():
        if not logdir.exists():
            print(f"{config_name}: [not yet run] ({logdir})")
            continue

        summary = summarize(load_results(logdir))
        summaries[config_name] = summary
        print(f"{config_name}:")
        print(f"  Files: {summary.total}")
        print(f"  Utility: {format_percent(summary.utility_rate)}")
        print(f"  Security pass rate: {format_percent(summary.security_rate)}")
        print(f"  Attack success rate: {format_percent(summary.attack_success_rate)}")
        print()

    if "Baseline" in summaries:
        baseline = summaries["Baseline"]
        print("Deltas vs Baseline:")
        for config_name in ("Attack", "Boundary formatting", "Sanitization"):
            summary = summaries.get(config_name)
            if summary is None:
                continue
            utility_delta = summary.utility_rate - baseline.utility_rate
            security_delta = summary.security_rate - baseline.security_rate
            attack_success_delta = summary.attack_success_rate - baseline.attack_success_rate
            print(f"  {config_name}:")
            print(f"    Utility delta: {utility_delta:+.2%}")
            print(f"    Security pass rate delta: {security_delta:+.2%}")
            print(f"    Attack success delta: {attack_success_delta:+.2%}")
        print()

    print("Notes:")
    print("  - Higher utility is better.")
    print("  - Higher security pass rate is better.")
    print("  - Lower attack success rate is better.")
    print()


if __name__ == "__main__":
    main()
