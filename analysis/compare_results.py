"""
Compare results across experiment configurations.
Usage:
    python analysis/compare_results.py
TODO: Update paths and parsing once we have actual result formats from AgentDojo.
"""
import json
import os
from pathlib import Path
RESULTS_DIR = Path("results")
CONFIGS = {
    "Baseline (no attack)": "baseline_no_attack",
    "Baseline + attack": "baseline_with_attack",
    # "Formatting + attack": "formatting_with_attack",    # Week 10-11
    # "Sanitization + attack": "sanitization_with_attack", # Week 12-13
}
def load_results(logdir: Path) -> dict:
    """
    Load benchmark results from a logdir.
    TODO: Inspect actual AgentDojo output format and parse accordingly.
    AgentDojo saves results as JSON files in the logdir.
    """
    results = {}
    for f in logdir.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
            results[f.stem] = data
    return results
def summarize(results: dict) -> dict:
    """
    Compute summary metrics from raw results.
    TODO: Implement once we understand the result JSON schema.
    Expected metrics:
    - benign_utility: fraction of tasks solved
    - utility_under_attack: fraction of tasks solved under attack
    - targeted_asr: fraction of security cases where attacker goal was met
    """
    return {
        "total_files": len(results),
        # "benign_utility": ...,
        # "targeted_asr": ...,
    }
def main():
    print("=" * 60)
    print("AgentDojo Defense Comparison")
    print("=" * 60)
    print()
    for config_name, dirname in CONFIGS.items():
        logdir = RESULTS_DIR / dirname
        if not logdir.exists():
            print(f"  {config_name}: [not yet run]")
            continue
        results = load_results(logdir)
        summary = summarize(results)
        print(f"  {config_name}:")
        print(f"    Result files: {summary['total_files']}")
        print()
    print("=" * 60)
    print("TODO: Add comparison table and plots once all configs are run.")
if __name__ == "__main__":
    main()
