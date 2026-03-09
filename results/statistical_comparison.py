from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import ttest_ind_from_stats


def load_summary(path: Path) -> tuple[float, float, int]:
    payload = json.loads(path.read_text())
    acc = payload["accuracy"]
    return float(acc["mean"]), float(acc["std"]), int(payload["runs"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark summaries with a t-test")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/statistical_comparison.json"))
    args = parser.parse_args()

    base_mean, base_std, base_n = load_summary(args.baseline)
    cand_mean, cand_std, cand_n = load_summary(args.candidate)

    stat = ttest_ind_from_stats(
        mean1=base_mean,
        std1=max(base_std, 1e-12),
        nobs1=base_n,
        mean2=cand_mean,
        std2=max(cand_std, 1e-12),
        nobs2=cand_n,
        equal_var=False,
    )

    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_accuracy_mean": base_mean,
        "candidate_accuracy_mean": cand_mean,
        "difference": cand_mean - base_mean,
        "t_statistic": float(stat.statistic),
        "p_value": float(stat.pvalue),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
