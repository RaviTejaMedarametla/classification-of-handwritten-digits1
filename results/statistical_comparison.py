import argparse
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

from research.core.utils.metrics import save_json


def welch_ttest(a: np.ndarray, b: np.ndarray):
    result = ttest_ind(a, b, equal_var=False)
    return {"t_statistic": float(result.statistic), "p_value": float(result.pvalue)}


def main():
    parser = argparse.ArgumentParser(description="Welch's t-test for two model output arrays")
    parser.add_argument("--a", type=Path, required=True, help="Path to .npy outputs for model A")
    parser.add_argument("--b", type=Path, required=True, help="Path to .npy outputs for model B")
    parser.add_argument("--out", type=Path, default=Path("results/welch_ttest.json"))
    args = parser.parse_args()

    a = np.load(args.a)
    b = np.load(args.b)
    report = {
        "a_path": str(args.a),
        "b_path": str(args.b),
        "a_count": int(a.size),
        "b_count": int(b.size),
        "welch_ttest": welch_ttest(a.ravel().astype(float), b.ravel().astype(float)),
    }
    save_json(report, args.out)
    print(report)


if __name__ == "__main__":
    main()
