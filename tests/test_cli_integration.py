import subprocess
import sys


def test_train_cli_runs_digits():
    cmd = [sys.executable, "-m", "research.train_cli", "--model", "knn", "--dataset", "digits", "--sample-size", "200"]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "accuracy" in proc.stdout
