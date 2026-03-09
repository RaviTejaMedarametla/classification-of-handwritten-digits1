import subprocess
import sys


def _run(cmd):
    return subprocess.run([sys.executable, "-m", *cmd], capture_output=True, text=True)


def test_train_cli_smoke():
    proc = _run(["research.train_cli", "--dataset", "digits", "--sample-size", "100", "--model", "knn"])
    assert proc.returncode == 0, proc.stderr


def test_evaluate_cli_smoke():
    proc = _run(["research.evaluate_cli", "--dataset", "digits", "--sample-size", "200", "--runs", "1", "--model", "knn"])
    assert proc.returncode == 0, proc.stderr


def test_infer_cli_smoke():
    proc = _run(["research.infer_cli", "--dataset", "digits", "--model", "knn"])
    assert proc.returncode == 0, proc.stderr
