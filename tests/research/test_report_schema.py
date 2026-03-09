from benchmarks.benchmark import repeated_benchmark
from research.config import SystemConfig, TrainingConfig


def test_repeated_benchmark_report_schema(tmp_path):
    sys_cfg = SystemConfig(dataset="digits", sample_size=100, artifacts_dir=tmp_path)
    tr_cfg = TrainingConfig(model_name="knn")
    report = repeated_benchmark(sys_cfg, tr_cfg, runs=2)
    assert report["summary_version"] == "v2"
    for key in ["accuracy", "inference_latency_s", "training_time_s"]:
        assert set(["mean", "std", "ci95"]).issubset(report[key].keys())
