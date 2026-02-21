import argparse
from pathlib import Path

import joblib

from ml_system.config import DeploymentConfig, SystemConfig, TrainingConfig
from ml_system.data.dataset import load_mnist, split_and_normalize
from ml_system.deployment.onnx_pipeline import batch_inference_cpu, export_to_onnx, save_deployment_report, validate_onnx
from ml_system.training.trainer import train_once


def main():
    parser = argparse.ArgumentParser(description="Run inference / export pipeline")
    parser.add_argument("--model", choices=["knn", "rf"], default="knn")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    sys_cfg = SystemConfig()
    tr_cfg = TrainingConfig(model_name=args.model)
    dep_cfg = DeploymentConfig(batch_size=args.batch_size)

    trained = train_once(sys_cfg, tr_cfg)
    model = trained["model"]
    _, x_test, _, _ = trained["data"]
    preds = batch_inference_cpu(model, x_test, batch_size=dep_cfg.batch_size)
    print(f"batch_predictions={len(preds)}")

    if args.save_model:
        out = sys_cfg.artifacts_dir / f"{args.model}_model.joblib"
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out)
        print(f"saved_model={out}")

    if args.export_onnx:
        onnx_path = export_to_onnx(model, x_test[:1], sys_cfg.artifacts_dir / f"{args.model}.onnx", dep_cfg.onnx_opset)
        report = validate_onnx(onnx_path, model, x_test)
        save_deployment_report(report, sys_cfg.artifacts_dir / f"onnx_report_{args.model}.json")
        print(f"onnx_report={report}")


if __name__ == "__main__":
    main()
