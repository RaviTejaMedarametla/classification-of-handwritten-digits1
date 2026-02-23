from pathlib import Path
from typing import Dict

import numpy as np

from research.core.utils.metrics import save_json, timed_call


def export_to_onnx(model, sample_input: np.ndarray, out_path: Path, opset: int = 12) -> Path:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError("skl2onnx is required for ONNX export") from exc

    initial_type = [("input", FloatTensorType([None, sample_input.shape[1]]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=opset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onx.SerializeToString())
    return out_path


def validate_onnx(onnx_path: Path, sklearn_model, x_test: np.ndarray, min_agreement: float = 0.98) -> Dict:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for ONNX validation") from exc

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    sk_pred, sk_t = timed_call(sklearn_model.predict, x_test)

    def _onnx_call(inp):
        return sess.run(None, {input_name: inp.astype(np.float32)})[0]

    onnx_pred, onnx_t = timed_call(_onnx_call, x_test)
    if onnx_pred.ndim > 1:
        onnx_pred = np.argmax(onnx_pred, axis=1)

    agreement = float((onnx_pred == sk_pred).mean())
    throughput = float(len(x_test) / max(onnx_t, 1e-9))
    return {
        "agreement": agreement,
        "min_agreement": float(min_agreement),
        "parity_pass": bool(agreement >= min_agreement),
        "sklearn_latency_s": float(sk_t),
        "onnx_latency_s": float(onnx_t),
        "onnx_throughput_samples_per_s": throughput,
    }


def batch_inference_cpu(model, x: np.ndarray, batch_size: int = 64):
    outputs = []
    for i in range(0, len(x), batch_size):
        outputs.append(model.predict(x[i : i + batch_size]))
    return np.concatenate(outputs)


def save_deployment_report(report: Dict, out_path: Path):
    save_json(report, out_path)
