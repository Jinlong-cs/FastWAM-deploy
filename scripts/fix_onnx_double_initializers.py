#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ONNX DOUBLE initializers to FLOAT while preserving existing external weights."
    )
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def external_data_dict(tensor: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def load_double_values(tensor: onnx.TensorProto, *, model_dir: Path) -> np.ndarray:
    expected = int(np.prod(tensor.dims, dtype=np.int64))
    if tensor.raw_data:
        data = np.frombuffer(tensor.raw_data, dtype=np.float64)
    elif tensor.double_data:
        data = np.asarray(tensor.double_data, dtype=np.float64)
    else:
        ext = external_data_dict(tensor)
        if not ext:
            raise ValueError(f"DOUBLE initializer has no data: {tensor.name}")
        location = ext.get("location")
        if not location:
            raise ValueError(f"DOUBLE initializer external_data has no location: {tensor.name}")
        offset = int(ext.get("offset", "0"))
        length = int(ext.get("length", "0"))
        with (model_dir / location).open("rb") as handle:
            handle.seek(offset)
            data = np.frombuffer(handle.read(length), dtype=np.float64)
    if data.size != expected:
        raise ValueError(f"{tensor.name}: expected {expected} DOUBLE values, loaded {data.size}")
    return data.reshape(tuple(int(dim) for dim in tensor.dims))


def main() -> None:
    args = parse_args()
    output = args.output or args.onnx_path
    model = onnx.load(str(args.onnx_path), load_external_data=False)
    converted: list[dict[str, object]] = []
    for tensor in model.graph.initializer:
        if tensor.data_type != onnx.TensorProto.DOUBLE:
            continue
        values = load_double_values(tensor, model_dir=args.onnx_path.parent).astype(np.float32, copy=False)
        tensor.ClearField("external_data")
        tensor.ClearField("double_data")
        tensor.ClearField("raw_data")
        tensor.data_location = onnx.TensorProto.DEFAULT
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.raw_data = values.tobytes()
        converted.append(
            {
                "name": tensor.name,
                "shape": [int(dim) for dim in tensor.dims],
                "float32_bytes": len(tensor.raw_data),
            }
        )

    onnx.save(model, str(output))
    summary = {
        "input": str(args.onnx_path),
        "output": str(output),
        "converted_count": len(converted),
        "converted": converted,
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
