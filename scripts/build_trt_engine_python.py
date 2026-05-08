#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorrt as trt


LOGGER_SEVERITY = {
    "internal_error": trt.Logger.INTERNAL_ERROR,
    "error": trt.Logger.ERROR,
    "warning": trt.Logger.WARNING,
    "info": trt.Logger.INFO,
    "verbose": trt.Logger.VERBOSE,
}

PROFILING_VERBOSITY = {
    "none": trt.ProfilingVerbosity.NONE,
    "layer_names_only": trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
    "detailed": trt.ProfilingVerbosity.DETAILED,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine through the TensorRT Python API.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--strongly-typed", action="store_true")
    parser.add_argument("--builder-optimization-level", type=int, default=5)
    parser.add_argument("--workspace-gib", type=int, default=16)
    parser.add_argument("--profiling-verbosity", choices=sorted(PROFILING_VERBOSITY), default="detailed")
    parser.add_argument("--logger-severity", choices=sorted(LOGGER_SEVERITY), default="info")
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def network_flags(*, strongly_typed: bool) -> int:
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if strongly_typed:
        flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
        if flag is None:
            raise RuntimeError("This TensorRT build does not expose STRONGLY_TYPED network creation.")
        flags |= 1 << int(flag)
    return flags


def main() -> None:
    args = parse_args()
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    summary_json = args.summary_json or args.engine.with_suffix(".build_summary.json")

    logger = trt.Logger(LOGGER_SEVERITY[args.logger_severity])
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags(strongly_typed=args.strongly_typed))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.onnx)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(f"Failed to parse ONNX {args.onnx}:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.builder_optimization_level = int(args.builder_optimization_level)
    config.profiling_verbosity = PROFILING_VERBOSITY[args.profiling_verbosity]
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib) * (1 << 30))

    if not args.strongly_typed:
        if args.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif args.precision == "int8":
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)

    input_names = [network.get_input(index).name for index in range(network.num_inputs)]
    output_names = [network.get_output(index).name for index in range(network.num_outputs)]
    print(f"Building TensorRT engine: {args.engine}")
    print(f"ONNX: {args.onnx}")
    print(f"precision={args.precision} strongly_typed={args.strongly_typed}")
    print(f"inputs={input_names}")
    print(f"outputs={output_names}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT builder returned no serialized engine.")
    args.engine.write_bytes(bytes(serialized))

    summary = {
        "onnx": str(args.onnx),
        "engine": str(args.engine),
        "precision": args.precision,
        "strongly_typed": bool(args.strongly_typed),
        "builder_optimization_level": int(args.builder_optimization_level),
        "workspace_gib": int(args.workspace_gib),
        "profiling_verbosity": args.profiling_verbosity,
        "num_inputs": len(input_names),
        "num_outputs": len(output_names),
        "input_names": input_names,
        "output_names": output_names,
        "engine_size_bytes": args.engine.stat().st_size,
        "tensorrt_version": trt.__version__,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
