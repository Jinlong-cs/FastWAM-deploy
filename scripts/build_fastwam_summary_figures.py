#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FastWAM AGX/DGX deployment summary PNG figures.")
    parser.add_argument("--agx-eager", type=Path, required=True)
    parser.add_argument("--agx-trt", type=Path, required=True)
    parser.add_argument("--agx-drift", type=Path, required=True)
    parser.add_argument("--dgx-eager", type=Path, required=True)
    parser.add_argument("--dgx-trt", type=Path, required=True)
    parser.add_argument("--dgx-drift", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docs-output", type=Path, default=Path("docs/fastwam_deployment_pipeline_agx_orin.png"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def metric(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    return float(value) if value is not None else default


def stage(data: dict[str, Any], key: str) -> float:
    return float(data.get("stage_summary_ms", {}).get(key, {}).get("mean", 0.0))


def engine_gib(data: dict[str, Any], key: str) -> float:
    value = float(data.get("engine_size_bytes", {}).get(key, 0.0))
    return value / (1024.0**3)


def drift_summary(data: dict[str, Any]) -> tuple[str, str]:
    summary = data.get("summary", {})
    comparisons = data.get("comparisons", {})
    action = comparisons.get("action_decoded", {})
    worst = summary.get("worst_max_abs")
    if isinstance(worst, dict):
        worst_text = f"{worst.get('name', 'worst')} max_abs={float(worst.get('max_abs', 0.0)):.3f}"
    elif worst is not None:
        worst_text = f"worst max_abs={float(worst):.3f}"
    else:
        worst_text = "worst drift not available"
    action_text = (
        f"action max_abs={float(action.get('max_abs', 0.0)):.3f}, "
        f"mean_abs={float(action.get('mean_abs', 0.0)):.3f}"
        if action
        else "action drift not available"
    )
    finite = summary.get("all_compared_finite", data.get("all_compared_finite"))
    count = summary.get("num_compared_tensors", summary.get("num_compared", len(comparisons)))
    return f"{count} tensors compared, all finite={finite}", f"{worst_text}; {action_text}"


def fonts():
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    regular = next((p for p in candidates if Path(p).exists() and "Bold" not in p), None)
    bold = next((p for p in candidates if Path(p).exists() and "Bold" in p), regular)

    def f(size: int, *, b: bool = False):
        path = bold if b else regular
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()

    return f


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        from PIL import Image, ImageDraw

        self.image = Image.new("RGB", (width, height), "#f4f7fb")
        self.draw = ImageDraw.Draw(self.image)
        self.font = fonts()
        self.width = width
        self.height = height

    def text(self, xy, text: str, size: int, fill: str = "#111827", bold: bool = False, anchor: str | None = None) -> None:
        self.draw.text(xy, text, font=self.font(size, b=bold), fill=fill, anchor=anchor)

    def rounded(self, box, radius: int = 18, fill: str = "#ffffff", outline: str = "#cbd5e1", width: int = 2) -> None:
        self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def pill(self, xy, text: str, fill: str, outline: str, color: str = "#0f172a") -> None:
        x, y = xy
        w = max(120, 18 * len(text) + 34)
        h = 40
        self.draw.rounded_rectangle((x, y, x + w, y + h), radius=20, fill=fill, outline=outline, width=2)
        self.text((x + w / 2, y + h / 2 + 1), text, 20, color, True, anchor="mm")

    def arrow(self, x1: int, y: int, x2: int, color: str = "#475569") -> None:
        self.draw.line((x1, y, x2, y), fill=color, width=5)
        self.draw.polygon([(x2, y), (x2 - 22, y - 12), (x2 - 22, y + 12)], fill=color)

    def bullet(self, x: int, y: int, text: str, size: int = 20, fill: str = "#334155", dot: str = "#0f766e") -> None:
        self.draw.ellipse((x, y + 8, x + 10, y + 18), fill=dot)
        self.text((x + 20, y), text, size, fill)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)


def draw_module(canvas: Canvas, box, tag: str, title: str, bullets: list[str], lat: str, mem: str, accent: str) -> None:
    x1, y1, x2, y2 = box
    canvas.rounded(box, 22, "#eff6ff" if accent == "blue" else "#ecfdf5", "#2563eb" if accent == "blue" else "#15803d", 4)
    tag_color = "#2563eb" if accent == "blue" else "#15803d"
    canvas.draw.rounded_rectangle((x1 + 26, y1 + 26, x1 + 178, y1 + 64), radius=19, fill=tag_color)
    canvas.text((x1 + 102, y1 + 46), tag, 18, "#ffffff", True, anchor="mm")
    canvas.text((x1 + 30, y1 + 96), title, 26, "#111827", True)
    canvas.draw.line((x1 + 30, y1 + 152, x2 - 30, y1 + 152), fill="#cbd5e1", width=2)
    y = y1 + 174
    for item in bullets:
        canvas.bullet(x1 + 34, y, item, size=18, dot=tag_color)
        y += 32
    canvas.draw.rounded_rectangle((x1 + 28, y2 - 58, x1 + 180, y2 - 18), radius=16, fill="#ffffff", outline="#bfdbfe", width=2)
    canvas.text((x1 + 104, y2 - 38), lat, 18, "#0f172a", True, anchor="mm")
    canvas.draw.rounded_rectangle((x1 + 196, y2 - 58, x2 - 24, y2 - 18), radius=16, fill="#ffffff", outline="#bfdbfe", width=2)
    canvas.text(((x1 + 196 + x2 - 24) / 2, y2 - 38), mem, 17, "#0f172a", True, anchor="mm")


def draw_pipeline(path: Path, *, title: str, platform_label: str, eager: dict[str, Any], trt: dict[str, Any], drift: dict[str, Any]) -> None:
    c = Canvas(3600, 2060)
    c.rounded((55, 55, 3545, 2000), 34, "#ffffff", "#d8e0eb", 3)
    c.text((120, 105), title, 52, "#111827", True)
    c.text((120, 174), "FastWAM RoboTwin real-text path: same checkpoint, same sample contract, same num_inference_steps=1.", 28, "#475569")

    e2e = metric(trt, "mean_end_to_end_ms")
    p95 = metric(trt, "p95_end_to_end_ms")
    eager_e2e = metric(eager, "mean_end_to_end_ms")
    c.pill((120, 245), "3 TensorRT FP16 engines", "#e6fffb", "#0f766e", "#0f766e")
    c.pill((430, 245), "real T5 text cache", "#eef2ff", "#4f46e5", "#4338ca")
    c.pill((700, 245), "batch=1", "#fff7ed", "#ea580c", "#c2410c")
    c.pill((870, 245), "1 denoise step", "#fff7ed", "#ea580c", "#c2410c")
    c.pill((1130, 245), f"E2E mean {e2e:.2f} ms", "#fef2f2", "#dc2626", "#b91c1c")
    c.pill((1420, 245), platform_label, "#f8fafc", "#0f172a", "#0f172a")

    y = 420
    boxes = [
        (120, y, 495, y + 360),
        (620, y, 1085, y + 360),
        (1225, y, 1690, y + 360),
        (1830, y, 2295, y + 360),
        (2435, y, 2900, y + 360),
        (3040, y, 3420, y + 360),
    ]
    c.rounded(boxes[0], 22, "#f8fafc", "#64748b", 4)
    c.draw.rounded_rectangle((150, y + 26, 300, y + 64), radius=19, fill="#64748b")
    c.text((225, y + 46), "Input", 18, "#ffffff", True, anchor="mm")
    c.text((150, y + 98), "Host Preprocess", 27, "#111827", True)
    c.draw.line((150, y + 140, 465, y + 140), fill="#cbd5e1", width=2)
    c.bullet(155, y + 170, "real RoboTwin 3-camera sample")
    c.bullet(155, y + 205, "14D state -> proprio token")
    c.bullet(155, y + 240, "precomputed T5 context")
    c.text((170, y + 322), f"eager e2e {eager_e2e:.2f} ms", 17, "#475569")

    draw_module(
        c,
        boxes[1],
        "TRT 1",
        "VAE Image\nEncoder",
        ["image [1,3,384,320]", "out first-frame latents", "[1,48,1,24,20]"],
        f"{stage(trt, 'vae_image_encoder_ms'):.2f} ms",
        f"{engine_gib(trt, 'vae_image_encoder'):.2f} GiB",
        "blue",
    )
    draw_module(
        c,
        boxes[2],
        "TRT 2",
        "Video Prefill\n+ KV Cache",
        ["first-frame latents", "context/mask [1,129]", "30 layers x K/V tensors"],
        f"{stage(trt, 'video_prefill_ms'):.2f} ms",
        f"{engine_gib(trt, 'video_prefill'):.2f} GiB",
        "blue",
    )
    draw_module(
        c,
        boxes[3],
        "TRT 3",
        "Action Denoise\nStep",
        ["action latent [1,32,14]", "context + video K/V", "out pred_action"],
        f"{stage(trt, 'action_denoise_loop_ms'):.2f} ms",
        f"{engine_gib(trt, 'action_step_dynamic_kv'):.2f} GiB",
        "blue",
    )
    draw_module(
        c,
        boxes[4],
        "Host",
        "Flow-Match\nScheduler",
        ["latents <- step(...)", "N=1 in benchmark", "host-side tensor op"],
        f"{stage(trt, 'scheduler_total_ms'):.2f} ms",
        "no engine",
        "green",
    )
    draw_module(
        c,
        boxes[5],
        "Host",
        "Action Decode\n/ Denorm",
        ["inverse z-score", "action chunk [1,32,14]", f"finite={trt.get('output_finite')}"],
        f"{stage(trt, 'action_decode_ms'):.2f} ms",
        "no engine",
        "green",
    )
    for left, right in zip(boxes, boxes[1:]):
        c.arrow(left[2] + 12, y + 190, right[0] - 20)

    c.rounded((120, 880, 3420, 1125), 22, "#f8fafc", "#cbd5e1", 2)
    c.text((160, 930), "Tensor Handoff Lane", 30, "#111827", True)
    handoffs = [
        "Host -> TRT1: image [1,3,384,320]",
        "TRT1 -> TRT2: first_frame_latents [1,48,1,24,20]",
        "TRT2 -> TRT3: 60 K/V tensors, each [1,120,3072]",
        "TRT3 -> Host: pred_action [1,32,14]",
    ]
    hx = 160
    for item in handoffs:
        c.rounded((hx, 980, hx + 740, 1085), 16, "#ffffff", "#cbd5e1", 2)
        c.text((hx + 24, 1015), item.split(":")[0], 20, "#111827", True)
        c.text((hx + 24, 1050), item.split(": ", 1)[1], 18, "#334155")
        hx += 805

    c.rounded((120, 1215, 1700, 1655), 22, "#f8fafc", "#cbd5e1", 2)
    c.text((165, 1265), "Latency Lane", 32, "#111827", True)
    c.text((165, 1310), f"Measured on {platform_label}; FP16 TensorRT, warmup={trt.get('warmup_batches')}, measure={trt.get('measure_batches')}.", 22, "#475569")
    parts = [
        ("vae", stage(trt, "vae_image_encoder_ms"), "#2563eb"),
        ("prefill", stage(trt, "video_prefill_ms"), "#0f766e"),
        ("kv cast", stage(trt, "kv_cast_ms"), "#94a3b8"),
        ("denoise", stage(trt, "action_denoise_loop_ms"), "#f97316"),
        ("sched", stage(trt, "scheduler_total_ms"), "#22c55e"),
        ("decode", stage(trt, "action_decode_ms"), "#64748b"),
    ]
    total = max(sum(v for _, v, _ in parts), 1.0)
    x = 165
    for name, value, color in parts:
        w = max(28, int(1300 * value / total))
        c.draw.rounded_rectangle((x, 1380, x + w, 1465), radius=10, fill=color)
        c.text((x + w / 2, 1410), name, 18, "#ffffff", True, anchor="mm")
        c.text((x + w / 2, 1442), f"{value:.2f} ms", 17, "#ffffff", anchor="mm")
        x += w + 4
    c.text((170, 1535), f"end-to-end mean {e2e:.2f} ms", 26, "#b91c1c", True)
    c.text((650, 1535), f"p95 {p95:.2f} ms", 26, "#c2410c", True)

    c.rounded((1825, 1215, 3420, 1655), 22, "#f8fafc", "#cbd5e1", 2)
    c.text((1870, 1265), "Plan Size / Runtime Validity", 32, "#111827", True)
    sizes = [
        ("vae_image_encoder", engine_gib(trt, "vae_image_encoder"), "#2563eb"),
        ("video_prefill", engine_gib(trt, "video_prefill"), "#0f766e"),
        ("action_step", engine_gib(trt, "action_step_dynamic_kv"), "#f97316"),
    ]
    max_size = max([s for _, s, _ in sizes] + [1.0])
    yy = 1340
    for name, value, color in sizes:
        c.text((1870, yy), name, 21, "#111827")
        c.draw.rounded_rectangle((2160, yy - 6, 3180, yy + 34), radius=9, fill="#e2e8f0")
        c.draw.rounded_rectangle((2160, yy - 6, 2160 + int(1020 * value / max_size), yy + 34), radius=9, fill=color)
        c.text((3220, yy), f"{value:.2f} GiB", 22, "#0f172a", True)
        yy += 78
    d1, d2 = drift_summary(drift)
    c.text((1870, 1590), d1, 20, "#15803d", True)
    c.text((1870, 1625), d2, 18, "#92400e")

    c.rounded((120, 1730, 3420, 1905), 22, "#fff7ed", "#fed7aa", 2)
    c.text((160, 1775), "Interpretation", 28, "#111827", True)
    c.text((160, 1820), "Runtime is finite-output-valid and aligned to AGX protocol. Drift/simulator success-rate remains the accuracy gate.", 21, "#334155")
    c.text((160, 1860), "Sources: FastWAM deploy JSON, real RoboTwin sample, precomputed T5 cache.", 18, "#64748b")
    c.save(path)


def draw_comparison(path: Path, *, agx_eager: dict[str, Any], agx_trt: dict[str, Any], agx_drift: dict[str, Any], dgx_eager: dict[str, Any], dgx_trt: dict[str, Any], dgx_drift: dict[str, Any]) -> None:
    c = Canvas(3600, 2520)
    c.text((90, 75), "FastWAM Baseline: AGX Orin vs. DGX Spark Runtime Comparison", 52, "#111827", True)
    c.text((90, 145), "Same checkpoint, same real RoboTwin sample contract, same precomputed T5 text cache, batch=1, num_inference_steps=1.", 27, "#475569")
    c.pill((90, 205), "TensorRT FP16 runtime", "#e6fffb", "#0f766e", "#0f766e")
    c.pill((390, 205), "real-text cache", "#eef2ff", "#4f46e5", "#4338ca")
    c.pill((615, 205), "cross-device comparison", "#fef2f2", "#dc2626", "#b91c1c")

    agx_e2e = metric(agx_trt, "mean_end_to_end_ms")
    dgx_e2e = metric(dgx_trt, "mean_end_to_end_ms")
    e2e_speed = agx_e2e / dgx_e2e if dgx_e2e else 0.0
    eager_speed = metric(agx_eager, "mean_end_to_end_ms") / metric(dgx_eager, "mean_end_to_end_ms") if metric(dgx_eager, "mean_end_to_end_ms") else 0.0

    c.rounded((80, 300, 1740, 720), 18, "#ffffff", "#cbd5e1", 2)
    c.text((120, 350), "(a) Shared workload and deployment path", 30, "#111827", True)
    for x, title, lines in [
        (130, "Model / data", ["FastWAM released RoboTwin checkpoint", "RoboTwin unified ep0 frame0", "3 cameras + 14D state"]),
        (560, "Runtime", ["VAE image encoder TRT", "video prefill + K/V TRT", "action step dynamic-KV TRT"]),
        (990, "Protocol", ["batch size = 1", "warmup / measure = 2 / 5", "num_inference_steps = 1"]),
        (1340, "Topology", ["TRT1 -> TRT2 -> TRT3", "host scheduler", "host action denorm"]),
    ]:
        c.text((x, 405), title, 24, "#111827", True)
        yy = 450
        for line in lines:
            c.bullet(x, yy, line, 18)
            yy += 34
    c.rounded((220, 615, 1550, 680), 16, "#f8fafc", "#94a3b8", 2)
    c.text((250, 638), "Host -> VAE TRT -> Video Prefill TRT -> Action Step TRT -> Scheduler -> Decode", 22, "#0f172a", True)

    c.rounded((1810, 300, 3520, 720), 18, "#ffffff", "#cbd5e1", 2)
    c.text((1850, 350), "(b) Platform and environment delta", 30, "#111827", True)
    c.text((1900, 425), "Jetson AGX Orin", 27, "#1d4ed8", True)
    c.text((2780, 425), "DGX Spark", 27, "#c2410c", True)
    rows = [
        ("Accelerator", "AGX Orin 64GB unified memory", "NVIDIA GB10"),
        ("CUDA / TRT", "AGX TensorRT stack", f"TensorRT {dgx_trt.get('tensorrt_version', '10.16.1.11')}"),
        ("PyTorch", "AGX eager BF16 baseline", "torch 2.11.0+cu130"),
        ("Power mode", "MAXN + jetson_clocks", "default driver boost; not locked"),
        ("Memory", "unified RAM visible", "nvidia-smi GPU memory N/A"),
    ]
    yy = 475
    for label, agx, dgx in rows:
        c.text((1850, yy), label, 18, "#475569", True)
        c.text((2050, yy), agx, 18, "#111827")
        c.text((2780, yy), dgx, 18, "#111827")
        yy += 43

    c.rounded((80, 770, 1740, 1280), 18, "#ffffff", "#cbd5e1", 2)
    c.text((120, 820), "(c) End-to-end runtime", 30, "#111827", True)
    c.text((145, 925), f"{agx_e2e:.2f} ms", 48, "#2563eb", True)
    c.text((430, 930), "->", 42, "#475569", True)
    c.text((540, 925), f"{dgx_e2e:.2f} ms", 48, "#ea580c", True)
    c.rounded((930, 900, 1245, 1015), 14, "#fef2f2", "#dc2626", 2)
    c.text((1088, 940), f"{e2e_speed:.2f}x", 42, "#b91c1c", True, anchor="mm")
    c.text((1088, 985), "TRT E2E speedup", 18, "#b91c1c", True, anchor="mm")
    c.text((1330, 930), f"{1000/agx_e2e:.2f} Hz -> {1000/dgx_e2e:.2f} Hz", 24, "#0f172a", True)
    c.text((145, 1095), f"Eager .pth E2E: {metric(agx_eager, 'mean_end_to_end_ms'):.2f} ms -> {metric(dgx_eager, 'mean_end_to_end_ms'):.2f} ms ({eager_speed:.2f}x)", 23, "#334155")

    max_e2e = max(agx_e2e, dgx_e2e, 1.0)
    for yy, name, value, color in [(1165, "AGX TRT", agx_e2e, "#2563eb"), (1225, "DGX TRT", dgx_e2e, "#ea580c")]:
        c.text((145, yy), name, 20, "#111827")
        c.draw.rounded_rectangle((290, yy - 8, 1280, yy + 28), radius=8, fill="#e2e8f0")
        c.draw.rounded_rectangle((290, yy - 8, 290 + int(990 * value / max_e2e), yy + 28), radius=8, fill=color)
        c.text((1300, yy - 4), f"{value:.2f} ms", 20, "#111827")

    c.rounded((1810, 770, 3520, 1280), 18, "#ffffff", "#cbd5e1", 2)
    c.text((1850, 820), "(d) Per-stage latency, aligned runtime modules", 30, "#111827", True)
    stage_keys = [
        ("VAE", "vae_image_encoder_ms"),
        ("Video prefill", "video_prefill_ms"),
        ("KV cast", "kv_cast_ms"),
        ("Action loop", "action_denoise_loop_ms"),
        ("Scheduler", "scheduler_total_ms"),
        ("Decode", "action_decode_ms"),
    ]
    x = 1880
    for label, key in stage_keys:
        av = stage(agx_trt, key)
        dv = stage(dgx_trt, key)
        sp = av / dv if dv else 0.0
        c.text((x, 895), label, 20, "#111827", True)
        maxv = max(av, dv, 1.0)
        c.draw.rounded_rectangle((x, 950, x + 250, 990), radius=8, fill="#e2e8f0")
        c.draw.rounded_rectangle((x, 950, x + int(250 * av / maxv), 990), radius=8, fill="#2563eb")
        c.text((x + 262, 958), f"{av:.2f}", 18, "#0f172a")
        c.draw.rounded_rectangle((x, 1010, x + 250, 1050), radius=8, fill="#e2e8f0")
        c.draw.rounded_rectangle((x, 1010, x + int(250 * dv / maxv), 1050), radius=8, fill="#ea580c")
        c.text((x + 262, 1018), f"{dv:.2f}", 18, "#0f172a")
        c.rounded((x + 20, 1100, x + 235, 1145), 18, "#fef2f2", "#dc2626", 2)
        c.text((x + 128, 1122), f"{sp:.2f}x", 20, "#b91c1c", True, anchor="mm")
        x += 270
    c.draw.rectangle((1880, 1190, 1910, 1220), fill="#2563eb")
    c.text((1922, 1188), "AGX Orin", 18)
    c.draw.rectangle((2070, 1190, 2100, 1220), fill="#ea580c")
    c.text((2112, 1188), "DGX Spark", 18)
    c.text((2500, 1188), "Values are milliseconds; pill values are AGX/DGX speedup.", 18, "#475569")

    c.rounded((80, 1340, 1740, 1840), 18, "#ffffff", "#cbd5e1", 2)
    c.text((120, 1390), "(e) Latency anatomy of TensorRT path", 30, "#111827", True)
    colors = {
        "vae": "#2563eb",
        "prefill": "#0f766e",
        "kv": "#94a3b8",
        "denoise": "#f97316",
        "host": "#64748b",
    }
    for y0, name, data in [(1515, "AGX", agx_trt), (1650, "DGX", dgx_trt)]:
        total = max(metric(data, "mean_end_to_end_ms"), 1.0)
        x0 = 300
        c.text((160, y0 + 20), name, 28, "#1d4ed8" if name == "AGX" else "#c2410c", True)
        parts = [
            ("vae", stage(data, "vae_image_encoder_ms"), colors["vae"]),
            ("prefill", stage(data, "video_prefill_ms"), colors["prefill"]),
            ("kv", stage(data, "kv_cast_ms"), colors["kv"]),
            ("denoise", stage(data, "action_denoise_loop_ms"), colors["denoise"]),
            ("host", stage(data, "scheduler_total_ms") + stage(data, "action_decode_ms"), colors["host"]),
        ]
        for label, value, color in parts:
            w = max(8, int(1100 * value / total))
            c.draw.rounded_rectangle((x0, y0, x0 + w, y0 + 70), radius=8, fill=color)
            if w > 90:
                c.text((x0 + w / 2, y0 + 35), label, 18, "#ffffff", True, anchor="mm")
            x0 += w + 2
        c.text((1430, y0 + 20), f"{total:.2f} ms", 24, "#111827", True)
    c.text((140, 1770), "Dominant blocks identify where DGX Spark gains or remaining bottlenecks are concentrated.", 22, "#334155")

    c.rounded((1810, 1340, 3520, 1840), 18, "#ffffff", "#cbd5e1", 2)
    c.text((1850, 1390), "(f) Telemetry and validity boundaries", 30, "#111827", True)
    c.rounded((1870, 1450, 2570, 1535), 14, "#f8fafc", "#cbd5e1", 1)
    c.text((1900, 1475), "AGX power state", 20, "#111827", True)
    c.text((2200, 1472), "MAXN + jetson_clocks", 25, "#1d4ed8", True)
    c.rounded((2650, 1450, 3360, 1535), 14, "#f8fafc", "#cbd5e1", 1)
    c.text((2680, 1475), "Spark GPU", 20, "#111827", True)
    c.text((2880, 1472), "NVIDIA GB10", 25, "#0f766e", True)
    d1, d2 = drift_summary(dgx_drift)
    c.text((1870, 1600), "Controlled / aligned", 24, "#047857", True)
    for i, line in enumerate(["model checkpoint", "sample contract", "text cache semantics", "batch=1, steps=1, FP16 TRT"]):
        c.bullet(1880, 1645 + i * 35, line, 18, dot="#047857")
    c.text((2650, 1600), "Not controlled", 24, "#b91c1c", True)
    for i, line in enumerate(["CUDA/TRT/PyTorch versions", "TensorRT tactic selection", "power-lock mechanism", "sim success-rate accuracy"]):
        c.bullet(2660, 1645 + i * 35, line, 18, dot="#b91c1c")
    c.text((1870, 1810), f"DGX drift: {d1}; {d2}", 18, "#92400e")

    c.rounded((80, 1900, 3520, 2390), 18, "#ffffff", "#cbd5e1", 2)
    c.text((120, 1950), "(g) Numerical summary", 30, "#111827", True)
    columns = [120, 900, 1320, 1740, 2180]
    headers = ["Metric", "AGX Orin", "DGX Spark", "Speedup", "Comment"]
    for x, h in zip(columns, headers):
        c.text((x, 2010), h, 22, "#111827", True)
    c.draw.line((115, 2052, 3470, 2052), fill="#cbd5e1", width=2)
    rows = [
        ("TRT E2E mean", agx_e2e, dgx_e2e, e2e_speed, "full partitioned runtime"),
        ("Eager .pth E2E", metric(agx_eager, "mean_end_to_end_ms"), metric(dgx_eager, "mean_end_to_end_ms"), eager_speed, "BF16 PyTorch baseline"),
        ("VAE image encoder", stage(agx_trt, "vae_image_encoder_ms"), stage(dgx_trt, "vae_image_encoder_ms"), 0, "TRT 1"),
        ("Video prefill", stage(agx_trt, "video_prefill_ms"), stage(dgx_trt, "video_prefill_ms"), 0, "TRT 2 largest graph"),
        ("Action denoise loop", stage(agx_trt, "action_denoise_loop_ms"), stage(dgx_trt, "action_denoise_loop_ms"), 0, "TRT 3 + host scheduler loop"),
    ]
    yy = 2090
    for name, av, dv, sp, comment in rows:
        sp = sp or (av / dv if dv else 0.0)
        if (yy // 55) % 2 == 0:
            c.draw.rectangle((115, yy - 10, 3470, yy + 38), fill="#f8fafc")
        c.text((columns[0], yy), name, 19)
        c.text((columns[1], yy), f"{av:.2f} ms", 19, "#1d4ed8")
        c.text((columns[2], yy), f"{dv:.2f} ms", 19, "#c2410c")
        c.text((columns[3], yy), f"{sp:.2f}x", 19, "#b91c1c", True)
        c.text((columns[4], yy), comment, 19, "#475569")
        yy += 55
    c.text((90, 2460), "Figure. Cross-device FastWAM deployment-chain reproduction. Results compare measured runtime, not hardware-only theoretical peak.", 20, "#475569")
    c.save(path)


def main() -> None:
    args = parse_args()
    agx_eager = load_json(args.agx_eager)
    agx_trt = load_json(args.agx_trt)
    agx_drift = load_json(args.agx_drift)
    dgx_eager = load_json(args.dgx_eager)
    dgx_trt = load_json(args.dgx_trt)
    dgx_drift = load_json(args.dgx_drift)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_pipeline(
        args.docs_output,
        title="FastWAM Deployment Pipeline on Jetson AGX Orin",
        platform_label="Jetson AGX Orin",
        eager=agx_eager,
        trt=agx_trt,
        drift=agx_drift,
    )
    draw_pipeline(
        args.output_dir / "fastwam_deployment_pipeline_dgx_spark.png",
        title="FastWAM Deployment Pipeline on DGX Spark",
        platform_label="NVIDIA GB10 / DGX Spark",
        eager=dgx_eager,
        trt=dgx_trt,
        drift=dgx_drift,
    )
    draw_comparison(
        args.output_dir / "fastwam_agx_vs_dgx_spark_comparison.png",
        agx_eager=agx_eager,
        agx_trt=agx_trt,
        agx_drift=agx_drift,
        dgx_eager=dgx_eager,
        dgx_trt=dgx_trt,
        dgx_drift=dgx_drift,
    )
    print(
        json.dumps(
            {
                "docs_pipeline": str(args.docs_output),
                "dgx_pipeline": str(args.output_dir / "fastwam_deployment_pipeline_dgx_spark.png"),
                "comparison": str(args.output_dir / "fastwam_agx_vs_dgx_spark_comparison.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
