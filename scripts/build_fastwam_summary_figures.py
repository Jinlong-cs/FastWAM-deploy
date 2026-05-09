#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import wrap
from typing import Any


COLORS = {
    "bg": "#eef3f8",
    "card": "#ffffff",
    "ink": "#111827",
    "muted": "#475569",
    "line": "#cbd5e1",
    "blue": "#2563eb",
    "blue_soft": "#eaf2ff",
    "teal": "#0f766e",
    "teal_soft": "#e7f8f5",
    "orange": "#ea580c",
    "orange_soft": "#fff3e7",
    "red": "#b91c1c",
    "red_soft": "#fff1f2",
    "green": "#15803d",
    "green_soft": "#ecfdf5",
    "slate": "#64748b",
    "slate_soft": "#f8fafc",
    "violet": "#4f46e5",
    "violet_soft": "#eef2ff",
}

FONT_SCALE = 1.08


def scaled_font_size(size: int) -> int:
    return max(1, int(round(size * FONT_SCALE)))


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
    return float(data.get("engine_size_bytes", {}).get(key, 0.0)) / (1024.0**3)


def pct(part: float, total: float) -> float:
    return 100.0 * part / total if total else 0.0


def drift_summary(data: dict[str, Any]) -> tuple[str, str]:
    summary = data.get("summary", {})
    comparisons = data.get("comparisons", {})
    action = comparisons.get("action_decoded", {})
    worst = summary.get("worst_max_abs")
    if isinstance(worst, dict):
        worst_text = f"worst {worst.get('name', '')} max_abs={float(worst.get('max_abs', 0.0)):.3f}"
    elif worst is not None:
        worst_text = f"worst max_abs={float(worst):.3f}"
    else:
        worst_text = "worst drift not available"
    action_text = (
        f"decoded action max_abs={float(action.get('max_abs', 0.0)):.3f}, "
        f"mean_abs={float(action.get('mean_abs', 0.0)):.3f}"
        if action
        else "decoded action drift not available"
    )
    count = summary.get("num_compared_tensors", summary.get("num_compared", len(comparisons)))
    finite = summary.get("all_compared_finite", data.get("all_compared_finite"))
    return f"{count} tensors compared; all finite={finite}", f"{worst_text}; {action_text}"


def fonts():
    from PIL import ImageFont

    candidates = {
        "regular": [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        "bold": [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
    }
    regular = next((p for p in candidates["regular"] if Path(p).exists()), None)
    bold = next((p for p in candidates["bold"] if Path(p).exists()), regular)

    def f(size: int, *, b: bool = False):
        path = bold if b else regular
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()

    return f


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        from PIL import Image, ImageDraw

        self.image = Image.new("RGB", (width, height), COLORS["bg"])
        self.draw = ImageDraw.Draw(self.image)
        self.font = fonts()
        self.width = width
        self.height = height

    def text(
        self,
        xy: tuple[float, float],
        text: str,
        size: int,
        fill: str = COLORS["ink"],
        bold: bool = False,
        anchor: str | None = None,
    ) -> None:
        self.draw.text(xy, text, font=self.font(scaled_font_size(size), b=bold), fill=fill, anchor=anchor)

    def rounded(
        self,
        box: tuple[int, int, int, int],
        radius: int = 24,
        fill: str = COLORS["card"],
        outline: str = COLORS["line"],
        width: int = 2,
    ) -> None:
        self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def pill(self, x: int, y: int, text: str, fill: str, outline: str, color: str, size: int = 21) -> int:
        scaled_size = scaled_font_size(size)
        w = max(155, int(scaled_size * 0.62 * len(text)) + 62)
        h = 52
        self.draw.rounded_rectangle((x, y, x + w, y + h), radius=26, fill=fill, outline=outline, width=2)
        self.text((x + w / 2, y + h / 2), text, size, color, True, anchor="mm")
        return w

    def wrapped(
        self,
        x: int,
        y: int,
        text: str,
        size: int,
        width_chars: int,
        fill: str = COLORS["muted"],
        bold: bool = False,
        line_gap: int = 8,
    ) -> int:
        line_h = scaled_font_size(size) + line_gap
        for line in wrap(text, width_chars):
            self.text((x, y), line, size, fill, bold)
            y += line_h
        return y

    def bullet(self, x: int, y: int, text: str, size: int = 21, dot: str = COLORS["teal"]) -> None:
        self.draw.ellipse((x, y + 8, x + 11, y + 19), fill=dot)
        self.text((x + 24, y), text, size, COLORS["muted"])

    def arrow(self, x1: int, y: int, x2: int, color: str = COLORS["slate"]) -> None:
        self.draw.line((x1, y, x2, y), fill=color, width=4)
        self.draw.polygon([(x2, y), (x2 - 16, y - 9), (x2 - 16, y + 9)], fill=color)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)


def header(c: Canvas, title: str, subtitle: str) -> None:
    c.text((90, 70), title, 58, COLORS["ink"], True)
    c.wrapped(90, 145, subtitle, 28, 128, COLORS["muted"])


def stage_card(
    c: Canvas,
    box: tuple[int, int, int, int],
    tag: str,
    title: str,
    bullets: list[str],
    latency: str,
    plan: str,
    color: str,
    fill: str,
) -> None:
    x1, y1, x2, y2 = box
    c.rounded(box, 26, fill, color, 4)
    c.draw.rounded_rectangle((x1 + 28, y1 + 26, x1 + 170, y1 + 68), radius=21, fill=color)
    c.text((x1 + 99, y1 + 47), tag, 19, "#ffffff", True, anchor="mm")
    c.wrapped(x1 + 30, y1 + 95, title, 28, 18, COLORS["ink"], True, line_gap=6)
    c.draw.line((x1 + 30, y1 + 166, x2 - 30, y1 + 166), fill=COLORS["line"], width=2)
    yy = y1 + 190
    for item in bullets:
        c.bullet(x1 + 34, yy, item, 20, color)
        yy += 38
    c.draw.rounded_rectangle((x1 + 28, y2 - 66, x1 + 205, y2 - 22), radius=18, fill=COLORS["card"], outline=color, width=2)
    c.text((x1 + 116, y2 - 43), latency, 20, COLORS["ink"], True, anchor="mm")
    c.draw.rounded_rectangle((x1 + 224, y2 - 66, x2 - 26, y2 - 22), radius=18, fill=COLORS["card"], outline=color, width=2)
    c.text(((x1 + 224 + x2 - 26) / 2, y2 - 43), plan, 19, COLORS["ink"], True, anchor="mm")


def draw_pipeline(path: Path, *, title: str, platform_label: str, eager: dict[str, Any], trt: dict[str, Any], drift: dict[str, Any]) -> None:
    c = Canvas(3300, 2050)
    header(
        c,
        title,
        "Same FastWAM RoboTwin checkpoint, real RoboTwin sample, precomputed T5 cache, batch=1, denoise steps=1.",
    )

    e2e = metric(trt, "mean_end_to_end_ms")
    p95 = metric(trt, "p95_end_to_end_ms")
    prefill = stage(trt, "video_prefill_ms")

    x = 90
    for text, fill, outline, color in [
        ("FP16 TensorRT", COLORS["teal_soft"], COLORS["teal"], COLORS["teal"]),
        ("real-text cache", COLORS["violet_soft"], COLORS["violet"], COLORS["violet"]),
        ("3 engines", COLORS["blue_soft"], COLORS["blue"], COLORS["blue"]),
        (f"E2E {e2e:.2f} ms", COLORS["red_soft"], COLORS["red"], COLORS["red"]),
        (platform_label, COLORS["slate_soft"], COLORS["ink"], COLORS["ink"]),
    ]:
        x += c.pill(x, 235, text, fill, outline, color) + 18

    c.rounded((80, 330, 3220, 980), 30, COLORS["card"], COLORS["line"], 2)
    c.text((125, 385), "Runtime Topology", 34, COLORS["ink"], True)
    c.text((125, 430), "Short arrows show tensor handoff; large K/V tensors stay on device in the partitioned runtime.", 23, COLORS["muted"])

    y = 500
    boxes = [
        (125, y, 520, y + 360),
        (640, y, 1050, y + 360),
        (1170, y, 1580, y + 360),
        (1700, y, 2110, y + 360),
        (2230, y, 2640, y + 360),
        (2760, y, 3175, y + 360),
    ]
    stage_card(
        c,
        boxes[0],
        "Input",
        "Host Preprocess",
        ["3 cameras, 384x320", "14D state", "T5 context"],
        "real sample",
        "host",
        COLORS["slate"],
        COLORS["slate_soft"],
    )
    stage_card(
        c,
        boxes[1],
        "TRT 1",
        "VAE Image Encoder",
        ["image [1,3,384,320]", "first-frame latents", "[1,48,1,24,20]"],
        f"{stage(trt, 'vae_image_encoder_ms'):.2f} ms",
        f"{engine_gib(trt, 'vae_image_encoder'):.2f} GiB",
        COLORS["blue"],
        COLORS["blue_soft"],
    )
    stage_card(
        c,
        boxes[2],
        "TRT 2",
        "Video Prefill + K/V",
        ["30 Transformer layers", "context [1,129,4096]", "60 K/V outputs"],
        f"{prefill:.2f} ms",
        f"{engine_gib(trt, 'video_prefill'):.2f} GiB",
        COLORS["teal"],
        COLORS["teal_soft"],
    )
    stage_card(
        c,
        boxes[3],
        "TRT 3",
        "Action Denoise Step",
        ["action latent [1,32,14]", "video K/V cache", "pred_action"],
        f"{stage(trt, 'action_denoise_loop_ms'):.2f} ms",
        f"{engine_gib(trt, 'action_step_dynamic_kv'):.2f} GiB",
        COLORS["blue"],
        COLORS["blue_soft"],
    )
    stage_card(
        c,
        boxes[4],
        "Host",
        "Flow-Match Scheduler",
        ["N=1 benchmark", "delta = -1.0", "FP16 timestep"],
        f"{stage(trt, 'scheduler_total_ms'):.2f} ms",
        "no engine",
        COLORS["green"],
        COLORS["green_soft"],
    )
    stage_card(
        c,
        boxes[5],
        "Host",
        "Action Decode / Denorm",
        ["inverse z-score", "chunk [1,32,14]", f"finite={trt.get('output_finite')}"],
        f"{stage(trt, 'action_decode_ms'):.2f} ms",
        "no engine",
        COLORS["green"],
        COLORS["green_soft"],
    )
    for left, right in zip(boxes, boxes[1:]):
        c.arrow(left[2] + 18, y + 180, right[0] - 22)

    c.rounded((80, 1050, 1575, 1620), 28, COLORS["card"], COLORS["line"], 2)
    c.text((125, 1100), "Latency Anatomy", 33, COLORS["ink"], True)
    c.text((125, 1144), f"Mean E2E {e2e:.2f} ms; p95 {p95:.2f} ms. Video prefill dominates this platform.", 23, COLORS["muted"])
    parts = [
        ("VAE", stage(trt, "vae_image_encoder_ms"), COLORS["blue"]),
        ("Prefill", prefill, COLORS["teal"]),
        ("KV cast", stage(trt, "kv_cast_ms"), COLORS["slate"]),
        ("Denoise", stage(trt, "action_denoise_loop_ms"), COLORS["orange"]),
        ("Sched", stage(trt, "scheduler_total_ms"), COLORS["green"]),
        ("Decode", stage(trt, "action_decode_ms"), COLORS["slate"]),
    ]
    total = max(sum(value for _, value, _ in parts), 1.0)
    bx = 125
    by = 1225
    max_w = 1320
    for name, value, color in parts:
        w = max(18, int(max_w * value / total))
        c.draw.rounded_rectangle((bx, by, bx + w, by + 82), radius=12, fill=color)
        if w > 95:
            c.text((bx + w / 2, by + 28), name, 19, "#ffffff", True, anchor="mm")
            c.text((bx + w / 2, by + 58), f"{value:.2f} ms", 18, "#ffffff", True, anchor="mm")
        bx += w + 5
    c.text((125, 1365), f"Prefill share: {prefill:.2f} / {e2e:.2f} = {pct(prefill, e2e):.1f}%", 28, COLORS["red"], True)
    c.text((125, 1410), "This is the stage to optimize first on both AGX and DGX.", 23, COLORS["muted"])

    c.rounded((1665, 1050, 3220, 1620), 28, COLORS["card"], COLORS["line"], 2)
    c.text((1710, 1100), "Validity Boundaries", 33, COLORS["ink"], True)
    d1, d2 = drift_summary(drift)
    for yy, key, val, color in [
        (1160, "Aligned", "checkpoint, sample, text cache, batch=1, FP16 TRT topology", COLORS["green"]),
        (1235, "Not Hardware-Only", "TRT/CUDA/export stack and tactics differ across platforms", COLORS["red"]),
        (1310, "Drift", d1, COLORS["teal"]),
        (1385, "Action Drift", d2, COLORS["orange"]),
    ]:
        c.text((1710, yy), key, 25, color, True)
        c.wrapped(2055, yy, val, 22, 62, COLORS["muted"])
    c.text((1710, 1535), "Report as: same deployment chain reproduced, not a strict hardware-only A/B.", 24, COLORS["ink"], True)

    c.rounded((80, 1715, 3220, 1945), 28, COLORS["orange_soft"], "#fed7aa", 2)
    c.text((125, 1765), "Interpretation", 32, COLORS["ink"], True)
    c.wrapped(
        125,
        1815,
        "The runtime is finite-output-valid. The main bottleneck is the Transformer video prefill/K/V partition, not image preprocessing or action decode.",
        24,
        135,
        COLORS["muted"],
    )
    c.save(path)


def small_bar(c: Canvas, x: int, y: int, width: int, value: float, max_value: float, color: str, label: str) -> None:
    c.draw.rounded_rectangle((x, y, x + width, y + 34), radius=8, fill="#e2e8f0")
    c.draw.rounded_rectangle((x, y, x + max(2, int(width * value / max_value)), y + 34), radius=8, fill=color)
    c.text((x + width + 16, y + 4), label, 20, COLORS["ink"], True)


def draw_comparison(
    path: Path,
    *,
    agx_eager: dict[str, Any],
    agx_trt: dict[str, Any],
    agx_drift: dict[str, Any],
    dgx_eager: dict[str, Any],
    dgx_trt: dict[str, Any],
    dgx_drift: dict[str, Any],
) -> None:
    del agx_drift
    c = Canvas(3400, 2600)
    header(
        c,
        "FastWAM AGX Orin vs. DGX Spark",
        "Cross-platform reproduction of the same FastWAM deployment chain. The result is aligned for runtime topology, but not a strict hardware-only A/B.",
    )
    x = 90
    for text, fill, outline, color in [
        ("same model + sample", COLORS["teal_soft"], COLORS["teal"], COLORS["teal"]),
        ("FP16 TensorRT engines", COLORS["blue_soft"], COLORS["blue"], COLORS["blue"]),
        ("batch=1, steps=1", COLORS["orange_soft"], COLORS["orange"], COLORS["orange"]),
        ("not hardware-only", COLORS["red_soft"], COLORS["red"], COLORS["red"]),
    ]:
        x += c.pill(x, 235, text, fill, outline, color) + 18

    agx_e2e = metric(agx_trt, "mean_end_to_end_ms")
    dgx_e2e = metric(dgx_trt, "mean_end_to_end_ms")
    e2e_speed = agx_e2e / dgx_e2e
    agx_prefill = stage(agx_trt, "video_prefill_ms")
    dgx_prefill = stage(dgx_trt, "video_prefill_ms")
    prefill_speed = agx_prefill / dgx_prefill
    target_3x = agx_e2e / 3.0
    dgx_prefill_share = pct(dgx_prefill, dgx_e2e)
    del agx_eager, dgx_eager

    c.rounded((80, 330, 1570, 820), 28, COLORS["card"], COLORS["line"], 2)
    c.text((125, 385), "Aligned Runtime Variables", 34, COLORS["ink"], True)
    aligned = [
        ("Model", "FastWAM released RoboTwin checkpoint"),
        ("Input", "same real RoboTwin sample and instruction"),
        ("Precision", "FP16 TensorRT engines"),
        ("Protocol", "batch=1, warmup/measure=2/5, steps=1"),
        ("Topology", "VAE -> video prefill -> action step -> host decode"),
    ]
    yy = 450
    for key, val in aligned:
        c.text((125, yy), key, 23, COLORS["teal"], True)
        c.text((330, yy), val, 23, COLORS["muted"])
        yy += 58

    c.rounded((1665, 330, 3320, 820), 28, COLORS["card"], COLORS["line"], 2)
    c.text((1710, 385), "Not Fully Controlled", 34, COLORS["ink"], True)
    uncontrolled = [
        ("TRT stack", "AGX 10.3.0 vs DGX 10.16.1"),
        ("Builder/tactics", "engines are platform-specific plan files"),
        ("Power", "AGX MAXN + jetson_clocks; DGX default boost"),
        ("KV dtype", "AGX casts 29 K/V bindings; DGX direct FP16"),
    ]
    yy = 450
    for key, val in uncontrolled:
        c.text((1710, yy), key, 23, COLORS["red"], True)
        c.text((1945, yy), val, 23, COLORS["muted"])
        yy += 58

    c.rounded((80, 890, 1570, 1390), 28, COLORS["card"], COLORS["line"], 2)
    c.text((125, 945), "Measured Speedup", 34, COLORS["ink"], True)
    c.text((125, 1015), f"TRT E2E: {agx_e2e:.2f} ms / {dgx_e2e:.2f} ms = {e2e_speed:.2f}x", 36, COLORS["red"], True)
    max_e2e = max(agx_e2e, dgx_e2e)
    small_bar(c, 125, 1135, 850, agx_e2e, max_e2e, COLORS["blue"], f"AGX TRT {agx_e2e:.2f} ms")
    small_bar(c, 125, 1210, 850, dgx_e2e, max_e2e, COLORS["orange"], f"DGX TRT {dgx_e2e:.2f} ms")
    c.text((125, 1310), f"TRT throughput: {1000/agx_e2e:.2f} Hz -> {1000/dgx_e2e:.2f} Hz", 25, COLORS["ink"], True)

    c.rounded((1665, 890, 3320, 1390), 28, COLORS["red_soft"], "#fecdd3", 3)
    c.text((1710, 945), "Why DGX Is < 2x Faster", 34, COLORS["red"], True)
    c.text((1710, 1010), f"3x target budget = AGX E2E / 3 = {agx_e2e:.2f} / 3 = {target_3x:.2f} ms", 27, COLORS["ink"], True)
    c.text((1710, 1075), f"DGX video prefill alone = {dgx_prefill:.2f} ms", 32, COLORS["red"], True)
    c.text((1710, 1140), f"{dgx_prefill:.2f} ms > {target_3x:.2f} ms, so 3x is impossible without prefill optimization.", 27, COLORS["ink"], True)
    c.text((1710, 1215), f"Prefill speedup = {agx_prefill:.2f} / {dgx_prefill:.2f} = {prefill_speed:.2f}x", 28, COLORS["teal"], True)
    c.text((1710, 1280), f"Prefill share on DGX = {dgx_prefill:.2f} / {dgx_e2e:.2f} = {dgx_prefill_share:.1f}%", 28, COLORS["teal"], True)

    c.rounded((80, 1460, 3320, 1995), 28, COLORS["card"], COLORS["line"], 2)
    c.text((125, 1515), "Stage Split", 34, COLORS["ink"], True)
    stage_rows = [
        ("VAE image encoder", "vae_image_encoder_ms", "TRT 1"),
        ("Video prefill + K/V", "video_prefill_ms", "TRT 2, dominant"),
        ("K/V cast", "kv_cast_ms", "platform dtype artifact"),
        ("Action denoise loop", "action_denoise_loop_ms", "TRT 3"),
        ("Scheduler", "scheduler_total_ms", "host"),
        ("Decode", "action_decode_ms", "host"),
    ]
    x0 = 125
    c.text((x0, 1582), "Stage", 23, COLORS["ink"], True)
    c.text((800, 1582), "AGX", 23, COLORS["blue"], True)
    c.text((1110, 1582), "DGX", 23, COLORS["orange"], True)
    c.text((1420, 1582), "Speedup", 23, COLORS["red"], True)
    c.text((1700, 1582), "Interpretation", 23, COLORS["ink"], True)
    yy = 1635
    for idx, (name, key, note) in enumerate(stage_rows):
        if idx % 2 == 0:
            c.draw.rectangle((110, yy - 12, 3290, yy + 42), fill=COLORS["slate_soft"])
        av = stage(agx_trt, key)
        dv = stage(dgx_trt, key)
        sp = av / dv if dv else 0.0
        c.text((x0, yy), name, 22, COLORS["ink"], True if "Video" in name else False)
        c.text((800, yy), f"{av:.2f} ms", 22, COLORS["blue"], True if "Video" in name else False)
        c.text((1110, yy), f"{dv:.2f} ms", 22, COLORS["orange"], True if "Video" in name else False)
        c.text((1420, yy), f"{sp:.2f}x", 22, COLORS["red"], True)
        c.text((1700, yy), note, 22, COLORS["muted"])
        yy += 58

    c.rounded((80, 2070, 1570, 2385), 28, COLORS["card"], COLORS["line"], 2)
    c.text((125, 2120), "Video Prefill Profile", 32, COLORS["ink"], True)
    c.wrapped(
        125,
        2178,
        "AGX profile: FFN/MLP 33.37 ms (44.5%), self-attn 15.42 ms (20.6%), cross-attn/KV 14.17 ms (18.9%). DGX profile shows the same Transformer prefill stage remains dominant.",
        24,
        78,
        COLORS["muted"],
    )
    c.text((125, 2310), "Bottleneck: batch=1 Transformer prefill, not image preprocessing.", 25, COLORS["red"], True)

    c.rounded((1665, 2070, 3320, 2385), 28, COLORS["card"], COLORS["line"], 2)
    c.text((1710, 2120), "Optimization Direction", 32, COLORS["ink"], True)
    for i, line in enumerate(
        [
            "Prioritize video_prefill INT8/PTQ/QDQ calibration.",
            "Consider fusing prefill + action step to avoid external K/V traffic.",
            "Use Nsight for achieved Tensor Core and memory bandwidth, not nvidia-smi busy.",
            "CUDA Graph cuts enqueue overhead but only changed prefill latency by a few percent.",
        ]
    ):
        c.bullet(1710, 2178 + i * 43, line, 22, COLORS["teal"])

    c.text((90, 2520), "Figure. FastWAM AGX/DGX deployment-chain comparison. Values are measured runtime; software-stack differences are documented.", 22, COLORS["muted"])
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
