"""
deployment/edge_inference/benchmark_pi.py

Run ON THE RASPBERRY PI 5. Needs only the Pi (no camera, GPS or ESP32):
uses the 38 test photos in deployment/bundle/test_images/.

Measures, for each available format (ONNX, NCNN):
  - 3-stage pipeline: ms per photo, split by stage
  - single-stage baseline: ms per photo
  - agreement with the PC: do we get the same disease labels as the dev PC?
Also records CPU temperature, to spot thermal throttling.

Run from the project root on the Pi:
    python deployment/edge_inference/benchmark_pi.py
"""

import json
import platform
import statistics as st
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from edge_pipeline import BUNDLE, EdgePipeline  # noqa: E402

REPEATS = 3
OUT = HERE.parents[1] / "results" / "metrics" / "pi5_benchmark.md"


def cpu_temp():
    try:
        out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True).stdout
        return out.strip().replace("temp=", "")
    except FileNotFoundError:
        return "n/a"


def summary(values):
    return f"{st.median(values):.0f} / {st.mean(values):.0f}"


def run_format(fmt, images, reference):
    pipe = EdgePipeline(fmt=fmt)
    baseline = pipe.load_baseline()

    for _, img in images[:3]:                                # warm up
        pipe.predict(img)
        baseline.predict(img, conf=0.25, imgsz=pipe.imgsz, verbose=False)

    totals, stages, base_ms = [], {"stage1": [], "stage2": [], "stage3": []}, []
    agree = 0
    for rep in range(REPEATS):
        for name, img in images:
            t0 = time.perf_counter()
            leaves, t = pipe.predict(img)
            totals.append((time.perf_counter() - t0) * 1000)
            for k in stages:
                stages[k].append(t[k])
            if rep == 0:
                agree += Counter(l["class_name"] for l in leaves) == Counter(reference.get(name, []))

            t0 = time.perf_counter()
            baseline.predict(img, conf=0.25, imgsz=pipe.imgsz, verbose=False)
            base_ms.append((time.perf_counter() - t0) * 1000)

    return {"fmt": fmt, "total": totals, "stages": stages, "baseline": base_ms,
            "agree": agree, "n": len(images), "temp": cpu_temp()}


def main():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    reference = json.loads((BUNDLE / "reference.json").read_text())
    paths = sorted((BUNDLE / "test_images").glob("*.jpg"))
    images = [(p.name, cv2.imread(str(p))) for p in paths]
    formats = [f for f in ("onnx", "ncnn") if f in manifest["leaf_segment"]]
    print(f"Photos: {len(images)}, formats: {formats}, CPU temp at start: {cpu_temp()}")

    results = []
    for fmt in formats:
        print(f"Benchmarking {fmt}...")
        results.append(run_format(fmt, images, reference))

    rows = "\n".join(
        f"| {r['fmt']} | {summary(r['total'])} | {1000 / st.mean(r['total']):.2f} | "
        f"{st.mean(r['stages']['stage1']):.0f} | {st.mean(r['stages']['stage2']):.0f} | "
        f"{st.mean(r['stages']['stage3']):.0f} | {summary(r['baseline'])} | "
        f"{r['agree']}/{r['n']} | {r['temp']} |" for r in results)

    report = f"""# ARIS - Raspberry Pi 5 benchmark

**Run:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Device:** {platform.machine()}, Python {platform.python_version()}
**Photos:** {len(images)} test photos x {REPEATS} repeats, image size {manifest['imgsz']}
**Classifier:** {manifest['classifier']['source']}

| Format | Pipeline ms (median, mean) | Photos per second | Stage 1 ms | Stage 2 ms | Stage 3 ms | Baseline ms (median, mean) | Same labels as PC | CPU temp at end |
|---|---|---|---|---|---|---|---|---|
{rows}

"Same labels as PC" = photos where the Pi gives exactly the same disease labels as
the dev PC pipeline. Small differences can come from numeric precision in export;
large differences mean an export problem.
"""
    print(report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()