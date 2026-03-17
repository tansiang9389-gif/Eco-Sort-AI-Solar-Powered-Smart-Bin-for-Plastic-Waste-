#!/usr/bin/env python3
"""
=============================================================================
Eco-Sort AI — Model Evaluation Script
=============================================================================
Computes the full Ultralytics metric suite on the validation (or test) split:

    Primary metrics
    ───────────────
    mAP50       — mean Average Precision @ IoU=0.50
                  (the standard single-threshold metric; target ≥ 0.98)
    mAP50-95    — mean AP averaged over IoU thresholds 0.50–0.95 step 0.05
                  (COCO primary metric; more stringent; target ≥ 0.80)
    Precision   — TP / (TP + FP)  at the confidence threshold that maximises F1
    Recall      — TP / (TP + FN)  at the same threshold
    F1          — harmonic mean of Precision and Recall

    Per-class breakdown
    ───────────────────
    AP50 per class, Precision per class, Recall per class

    Confusion matrix
    ────────────────
    N×N matrix (N = nc + 1 for background).  Saved as PNG.
    Shows where the model confuses classes, e.g., PS ↔ PP.

    PR curves & F1 curves
    ─────────────────────
    Saved as PNG plots by Ultralytics automatically.

98 % Accuracy Interpretation
─────────────────────────────
"98 % accuracy" in object detection typically means mAP50 ≥ 0.98 on the
held-out validation set.  This is an ambitious target; typical YOLO11n
on a well-annotated single-domain dataset achieves 0.90–0.97 mAP50.
To reach 0.98:
  1. Annotate ≥ 1 000 high-quality instances per class.
  2. Use the full augmentation pipeline in train_yolo11.py.
  3. Fine-tune for 300+ epochs with patience=50.
  4. Optionally use yolo11s or yolo11m if latency budget allows.

Usage:
    python evaluate.py --weights runs/detect/.../best.pt --split val
    python evaluate.py --weights best.pt --split test --conf 0.45 --iou 0.5

Dependencies:
    pip install ultralytics matplotlib seaborn pandas
=============================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Install ultralytics:  pip install ultralytics")


# Target thresholds — update if project requirements change
MAP50_TARGET    = 0.98
MAP5095_TARGET  = 0.80


# =============================================================================
# CORE EVALUATION
# =============================================================================

def evaluate(
    weights_path: str,
    data_yaml:    str,
    split:        str  = "val",
    imgsz:        int  = 640,
    conf:         float = 0.001,   # low conf → high recall; NMS handles FP
    iou:          float = 0.6,
    device:       str  = "0",
    save_dir:     str  = None,
    verbose:      bool = True,
) -> dict:
    """
    Run model.val() and return a structured metrics dictionary.

    Parameters
    ----------
    weights_path  : Path to .pt checkpoint.
    data_yaml     : Path to data.yaml.
    split         : Dataset split to evaluate on ('val' or 'test').
    imgsz         : Evaluation image size (should match training imgsz).
    conf          : Detection confidence threshold.
                    Use 0.001 for standard COCO-style mAP evaluation
                    (sweeps all thresholds; Ultralytics default).
    iou           : NMS IoU threshold for deduplication.
    device        : Inference device ('cpu', '0', 'mps').
    save_dir      : Output directory for confusion matrix + plots.
    verbose       : Print per-class table.

    Returns
    -------
    dict with keys: mAP50, mAP50_95, precision, recall, f1,
                    per_class_ap50, per_class_precision, per_class_recall
    """
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model = YOLO(weights_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = save_dir or f"runs/eval_{timestamp}"

    print(f"\n[EVAL] Starting evaluation")
    print(f"       Weights : {weights_path}")
    print(f"       Data    : {data_yaml}")
    print(f"       Split   : {split}")
    print(f"       Device  : {device}\n")

    # model.val() runs the full COCO-style evaluation loop:
    #   1. Iterates the dataset split in batches.
    #   2. Runs model.predict() at conf=0.001 (all predictions kept).
    #   3. Sorts predictions by score and computes AP per class via
    #      the standard 11-point interpolation (COCO mAP50-95 uses 101-point).
    #   4. Saves confusion matrix, PR curve, F1 curve, and label/pred plots.
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        project=project_dir,
        name="metrics",
        save_json=True,       # saves COCO-format predictions.json
        plots=True,           # confusion matrix, PR curve, F1 curve
        verbose=verbose,
    )

    return metrics, project_dir


# =============================================================================
# METRIC EXTRACTION & REPORTING
# =============================================================================

def extract_metrics(metrics) -> dict:
    """
    Extract scalar and per-class metrics from the Ultralytics DetMetrics object.

    DetMetrics attributes (Ultralytics ≥ 8.x):
        metrics.box.map50      → mAP50
        metrics.box.map        → mAP50-95
        metrics.box.mp         → mean Precision
        metrics.box.mr         → mean Recall
        metrics.box.maps       → per-class AP50 array
        metrics.box.ap50       → per-class AP @ 0.50 (same as maps)
        metrics.box.ap         → per-class AP50-95 array
    """
    box = metrics.box

    # Overall metrics
    map50   = float(box.map50)
    map5095 = float(box.map)
    prec    = float(box.mp)
    rec     = float(box.mr)
    f1      = 2 * prec * rec / (prec + rec + 1e-9)

    # Per-class (returns np.ndarray)
    per_cls_ap50 = box.ap50.tolist() if hasattr(box, "ap50") else []
    per_cls_ap   = box.ap.tolist()   if hasattr(box, "ap")   else []

    return {
        "mAP50":              map50,
        "mAP50_95":           map5095,
        "precision":          prec,
        "recall":             rec,
        "f1":                 f1,
        "per_class_ap50":     per_cls_ap50,
        "per_class_ap5095":   per_cls_ap,
    }


def print_report(
    metrics_dict: dict,
    class_names: list[str],
    save_path: str = None,
) -> None:
    """
    Print a formatted evaluation report and optionally save it as JSON.
    Includes clear PASS/FAIL against the 98 % mAP50 target.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  ECO-SORT AI — EVALUATION REPORT")
    print(sep)

    map50   = metrics_dict["mAP50"]
    map5095 = metrics_dict["mAP50_95"]
    prec    = metrics_dict["precision"]
    rec     = metrics_dict["recall"]
    f1      = metrics_dict["f1"]

    # ── Overall metrics ───────────────────────────────────────────────────────
    status50   = "✓ PASS" if map50   >= MAP50_TARGET   else "✗ FAIL"
    status5095 = "✓ PASS" if map5095 >= MAP5095_TARGET else "✗ FAIL"

    print(f"\n  OVERALL METRICS")
    print(f"  {'mAP50':<20} {map50:.4f}   target ≥ {MAP50_TARGET:.2f}  [{status50}]")
    print(f"  {'mAP50-95':<20} {map5095:.4f}   target ≥ {MAP5095_TARGET:.2f}  [{status5095}]")
    print(f"  {'Precision':<20} {prec:.4f}")
    print(f"  {'Recall':<20} {rec:.4f}")
    print(f"  {'F1 Score':<20} {f1:.4f}")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    per_cls_ap50   = metrics_dict.get("per_class_ap50",   [])
    per_cls_ap5095 = metrics_dict.get("per_class_ap5095", [])

    if per_cls_ap50 and class_names:
        print(f"\n  PER-CLASS BREAKDOWN")
        print(f"  {'Class':<14} {'AP50':>8} {'AP50-95':>10}")
        print(f"  {'-'*36}")
        for i, name in enumerate(class_names):
            ap50  = per_cls_ap50[i]   if i < len(per_cls_ap50)   else float("nan")
            ap95  = per_cls_ap5095[i] if i < len(per_cls_ap5095) else float("nan")
            flag  = " ← LOW" if ap50 < 0.90 else ""
            print(f"  {name:<14} {ap50:>8.4f} {ap95:>10.4f}{flag}")

    print(f"\n{sep}\n")

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    if map50 < MAP50_TARGET:
        print("  DIAGNOSIS — mAP50 below target. Common causes:")
        print("  • Insufficient training data for low-AP classes (annotate more)")
        print("  • Class imbalance (use class_weights or oversample rare classes)")
        print("  • Model too small (try yolo11s or yolo11m)")
        print("  • Resolution too low (increase imgsz from 640 to 1280)")
        print("  • Training not converged (run more epochs or lower lr0)\n")
    else:
        print("  TARGET ACHIEVED — model is ready for edge export.\n")
        print("  Next: python export_edge.py --weights <best.pt> --format tflite_int8\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"  Metrics saved → {out.resolve()}")


# =============================================================================
# CONFUSION MATRIX ANALYSIS
# =============================================================================

def analyse_confusion_matrix(eval_dir: str, class_names: list[str]) -> None:
    """
    Load the confusion matrix saved by Ultralytics (confusion_matrix.csv or
    confusion_matrix_normalized.csv) and print a text summary of the most
    common confusions.

    Ultralytics saves:
        runs/eval_.../metrics/confusion_matrix.png
        runs/eval_.../metrics/confusion_matrix_normalized.png

    For a numeric analysis, we re-derive the matrix from predictions.json.
    This function prints which classes are most often confused, guiding
    targeted data collection to close the gap to 98 % mAP.
    """
    csv_path = Path(eval_dir) / "metrics" / "confusion_matrix.csv"
    if not csv_path.exists():
        print(f"[INFO] confusion_matrix.csv not found at {csv_path}.")
        print(f"       See confusion_matrix.png in {eval_dir}/metrics/ instead.")
        return

    try:
        import pandas as pd
    except ImportError:
        print("[SKIP] pandas not installed — skipping confusion matrix text analysis.")
        return

    cm = pd.read_csv(csv_path, index_col=0).values.astype(float)
    n  = cm.shape[0]

    # Normalise row-wise (true-class normalisation)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0)

    print("\n  CONFUSION MATRIX ANALYSIS (normalised by true class)")
    header = "True \\ Pred"
    print(f"  {header:<14}", end="")
    all_names = (class_names or [str(i) for i in range(n - 1)]) + ["background"]
    for name in all_names[:n]:
        print(f" {name[:8]:>8}", end="")
    print()

    for i in range(n):
        row_name = all_names[i] if i < len(all_names) else str(i)
        print(f"  {row_name:<14}", end="")
        for j in range(n):
            val = cm_norm[i, j]
            marker = " ←" if i != j and val > 0.05 else "  "
            print(f" {val:>7.2f}{marker[:1]}", end="")
        print()

    print()
    # Flag top-3 off-diagonal confusions
    off_diag = [(cm_norm[i, j], i, j)
                for i in range(n) for j in range(n) if i != j and cm_norm[i, j] > 0.01]
    off_diag.sort(reverse=True)

    if off_diag:
        print("  TOP CONFUSIONS (fix these to reach 98 % mAP):")
        for rate, i, j in off_diag[:5]:
            ti = all_names[i] if i < len(all_names) else str(i)
            tj = all_names[j] if j < len(all_names) else str(j)
            print(f"    {ti} → predicted as {tj}:  {rate:.1%}")
        print()


# =============================================================================
# BENCHMARK: LATENCY vs. ACCURACY TRADE-OFF
# =============================================================================

def benchmark_inference_speed(
    weights_path: str,
    imgsz: int = 640,
    runs: int = 50,
    device: str = "cpu",
) -> None:
    """
    Measure average inference latency (ms/image) and throughput (FPS) on
    the host device.  Useful for deciding if the model meets the edge
    real-time requirement before export.

    For edge latency, see the EMBEDDED_NOTES in export_edge.py.
    """
    import time
    import torch

    model = YOLO(weights_path)
    dummy = torch.zeros(1, 3, imgsz, imgsz)

    # Warm-up
    for _ in range(5):
        model.predict(source=dummy, imgsz=imgsz, verbose=False)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(source=dummy, imgsz=imgsz, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"\n  INFERENCE LATENCY  ({runs} runs, device={device}, imgsz={imgsz})")
    print(f"  Mean  : {latencies.mean():.1f} ms   ({1000/latencies.mean():.1f} FPS)")
    print(f"  P50   : {np.percentile(latencies, 50):.1f} ms")
    print(f"  P95   : {np.percentile(latencies, 95):.1f} ms")
    print(f"  P99   : {np.percentile(latencies, 99):.1f} ms\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eco-Sort AI — YOLO11 Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",   required=True,
                   help="Path to best.pt")
    p.add_argument("--data",      default="data.yaml",
                   help="Path to data.yaml")
    p.add_argument("--split",     default="val",
                   choices=["val", "test"],
                   help="Dataset split to evaluate")
    p.add_argument("--imgsz",     type=int,   default=640,
                   help="Evaluation image size")
    p.add_argument("--conf",      type=float, default=0.001,
                   help="Confidence threshold (0.001 for COCO-style mAP)")
    p.add_argument("--iou",       type=float, default=0.6,
                   help="NMS IoU threshold")
    p.add_argument("--device",    default="0",
                   help="'cpu', '0', '0,1', 'mps'")
    p.add_argument("--save-dir",  default=None,
                   help="Directory to save plots and JSON report")
    p.add_argument("--benchmark", action="store_true",
                   help="Also run inference latency benchmark")
    p.add_argument("--bench-runs", type=int, default=50,
                   help="Number of inference runs for benchmarking")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load class names from data.yaml for the report
    import yaml
    class_names = []
    if Path(args.data).exists():
        with open(args.data) as f:
            cfg = yaml.safe_load(f)
        class_names = list(cfg.get("names", {}).values()) if isinstance(
            cfg.get("names"), dict) else cfg.get("names", [])

    # ── Run evaluation ────────────────────────────────────────────────────────
    metrics_obj, eval_dir = evaluate(
        weights_path = args.weights,
        data_yaml    = args.data,
        split        = args.split,
        imgsz        = args.imgsz,
        conf         = args.conf,
        iou          = args.iou,
        device       = args.device,
        save_dir     = args.save_dir,
    )

    # ── Extract and report ────────────────────────────────────────────────────
    mdict = extract_metrics(metrics_obj)
    print_report(
        metrics_dict = mdict,
        class_names  = class_names,
        save_path    = str(Path(eval_dir) / "metrics" / "report.json"),
    )

    # ── Confusion matrix analysis ─────────────────────────────────────────────
    analyse_confusion_matrix(eval_dir, class_names)

    # ── Latency benchmark ─────────────────────────────────────────────────────
    if args.benchmark:
        benchmark_inference_speed(
            weights_path = args.weights,
            imgsz        = args.imgsz,
            runs         = args.bench_runs,
            device       = args.device,
        )

    print(f"[INFO] All evaluation artefacts saved → {Path(eval_dir).resolve()}")
    print(f"       (confusion matrix, PR curve, F1 curve, predictions.json)")
