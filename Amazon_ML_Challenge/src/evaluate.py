from __future__ import annotations

"""
Evaluation and threshold tuning module for Amazon ML Challenge - Business Entity Resolution.
Computes Precision, Recall, F0.5 (primary competition metric), and performs group-level validation.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - runtime guard for restricted environments
    raise ImportError("numpy is required to run evaluate.py") from exc

def compute_f_beta(p: float, r: float, beta: float = 0.5) -> float:
    b2 = beta * beta
    return ((1.0 + b2) * p * r) / (b2 * p + r) if (b2 * p + r) > 0 else 0.0

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(p, 5),
        "recall": round(r, 5),
        "f0_5": round(compute_f_beta(p, r, 0.5), 5),
        "f1": round(compute_f_beta(p, r, 1.0), 5)
    }

def split_entities(s1_ids: List[str], val_ratio: float = 0.20, seed: int = 42) -> Tuple[Set[str], Set[str]]:
    uniq = sorted(list(set(s1_ids)))
    shuffled = np.random.RandomState(seed).permutation(uniq)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return set(shuffled[n_val:]), set(shuffled[:n_val])

def tune_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, dict, list]:
    best_f, best_t, best_m, hist = -1.0, 0.60, {}, []
    for t in np.arange(0.20, 0.96, 0.05):
        t = round(float(t), 2)
        m = calculate_metrics(y_true, (y_probs >= t).astype(int))
        m["threshold"] = t
        hist.append(m)
        if m["f0_5"] > best_f:
            best_f, best_t, best_m = m["f0_5"], t, m
    return best_t, best_m, hist

def save_metrics_report(metrics: dict, output_file: Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
