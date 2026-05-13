#!/usr/bin/env python3
"""Loss plotting helpers for standalone testing.

This module provides pure-Python utilities to prepare data for loss plots
without relying on PyQt rendering. It is intended to be unit-tested and used
by non-GUI tests or quick scripted checks.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np


def smooth(data: List[float], window: Optional[int]) -> np.ndarray:
    if window is None or window <= 1:
        return np.asarray(data, dtype=float)
    arr = np.asarray(data, dtype=float)
    return np.convolve(arr, np.ones(window) / float(window), mode="same")


def build_plot_data(
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    smooth_window: Optional[int] = None,
    w: int = 800,
    h: int = 400,
    margin: int = 50,
) -> Dict:
    # Prepare data with optional smoothing
    train = smooth(train_loss, smooth_window)
    data_all = train
    val = None
    if val_loss is not None:
        val = smooth(val_loss, smooth_window)
        data_all = np.concatenate([train, val])

    min_v = float(np.min(data_all))
    max_v = float(np.max(data_all))
    eps = 1e-8

    def norm_y(v: float) -> float:
        return (v - min_v) / (max_v - min_v + eps)

    plot_w = w - 2 * margin
    plot_h = h - 2 * margin

    def compute_points(data_arr: np.ndarray) -> List[Tuple[int, int]]:
        n = len(data_arr)
        if n == 0:
            return []
        points: List[Tuple[int, int]] = []
        for i, v in enumerate(data_arr):
            x = margin + int(i / (n - 1) * plot_w)
            y = margin + int((1 - norm_y(v)) * plot_h)
            points.append((x, y))
        return points

    train_points = compute_points(train)
    val_points = compute_points(val) if val is not None else None

    return {
        "train_points": train_points,
        "val_points": val_points,
        "min_v": min_v,
        "max_v": max_v,
        "width": w,
        "height": h,
        "margin": margin,
        "smooth_window": smooth_window,
    }


__all__ = ["smooth", "build_plot_data"]
