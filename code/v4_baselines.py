"""Baselines, evaluation probes, and calibration metrics for V4 study.

Addresses:
- 1.4 Persistence baseline & Delta-HP metrics (R^2 over persistence, changed-slot MSE).
- 1.3 Non-circular frozen latent probing against Supervised and Random baselines.
- 3.3 Calibration metrics (Brier score, ECE, log loss).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_squared_error, mean_absolute_error
from typing import Any


def compute_hp_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_curr: np.ndarray,
) -> dict[str, float]:
    """Compute comprehensive HP dynamics metrics including persistence baseline and R^2 over persistence."""
    # Overall next-HP MSE and MAE
    hp_mse = float(mean_squared_error(y_true, y_pred))
    hp_mae = float(mean_absolute_error(y_true, y_pred))

    # Persistence baseline (predicting y_curr)
    persist_mse = float(mean_squared_error(y_true, y_curr))
    persist_mae = float(mean_absolute_error(y_true, y_curr))

    # Delta HP metrics: true delta vs predicted delta
    true_delta = y_true - y_curr
    pred_delta = y_pred - y_curr
    delta_mse = float(mean_squared_error(true_delta, pred_delta))
    delta_mae = float(mean_absolute_error(true_delta, pred_delta))
    zero_delta_mse = float(mean_squared_error(true_delta, np.zeros_like(true_delta)))

    # Changed-slots only metrics (where HP actually changed between turns)
    changed_mask = np.abs(true_delta) > 1e-4
    if changed_mask.any():
        changed_pred_mse = float(mean_squared_error(true_delta[changed_mask], pred_delta[changed_mask]))
        changed_zero_mse = float(mean_squared_error(true_delta[changed_mask], np.zeros_like(true_delta[changed_mask])))
        changed_mae = float(mean_absolute_error(true_delta[changed_mask], pred_delta[changed_mask]))
        changed_zero_mae = float(mean_absolute_error(true_delta[changed_mask], np.zeros_like(true_delta[changed_mask])))
    else:
        changed_pred_mse = 0.0
        changed_zero_mse = 0.0
        changed_mae = 0.0
        changed_zero_mae = 0.0

    # R^2 improvement over persistence baseline: 1 - MSE(pred, target) / MSE(curr, target)
    r2_persist = 1.0 - (hp_mse / max(1e-8, persist_mse))

    return {
        "hp_mse": hp_mse,
        "hp_mae": hp_mae,
        "persist_mse": persist_mse,
        "persist_mae": persist_mae,
        "delta_mse": delta_mse,
        "delta_mae": delta_mae,
        "zero_delta_mse": zero_delta_mse,
        "changed_slots_mse": changed_pred_mse,
        "changed_slots_zero_baseline_mse": changed_zero_mse,
        "changed_slots_mae": changed_mae,
        "changed_slots_zero_baseline_mae": changed_zero_mae,
        "r2_over_persistence": r2_persist,
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE), Brier score, and log loss."""
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    brier = float(brier_score_loss(y_true, probs))
    loss = float(log_loss(y_true, probs))

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        bin_count = int(bin_mask.sum())
        if bin_count > 0:
            bin_acc = float(y_true[bin_mask].mean())
            bin_conf = float(probs[bin_mask].mean())
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return {
        "brier_score": brier,
        "log_loss": loss,
        "expected_calibration_error": float(ece),
    }


def run_frozen_latent_probes(
    train_latents: np.ndarray,
    train_curr_hps: np.ndarray,
    train_next_hps: np.ndarray,
    train_winners: np.ndarray,
    test_latents: np.ndarray,
    test_curr_hps: np.ndarray,
    test_next_hps: np.ndarray,
    test_winners: np.ndarray,
) -> dict[str, Any]:
    """Train independent linear/logistic probes on frozen representations to evaluate downstream retention."""
    # 1. Delta HP dynamics probe: predict delta_hp from frozen latent + current HP
    # Ridge learns: delta_hat = W [z, y_curr] + b
    # next_pred = y_curr + delta_hat
    train_delta = train_next_hps - train_curr_hps
    
    # Concatenate latent with current HP so the probe knows the base HP state
    train_features = np.concatenate([train_latents, train_curr_hps], axis=-1)
    test_features = np.concatenate([test_latents, test_curr_hps], axis=-1)

    ridge_delta = Ridge(alpha=10.0)
    ridge_delta.fit(train_features, train_delta)
    test_pred_delta = ridge_delta.predict(test_features)
    test_pred_hp = np.clip(test_curr_hps + test_pred_delta, 0.0, 1.0)
    hp_metrics = compute_hp_metrics(test_next_hps, test_pred_hp, test_curr_hps)

    # 2. Logistic regression probe for Winner outcome
    clf_win = LogisticRegression(max_iter=1000, C=1.0)
    clf_win.fit(train_latents, train_winners)
    test_pred_win = clf_win.predict(test_latents)
    test_probs_win = clf_win.predict_proba(test_latents)[:, 1]

    win_acc = float(accuracy_score(test_winners, test_pred_win))
    calib_metrics = compute_calibration_metrics(test_winners, test_probs_win)

    return {
        "win_acc": win_acc,
        **calib_metrics,
        **hp_metrics,
    }
