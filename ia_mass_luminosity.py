#!/usr/bin/env python3
"""
ia_mass_luminosity.py

End-to-end analysis pipeline for Kepler DR25 stellar17 data:
- Load CSV or CSV.GZ with delimiter auto-detection
- Clean and filter to a main-sequence sample
- Compute luminosity and propagated uncertainties
- Fit the mass-luminosity relation in log-log space
- Optional Model 1: polynomial color-to-Teff regression
- Optional Model 2: probabilistic HR diagram classification layers
- Validate Model 2 and use high-confidence subset for robust alpha refit
- Save plots, cleaned data, and result summaries
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Canonical columns used downstream.
CANONICAL_COLUMNS: List[str] = [
    "teff",
    "teff_err_high",
    "teff_err_low",
    "mass",
    "mass_err_high",
    "mass_err_low",
    "radius",
    "radius_err_high",
    "radius_err_low",
    "logg",
    "logg_err_high",
    "logg_err_low",
    "jmag",
    "jmag_err",
    "kmag",
    "kmag_err",
]

# Aliases seen in some stellar17 exports.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "teff": ["teff"],
    "teff_err_high": ["teff_err_high", "teff_err1"],
    "teff_err_low": ["teff_err_low", "teff_err2"],
    "mass": ["mass"],
    "mass_err_high": ["mass_err_high", "mass_err1"],
    "mass_err_low": ["mass_err_low", "mass_err2"],
    "radius": ["radius", "st_radius"],
    "radius_err_high": ["radius_err_high", "radius_err1"],
    "radius_err_low": ["radius_err_low", "radius_err2"],
    "logg": ["logg"],
    "logg_err_high": ["logg_err_high", "logg_err1"],
    "logg_err_low": ["logg_err_low", "logg_err2"],
    "jmag": ["jmag"],
    "jmag_err": ["jmag_err"],
    "kmag": ["kmag"],
    "kmag_err": ["kmag_err"],
}

CLASS_BINS: List[Tuple[str, float, float]] = [
    ("M", 3000.0, 3700.0),
    ("K", 3700.0, 5200.0),
    ("G", 5200.0, 6000.0),
    ("F", 6000.0, 7500.0),
    ("A", 7500.0, 10000.0),
]

COLOR_MAP: Dict[str, str] = {
    "M": "#d73027",
    "K": "#fc8d59",
    "G": "#fee08b",
    "F": "#91bfdb",
    "A": "#4575b4",
}


def checkpoint(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kepler DR25 stellar17 mass-luminosity analysis pipeline"
    )
    parser.add_argument("--data", required=True, help="Path to kepler_stellar17.csv or .csv.gz")
    parser.add_argument("--out", default="./ia_outputs", help="Output base directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.7,
        help="High-confidence threshold for Model 2 max probability",
    )
    parser.add_argument("--disable_model1", action="store_true", help="Disable Model 1 (C_JK -> Teff)")
    parser.add_argument("--disable_model2", action="store_true", help="Disable Model 2 (HR probability layers)")
    parser.add_argument("--grid_size", type=int, default=250, help="Model 2 histogram grid size")
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=1.5,
        help="Model 2 Gaussian smoothing sigma (pixels)",
    )
    return parser.parse_args()


def open_text_any(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", errors="ignore")
    return open(path, mode="rt", encoding="utf-8", errors="ignore")


def detect_delimiter(path: Path) -> str:
    candidates = ["|", ",", "\t"]
    header_line: Optional[str] = None
    with open_text_any(path) as f:
        for raw in f:
            line = raw.strip()
            if line:
                header_line = line
                break

    if header_line is None:
        raise ValueError("Input file appears empty; could not find a non-empty header line.")

    counts = {d: header_line.count(d) for d in candidates}
    best = max(candidates, key=lambda d: counts[d])
    if counts[best] == 0:
        raise ValueError(
            "Could not detect delimiter from header. Expected one of '|', ',' or TAB in header line."
        )
    return best


def make_unique_run_dir(base_out: Path) -> Path:
    base_out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_out / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = base_out / f"{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_column_mapping(all_columns: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    missing: List[str] = []
    for canonical in CANONICAL_COLUMNS:
        aliases = COLUMN_ALIASES.get(canonical, [canonical])
        found = next((name for name in aliases if name in all_columns), None)
        if found is None:
            missing.append(canonical)
        else:
            mapping[canonical] = found

    if missing:
        details = [f"{col} (aliases tried: {COLUMN_ALIASES.get(col, [col])})" for col in missing]
        raise ValueError(
            "Missing required columns needed for analysis:\n  - " + "\n  - ".join(details)
        )
    return mapping


def load_catalog(path: Path, sep: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    header_df = pd.read_csv(path, sep=sep, compression="infer", nrows=0)
    all_columns = [str(c).strip() for c in header_df.columns]
    mapping = resolve_column_mapping(all_columns)

    raw_usecols = sorted(set(mapping.values()))
    df = pd.read_csv(
        path,
        sep=sep,
        compression="infer",
        usecols=raw_usecols,
        low_memory=False,
    )

    rename_map = {raw_name: canonical for canonical, raw_name in mapping.items()}
    df = df.rename(columns=rename_map)

    # Keep only canonical columns in stable order.
    df = df[CANONICAL_COLUMNS].copy()

    # Coerce all used columns to numeric.
    for col in CANONICAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, mapping


def apply_main_sequence_filter(df: pd.DataFrame, mass_max: Optional[float] = 3.8) -> pd.DataFrame:
    mask = (
        (df["logg"] >= 4.1)
        & (df["radius"] <= 2.0)
        & (df["teff"] >= 3000.0)
        & (df["teff"] <= 10000.0)
        & (df["mass"] > 0.0)
    )
    if mass_max is not None:
        mask &= df["mass"] <= mass_max
    return df.loc[mask].copy()


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["C_JK"] = out["jmag"] - out["kmag"]
    out["sigma_C"] = np.sqrt(out["jmag_err"] ** 2 + out["kmag_err"] ** 2)

    out["L_over_Lsun"] = (out["radius"] ** 2) * (out["teff"] / 5772.0) ** 4

    # Protect logs and divisions.
    out = out[
        (out["L_over_Lsun"] > 0.0)
        & (out["teff"] > 0.0)
        & (out["mass"] > 0.0)
        & (out["radius"] > 0.0)
    ].copy()

    out["x_logM"] = np.log10(out["mass"])
    out["y_logL"] = np.log10(out["L_over_Lsun"])

    out["u_logTeff"] = np.log10(out["teff"])
    out["v_logL"] = out["y_logL"]
    out["w_logR"] = np.log10(out["radius"])

    out["sigma_teff"] = 0.5 * (out["teff_err_low"].abs() + out["teff_err_high"])
    out["sigma_radius"] = 0.5 * (out["radius_err_low"].abs() + out["radius_err_high"])
    out["sigma_mass"] = 0.5 * (out["mass_err_low"].abs() + out["mass_err_high"])

    ln10 = np.log(10.0)
    out["sigma_L_fraction"] = np.sqrt(
        (2.0 * out["sigma_radius"] / out["radius"]) ** 2
        + (4.0 * out["sigma_teff"] / out["teff"]) ** 2
    )
    out["sigma_logL"] = out["sigma_L_fraction"] / ln10
    out["sigma_logM"] = (out["sigma_mass"] / out["mass"]) / ln10

    return out


def sample_for_plot(df: pd.DataFrame, nmax: int = 50000, seed: int = 42) -> pd.DataFrame:
    if len(df) <= nmax:
        return df
    return df.sample(n=nmax, random_state=seed)


def fit_degree3_color_teff(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "enabled": True,
        "degree": 3,
    }

    train = train_df[["C_JK", "teff"]].replace([np.inf, -np.inf], np.nan).dropna()
    test = test_df[["C_JK", "teff"]].replace([np.inf, -np.inf], np.nan).dropna()

    result["n_train_used"] = int(len(train))
    result["n_test_used"] = int(len(test))

    if len(train) < 10 or len(test) < 5:
        result["status"] = "skipped"
        result["reason"] = "Insufficient rows after NaN/inf filtering for Model 1."
        return result

    c_train = train["C_JK"].to_numpy()
    t_train = train["teff"].to_numpy()
    c_test = test["C_JK"].to_numpy()
    t_test = test["teff"].to_numpy()

    def design(c: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(c), c, c**2, c**3])

    reg = LinearRegression(fit_intercept=False)
    reg.fit(design(c_train), t_train)

    t_pred = reg.predict(design(c_test))
    rmse = float(np.sqrt(mean_squared_error(t_test, t_pred)))
    r2 = float(r2_score(t_test, t_pred))

    coeffs = [float(v) for v in reg.coef_]
    result.update(
        {
            "status": "ok",
            "coefficients": {
                "a0": coeffs[0],
                "a1": coeffs[1],
                "a2": coeffs[2],
                "a3": coeffs[3],
            },
            "rmse_teff_K": rmse,
            "r2_teff": r2,
        }
    )

    # Plot: C_JK vs Teff with degree-3 curve.
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_train = sample_for_plot(train, seed=seed)
    plot_test = sample_for_plot(test, seed=seed)

    ax.scatter(plot_train["C_JK"], plot_train["teff"], s=8, alpha=0.35, label="Train")
    ax.scatter(plot_test["C_JK"], plot_test["teff"], s=8, alpha=0.35, label="Test")

    c_all = np.concatenate([c_train, c_test])
    c_min, c_max = float(np.nanmin(c_all)), float(np.nanmax(c_all))
    c_grid = np.linspace(c_min, c_max, 500)
    t_curve = reg.predict(design(c_grid))
    ax.plot(c_grid, t_curve, color="black", linewidth=2.0, label="Degree-3 fit")

    ax.set_xlabel("C_JK = J - K [mag]")
    ax.set_ylabel("Effective Temperature Teff [K]")
    ax.set_title("Model 1: Teff from J-K Color")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "model1_color_vs_teff_fit.png", dpi=220)
    plt.close(fig)

    # Optional diagnostic plot: predicted vs true on test.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t_test, t_pred, s=10, alpha=0.5)
    lo = float(min(np.min(t_test), np.min(t_pred)))
    hi = float(max(np.max(t_test), np.max(t_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.5)
    ax.set_xlabel("True Teff [K]")
    ax.set_ylabel("Predicted Teff [K]")
    ax.set_title("Model 1 Test: Predicted vs True Teff")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "model1_teff_pred_vs_true.png", dpi=220)
    plt.close(fig)

    return result


def assign_teff_classes(teff: pd.Series) -> pd.Series:
    t = teff.to_numpy()
    labels = np.full(shape=t.shape, fill_value="", dtype=object)

    labels[(t >= 3000.0) & (t < 3700.0)] = "M"
    labels[(t >= 3700.0) & (t < 5200.0)] = "K"
    labels[(t >= 5200.0) & (t < 6000.0)] = "G"
    labels[(t >= 6000.0) & (t < 7500.0)] = "F"
    labels[(t >= 7500.0) & (t <= 10000.0)] = "A"

    out = pd.Series(labels, index=teff.index, dtype="object")
    out = out.replace("", np.nan)
    return out


def run_model2(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    grid_size: int,
    smooth_sigma: float,
    p_threshold: float,
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, List[str], Optional[pd.DataFrame]]:
    result: Dict[str, Any] = {
        "enabled": True,
        "grid_size": int(grid_size),
        "smooth_sigma": float(smooth_sigma),
        "p_threshold": float(p_threshold),
    }

    train_labels = assign_teff_classes(train_df["teff"])
    test_labels = assign_teff_classes(test_df["teff"])

    train_mask = (
        train_labels.notna()
        & np.isfinite(train_df["u_logTeff"])
        & np.isfinite(train_df["v_logL"])
        & np.isfinite(train_df["w_logR"])
    )
    test_mask = (
        test_labels.notna()
        & np.isfinite(test_df["u_logTeff"])
        & np.isfinite(test_df["v_logL"])
        & np.isfinite(test_df["w_logR"])
    )

    tr = train_df.loc[train_mask].copy()
    te = test_df.loc[test_mask].copy()
    y_tr = train_labels.loc[train_mask]
    y_te = test_labels.loc[test_mask]

    result["n_train_used"] = int(len(tr))
    result["n_test_used"] = int(len(te))

    if len(tr) < 10 or len(te) < 5:
        raise RuntimeError("Model 2 enabled but insufficient rows to train/validate after filtering.")

    full_class_order = [name for name, _, _ in CLASS_BINS]
    class_counts = {c: int((y_tr == c).sum()) for c in full_class_order}
    active_classes = [c for c in full_class_order if class_counts[c] > 0]

    if len(active_classes) == 0:
        raise RuntimeError("Model 2 enabled but no class bins contain training stars.")

    result["active_classes"] = active_classes
    result["train_class_counts"] = class_counts

    u_train = tr["u_logTeff"].to_numpy()
    v_train = tr["v_logL"].to_numpy()
    w_train = tr["w_logR"].to_numpy()

    u_test = te["u_logTeff"].to_numpy()
    v_test = te["v_logL"].to_numpy()
    w_test = te["w_logR"].to_numpy()

    # Build grid with small margins.
    u_min, u_max = float(np.min(u_train)), float(np.max(u_train))
    v_min, v_max = float(np.min(v_train)), float(np.max(v_train))
    du = max(0.02 * (u_max - u_min), 1e-4)
    dv = max(0.02 * (v_max - v_min), 1e-4)

    u_edges = np.linspace(u_min - du, u_max + du, grid_size + 1)
    v_edges = np.linspace(v_min - dv, v_max + dv, grid_size + 1)

    k = len(active_classes)
    layers = np.zeros((k, grid_size, grid_size), dtype=float)

    mu = np.zeros(k, dtype=float)
    sigma = np.zeros(k, dtype=float)

    for idx, cls in enumerate(active_classes):
        cmask = y_tr.to_numpy() == cls

        hist2d, _, _ = np.histogram2d(
            u_train[cmask],
            v_train[cmask],
            bins=[u_edges, v_edges],
        )
        layers[idx] = gaussian_filter(hist2d, sigma=smooth_sigma, mode="nearest")

        w_cls = w_train[cmask]
        mu[idx] = float(np.mean(w_cls))
        s = float(np.std(w_cls, ddof=1)) if w_cls.size > 1 else 0.0
        sigma[idx] = max(s, 1e-3)

    layer_sum = np.sum(layers, axis=0)
    p_uv = np.zeros_like(layers)

    nonzero = layer_sum > 0
    p_uv[:, nonzero] = layers[:, nonzero] / layer_sum[nonzero]

    if np.any(~nonzero):
        p_uv[:, ~nonzero] = 1.0 / float(k)

    def predict_probabilities(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        iu = np.searchsorted(u_edges, u, side="right") - 1
        iv = np.searchsorted(v_edges, v, side="right") - 1
        iu = np.clip(iu, 0, grid_size - 1)
        iv = np.clip(iv, 0, grid_size - 1)

        base = p_uv[:, iu, iv]  # shape: (k, n)

        w_row = w[None, :]
        mu_col = mu[:, None]
        sig_col = sigma[:, None]
        radius_like = np.exp(-((w_row - mu_col) ** 2) / (2.0 * sig_col**2))

        raw = base * radius_like
        denom = raw.sum(axis=0)

        zero = denom <= 0
        if np.any(zero):
            raw[:, zero] = 1.0
            denom = raw.sum(axis=0)

        probs = raw / denom[None, :]
        return probs.T  # shape: (n, k)

    probs_test = predict_probabilities(u_test, v_test, w_test)
    pred_idx = np.argmax(probs_test, axis=1)
    pred_labels = np.array(active_classes, dtype=object)[pred_idx]
    pmax = np.max(probs_test, axis=1)

    true_labels = y_te.to_numpy()
    acc = float(accuracy_score(true_labels, pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=active_classes)

    result["accuracy"] = acc
    result["confusion_matrix_labels"] = active_classes
    result["confusion_matrix"] = cm.tolist()

    # Save confusion matrix CSV.
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in active_classes],
        columns=[f"pred_{c}" for c in active_classes],
    )
    cm_csv_path = out_dir / "model2_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path, index=True)

    # Save confusion matrix heatmap.
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(active_classes)))
    ax.set_yticks(np.arange(len(active_classes)))
    ax.set_xticklabels(active_classes)
    ax.set_yticklabels(active_classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Model 2 Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(out_dir / "model2_confusion_matrix.png", dpi=220)
    plt.close(fig)

    # Save HR scatter: v vs u, colored by true class.
    hr_df = te.copy()
    hr_df["true_class"] = true_labels
    hr_plot = sample_for_plot(hr_df, seed=seed)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in active_classes:
        m = hr_plot["true_class"] == cls
        if m.sum() == 0:
            continue
        ax.scatter(
            hr_plot.loc[m, "u_logTeff"],
            hr_plot.loc[m, "v_logL"],
            s=8,
            alpha=0.55,
            color=COLOR_MAP.get(cls, None),
            label=cls,
        )
    ax.set_xlabel("u = log10(Teff [K])")
    ax.set_ylabel("v = log10(L/Lsun)")
    ax.set_title("HR Diagram (Test Set, True Class)")
    ax.invert_xaxis()
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "model2_hr_scatter_true_class.png", dpi=220)
    plt.close(fig)

    # Save one probability layer per active class.
    for idx, cls in enumerate(active_classes):
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            p_uv[idx].T,
            origin="lower",
            extent=[u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]],
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlabel("u = log10(Teff [K])")
        ax.set_ylabel("v = log10(L/Lsun)")
        ax.set_title(f"Model 2 Probability Layer: P({cls} | u,v)")
        fig.colorbar(im, ax=ax, label="Probability")
        fig.tight_layout()
        fig.savefig(out_dir / f"model2_probability_layer_{cls}.png", dpi=220)
        plt.close(fig)

    # Save per-star predictions for traceability.
    pred_df = te.copy()
    pred_df["true_class"] = true_labels
    pred_df["pred_class"] = pred_labels
    pred_df["Pmax"] = pmax
    for idx, cls in enumerate(active_classes):
        pred_df[f"P_{cls}"] = probs_test[:, idx]
    pred_df.to_csv(out_dir / "model2_test_predictions.csv", index=True)

    # Build high-confidence subset from the test set.
    high_conf = pred_df[pred_df["Pmax"] >= p_threshold].copy()
    result["high_conf_count"] = int(len(high_conf))

    return result, pred_df, active_classes, high_conf


def fit_loglog_relation(
    df: pd.DataFrame, x_col: str, y_col: str
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    work = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 2:
        return None, None

    x = work[x_col].to_numpy().reshape(-1, 1)
    y = work[y_col].to_numpy()

    reg = LinearRegression()
    reg.fit(x, y)

    y_hat = reg.predict(x)
    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    r2 = float(r2_score(y, y_hat))
    rmse = float(np.sqrt(mean_squared_error(y, y_hat)))

    out = {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "rmse": rmse,
    }
    work["y_hat"] = y_hat
    work["residual"] = y - y_hat
    return out, work


def plot_mass_luminosity_fit(df_fit: pd.DataFrame, fit: Dict[str, Any], out_dir: Path, tag: str, seed: int) -> None:
    plot_df = sample_for_plot(df_fit, seed=seed)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot_df["x_logM"], plot_df["y_logL"], s=8, alpha=0.45, label="Stars")

    x_line = np.linspace(float(df_fit["x_logM"].min()), float(df_fit["x_logM"].max()), 300)
    y_line = fit["intercept"] + fit["slope"] * x_line
    ax.plot(x_line, y_line, color="black", linewidth=2.0, label="Linear fit")

    ax.set_xlabel("log10(M/Msun)")
    ax.set_ylabel("log10(L/Lsun)")
    ax.set_title(f"Mass-Luminosity Fit ({tag})")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"mass_luminosity_fit_{tag}.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(plot_df["x_logM"], plot_df["residual"], s=8, alpha=0.45)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.3)
    ax.set_xlabel("log10(M/Msun)")
    ax.set_ylabel("Residual y - y_hat [dex]")
    ax.set_title(f"Mass-Luminosity Residuals ({tag})")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"mass_luminosity_residuals_{tag}.png", dpi=220)
    plt.close(fig)


def run_mass_bin_fits(df: pd.DataFrame) -> Dict[str, Any]:
    bins = [
        ("0.1_to_0.7", 0.1, 0.7, True, True),
        ("0.7_to_1.5", 0.7, 1.5, False, True),
        ("1.5_to_3.5", 1.5, 3.5, False, True),
    ]

    results: Dict[str, Any] = {}
    for name, lo, hi, lo_inc, hi_inc in bins:
        if lo_inc:
            mask_lo = df["mass"] >= lo
        else:
            mask_lo = df["mass"] > lo
        if hi_inc:
            mask_hi = df["mass"] <= hi
        else:
            mask_hi = df["mass"] < hi

        subset = df[mask_lo & mask_hi].copy()
        n = int(len(subset))

        bin_result: Dict[str, Any] = {
            "n": n,
            "mass_range": [lo, hi],
            "rule": f"{'[' if lo_inc else '('}{lo}, {hi}{']' if hi_inc else ')'}",
        }

        if n >= 200:
            fit, _ = fit_loglog_relation(subset, "x_logM", "y_logL")
            if fit is not None:
                bin_result.update(
                    {
                        "alpha": fit["slope"],
                        "intercept": fit["intercept"],
                        "r2": fit["r2"],
                        "rmse_logL": fit["rmse"],
                    }
                )
            else:
                bin_result["status"] = "skipped"
                bin_result["reason"] = "Not enough finite rows for fitting after cleanup."
        else:
            bin_result["status"] = "skipped"
            bin_result["reason"] = "Fewer than 200 points in mass bin."

        results[name] = bin_result

    return results


def stefan_boltzmann_consistency(
    df: pd.DataFrame,
    alpha_reference: Optional[float],
    out_dir: Path,
    tag: str,
    seed: int,
) -> Dict[str, Any]:
    work = df[["mass", "radius", "teff"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = work[(work["mass"] > 0.0) & (work["radius"] > 0.0) & (work["teff"] > 0.0)].copy()

    if len(work) < 2:
        return {
            "status": "skipped",
            "reason": "Insufficient data for Stefan-Boltzmann consistency check.",
        }

    work["logM"] = np.log10(work["mass"])
    work["logR"] = np.log10(work["radius"])
    work["logT"] = np.log10(work["teff"])

    x = work["logM"].to_numpy().reshape(-1, 1)

    reg_r = LinearRegression().fit(x, work["logR"].to_numpy())
    reg_t = LinearRegression().fit(x, work["logT"].to_numpy())

    beta = float(reg_r.coef_[0])
    gamma = float(reg_t.coef_[0])
    c = float(reg_r.intercept_)
    d = float(reg_t.intercept_)

    alpha_sb = float(2.0 * beta + 4.0 * gamma)
    alpha_diff = None if alpha_reference is None else float(alpha_reference - alpha_sb)

    # Plot logR vs logM.
    plot_df = sample_for_plot(work, seed=seed)
    x_line = np.linspace(float(work["logM"].min()), float(work["logM"].max()), 300)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot_df["logM"], plot_df["logR"], s=8, alpha=0.45, label="Stars")
    ax.plot(x_line, c + beta * x_line, color="black", linewidth=2.0, label="Fit")
    ax.set_xlabel("log10(M/Msun)")
    ax.set_ylabel("log10(R/Rsun)")
    ax.set_title(f"Stefan-Boltzmann Check: logR vs logM ({tag})")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"sb_logR_vs_logM_{tag}.png", dpi=220)
    plt.close(fig)

    # Plot logT vs logM.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot_df["logM"], plot_df["logT"], s=8, alpha=0.45, label="Stars")
    ax.plot(x_line, d + gamma * x_line, color="black", linewidth=2.0, label="Fit")
    ax.set_xlabel("log10(M/Msun)")
    ax.set_ylabel("log10(Teff [K])")
    ax.set_title(f"Stefan-Boltzmann Check: logT vs logM ({tag})")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"sb_logT_vs_logM_{tag}.png", dpi=220)
    plt.close(fig)

    return {
        "status": "ok",
        "n_used": int(len(work)),
        "beta": beta,
        "gamma": gamma,
        "c_intercept_logR": c,
        "d_intercept_logT": d,
        "alpha_SB": alpha_sb,
        "alpha_minus_alpha_SB": alpha_diff,
    }


def to_native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def build_narrative(results: Dict[str, Any]) -> List[str]:
    counts = results["counts"]
    reg = results["regression"]
    sb = results.get("stefan_boltzmann", {})
    model1 = results.get("model1", {})
    model2 = results.get("model2", {})

    lines: List[str] = []
    lines.append(
        f"Loaded {counts['N_raw_rows']} stellar rows; {counts['N_after_ms_filter']} remain after cleaning and main-sequence cuts."
    )
    lines.append(
        f"Main fit on full sample gives alpha = {reg['alpha_full'].get('alpha')} with R^2 = {reg['alpha_full'].get('r2')}."
    )
    lines.append(
        f"Held-out test fit gives alpha = {reg['alpha_test'].get('alpha')} and RMSE = {reg['alpha_test'].get('rmse_logL')} dex."
    )

    if reg.get("alpha_highconf", {}).get("status") == "ok":
        lines.append(
            f"Using Model 2 high-confidence stars (Pmax >= {results['config']['p_threshold']}) gives alpha = {reg['alpha_highconf'].get('alpha')}."
        )
    else:
        lines.append("High-confidence alpha refit was not available (Model 2 disabled or insufficient stars).")

    if model1.get("enabled") and model1.get("status") == "ok":
        lines.append(
            f"Model 1 (J-K -> Teff, degree 3) achieved RMSE = {model1.get('rmse_teff_K')} K and R^2 = {model1.get('r2_teff')}."
        )

    if model2.get("enabled") and model2.get("status") == "ok":
        lines.append(
            f"Model 2 HR classification reached accuracy = {model2.get('accuracy')} on the held-out test set."
        )

    if sb.get("status") == "ok":
        lines.append(
            f"Stefan-Boltzmann consistency gives alpha_SB = {sb.get('alpha_SB')} and alpha - alpha_SB = {sb.get('alpha_minus_alpha_SB')}."
        )

    # Keep narrative compact (5-10 lines).
    if len(lines) > 10:
        lines = lines[:10]
    while len(lines) < 5:
        lines.append("The resulting plots and tables support the robustness checks requested for the IA analysis.")

    return lines


def write_results_text(path: Path, results: Dict[str, Any], narrative: List[str]) -> None:
    counts = results["counts"]
    cfg = results["config"]
    m1 = results["model1"]
    m2 = results["model2"]
    reg = results["regression"]
    sb = results["stefan_boltzmann"]

    lines: List[str] = []
    lines.append("Kepler DR25 Mass-Luminosity Analysis Results")
    lines.append("=" * 44)
    lines.append(f"Timestamp: {results['run']['timestamp']}")
    lines.append(f"Input data: {results['run']['data_path']}")
    lines.append(f"Output folder: {results['run']['output_dir']}")
    lines.append("")

    lines.append("Counts")
    lines.append("-" * 20)
    lines.append(f"N_raw_rows: {counts['N_raw_rows']}")
    lines.append(f"N_after_dropna: {counts['N_after_dropna']}")
    lines.append(f"N_after_ms_filter: {counts['N_after_ms_filter']}")
    lines.append(f"N_train: {counts['N_train']}")
    lines.append(f"N_test: {counts['N_test']}")
    lines.append("")

    lines.append("Main-Sequence Filters")
    lines.append("-" * 20)
    for k, v in cfg["main_sequence_filter"].items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("Model 1 (Color -> Teff)")
    lines.append("-" * 20)
    lines.append(f"enabled: {m1.get('enabled', False)}")
    lines.append(f"status: {m1.get('status', 'disabled')}")
    if m1.get("status") == "ok":
        coeff = m1["coefficients"]
        lines.append(
            "coefficients: "
            f"a0={coeff['a0']}, a1={coeff['a1']}, a2={coeff['a2']}, a3={coeff['a3']}"
        )
        lines.append(f"RMSE (K): {m1['rmse_teff_K']}")
        lines.append(f"R^2: {m1['r2_teff']}")
    elif m1.get("enabled"):
        lines.append(f"reason: {m1.get('reason', 'n/a')}")
    lines.append("")

    lines.append("Model 2 (HR Probability Layers)")
    lines.append("-" * 20)
    lines.append(f"enabled: {m2.get('enabled', False)}")
    lines.append(f"status: {m2.get('status', 'disabled')}")
    if m2.get("status") == "ok":
        lines.append(f"grid_size: {m2['grid_size']}")
        lines.append(f"smooth_sigma: {m2['smooth_sigma']}")
        lines.append(f"accuracy: {m2['accuracy']}")
        lines.append(f"active_classes: {m2['active_classes']}")
        lines.append(f"high_conf_count: {m2['high_conf_count']}")
    elif m2.get("enabled"):
        lines.append(f"reason: {m2.get('reason', 'n/a')}")
    lines.append("")

    lines.append("Mass-Luminosity Regression")
    lines.append("-" * 20)
    lines.append(f"alpha_full: {reg['alpha_full']}")
    lines.append(f"alpha_test: {reg['alpha_test']}")
    lines.append(f"alpha_highconf: {reg['alpha_highconf']}")
    lines.append("alpha_bins:")
    for name, item in reg["alpha_bins"].items():
        lines.append(f"  {name}: {item}")
    lines.append("")

    lines.append("Stefan-Boltzmann Consistency")
    lines.append("-" * 20)
    lines.append(str(sb))
    lines.append("")

    lines.append("Narrative Summary")
    lines.append("-" * 20)
    for line in narrative:
        lines.append(line)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if args.grid_size < 20:
        raise ValueError("--grid_size must be >= 20")
    if args.smooth_sigma <= 0:
        raise ValueError("--smooth_sigma must be > 0")
    if not (0.0 <= args.p_threshold <= 1.0):
        raise ValueError("--p_threshold must be in [0, 1]")

    model1_enabled = not args.disable_model1
    model2_enabled = not args.disable_model2

    checkpoint("Creating output directory.")
    run_dir = make_unique_run_dir(Path(args.out))

    checkpoint("Detecting delimiter from first non-empty header line.")
    sep = detect_delimiter(data_path)
    checkpoint(f"Detected delimiter: {repr(sep)}")

    checkpoint("Loading catalog columns and coercing to numeric.")
    df_raw, col_map = load_catalog(data_path, sep)
    n_raw_rows = int(len(df_raw))

    checkpoint("Applying mandatory NaN filtering.")
    dropna_cols = ["teff", "mass", "radius", "logg"]
    if model1_enabled:
        dropna_cols.extend(["jmag", "kmag"])
    df_clean = df_raw.dropna(subset=dropna_cols).copy()
    n_after_dropna = int(len(df_clean))

    checkpoint("Applying main-sequence filters.")
    df_ms = apply_main_sequence_filter(df_clean, mass_max=3.8)
    df_ms = add_derived_columns(df_ms)
    n_after_ms = int(len(df_ms))

    if n_after_ms < 10:
        raise RuntimeError("Too few rows remain after filtering; cannot continue analysis.")

    checkpoint("Saving cleaned main-sequence dataset.")
    cleaned_path = run_dir / "cleaned_main_sequence.csv"
    df_ms.to_csv(cleaned_path, index=False)

    checkpoint("Performing train/test split.")
    train_df, test_df = train_test_split(
        df_ms,
        test_size=0.2,
        random_state=args.seed,
        shuffle=True,
    )

    n_train = int(len(train_df))
    n_test = int(len(test_df))

    if n_train < 5 or n_test < 5:
        raise RuntimeError("Train/test split produced too few rows in one split.")

    model1_result: Dict[str, Any] = {"enabled": False, "status": "disabled"}
    if model1_enabled:
        checkpoint("Training and evaluating Model 1 (J-K -> Teff).")
        model1_result = fit_degree3_color_teff(train_df, test_df, run_dir, seed=args.seed)

    model2_result: Dict[str, Any] = {"enabled": False, "status": "disabled"}
    model2_pred_df: Optional[pd.DataFrame] = None
    high_conf_df: Optional[pd.DataFrame] = None

    if model2_enabled:
        checkpoint("Training and validating Model 2 (probability layers on HR diagram).")
        m2_result, pred_df, _active_classes, high_conf = run_model2(
            train_df=train_df,
            test_df=test_df,
            out_dir=run_dir,
            grid_size=args.grid_size,
            smooth_sigma=args.smooth_sigma,
            p_threshold=args.p_threshold,
            seed=args.seed,
        )
        m2_result["status"] = "ok"
        model2_result = m2_result
        model2_pred_df = pred_df
        high_conf_df = high_conf

    # Mass-luminosity regressions.
    checkpoint("Fitting mass-luminosity regressions.")

    alpha_full_fit, alpha_full_df = fit_loglog_relation(df_ms, "x_logM", "y_logL")
    alpha_test_fit, alpha_test_df = fit_loglog_relation(test_df, "x_logM", "y_logL")

    if alpha_full_fit is None or alpha_full_df is None:
        raise RuntimeError("Could not fit alpha on full filtered sample.")
    if alpha_test_fit is None or alpha_test_df is None:
        raise RuntimeError("Could not fit alpha on test sample.")

    alpha_highconf_fit: Optional[Dict[str, Any]] = None
    alpha_highconf_df: Optional[pd.DataFrame] = None

    if model2_enabled and high_conf_df is not None and len(high_conf_df) >= 2:
        alpha_highconf_fit, alpha_highconf_df = fit_loglog_relation(high_conf_df, "x_logM", "y_logL")

    plot_mass_luminosity_fit(alpha_full_df, alpha_full_fit, run_dir, tag="full", seed=args.seed)
    plot_mass_luminosity_fit(alpha_test_df, alpha_test_fit, run_dir, tag="test", seed=args.seed)
    if alpha_highconf_fit is not None and alpha_highconf_df is not None:
        plot_mass_luminosity_fit(alpha_highconf_df, alpha_highconf_fit, run_dir, tag="highconf", seed=args.seed)

    # Mass-bin fits (default on).
    alpha_bins = run_mass_bin_fits(df_ms)

    # Stefan-Boltzmann consistency check using the same subset as alpha_full (full sample fit).
    checkpoint("Running Stefan-Boltzmann consistency check.")
    sb_result = stefan_boltzmann_consistency(
        df=df_ms,
        alpha_reference=alpha_full_fit["slope"],
        out_dir=run_dir,
        tag="full",
        seed=args.seed,
    )

    regression_summary: Dict[str, Any] = {
        "alpha_full": {
            "status": "ok",
            "alpha": alpha_full_fit["slope"],
            "intercept": alpha_full_fit["intercept"],
            "r2": alpha_full_fit["r2"],
            "rmse_logL": alpha_full_fit["rmse"],
        },
        "alpha_test": {
            "status": "ok",
            "alpha": alpha_test_fit["slope"],
            "intercept": alpha_test_fit["intercept"],
            "r2": alpha_test_fit["r2"],
            "rmse_logL": alpha_test_fit["rmse"],
        },
        "alpha_highconf": {
            "status": "ok",
            "alpha": alpha_highconf_fit["slope"],
            "intercept": alpha_highconf_fit["intercept"],
            "r2": alpha_highconf_fit["r2"],
            "rmse_logL": alpha_highconf_fit["rmse"],
        }
        if alpha_highconf_fit is not None
        else {
            "status": "skipped",
            "reason": "Model 2 disabled or insufficient high-confidence stars.",
        },
        "alpha_bins": alpha_bins,
    }

    results: Dict[str, Any] = {
        "run": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "data_path": str(data_path.resolve()),
            "output_dir": str(run_dir.resolve()),
            "delimiter_detected": sep,
            "column_mapping": col_map,
        },
        "config": {
            "seed": int(args.seed),
            "test_size": 0.2,
            "p_threshold": float(args.p_threshold),
            "grid_size": int(args.grid_size),
            "smooth_sigma": float(args.smooth_sigma),
            "model1_enabled": bool(model1_enabled),
            "model2_enabled": bool(model2_enabled),
            "main_sequence_filter": {
                "logg_min": 4.1,
                "radius_max": 2.0,
                "teff_min": 3000.0,
                "teff_max": 10000.0,
                "mass_min_exclusive": 0.0,
                "mass_max": 3.8,
            },
        },
        "counts": {
            "N_raw_rows": n_raw_rows,
            "N_after_dropna": n_after_dropna,
            "N_after_ms_filter": n_after_ms,
            "N_train": n_train,
            "N_test": n_test,
        },
        "model1": model1_result,
        "model2": model2_result,
        "regression": regression_summary,
        "stefan_boltzmann": sb_result,
        "artifacts": {
            "cleaned_dataset_csv": str(cleaned_path.resolve()),
            "results_json": str((run_dir / "results.json").resolve()),
            "results_txt": str((run_dir / "results.txt").resolve()),
            "model2_confusion_matrix_csv": str((run_dir / "model2_confusion_matrix.csv").resolve())
            if model2_enabled
            else None,
            "model2_confusion_matrix_png": str((run_dir / "model2_confusion_matrix.png").resolve())
            if model2_enabled
            else None,
        },
    }

    if model2_enabled and model2_pred_df is not None:
        # Keep compact metrics in JSON; detailed per-star predictions are in CSV artifact.
        results["model2"]["n_predictions_saved"] = int(len(model2_pred_df))

    narrative = build_narrative(results)

    checkpoint("Writing results summary files.")
    with (run_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(to_native(results), f, indent=2)

    write_results_text(run_dir / "results.txt", to_native(results), narrative)

    checkpoint("Analysis complete.")
    checkpoint(f"Outputs written to: {run_dir}")


def main() -> None:
    try:
        run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
