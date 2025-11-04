from math import erf
from random import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from scipy.integrate import quad

EPS = np.finfo(float).tiny
HALF = 0.5
SQRT_TWO = np.sqrt(2.0)
SQRT_TWO_PI = np.sqrt(2 * np.pi)
LOG_MAX_FLOAT = np.log(np.finfo(float).max)
SCALED_CLIP = np.sqrt(2 * LOG_MAX_FLOAT)


def _safe_gaussian_pdf(x, sigma):
    x_arr = np.asarray(x, dtype=np.float64)
    sigma = max(float(sigma), EPS)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = x_arr / sigma
        exponent = -0.5 * scaled * scaled
        pdf = np.exp(exponent) / (sigma * SQRT_TWO_PI)
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    return float(pdf) if np.ndim(pdf) == 0 else pdf


def _split_pdf(x, sigma_neg, sigma_pos, scale_below, scale_above):
    x_arr = np.asarray(x, dtype=np.float64)
    below = scale_below * _safe_gaussian_pdf(x_arr, sigma_neg) * (x_arr < 0)
    above = scale_above * _safe_gaussian_pdf(x_arr, sigma_pos) * (x_arr >= 0)
    total = below + above
    total_arr = np.asarray(total)
    return float(total_arr) if np.ndim(total_arr) == 0 else total_arr


def _validate_split_pdf_normalization(sigma_neg, sigma_pos, scale_below, scale_above):
    neg_integrand = (
        lambda t: scale_below
        * _safe_gaussian_pdf(-t * sigma_neg, sigma_neg)
        * sigma_neg
    )
    pos_integrand = (
        lambda t: scale_above
        * _safe_gaussian_pdf(t * sigma_pos, sigma_pos)
        * sigma_pos
    )

    below_area, _ = quad(neg_integrand, 0.0, SCALED_CLIP, limit=200, epsabs=1e-12, epsrel=1e-12)
    above_area, _ = quad(pos_integrand, 0.0, SCALED_CLIP, limit=200, epsabs=1e-12, epsrel=1e-12)
    return below_area + above_area


def _normal_cdf(x: float, sigma: float) -> float:
    if np.isneginf(x):
        return 0.0
    if np.isposinf(x):
        return 1.0

    sigma = max(abs(sigma), EPS)
    scaled = float(x) / (sigma * SQRT_TWO)
    return 0.5 * (1.0 + erf(scaled))


def _split_interval_area(
    interval_min: float,
    interval_max: float,
    sigma_neg: float,
    sigma_pos: float,
    scale_below: float,
    scale_above: float,
) -> float:
    area = 0.0

    if interval_min < 0:
        below_upper = min(interval_max, 0.0)
        cdf_upper = _normal_cdf(below_upper, sigma_neg)
        cdf_lower = _normal_cdf(interval_min, sigma_neg)
        area += scale_below * max(cdf_upper - cdf_lower, 0.0)

    if interval_max > 0:
        above_lower = max(interval_min, 0.0)
        cdf_upper = _normal_cdf(interval_max, sigma_pos)
        cdf_lower = _normal_cdf(above_lower, sigma_pos)
        area += scale_above * max(cdf_upper - cdf_lower, 0.0)

    return area


def cal_sigmas(X_train, X_test, feature_names, test_ids=None):
    """Calculate sigma_plus and sigma_minus for each feature in ``X_test`` based on ``X_train``.

    If ``test_ids`` is provided, the outer dictionary will use those IDs instead of numeric indices.
    """

    sigmas_all = {}
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    if test_ids is None:
        test_ids = list(range(len(X_test_df)))

    if len(test_ids) != len(X_test_df):
        raise ValueError(
            f"Length mismatch: len(test_ids)={len(test_ids)} vs len(X_test)={len(X_test_df)}"
        )

    for sample_id, (_, row) in zip(test_ids, X_test_df.iterrows()):
        sigmas_all[sample_id] = {}
        for feature in feature_names:
            tmp = np.array(X_train_df[feature]) - row[feature]
            delta_pos = tmp[tmp >= 0]
            delta_neg = np.abs(tmp[tmp < 0])

            n_above = np.float64(delta_pos.shape[0])
            n_below = np.float64(delta_neg.shape[0])
            n = n_above + n_below

            if n == 0:
                continue

            sum_pos = np.sum(delta_pos ** 2)
            sum_neg = np.sum(delta_neg ** 2)

            sigma_plus = float(np.sqrt(sum_pos / n_above)) if n_above > 0 else 0.0
            sigma_minus = float(np.sqrt(sum_neg / n_below)) if n_below > 0 else 0.0

            sigmas_all[sample_id][feature] = {
                "sigma_plus": sigma_plus,
                "sigma_minus": sigma_minus,
                "ratio_above_mean": float(n_above / n) if n > 0 else 0.0,
                "ratio_below_mean": float(n_below / n) if n > 0 else 0.0,
            }

    return sigmas_all


def cost_function(
    sample: Dict[str, float] = None,
    icf: Dict[str, Tuple[float, float]] = None,
    sigmas: Dict[str, Dict[str, dict]] = None,
    verbose: bool = False,
) -> float:
    if sigmas is None:
        raise ValueError("Sigmas must be provided")
    if icf is None:
        raise ValueError("ICF must be provided")
    if sample is None:
        raise ValueError("Sample must be provided")

    cost = 0.0
    for key in icf.keys():
        if verbose:
            print(f"Processing key: {key}")

        sigma_pos = sigmas[key]["sigma_plus"]
        sigma_neg = sigmas[key]["sigma_minus"]
        percent_above = sigmas[key]["ratio_above_mean"]
        percent_below = sigmas[key]["ratio_below_mean"]
        interval_min, interval_max = icf[key]

        if verbose:
            print(f"  Interval: [{interval_min:.4f}, {interval_max:.4f}]")
            print(f"  Sigmas: sigma_pos={sigma_pos:.4f}, sigma_neg={sigma_neg:.4f}")
            print(f"  Percentages: above={percent_above:.4f}, below={percent_below:.4f}")

        if not np.isclose(percent_above + percent_below, 1.0):
            print(f"Error in percentages for key {key}: sum={percent_above + percent_below}")
            break

        sigma_neg = max(abs(sigma_neg), EPS)
        sigma_pos = max(abs(sigma_pos), EPS)

        low_norm = HALF
        above_norm = HALF

        interval_min = interval_min - sample[key]
        interval_max = interval_max - sample[key]

        if verbose:
            print(f"  Interval repose: [{interval_min:.4f}, {interval_max:.4f}]")

        scale_below = percent_below / low_norm
        scale_above = percent_above / above_norm

        total_area = _validate_split_pdf_normalization(
            sigma_neg, sigma_pos, scale_below, scale_above
        )
        expected_area = scale_below * low_norm + scale_above * above_norm
        if not np.isclose(total_area, expected_area, rtol=1e-6, atol=1e-8):
            raise RuntimeError(
                "Split Gaussian PDF normalization check failed: "
                f"expected {expected_area:.6f}, got {total_area:.6f}"
            )

        area = _split_interval_area(
            interval_min,
            interval_max,
            sigma_neg,
            sigma_pos,
            scale_below,
            scale_above,
        )
        cost += area

        if verbose:
            print(
                f"Area under split Gaussian from {interval_min:.4f} to {interval_max:.4f}: {area:.4f}"
            )
            print(f"  Area under curve in interval: {area:.4f}, Cost total: {cost:.4f}")

            if random() < 0.3:
                area_below_actual = _split_interval_area(
                    -np.inf,
                    0.0,
                    sigma_neg,
                    sigma_pos,
                    scale_below,
                    scale_above,
                )
                area_above_actual = _split_interval_area(
                    0.0,
                    np.inf,
                    sigma_neg,
                    sigma_pos,
                    scale_below,
                    scale_above,
                )
                print(
                    "Area left (<0): "
                    f"{area_below_actual:.4f}, Area right (>=0): {area_above_actual:.4f}, "
                    f"Area sum: {area_below_actual + area_above_actual:.4f}"
                )

                x_vals = np.linspace(-5, 5, 400)
                y_vals = _split_pdf(x_vals, sigma_neg, sigma_pos, scale_below, scale_above)
                plt.figure(figsize=(8, 4))
                plt.plot(x_vals, y_vals)
                plt.fill_between(
                    x_vals,
                    0,
                    y_vals,
                    where=(x_vals < 0),
                    color="red",
                    alpha=0.3,
                    label=f"Below_Area={area_below_actual:.2f}",
                )
                plt.fill_between(
                    x_vals,
                    0,
                    y_vals,
                    where=(x_vals >= 0),
                    color="green",
                    alpha=0.3,
                    label=f"Above_Area={area_above_actual:.2f}",
                )

                plot_min = max(-5, interval_min if not np.isinf(interval_min) else -5)
                plot_max = min(5, interval_max if not np.isinf(interval_max) else 5)
                plt.axvspan(plot_min, plot_max, color="black", alpha=0.4, label="Interval")

                interval_str = f"[{interval_min:.4f}, {interval_max:.4f}]"
                if np.isinf(interval_min):
                    interval_str = f"[-∞, {interval_max:.4f}]"
                if np.isinf(interval_max):
                    interval_str = f"[{interval_min:.4f}, ∞]"
                if np.isinf(interval_min) and np.isinf(interval_max):
                    interval_str = "[-∞, ∞]"

                plt.title(
                    f"Feature: {key} | Cost contribution: {area:.4f} | Interval: {interval_str} | "
                    f"Sigmas:+{sigma_pos:.2f}, -{sigma_neg:.2f}"
                )
                plt.axvline(0, color="black", linestyle="--")
                plt.legend()
                plt.savefig(f"fig/feature_{key}_cost_plot.png")
                plt.close()

    return cost
