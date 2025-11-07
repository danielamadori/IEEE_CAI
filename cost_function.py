from random import random
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from scipy.integrate import quad

EPS = np.finfo(float).tiny
MIN_SIGMA = 1e-10  # Minimum meaningful sigma value to prevent numerical issues
HALF = 0.5


def cal_sigmas(X_train, X_test, feature_names, test_ids=None):
	"""
	Calculate sigma_plus and sigma_minus for each feature in X_test based on X_train.
	If test_ids is provided, the outer dictionary will use those IDs instead of numeric indices.

	Parameters
	----------
	X_train : array-like
		Training samples (n_train, n_features)
	X_test : array-like
		Test samples (n_test, n_features)
	feature_names : list of str
		Names of the features (columns)
	test_ids : list, optional
		Custom identifiers for each row of X_test (e.g., keys of tests_sample).
		If None, the function defaults to numeric indices 0..n_test-1.

	Returns
	-------
	sigmas_all : dict
		Dictionary keyed by either numeric indices or test_ids.
		Each entry maps feature_name -> dict with:
			sigma_plus, sigma_minus, ratio_above_mean, ratio_below_mean
	"""

	sigmas_all = {}
	X_train_df = pd.DataFrame(X_train, columns=feature_names)
	X_test_df = pd.DataFrame(X_test, columns=feature_names)

	# Default behaviour preserved if test_ids not given
	if test_ids is None:
		test_ids = list(range(len(X_test_df)))

	# Sanity check
	if len(test_ids) != len(X_test_df):
		raise ValueError(
			f"Length mismatch: len(test_ids)={len(test_ids)} vs len(X_test)={len(X_test_df)}"
		)

	# Main computation
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

def cost_function(sample: Dict[str, float] = None,  icf: Dict[str, Tuple[float, float]] = None, sigmas: Dict[str, Dict[str, dict]] = None, verbose: bool = False) -> float:
	"""
	Calculate cost function based on split Gaussian distributions.

	Parameters
	----------
	sample : dict
		Sample values for each feature
	icf : dict
		Interval for each feature (min, max)
	sigmas : dict
		Sigma values and ratios for each feature
	verbose : bool
		Print debug information

	Returns
	-------
	float
		Total cost across all features
	"""
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

		# Skip if key not in sigmas (feature might have been skipped in cal_sigmas)
		if key not in sigmas:
			if verbose:
				print(f"  Warning: key {key} not in sigmas, skipping")
			continue

		sigma_pos = sigmas[key]['sigma_plus']
		sigma_neg = sigmas[key]['sigma_minus']
		percent_above = sigmas[key]['ratio_above_mean']
		percent_below = sigmas[key]['ratio_below_mean']
		interval_min, interval_max = icf[key]

		if verbose:
			print(f"  Interval: [{interval_min:.4f}, {interval_max:.4f}]")
			print(f"  Sigmas: sigma_pos={sigma_pos:.4f}, sigma_neg={sigma_neg:.4f}")
			print(f"  Percentages: above={percent_above:.4f}, below={percent_below:.4f}")

		# Validate percentages
		if not np.isclose(percent_above + percent_below, 1.0, rtol=1e-5):
			if verbose:
				print(f"  Warning: percentages for key {key} don't sum to 1.0: sum={percent_above + percent_below}")
			# Normalize percentages if close enough
			total_percent = percent_above + percent_below
			if total_percent > 0:
				percent_above /= total_percent
				percent_below /= total_percent
			else:
				# If both are zero, skip this feature
				if verbose:
					print(f"  Skipping feature {key} due to zero percentages")
				continue

		# Protect against zero or negative sigmas
		sigma_neg = max(abs(sigma_neg), MIN_SIGMA)
		sigma_pos = max(abs(sigma_pos), MIN_SIGMA)

		# If both sigmas are at minimum threshold, it means no variance - skip this feature
		if sigma_neg <= MIN_SIGMA and sigma_pos <= MIN_SIGMA:
			if verbose:
				print(f"  Skipping feature {key} due to zero variance (both sigmas at minimum)")
			continue

		# Calculate normalization constants for split Gaussian
		# These should be ~0.5 for a standard normal distribution
		try:
			low_norm, _ = quad(
				lambda x: (1 / (sigma_neg * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_neg) ** 2),
				-np.inf, 0,
				limit=50
			)
			above_norm, _ = quad(
				lambda x: (1 / (sigma_pos * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_pos) ** 2),
				0, np.inf,
				limit=50
			)
		except Exception as e:
			if verbose:
				print(f"  Warning: integration error for key {key}: {e}")
			continue

		# Protect against zero normalization
		low_norm = max(low_norm, MIN_SIGMA)
		above_norm = max(above_norm, MIN_SIGMA)

		# Shift interval to be relative to sample value
		interval_min_shifted = interval_min - sample[key]
		interval_max_shifted = interval_max - sample[key]

		if verbose:
			print(f"  Interval shifted: [{interval_min_shifted:.4f}, {interval_max_shifted:.4f}]")

		# Calculate scaling factors to ensure PDF integrates to proper percentages
		scale_below = percent_below / low_norm
		scale_above = percent_above / above_norm

		# Protect against infinite or NaN scaling
		if not np.isfinite(scale_below):
			scale_below = 0.0
		if not np.isfinite(scale_above):
			scale_above = 0.0

		def split_pdf(x, sigma_neg, sigma_pos, scale_below, scale_above):
			"""Split Gaussian PDF: different sigmas for x < 0 and x >= 0"""
			x = np.asarray(x)
			below = scale_below * (1 / (sigma_neg * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_neg) ** 2) * (x < 0)
			above = scale_above * (1 / (sigma_pos * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_pos) ** 2) * (x >= 0)
			return below + above

		# Integrate over the interval to get cost contribution
		try:
			area_interval, error = quad(
				lambda x: split_pdf(x, sigma_neg, sigma_pos, scale_below, scale_above),
				interval_min_shifted,
				interval_max_shifted,
				limit=50
			)
			area = abs(area_interval)

			# Sanity check: area should be between 0 and 1
			if not (0 <= area <= 1.0 + 1e-6):
				if verbose:
					print(f"  Warning: area {area:.6f} outside [0,1] for key {key}, clamping")
				area = np.clip(area, 0.0, 1.0)

		except Exception as e:
			if verbose:
				print(f"  Warning: integration error for interval in key {key}: {e}")
			area = 0.0

		# Cost is area under the curve in the interval
		cost += area

		if verbose:
			print(f"  Area under curve in interval: {area:.4f}, Cost total: {cost:.4f}")

			# Plot the curve and highlight the interval (occasionally)
			if random() < 0.3:  # Plot only 30% of features for brevity
				try:
					area_below_actual, _ = quad(
						lambda x: split_pdf(x, sigma_neg, sigma_pos, scale_below, scale_above),
						-np.inf, 0, limit=50
					)
					area_above_actual, _ = quad(
						lambda x: split_pdf(x, sigma_neg, sigma_pos, scale_below, scale_above),
						0, np.inf, limit=50
					)
					print(f"  Area left (<0): {area_below_actual:.4f}, Area right (>=0): {area_above_actual:.4f}, Area sum: {area_below_actual + area_above_actual:.4f}")

					x_vals = np.linspace(-5, 5, 400)
					y_vals = split_pdf(x_vals, sigma_neg, sigma_pos, scale_below, scale_above)
					plt.figure(figsize=(8, 4))
					plt.plot(x_vals, y_vals)
					plt.fill_between(x_vals, 0, y_vals, where=(x_vals < 0), color='red', alpha=0.3, label=f'Below_Area={area_below_actual:.2f}')
					plt.fill_between(x_vals, 0, y_vals, where=(x_vals >= 0), color='green', alpha=0.3, label=f'Above_Area={area_above_actual:.2f}')

					# Handle infinite intervals for plotting
					plot_min = max(-5, interval_min_shifted if not np.isinf(interval_min_shifted) else -5)
					plot_max = min(5, interval_max_shifted if not np.isinf(interval_max_shifted) else 5)

					plt.axvspan(plot_min, plot_max, color='black', alpha=0.4, label='Interval')

					# Create title with proper inf handling
					interval_str = f"[{interval_min_shifted:.4f}, {interval_max_shifted:.4f}]"
					if np.isinf(interval_min_shifted):
						interval_str = f"[-∞, {interval_max_shifted:.4f}]"
					if np.isinf(interval_max_shifted):
						interval_str = f"[{interval_min_shifted:.4f}, ∞]"
					if np.isinf(interval_min_shifted) and np.isinf(interval_max_shifted):
						interval_str = "[-∞, ∞]"

					plt.title(f'Feature: {key} | Cost contribution: {area:.4f} | Interval: {interval_str} | Sigmas: +{sigma_pos:.2f}, -{sigma_neg:.2f}')
					plt.axvline(0, color='black', linestyle='--')
					plt.legend()
					plt.savefig(f'fig/feature_{key}_cost_plot.png')
					plt.close()
				except Exception as e:
					if verbose:
						print(f"  Warning: plotting error for key {key}: {e}")

	return cost
