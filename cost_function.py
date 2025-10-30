from random import random
import numpy as np
from typing import Dict, Tuple
import pandas as pd
from scipy.integrate import quad
import matplotlib.pyplot as plt

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
	import numpy as np
	import pandas as pd

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

			n_above = float(delta_pos.shape[0])
			n_below = float(delta_neg.shape[0])
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
		sigma_pos = sigmas[key]['sigma_plus']
		sigma_neg = sigmas[key]['sigma_minus']
		percent_above = sigmas[key]['ratio_above_mean']
		percent_below = sigmas[key]['ratio_below_mean']
		interval_min, interval_max = icf[key]

		if verbose:
			print(f"  Interval: [{interval_min:.4f}, {interval_max:.4f}]")
			print(f"  Sigmas: sigma_pos={sigma_pos:.4f}, sigma_neg={sigma_neg:.4f}")
			print(f"  Percentages: above={percent_above:.4f}, below={percent_below:.4f}")
		if not np.isclose(percent_above + percent_below, 1.0):
			print(f"Error in percentages for key {key}: sum={percent_above + percent_below}")
			break
		low_norm, _ = quad(lambda x: (1 / (sigma_neg * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_neg) ** 2), -np.inf, 0)
		above_norm, _ = quad(lambda x: (1 / (sigma_pos * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_pos) ** 2), 0, np.inf)
		interval_min = interval_min - sample[key]
		interval_max = interval_max - sample[key]
		if verbose:
			print(f"  Interval repose: [{interval_min:.4f}, {interval_max:.4f}]")

		# Integrate the Gaussian over the interval [interval_min, interval_max]
		scale_below = percent_below / low_norm
		scale_above = percent_above / above_norm

		def split_pdf(x, sigma_neg, sigma_pos, scale_below, scale_above):
			x = np.array(x)
			below = scale_below * (1 / (sigma_neg * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_neg) ** 2) * (x < 0)
			above = scale_above * (1 / (sigma_pos * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_pos) ** 2) * (x >= 0)
			return below + above
		area_interval, _ = quad(lambda x: split_pdf(x, sigma_neg, sigma_pos, scale_below, scale_above), interval_min, interval_max)
		area = abs(area_interval) 
		# Cost is area under the curve in the interval
		cost += area
		if verbose:
			print(f"Area under split Gaussian from {interval_min:.4f} to {interval_max:.4f}: {area:.4f}")
		if verbose:
			print(f"  Area under curve in interval: {area:.4f}, Cost total: {cost:.4f}")
			# Plot the curve and highlight the interval
			if random() < 0.3:  # Plot only 30% of features for brevity
				area_below_actual, _ = quad(lambda x: split_pdf(x, sigma_neg, sigma_pos, scale_below, scale_above), -np.inf, 0)
				area_above_actual, _ = quad(lambda x: split_pdf(x, sigma_neg, sigma_pos, scale_below, scale_above), 0, np.inf)
				print(f"Area left (<0): {area_below_actual:.4f}, Area right (>=0): {area_above_actual:.4f}, Area sum: {area_below_actual + area_above_actual:.4f}")
				x_vals = np.linspace(-5, 5, 400)
				y_vals = split_pdf(x_vals, sigma_neg, sigma_pos, scale_below, scale_above)
				plt.figure(figsize=(8, 4))
				plt.plot(x_vals, y_vals)
				plt.fill_between(x_vals, 0, y_vals, where=(x_vals < 0), color='red', alpha=0.3, label= f'Below_Area={area_below_actual:.2f}')
				plt.fill_between(x_vals, 0, y_vals, where=(x_vals >= 0), color='green', alpha=0.3, label=f'Above_Area={area_above_actual:.2f}')
				
				# Handle infinite intervals for plotting
				plot_min = max(-5, interval_min if not np.isinf(interval_min) else -5)
				plot_max = min(5, interval_max if not np.isinf(interval_max) else 5)
				
				plt.axvspan(plot_min, plot_max, color='black', alpha=0.4, label='Interval')
				
				# Create title with proper inf handling
				interval_str = f"[{interval_min:.4f}, {interval_max:.4f}]"
				if np.isinf(interval_min):
					interval_str = f"[-∞, {interval_max:.4f}]"
				if np.isinf(interval_max):
					interval_str = f"[{interval_min:.4f}, ∞]"
				if np.isinf(interval_min) and np.isinf(interval_max):
					interval_str = "[-∞, ∞]"
					
				plt.title(f'Feature: {key} | Cost contribution: {area:.4f} | Interval: {interval_str} | Sigmas: +{sigma_pos:.2f}, -{sigma_neg:.2f}')
				plt.axvline(0, color='black', linestyle='--' ) #, label='Center (0)')
				plt.legend()
				# plt.show()
				plt.savefig(f'fig/feature_{key}_cost_plot.png')
				plt.close()


		
		return cost