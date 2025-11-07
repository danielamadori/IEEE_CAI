from redis_helpers.forest import retrieve_forest
from redis_helpers.endpoints import retrieve_monotonic_dict
from redis_helpers.samples import retrieve_sample
from redis_helpers.icf import key_dominates
from icf_eu_encoding import icf_to_bitmap_mask, bitmap_mask_to_string
from typing import List, Dict, Any, Tuple
import json

def count_containing_keys(sample_bitmap: str,
                          key_list: List[str]) -> int:
	"""
	Count how many keys in the list contain (dominate) the sample

	Args:
		sample_bitmap: Bitmap string of the sample's ICF
		key_list: List of bitmap strings to check

	Returns:
		Count of keys that contain this sample
	"""
	count = 0
	for key in key_list:
		if key_dominates(key, sample_bitmap, reverse=False):
			count += 1
	return count


def find_containing_reasons(sample_bitmap: str,
                            maximal_reasons: List[str]) -> List[str]:
	"""
	Find all maximal reasons that contain (dominate) the sample

	Args:
		sample_bitmap: Bitmap string of the sample's ICF
		maximal_reasons: List of maximal reason bitmaps

	Returns:
		List of maximal reasons that contain this sample
	"""
	containing_reasons = []

	for reason in maximal_reasons:
		# A reason contains a sample if the reason dominates the sample
		# i.e., for all positions where the reason has '0', the sample also has '0'
		if key_dominates(reason, sample_bitmap, reverse=False):
			containing_reasons.append(reason)

	return containing_reasons


def analyze_samples_and_maximal_reasons(db, maximal_reasons: List[str], verbose: bool = True) -> Dict[str, Any]:
	"""
	For each sample in DATA, find which maximal reasons, R keys, and CAN keys contain it

	Args:
		db: Dictionary containing data, reasons and candidates
		maximal_reasons: List of maximal reason bitmaps
		verbose: Whether to print progress

	Returns:
		Dictionary mapping sample keys to their containing reasons
	"""

	# Load necessary data
	if verbose:
		print("Loading forest and endpoints...")

	rf_data = retrieve_forest(db["data"]['RF'])
	if not rf_data:
		raise Exception("Could not load RF from DATA")

	eu_data = retrieve_monotonic_dict(db["data"]['EU'])
	if not eu_data:
		raise Exception("Could not load EU from DATA")

	if verbose:
		print(f"Loaded forest with {len(rf_data)} trees")
		print(f"Loaded EU with {len(eu_data)} features")

	# Get all keys from R and CAN
	if verbose:
		print(f"\nLoading all keys from R and CAN databases...")

	all_r_keys = list(db["reasons"].keys())
	all_can_keys = list(db.get("candidates", {}).keys())

	if verbose:
		print(f"Loaded {len(all_r_keys)} keys from R")
		print(f"Loaded {len(all_can_keys)} keys from CAN")

	# Get all samples from data
	prefix = "sample_"
	suffix = "_meta"
	sample_keys = [k[len(prefix):-len(suffix)] for k in db["data"].keys()
	               if isinstance(k, str) and k.startswith(prefix) and k.endswith(suffix)]

	if verbose:
		print(f"\nFound {len(sample_keys)} samples in DATA")
		print(f"Have {len(maximal_reasons)} maximal reasons to check")
		print(f"{ '='*80 }")

	if len(sample_keys) == 0:
		print("No samples found in DATA")
		return {}

	# Analyze each sample
	results = {}
	samples_with_maximal = 0
	samples_with_r = 0
	samples_with_can = 0

	for i, sample_key in enumerate(sample_keys):
		if verbose and (i % 10 == 0 or i == len(sample_keys) - 1):
			print(f"Progress: {i+1}/{len(sample_keys)} samples analyzed...")

		try:
			# Load sample
			sample_dict = retrieve_sample(db['data'], sample_key)
			if not sample_dict:
				if verbose:
					print(f"Could not load sample: {sample_key}")
				continue

			# Load metadata if available
			metadata = None
			try:
				meta_key = f"{sample_key}_meta"
				if meta_key in db['data']:
					metadata = db['data'][meta_key].get('value_json', None)
			except:
				pass

			# Extract ICF and convert to bitmap
			sample_icf = rf_data.extract_icf(sample_dict)
			sample_bitmap = bitmap_mask_to_string(icf_to_bitmap_mask(sample_icf, eu_data))

			# Count containing keys in each database
			num_maximal = count_containing_keys(sample_bitmap, maximal_reasons)
			num_r = count_containing_keys(sample_bitmap, all_r_keys)
			num_can = count_containing_keys(sample_bitmap, all_can_keys)

			# Find actual containing maximal reasons (for detailed analysis)
			containing_reasons = find_containing_reasons(sample_bitmap, maximal_reasons)

			# Store results
			results[sample_key] = {
				'sample_bitmap': sample_bitmap,
				'sample_icf': sample_icf,
				'containing_reasons': containing_reasons,
				'num_maximal_reasons': num_maximal,
				'num_r_containing': num_r,
				'num_can_containing': num_can,
				'metadata': metadata
			}

			if num_maximal > 0:
				samples_with_maximal += 1
			if num_r > 0:
				samples_with_r += 1
			if num_can > 0:
				samples_with_can += 1

		except Exception as e:
			if verbose:
				print(f"Error processing sample {sample_key}: {e}")
			continue

	if verbose:
		print(f"\n{ '='*80 }")
		print("Analysis complete!")
		print("Summary:")
		print(f"   Total samples: {len(sample_keys)}")
		print(f"   Successfully analyzed: {len(results)}")
		print(f"   Samples with maximal reasons: {samples_with_maximal}")
		print(f"   Samples with R containment: {samples_with_r}")
		print(f"   Samples with CAN containment: {samples_with_can}")

		if len(results) > 0:
			maximal_counts = [r['num_maximal_reasons'] for r in results.values()]
			r_counts = [r['num_r_containing'] for r in results.values()]
			can_counts = [r['num_can_containing'] for r in results.values()]

			print(f"\nMaximal reasons statistics:")
			print(f"   Min: {min(maximal_counts)}, Max: {max(maximal_counts)}, Avg: {sum(maximal_counts)/len(maximal_counts):.2f}")

			print(f"\nR containment statistics:")
			print(f"   Min: {min(r_counts)}, Max: {max(r_counts)}, Avg: {sum(r_counts)/len(r_counts):.2f}")

			print(f"\nCAN containment statistics:")
			print(f"   Min: {min(can_counts)}, Max: {max(can_counts)}, Avg: {sum(can_counts)/len(can_counts):.2f}")

	return results


def analyze_maximal_reasons_coverage(maximal_reasons: List[str],
                                     sample_reason_mapping: Dict[str, Any],
                                     verbose: bool = True) -> Dict[str, Any]:
	"""
	For each maximal reason, find which samples it covers

	Args:
		maximal_reasons: List of maximal reason bitmaps
		sample_reason_mapping: Dictionary from analyze_samples_and_maximal_reasons
		verbose: Whether to print progress

	Returns:
		Dictionary mapping maximal reason index to covered samples info
	"""

	if verbose:
		print("\nAnalyzing maximal reason coverage...")
		print(f"{ '='*80 }")

	reason_coverage = {}

	for i, reason in enumerate(maximal_reasons):
		if verbose and (i % 10 == 0 or i == len(maximal_reasons) - 1):
			print(f"Progress: {i+1}/{len(maximal_reasons)} maximal reasons analyzed...")

		# Find all samples that this reason covers
		covered_samples = []

		for sample_key, sample_info in sample_reason_mapping.items():
			# Check if this reason is in the sample's containing reasons
			if reason in sample_info['containing_reasons']:
				covered_samples.append(sample_key)

		reason_coverage[i] = {
			'reason_bitmap': reason,
			'covered_samples': covered_samples,
			'num_covered': len(covered_samples)
		}

	if verbose:
		print(f"\n{ '='*80 }")
		print("Maximal reason coverage analysis complete!")

		if len(reason_coverage) > 0:
			coverage_counts = [info['num_covered'] for info in reason_coverage.values()]
			print(f"\nCoverage statistics:")
			print(f"   Total maximal reasons: {len(reason_coverage)}")
			print(f"   Min samples covered: {min(coverage_counts)}")
			print(f"   Max samples covered: {max(coverage_counts)}")
			print(f"   Avg samples covered: {sum(coverage_counts)/len(coverage_counts):.2f}")

			# Count how many reasons cover 0 samples
			zero_coverage = sum(1 for count in coverage_counts if count == 0)
			if zero_coverage > 0:
				print(f"   Reasons with 0 coverage: {zero_coverage}")

	return reason_coverage


def display_maximal_reason_coverage(reason_coverage: Dict[str, Any],
                                    sample_reason_mapping: Dict[str, Any],
                                    max_display: int = 10,
                                    show_sample_details: bool = True):
	"""
	Display coverage report for maximal reasons

	Args:
		reason_coverage: Dictionary from analyze_maximal_reasons_coverage
		sample_reason_mapping: Dictionary from analyze_samples_and_maximal_reasons
		max_display: Maximum number of reasons to display in detail
		show_sample_details: Whether to show details of covered samples
	"""

	if len(reason_coverage) == 0:
		print("No maximal reasons to display")
		return

	print(f"\nMaximal Reason Coverage Report:")
	print("="*80)

	# Sort by number of covered samples (most coverage first)
	sorted_reasons = sorted(reason_coverage.items(),
	                        key=lambda x: x[1]['num_covered'],
	                        reverse=True)

	# Detailed view of top reasons
	print(f"\nDetailed Coverage (showing top {min(max_display, len(sorted_reasons))}):")
	print("="*80)

	for i, (reason_idx, info) in enumerate(sorted_reasons[:max_display]):
		print(f"\n{'='*80}")
		print(f"Maximal Reason #{reason_idx + 1}")
		print(f"{'='*80}")

		reason_bitmap = info['reason_bitmap']
		print(f"Bitmap info:")
		print(f"  Length: {len(reason_bitmap)} bits")
		print(f"  Ones: {reason_bitmap.count('1')} ({100*reason_bitmap.count('1')/len(reason_bitmap):.1f}%)")
		print(f"  Bitmap: {reason_bitmap[:60]}{'...' if len(reason_bitmap) > 60 else ''}")

		covered_samples = info['covered_samples']
		print(f"\nCovers {len(covered_samples)} sample(s)")

		if len(covered_samples) == 0:
			print("  This maximal reason covers NO samples!")
		else:
			print(f"\n  Covered sample IDs:")

			if show_sample_details:
				# Show detailed info for each covered sample
				for j, sample_key in enumerate(covered_samples[:10]):  # Limit to first 10
					sample_info = sample_reason_mapping.get(sample_key, {})
					metadata = sample_info.get('metadata', {})

					label = metadata.get('predicted_label', 'unknown') if metadata else 'unknown'
					correct = 'correct' if metadata.get('prediction_correct') else 'incorrect' if metadata else 'unknown'
					test_idx = metadata.get('test_index', '?') if metadata else '?'

					print(f"    {j+1}. {sample_key}")
					print(f"       Label: {label} | Correct: {correct} | Test index: {test_idx}")

				if len(covered_samples) > 10:
					print(f"\n    ... and {len(covered_samples) - 10} more samples")
			else:
				# Just show sample keys
				for j, sample_key in enumerate(covered_samples[:20]):
					print(f"    {j+1}. {sample_key}")

				if len(covered_samples) > 20:
					print(f"    ... and {len(covered_samples) - 20} more")

	if len(sorted_reasons) > max_display:
		print(f"\n{'='*80}")
		print(f"... and {len(sorted_reasons) - max_display} more maximal reasons")

	# Summary table of ALL maximal reasons
	print(f"\n{'='*80}")
	print(f"Complete Maximal Reason Coverage Table:")
	print(f"{'='*80}")
	print(f"{'Reason #':<12} {'# Samples':<12} {'Ones':<12} {'Ones %':<10}")
	print(f"{'-'*80}")

	for reason_idx, info in sorted_reasons:
		reason_bitmap = info['reason_bitmap']
		ones_count = reason_bitmap.count('1')
		ones_pct = 100 * ones_count / len(reason_bitmap)

		print(f"{reason_idx + 1:<12} {info['num_covered']:<12} {ones_count:<12} {ones_pct:<10.1f}")

	# Additional statistics
	print(f"\n{'='*80}")
	print(f"Coverage Distribution:")
	print(f"{'='*80}")

	coverage_counts = [info['num_covered'] for info in reason_coverage.values()]

	# Histogram of coverage
	print(f"\nCoverage histogram:")
	max_coverage = max(coverage_counts)

	# Create bins
	if max_coverage <= 10:
		bins = list(range(max_coverage + 1))
	else:
		bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, float('inf')]

	for i in range(len(bins) - 1):
		lower = bins[i]
		upper = bins[i + 1]

		if upper == float('inf'):
			count = sum(1 for c in coverage_counts if c >= lower)
			label = f"{lower}+"
		else:
			count = sum(1 for c in coverage_counts if lower <= c < upper)
			label = f"{lower}-{upper-1}" if upper != lower + 1 else f"{lower}"

		if count > 0:
			bar = '█' * min(50, count)
			print(f"  {label:>6} samples: {bar} ({count})")


def display_sample_reason_mapping(results: Dict[str, Any],
                                  max_display: int = 10,
                                  show_bitmaps: bool = False):
	"""
	Display the mapping of samples to containing maximal reasons

	Args:
		results: Dictionary from analyze_samples_and_maximal_reasons
		max_display: Maximum number of samples to display in detail
		show_bitmaps: Whether to show full bitmaps (can be very long)
	"""

	if len(results) == 0:
		print("No results to display")
		return

	print(f"\nSample to Maximal Reason Mapping:")
	print("="*80)

	# Sort by number of containing maximal reasons (most interesting first)
	sorted_samples = sorted(results.items(),
	                        key=lambda x: x[1]['num_maximal_reasons'],
	                        reverse=True)

	for i, (sample_key, info) in enumerate(sorted_samples[:max_display]):
		print(f"\n{'='*80}")
		print(f"Sample #{i+1}: {sample_key}")
		print(f"{'='*80}")

		# Display metadata if available
		if info['metadata']:
			metadata = info['metadata']
			print(f"Metadata:")
			print(f"  Dataset: {metadata.get('dataset_name', 'unknown')}")
			print(f"  Predicted label: {metadata.get('predicted_label', 'unknown')}")
			print(f"  Actual label: {metadata.get('actual_label', 'unknown')}")
			print(f"  Test index: {metadata.get('test_index', 'unknown')}")
			print(f"  Prediction correct: {metadata.get('prediction_correct', 'unknown')}")

		# Display bitmap info
		sample_bitmap = info['sample_bitmap']
		print(f"\nSample ICF bitmap:")
		print(f"  Length: {len(sample_bitmap)} bits")
		print(f"  Ones: {sample_bitmap.count('1')} ({100*sample_bitmap.count('1')/len(sample_bitmap):.1f}%)")

		if show_bitmaps:
			print(f"  Bitmap: {sample_bitmap[:100]}{'...' if len(sample_bitmap) > 100 else ''}")

		# Display containment statistics
		print(f"\nContainment statistics:")
		print(f"  Maximal reasons containing: {info['num_maximal_reasons']}")
		print(f"  Total R keys containing: {info['num_r_containing']}")
		print(f"  CAN candidates containing: {info['num_can_containing']}")

		# Calculate percentage if R has keys
		if info['num_r_containing'] > 0:
			maximal_pct = 100 * info['num_maximal_reasons'] / info['num_r_containing']
			print(f"  Maximal/R ratio: {maximal_pct:.1f}%")

		if info['num_maximal_reasons'] == 0:
			print("  No maximal reasons contain this sample!")
		else:
			print(f"  Sample is contained in {info['num_maximal_reasons']} maximal reason(s)")

	if len(sorted_samples) > max_display:
		print(f"\n{'='*80}")
		print(f"... and {len(sorted_samples) - max_display} more samples")

	# Enhanced summary table
	print(f"\n{'='*80}")
	print(f"Enhanced Summary Table:")
	print(f"{'='*80}")
	print(f"{'Sample':<35} {'Max/R/CAN':<15} {'Label':<10} {'Correct':<8}")
	print(f"{'-'*80}")

	for sample_key, info in sorted_samples[:40]:  # Show more in table format
		label = 'unknown'
		correct = '?'
		if info['metadata']:
			label = str(info['metadata'].get('predicted_label', 'unknown'))[:8]
			correct = 'correct' if info['metadata'].get('prediction_correct') else 'incorrect'

		containment_str = f"{info['num_maximal_reasons']}/{info['num_r_containing']}/{info['num_can_containing']}"

		print(f"{sample_key:<35} {containment_str:<15} {label:<10} {correct:<8}")

	# Additional statistics
	print(f"\n{'='*80}")
	print(f"Containment Analysis:")
	print(f"{'='*80}")

	# Samples with no maximal reasons but in R
	no_maximal_but_in_r = sum(1 for info in results.values()
	                          if info['num_maximal_reasons'] == 0 and info['num_r_containing'] > 0)

	# Samples in CAN but not in R
	in_can_not_r = sum(1 for info in results.values()
	                   if info['num_can_containing'] > 0 and info['num_r_containing'] == 0)

	# Samples nowhere
	nowhere = sum(1 for info in results.values()
	              if info['num_maximal_reasons'] == 0 and
	              info['num_r_containing'] == 0 and
	              info['num_can_containing'] == 0)

	print(f"Samples in maximal reasons: {sum(1 for info in results.values() if info['num_maximal_reasons'] > 0)}")
	print(f"Samples in R but not maximal: {no_maximal_but_in_r}")
	print(f"Samples in CAN but not R: {in_can_not_r}")
	print(f"Samples not contained anywhere: {nowhere}")

from redis_helpers.forest import retrieve_forest
from redis_helpers.endpoints import retrieve_monotonic_dict
from redis_helpers.samples import retrieve_sample
from redis_helpers.icf import bitmap_to_icf
from icf_eu_encoding import icf_to_bitmap_mask, bitmap_mask_to_string
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings

def load_training_bounds(dataset_name='ECG200'):
	"""
	Load training dataset and compute min/max bounds for each time point.
	Uses the training_set already loaded in the notebook.

	Args:
		dataset_name: Name of the dataset (used for fallback only)

	Returns:
		Tuple of (min_bounds, max_bounds) arrays
	"""
	print(f"Loading training bounds...")

	# Try to get training_set from global scope
	X_train = None
	if 'training_set' in globals() and isinstance(globals()['training_set'], dict):
		X_train = np.asarray(globals()['training_set'].get('X_train'))

	# Fallback: try to get from db
	if X_train is None and 'db' in globals():
		try:
			ts = globals()['db']['data'].get('TRAINING_SET', {}).get('value_json', {})
			if 'X_train' in ts:
				X_train = np.asarray(ts['X_train'])
		except Exception:
			pass

	# Last resort: try aeon
	if X_train is None:
		try:
			from aeon.datasets import load_classification
			X_train, _ = load_classification(dataset_name, split="train")
		except Exception:
			raise RuntimeError("Could not load training data. Please ensure training_set is available.")

	# Reshape to 2D: (n_samples, n_timepoints)
	if X_train.ndim > 2:
		X_train = X_train.reshape(X_train.shape[0], -1)

	print(f"Loaded {X_train.shape[0]} training samples")
	print(f"Each sample has {X_train.shape[1]} time points")

	# Compute min and max for each time point
	min_bounds = np.min(X_train, axis=0)
	max_bounds = np.max(X_train, axis=0)

	print(f"Computed bounds:")
	print(f"   Global min: {min_bounds.min():.3f}")
	print(f"   Global max: {max_bounds.max():.3f}")

	return min_bounds, max_bounds


def complete_icf_with_bounds(icf, feature_names, min_bounds, max_bounds):
	"""
	Complete ICF dictionary with bounds, replacing infinities

	Args:
		icf: Dictionary mapping feature names to (inf, sup) intervals
		feature_names: List of all feature names
		min_bounds: Array of minimum values for each time point
		max_bounds: Array of maximum values for each time point

	Returns:
		Dictionary with complete ICF intervals using actual bounds
	"""
	completed_icf = {}

	for i, feature in enumerate(feature_names):
		if feature in icf:
			inf_val, sup_val = icf[feature]

			# Replace -inf with min bound
			if inf_val == float('-inf'):
				inf_val = min_bounds[i]

			# Replace +inf with max bound
			if sup_val == float('inf'):
				sup_val = max_bounds[i]

			completed_icf[feature] = (inf_val, sup_val)
		else:
			# Feature not in ICF, use full bounds
			completed_icf[feature] = (min_bounds[i], max_bounds[i])

	return completed_icf


def plot_sample_with_reasons(sample_values,
                             original_icf,
                             maximal_icf,
                             feature_names,
                             sample_key,
                             sample_metadata,
                             maximal_reason_idx,
                             output_path):
	"""
	Plot time series with two ICF corridors

	Args:
		sample_values: Array of time series values
		original_icf: Completed ICF from original forest extraction
		maximal_icf: Completed ICF from maximal reason
		feature_names: List of feature names
		sample_key: Sample identifier
		sample_metadata: Sample metadata dictionary
		maximal_reason_idx: Index of the maximal reason
		output_path: Path to save the plot
	"""
	fig, ax = plt.subplots(figsize=(14, 6))

	# Time points
	time_points = np.arange(len(sample_values))

	# Extract corridor bounds
	original_lower = np.array([original_icf[feature_names[i]][0] for i in range(len(feature_names))])
	original_upper = np.array([original_icf[feature_names[i]][1] for i in range(len(feature_names))])

	maximal_lower = np.array([maximal_icf[feature_names[i]][0] for i in range(len(feature_names))])
	maximal_upper = np.array([maximal_icf[feature_names[i]][1] for i in range(len(feature_names))])

	# Plot maximal reason corridor (wider, behind) - transparent red
	ax.fill_between(time_points, maximal_lower, maximal_upper,
	                alpha=0.2, color='red', label='Maximal Reason Corridor')

	# Plot original ICF corridor (narrower, in front) - transparent blue
	ax.fill_between(time_points, original_lower, original_upper,
	                alpha=0.3, color='blue', label='Original ICF Corridor')

	# Plot actual time series - solid red
	ax.plot(time_points, sample_values, 'r-', linewidth=2, label='Time Series', zorder=10)

	# Labels and title
	ax.set_xlabel('Time Point', fontsize=12)
	ax.set_ylabel('Value', fontsize=12)
	ax.set_title(f'Sample: {sample_key}', fontsize=14, fontweight='bold')
	ax.legend(loc='upper right', fontsize=10)
	ax.grid(True, alpha=0.3)

	# Add text box with information
	info_text = []
	info_text.append(f"Sample: {sample_key}")

	if sample_metadata:
		info_text.append(f"Dataset: {sample_metadata.get('dataset_name', 'unknown')}")
		info_text.append(f"Predicted Label: {sample_metadata.get('predicted_label', 'unknown')}")
		info_text.append(f"Actual Label: {sample_metadata.get('actual_label', 'unknown')}")
		info_text.append(f"Correct: {'correct' if sample_metadata.get('prediction_correct') else 'incorrect'}")
		info_text.append(f"Test Index: {sample_metadata.get('test_index', 'unknown')}")

	info_text.append("")
	info_text.append(f"Maximal Reason: #{maximal_reason_idx + 1}")

	# Count non-infinite intervals in original and maximal
	original_finite = sum(1 for f in feature_names
	                      if original_icf[f][0] != float('-inf') or original_icf[f][1] != float('inf'))
	maximal_finite = sum(1 for f in feature_names
	                     if maximal_icf[f][0] != float('-inf') or maximal_icf[f][1] != float('inf'))

	info_text.append(f"Original ICF constraints: {original_finite}/{len(feature_names)}")
	info_text.append(f"Maximal ICF constraints: {maximal_finite}/{len(feature_names)}")

	# Place text box
	textstr = '\n'.join(info_text)
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
	ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
	        verticalalignment='top', bbox=props, family='monospace')

	plt.tight_layout()
	plt.savefig(output_path, dpi=150, bbox_inches='tight')
	plt.close()

	print(f"  Saved: {output_path.name}")


def generate_reason_visualizations(db,
                                   sample_reason_mapping,
                                   reason_coverage,
                                   maximal_reasons,
                                   dataset_name='ECG200',
                                   output_dir='reasons',
                                   verbose=True):
	"""
	Generate visualization plots for all samples and their maximal reasons

	Args:
		db: Dictionary containing data, reasons and candidates
		sample_reason_mapping: Sample to reasons mapping
		reason_coverage: Reason to samples mapping
		maximal_reasons: List of maximal reason bitmaps
		dataset_name: Name of the dataset
		output_dir: Directory to save plots
		verbose: Whether to print progress
	"""

	# Create output directory
	output_path = Path(output_dir)
	output_path.mkdir(exist_ok=True)

	if verbose:
		print(f"Created output directory: {output_path.absolute()}")

	# Load training bounds
	min_bounds, max_bounds = load_training_bounds(dataset_name)

	# Load forest and EU
	if verbose:
		print(f"\nLoading forest and endpoints...")

	rf_data = retrieve_forest(db['data']['RF'])
	eu_data = retrieve_monotonic_dict(db['data']['EU'])

	if not rf_data or not eu_data:
		raise Exception("Could not load RF or EU from DATA")

	# Get feature names (sorted)
	feature_names = sorted(eu_data.keys())

	if verbose:
		print(f"Loaded forest with {len(rf_data)} trees")
		print(f"Loaded EU with {len(feature_names)} features")

	# Filter samples that have at least one maximal reason
	samples_with_maximal = {k: v for k, v in sample_reason_mapping.items()
	                        if v['num_maximal_reasons'] > 0}

	if verbose:
		print(f"\nFound {len(samples_with_maximal)} samples with maximal reasons")
		print(f"{'='*80}")

	total_plots = 0

	# Iterate over samples
	for sample_idx, (sample_key, sample_info) in enumerate(samples_with_maximal.items()):
		if verbose:
			print(f"\nProcessing sample {sample_idx + 1}/{len(samples_with_maximal)}: {sample_key}")

		# Load sample
		sample_dict = retrieve_sample(db['data'], sample_key)
		if not sample_dict:
			if verbose:
				print(f"  Could not load sample, skipping")
			continue

		# Convert sample dict to array (in correct order)
		sample_values = np.array([sample_dict[f] for f in feature_names])

		# Get original ICF from forest
		original_icf_raw = rf_data.extract_icf(sample_dict)
		original_icf = complete_icf_with_bounds(original_icf_raw, feature_names, min_bounds, max_bounds)

		# Get metadata
		sample_metadata = sample_info.get('metadata', {})

		# Get all maximal reasons containing this sample (sorted lexicographically by bitmap)
		containing_reasons = sample_info['containing_reasons']

		# Create list of (reason_idx, reason_bitmap) and sort by bitmap lexicographically
		reason_indices = []
		for reason_bitmap in containing_reasons:
			# Find the index of this reason in maximal_reasons list
			try:
				reason_idx = maximal_reasons.index(reason_bitmap)
				reason_indices.append((reason_idx, reason_bitmap))
			except ValueError:
				continue

		# Sort by bitmap lexicographically
		reason_indices.sort(key=lambda x: x[1])

		if verbose:
			print(f"  {len(reason_indices)} maximal reasons contain this sample")

		# Generate plot for each maximal reason
		for progressive_id, (reason_idx, reason_bitmap) in enumerate(reason_indices):
			# Convert maximal reason bitmap to ICF
			maximal_icf_raw = bitmap_to_icf(reason_bitmap, eu_data)
			maximal_icf = complete_icf_with_bounds(maximal_icf_raw, feature_names, min_bounds, max_bounds)

			# Create filename
			filename = f"reason_{sample_key}_{progressive_id:03d}.png"
			output_file = output_path / filename

			# Generate plot
			plot_sample_with_reasons(
				sample_values=sample_values,
				original_icf=original_icf,
				maximal_icf=maximal_icf,
				feature_names=feature_names,
				sample_key=sample_key,
				sample_metadata=sample_metadata,
				maximal_reason_idx=reason_idx,
				output_path=output_file
			)

			total_plots += 1

	if verbose:
		print(f"\n{'='*80}")
		print(f"Visualization complete!")
		print(f"Generated {total_plots} plots")
		print(f"Saved to: {output_path.absolute()}")

