"""
Test suite for cost_function.py to ensure robustness against edge cases
and verify that the function behaves correctly after modifications.
"""

import numpy as np
import pytest
from cost_function import cal_sigmas, cost_function


class TestCalSigmas:
	"""Test suite for cal_sigmas function"""

	def test_basic_functionality(self):
		"""Test basic sigma calculation"""
		X_train = np.array([
			[1.0, 2.0, 3.0],
			[2.0, 3.0, 4.0],
			[3.0, 4.0, 5.0],
			[4.0, 5.0, 6.0],
		])
		X_test = np.array([
			[2.5, 3.5, 4.5],
		])
		feature_names = ['f1', 'f2', 'f3']

		sigmas = cal_sigmas(X_train, X_test, feature_names)

		assert 0 in sigmas
		assert 'f1' in sigmas[0]
		assert 'sigma_plus' in sigmas[0]['f1']
		assert 'sigma_minus' in sigmas[0]['f1']
		assert 'ratio_above_mean' in sigmas[0]['f1']
		assert 'ratio_below_mean' in sigmas[0]['f1']

		# Check that ratios sum to 1
		for feature in feature_names:
			ratio_sum = sigmas[0][feature]['ratio_above_mean'] + sigmas[0][feature]['ratio_below_mean']
			assert np.isclose(ratio_sum, 1.0), f"Ratios don't sum to 1 for {feature}: {ratio_sum}"

	def test_all_values_same(self):
		"""Test when all training values are identical"""
		X_train = np.array([
			[5.0, 5.0],
			[5.0, 5.0],
			[5.0, 5.0],
		])
		X_test = np.array([[5.0, 5.0]])
		feature_names = ['f1', 'f2']

		sigmas = cal_sigmas(X_train, X_test, feature_names)

		# Should handle all identical values gracefully
		assert 0 in sigmas
		for feature in feature_names:
			assert feature in sigmas[0]
			assert sigmas[0][feature]['sigma_plus'] == 0.0
			assert sigmas[0][feature]['sigma_minus'] == 0.0

	def test_all_above_test_value(self):
		"""Test when all training values are above test value"""
		X_train = np.array([
			[10.0],
			[11.0],
			[12.0],
		])
		X_test = np.array([[1.0]])
		feature_names = ['f1']

		sigmas = cal_sigmas(X_train, X_test, feature_names)

		assert 0 in sigmas
		assert sigmas[0]['f1']['sigma_plus'] > 0
		assert sigmas[0]['f1']['sigma_minus'] == 0.0
		assert np.isclose(sigmas[0]['f1']['ratio_above_mean'], 1.0)
		assert np.isclose(sigmas[0]['f1']['ratio_below_mean'], 0.0)

	def test_all_below_test_value(self):
		"""Test when all training values are below test value"""
		X_train = np.array([
			[1.0],
			[2.0],
			[3.0],
		])
		X_test = np.array([[10.0]])
		feature_names = ['f1']

		sigmas = cal_sigmas(X_train, X_test, feature_names)

		assert 0 in sigmas
		assert sigmas[0]['f1']['sigma_plus'] == 0.0
		assert sigmas[0]['f1']['sigma_minus'] > 0
		assert np.isclose(sigmas[0]['f1']['ratio_above_mean'], 0.0)
		assert np.isclose(sigmas[0]['f1']['ratio_below_mean'], 1.0)

	def test_custom_test_ids(self):
		"""Test with custom test IDs"""
		X_train = np.array([[1.0], [2.0], [3.0]])
		X_test = np.array([[2.0], [3.0]])
		feature_names = ['f1']
		test_ids = ['test_a', 'test_b']

		sigmas = cal_sigmas(X_train, X_test, feature_names, test_ids=test_ids)

		assert 'test_a' in sigmas
		assert 'test_b' in sigmas
		assert 'f1' in sigmas['test_a']
		assert 'f1' in sigmas['test_b']

	def test_empty_training_data(self):
		"""Test behavior with empty training data"""
		X_train = np.array([]).reshape(0, 2)
		X_test = np.array([[1.0, 2.0]])
		feature_names = ['f1', 'f2']

		sigmas = cal_sigmas(X_train, X_test, feature_names)

		# Should skip features when no training data
		assert 0 in sigmas
		assert len(sigmas[0]) == 0  # No features processed

	def test_single_training_sample(self):
		"""Test with only one training sample"""
		X_train = np.array([[5.0, 10.0]])
		X_test = np.array([[3.0, 8.0]])
		feature_names = ['f1', 'f2']

		sigmas = cal_sigmas(X_train, X_test, feature_names)

		assert 0 in sigmas
		for feature in feature_names:
			assert feature in sigmas[0]
			# With one sample, sigma will be 0
			assert sigmas[0][feature]['sigma_plus'] == 0.0 or sigmas[0][feature]['sigma_minus'] == 0.0


class TestCostFunction:
	"""Test suite for cost_function"""

	def test_basic_cost_calculation(self):
		"""Test basic cost function calculation"""
		sample = {'f1': 5.0, 'f2': 10.0}
		icf = {
			'f1': (4.0, 6.0),
			'f2': (9.0, 11.0)
		}
		sigmas = {
			'f1': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			},
			'f2': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			}
		}

		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert cost >= 0.0
		assert cost <= 2.0  # Maximum cost should be number of features
		assert np.isfinite(cost)

	def test_zero_sigmas_handling(self):
		"""Test that zero sigmas are handled gracefully"""
		sample = {'f1': 5.0}
		icf = {'f1': (4.0, 6.0)}
		sigmas = {
			'f1': {
				'sigma_plus': 0.0,
				'sigma_minus': 0.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			}
		}

		# Should not raise exception, feature should be skipped
		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		# When both sigmas are zero (no variance), the feature should be skipped
		assert cost == 0.0

	def test_all_above_case(self):
		"""Test when all training data is above test value"""
		sample = {'f1': 5.0}
		icf = {'f1': (4.0, 6.0)}
		sigmas = {
			'f1': {
				'sigma_plus': 2.0,
				'sigma_minus': 0.01,  # Very small but not zero
				'ratio_above_mean': 1.0,
				'ratio_below_mean': 0.0
			}
		}

		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		assert cost >= 0.0

	def test_all_below_case(self):
		"""Test when all training data is below test value"""
		sample = {'f1': 5.0}
		icf = {'f1': (4.0, 6.0)}
		sigmas = {
			'f1': {
				'sigma_plus': 0.01,  # Very small but not zero
				'sigma_minus': 2.0,
				'ratio_above_mean': 0.0,
				'ratio_below_mean': 1.0
			}
		}

		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		assert cost >= 0.0

	def test_infinite_intervals(self):
		"""Test handling of infinite intervals"""
		sample = {'f1': 5.0}
		icf = {'f1': (-np.inf, np.inf)}
		sigmas = {
			'f1': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			}
		}

		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		# With infinite interval, should capture all probability mass
		assert 0.9 <= cost <= 1.01  # Should be close to 1.0

	def test_semi_infinite_intervals(self):
		"""Test handling of semi-infinite intervals"""
		sample = {'f1': 5.0, 'f2': 5.0}
		icf = {
			'f1': (-np.inf, 6.0),
			'f2': (4.0, np.inf)
		}
		sigmas = {
			'f1': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			},
			'f2': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			}
		}

		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		assert cost >= 0.0

	def test_multiple_features(self):
		"""Test that cost accumulates correctly across multiple features"""
		sample = {'f1': 5.0, 'f2': 10.0, 'f3': 15.0}
		icf = {
			'f1': (4.0, 6.0),
			'f2': (9.0, 11.0),
			'f3': (14.0, 16.0)
		}
		sigmas = {
			'f1': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			},
			'f2': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			},
			'f3': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			}
		}

		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		assert cost >= 0.0
		# Cost should be greater than any single feature contribution
		assert cost > 0.0

	def test_missing_feature_in_sigmas(self):
		"""Test handling when a feature in ICF is missing from sigmas"""
		sample = {'f1': 5.0, 'f2': 10.0}
		icf = {
			'f1': (4.0, 6.0),
			'f2': (9.0, 11.0)
		}
		sigmas = {
			'f1': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			}
			# f2 is missing
		}

		# Should not raise exception, just skip missing feature
		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		assert cost >= 0.0

	def test_invalid_percentages(self):
		"""Test handling of percentages that don't sum to 1"""
		sample = {'f1': 5.0}
		icf = {'f1': (4.0, 6.0)}
		sigmas = {
			'f1': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.6,
				'ratio_below_mean': 0.6  # Sum > 1
			}
		}

		# Should normalize and handle gracefully
		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)

	def test_zero_percentages(self):
		"""Test handling when both percentages are zero"""
		sample = {'f1': 5.0}
		icf = {'f1': (4.0, 6.0)}
		sigmas = {
			'f1': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.0,
				'ratio_below_mean': 0.0
			}
		}

		# Should skip this feature gracefully
		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		assert cost == 0.0  # No features contributed

	def test_very_narrow_interval(self):
		"""Test with very narrow interval"""
		sample = {'f1': 5.0}
		icf = {'f1': (4.999, 5.001)}
		sigmas = {
			'f1': {
				'sigma_plus': 1.0,
				'sigma_minus': 1.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			}
		}

		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		assert cost >= 0.0
		# Narrow interval should give small cost
		assert cost < 0.1

	def test_very_wide_sigmas(self):
		"""Test with very large sigma values"""
		sample = {'f1': 5.0}
		icf = {'f1': (4.0, 6.0)}
		sigmas = {
			'f1': {
				'sigma_plus': 1000.0,
				'sigma_minus': 1000.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			}
		}

		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		assert cost >= 0.0

	def test_very_small_sigmas(self):
		"""Test with very small sigma values"""
		sample = {'f1': 5.0}
		icf = {'f1': (4.0, 6.0)}
		sigmas = {
			'f1': {
				'sigma_plus': 0.001,
				'sigma_minus': 0.001,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			}
		}

		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		assert cost >= 0.0

	def test_asymmetric_sigmas(self):
		"""Test with very different sigma values on each side"""
		sample = {'f1': 5.0}
		icf = {'f1': (3.0, 7.0)}
		sigmas = {
			'f1': {
				'sigma_plus': 0.1,
				'sigma_minus': 10.0,
				'ratio_above_mean': 0.5,
				'ratio_below_mean': 0.5
			}
		}

		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas, verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		assert cost >= 0.0

	def test_missing_required_parameters(self):
		"""Test that missing required parameters raise ValueError"""
		with pytest.raises(ValueError, match="Sample must be provided"):
			cost_function(sample=None, icf={}, sigmas={})

		with pytest.raises(ValueError, match="ICF must be provided"):
			cost_function(sample={}, icf=None, sigmas={})

		with pytest.raises(ValueError, match="Sigmas must be provided"):
			cost_function(sample={}, icf={}, sigmas=None)


class TestIntegration:
	"""Integration tests combining cal_sigmas and cost_function"""

	def test_end_to_end_workflow(self):
		"""Test complete workflow from training data to cost"""
		# Create synthetic training data
		np.random.seed(42)
		X_train = np.random.randn(100, 3) * 2 + 5
		X_test = np.array([[5.0, 5.0, 5.0]])
		feature_names = ['f1', 'f2', 'f3']

		# Calculate sigmas
		sigmas = cal_sigmas(X_train, X_test, feature_names)

		# Define ICF intervals
		icf = {
			'f1': (4.0, 6.0),
			'f2': (4.0, 6.0),
			'f3': (4.0, 6.0)
		}

		# Create sample
		sample = {'f1': 5.0, 'f2': 5.0, 'f3': 5.0}

		# Calculate cost
		cost = cost_function(sample=sample, icf=icf, sigmas=sigmas[0], verbose=False)

		assert isinstance(cost, float)
		assert np.isfinite(cost)
		assert cost >= 0.0
		print(f"End-to-end test cost: {cost:.4f}")

	def test_multiple_test_samples(self):
		"""Test with multiple test samples"""
		np.random.seed(42)
		X_train = np.random.randn(50, 2) * 2 + 10
		X_test = np.array([
			[9.0, 11.0],
			[10.0, 10.0],
			[11.0, 9.0]
		])
		feature_names = ['f1', 'f2']

		sigmas = cal_sigmas(X_train, X_test, feature_names)

		icf = {
			'f1': (8.0, 12.0),
			'f2': (8.0, 12.0)
		}

		costs = []
		for i in range(len(X_test)):
			sample = {'f1': X_test[i, 0], 'f2': X_test[i, 1]}
			cost = cost_function(sample=sample, icf=icf, sigmas=sigmas[i], verbose=False)
			costs.append(cost)
			assert np.isfinite(cost)
			assert cost >= 0.0

		print(f"Multiple sample costs: {costs}")
		assert len(costs) == 3
		assert all(c >= 0.0 for c in costs)


def run_all_tests():
	"""Run all tests and report results"""
	print("=" * 70)
	print("RUNNING COST FUNCTION TEST SUITE")
	print("=" * 70)

	# Run with pytest
	pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
	run_all_tests()

