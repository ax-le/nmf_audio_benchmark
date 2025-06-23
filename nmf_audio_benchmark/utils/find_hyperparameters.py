"""
Helper to find the best hyperparameters.
Only the cross validation scheme is developed for now.
A second option could be hyperparameter optimization, using optuna (for instance) on a learning dataset. This is TODO.
"""

import itertools
import numpy as np

from sklearn.model_selection import KFold

def generate_param_grid(param_grid):
    """
    Iterates over the parameter grid to generate all the possible combinations.
    """
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        yield params

def find_best_results_in_cv_scheme(all_results, cv, param_combinations, nb_metrics, test_metric_idx=0):
    """
    Find the best results in the cross validation, based on the accuracy.
    Works only if all the results are available in advance, i.e. if the model doesn't need to be trained before obtaining the test results.
    Seems to be the case for most of NMF algorithms.
    """
    # Initialize the final results
    final_results = -np.inf * np.ones((len(all_results), nb_metrics))
    final_best_params = np.empty(len(all_results), dtype=object)
    
    # Split the results in train and test sets
    kf = KFold(n_splits=cv, shuffle=False)#, random_state=42)
    
    # Loop over the splits
    for train_index, test_index in kf.split(all_results):
        # Keep only the results of the train set, to find the best params
        train_results = np.array([all_results[i] for i in train_index])

        # Aggregate train results across songs
        train_results_mean = np.mean(train_results, axis=0)

        # Find best params based on train set
        best_train_idx = np.argmax(train_results_mean[:, test_metric_idx])
        best_train_params = param_combinations[best_train_idx]

        # Update final results with the best params found
        for i in test_index:
            final_results[i] = all_results[i][best_train_idx]
            final_best_params[i] = best_train_params

    return final_results, final_best_params

if __name__ == "__main__":
    import unittest

    class TestFindBestResultsInCVScheme(unittest.TestCase):
        
        def test_find_best_results(self):
            # Define the parameters to test
            param_grid = {
                'param_1': [0.1, 0.2, 0.3, 0.4]
            }

            # Generate the parameter combinations
            param_combinations = list(generate_param_grid(param_grid))

            # Define the results
            all_results = np.array([
                [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
                [[0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]],
                [[0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7]]
            ])

            # Expected output
            expected_final_results = np.array([
                [0.4, 0.5], 
                [0.5, 0.6], 
                [0.6, 0.7]
            ])
            expected_final_best_params = [{'param_1': 0.4}, {'param_1': 0.4}, {'param_1': 0.4}]

            # Find the best results in the cross validation scheme
            final_results, final_best_params = find_best_results_in_cv_scheme(all_results, cv=3, param_combinations=param_combinations, nb_metrics=2, test_metric_idx=1)

            # Assert the results
            np.testing.assert_array_almost_equal(final_results, expected_final_results)
            self.assertEqual(final_best_params.tolist(), expected_final_best_params)

        def test_find_best_results_case_2(self):
            # Define the parameters to test
            param_grid = {
                'param_1': [0.1, 0.2, 0.3, 0.4]
            }

            # Generate the parameter combinations
            param_combinations = list(generate_param_grid(param_grid))

            # Define the results
            all_results = np.array([
                [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
                [[0.2, 0.3], [0.3, 0.4], [0.9, 1.0], [0.4, 0.5]],
                [[0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7]]
            ])

            # Expected output
            expected_final_results = np.array([
                [0.3, 0.4], 
                [0.4, 0.5], 
                [0.5, 0.6]
            ])
            expected_final_best_params = [{'param_1': 0.3}, {'param_1': 0.4}, {'param_1': 0.3}]

            # Find the best results in the cross validation scheme
            final_results, final_best_params = find_best_results_in_cv_scheme(all_results, cv=3, param_combinations=param_combinations, nb_metrics=2, test_metric_idx=1)

            # Assert the results
            np.testing.assert_array_almost_equal(final_results, expected_final_results)
            self.assertEqual(final_best_params.tolist(), expected_final_best_params)

        def test_find_best_results_2_params(self):
            # Define the parameters to test
            param_grid = {
                'param_1': [0.1, 0.2, 0.3],
                'param_2': [0.1, 0.2]
            }

            # Generate the parameter combinations
            param_combinations = list(generate_param_grid(param_grid))

            # Define the results
            all_results = np.array([
                [[0.1], [0.2], [0.3], [0.4], [0.5], [0.4]],
                [[0.1], [0.2], [1], [0.4], [0.5], [0.4]],
            ])

            # Expected output
            expected_final_results = np.array([
                [0.3], [0.5]
            ])
            expected_final_best_params = [{'param_1': 0.2, 'param_2':0.1}, {'param_1': 0.3, 'param_2': 0.1}]

            # Find the best results in the cross validation scheme
            final_results, final_best_params = find_best_results_in_cv_scheme(all_results, cv=2, param_combinations=param_combinations, nb_metrics=1, test_metric_idx=0)

            # Assert the results
            np.testing.assert_array_almost_equal(final_results, expected_final_results)
            self.assertEqual(final_best_params.tolist(), expected_final_best_params)

    unittest.main()
