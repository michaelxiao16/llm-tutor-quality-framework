#!/usr/bin/env python3
"""Regression tests for the mathematical invariants in analyze-efa.py."""

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name('analyze-efa.py')
SPEC = importlib.util.spec_from_file_location('analyze_efa', SCRIPT)
EFA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EFA)


class EfaMathTests(unittest.TestCase):
    def test_oblique_pattern_phi_structure_identity(self):
        rng = np.random.default_rng(20260827)
        latent = rng.multivariate_normal(
            mean=[0, 0], cov=[[1, 0.45], [0.45, 1]], size=500
        )
        population_loadings = np.array([
            [.85, .05], [.75, .10], [.70, .05],
            [.05, .85], [.10, .75], [.05, .70],
        ])
        scores = latent @ population_loadings.T + rng.normal(0, .45, size=(500, 6))
        frame = pd.DataFrame(scores, columns=[f'item_{i}' for i in range(6)])

        fitted = EFA.run_efa(frame, n_factors=2, rotation='oblimin')
        pattern, phi, structure, common = EFA.efa_model_matrices(fitted)

        np.testing.assert_allclose(pattern @ phi, structure, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(common, pattern @ structure.T, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(phi, phi.T, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(np.diag(phi), 1, atol=1e-10, rtol=1e-10)

    @unittest.skipUnless(
        os.environ.get('RSCRIPT') or shutil.which('Rscript'),
        'Rscript is required for the R-primary integration test',
    )
    def test_r_psych_is_authoritative_and_python_is_only_the_check(self):
        rng = np.random.default_rng(20260827)
        latent = rng.multivariate_normal(
            mean=[0, 0], cov=[[1, 0.40], [0.40, 1]], size=400
        )
        population_loadings = np.array([
            [.85, .05], [.78, .08], [.70, .05],
            [.05, .84], [.08, .76], [.05, .68],
        ])
        scores = latent @ population_loadings.T + rng.normal(0, .45, size=(400, 6))
        frame = pd.DataFrame(scores, columns=[f'item_{i}' for i in range(6)])
        python_check = EFA.run_efa(frame, n_factors=2, rotation='oblimin')

        with tempfile.TemporaryDirectory(prefix='efa-r-primary-test-') as temp_dir:
            root = Path(temp_dir)
            results = EFA.run_psych_primary_efa(
                df=frame,
                python_fa=python_check,
                n_factors=2,
                threshold=.40,
                iterations=10,
                parallel_output=root / 'parallel.csv',
                reliability_output=root / 'reliability.csv',
                r_pattern_output=root / 'pattern.csv',
                r_phi_output=root / 'phi.csv',
                r_structure_output=root / 'structure.csv',
                factorability_output=root / 'factorability.csv',
                congruence_output=root / 'congruence.csv',
                sl_loadings_output=root / 'sl_loadings.csv',
                sl_summary_output=root / 'sl_summary.csv',
                sl_omega_by_factor_output=root / 'sl_omega_by_factor.csv',
                sl_group_mapping_output=root / 'sl_group_mapping.csv',
                rotation='oblimin',
            )

            raw_r_pattern = pd.read_csv(root / 'pattern.csv').set_index('metric_id')
            final_pattern = results['loadings_df']
            np.testing.assert_allclose(
                final_pattern,
                raw_r_pattern.loc[final_pattern.index, final_pattern.columns],
            )
            pattern, phi, structure, common = EFA.efa_model_matrices(results['solution'])
            np.testing.assert_allclose(pattern @ phi, structure, atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(
                np.diag(common), raw_r_pattern['communality_h2_psych'],
                atol=1e-8, rtol=1e-8,
            )
            self.assertEqual(results['solution'].primary_engine, 'R psych::fa')
            self.assertGreaterEqual(
                results['congruence']['tucker_congruence_absolute'].min(), .85
            )
            self.assertIn('omega_hierarchical', results['sl_summary'])
            self.assertEqual(len(results['sl_group_mapping']), 2)
            self.assertTrue((root / 'sl_loadings.csv').exists())

    def test_surrogate_data_reproduces_correlation_matrix_and_fit(self):
        rng = np.random.default_rng(20260914)
        latent = rng.multivariate_normal(mean=[0, 0], cov=[[1, 0.4], [0.4, 1]], size=300)
        population_loadings = np.array([
            [.80, .05], [.72, .10], [.65, .05],
            [.05, .80], [.10, .72], [.05, .65],
        ])
        scores = latent @ population_loadings.T + rng.normal(0, .5, size=(300, 6))
        frame = pd.DataFrame(scores, columns=pd.Index([f'item_{i}' for i in range(6)], name='metric'))
        corr = frame.corr()

        surrogate = EFA.surrogate_data_from_correlation(corr, n_obs=len(frame))
        self.assertEqual(surrogate.shape, frame.shape)
        np.testing.assert_allclose(surrogate.corr(), corr, atol=1e-12)

        original = EFA.efa_model_matrices(EFA.run_efa(frame, n_factors=2, rotation='oblimin'))
        refit = EFA.efa_model_matrices(EFA.run_efa(surrogate, n_factors=2, rotation='oblimin'))
        for expected, actual in zip(original, refit):
            np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_applicability_tiers_core_conditional_rare(self):
        # 'retained' scored everywhere -> core; 'constant' scored everywhere but no variance
        # -> dropped; 'conditional_metric' scored in 3 of 10 (>= min_applicable=2) -> set
        # aside for subset analysis; 'too_rare' scored in 1 of 10 (< min_applicable) ->
        # dropped.
        rows = []
        for conversation in range(10):
            rows.extend([
                {
                    'conversation_id': f'c{conversation}', 'metric_id': 'retained',
                    'category': 'Test', 'subcategory': 'Retained',
                    'score': 1 + conversation % 5, 'score_status': 'scored',
                },
                {
                    'conversation_id': f'c{conversation}', 'metric_id': 'constant',
                    'category': 'Test', 'subcategory': 'Constant',
                    'score': 3, 'score_status': 'scored',
                },
                {
                    'conversation_id': f'c{conversation}', 'metric_id': 'conditional_metric',
                    'category': 'Test', 'subcategory': 'Conditional',
                    'score': (1 + conversation) if conversation < 3 else np.nan,
                    'score_status': 'scored' if conversation < 3 else 'not_applicable',
                },
                {
                    'conversation_id': f'c{conversation}', 'metric_id': 'too_rare',
                    'category': 'Test', 'subcategory': 'Rare',
                    'score': 2 if conversation < 1 else np.nan,
                    'score_status': 'scored' if conversation < 1 else 'not_applicable',
                },
            ])

        with tempfile.TemporaryDirectory(prefix='efa-math-test-') as temp_dir:
            path = Path(temp_dir) / 'scores.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            matrix = EFA.load_and_pivot_results(
                path, core_max_na_rate=.10, min_applicable_convos=2)

        # Pooled EFA matrix holds only the core, non-degenerate metric.
        self.assertEqual(list(matrix.columns), ['retained'])
        audit = matrix.attrs['preprocessing_audit'].set_index('metric_id')
        self.assertEqual(audit.loc['retained', 'applicability_tier'], 'core')
        self.assertEqual(audit.loc['constant', 'preprocessing_outcome'], 'zero_variance_after_imputation')
        self.assertEqual(audit.loc['conditional_metric', 'preprocessing_outcome'],
                         'conditional_set_aside_for_subset_efa')
        self.assertEqual(audit.loc['too_rare', 'preprocessing_outcome'],
                         'fewer_than_2_applicable_convos')
        # Conditional metric is held separately (un-imputed), not in the pooled matrix.
        conditional = matrix.attrs['conditional_matrix']
        self.assertEqual(list(conditional.columns), ['conditional_metric'])
        self.assertEqual(int(conditional['conditional_metric'].notna().sum()), 3)


if __name__ == '__main__':
    unittest.main()
