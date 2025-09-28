"""
Post-Hoc Tests Module
====================

Implements comprehensive post-hoc testing procedures for multiple comparisons.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import warnings

class PostHocTests:
    """
    Comprehensive post-hoc testing for multiple comparison procedures.
    """

    def __init__(self):
        self.supported_tests = [
            'tukey_hsd',
            'bonferroni',
            'holm_sidak',
            'benjamini_hochberg',
            'dunn_test',
            'nemenyi_test',
            'games_howell'
        ]

    def perform_post_hoc(self, data, dependent_var, independent_var, method='tukey_hsd', **kwargs):
        """
        Perform post-hoc analysis after significant omnibus test.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        dependent_var : str
            Dependent variable column name
        independent_var : str
            Independent variable column name
        method : str
            Post-hoc test method
        **kwargs : additional parameters

        Returns:
        --------
        dict : Post-hoc test results
        """
        try:
            # Clean data
            clean_data = data[[dependent_var, independent_var]].dropna()
            groups = clean_data[independent_var].unique()

            if len(groups) < 3:
                return {
                    'error': 'Post-hoc tests require at least 3 groups',
                    'method': method
                }

            # Route to appropriate method
            if method == 'tukey_hsd':
                return self._tukey_hsd(clean_data, dependent_var, independent_var)
            elif method == 'bonferroni':
                return self._bonferroni_correction(clean_data, dependent_var, independent_var)
            elif method == 'holm_sidak':
                return self._holm_sidak(clean_data, dependent_var, independent_var)
            elif method == 'benjamini_hochberg':
                return self._benjamini_hochberg(clean_data, dependent_var, independent_var)
            elif method == 'dunn_test':
                return self._dunn_test(clean_data, dependent_var, independent_var)
            elif method == 'games_howell':
                return self._games_howell(clean_data, dependent_var, independent_var)
            else:
                return {
                    'error': f'Unknown post-hoc method: {method}',
                    'supported_methods': self.supported_tests
                }

        except Exception as e:
            return {
                'error': f'Post-hoc analysis failed: {str(e)}',
                'method': method
            }

    def _tukey_hsd(self, data, dependent_var, independent_var):
        """
        Tukey's Honestly Significant Difference test.
        """
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            # Perform Tukey HSD
            tukey_result = pairwise_tukeyhsd(
                endog=data[dependent_var],
                groups=data[independent_var],
                alpha=0.05
            )

            # Extract results
            comparisons = []
            for i in range(len(tukey_result.groupsunique)):
                for j in range(i + 1, len(tukey_result.groupsunique)):
                    group1 = tukey_result.groupsunique[i]
                    group2 = tukey_result.groupsunique[j]

                    # Find the corresponding row in results
                    mask1 = (tukey_result.data['group1'] == group1) & (tukey_result.data['group2'] == group2)
                    mask2 = (tukey_result.data['group1'] == group2) & (tukey_result.data['group2'] == group1)
                    row_idx = np.where(mask1 | mask2)[0]

                    if len(row_idx) > 0:
                        idx = row_idx[0]
                        comparisons.append({
                            'group1': str(group1),
                            'group2': str(group2),
                            'mean_diff': tukey_result.data['meandiff'][idx],
                            'p_adj': tukey_result.data['p-adj'][idx],
                            'lower_ci': tukey_result.data['lower'][idx],
                            'upper_ci': tukey_result.data['upper'][idx],
                            'reject': tukey_result.data['reject'][idx]
                        })

            return {
                'method': 'Tukey HSD',
                'comparisons': comparisons,
                'family_wise_error_rate': 0.05,
                'summary_table': tukey_result.summary().as_html(),
                'interpretation': self._interpret_tukey_results(comparisons)
            }

        except ImportError:
            return self._bonferroni_correction(data, dependent_var, independent_var)
        except Exception as e:
            return {
                'error': f'Tukey HSD failed: {str(e)}',
                'method': 'Tukey HSD'
            }

    def _bonferroni_correction(self, data, dependent_var, independent_var):
        """
        Bonferroni correction for multiple comparisons.
        """
        groups = data[independent_var].unique()
        group_data = {group: data[data[independent_var] == group][dependent_var] for group in groups}

        comparisons = []
        p_values = []

        # Perform all pairwise t-tests
        for group1, group2 in combinations(groups, 2):
            data1 = group_data[group1]
            data2 = group_data[group2]

            # Independent t-test
            statistic, p_value = stats.ttest_ind(data1, data2)

            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(data1) - 1) * data1.var() +
                                 (len(data2) - 1) * data2.var()) /
                                (len(data1) + len(data2) - 2))
            cohens_d = (data1.mean() - data2.mean()) / pooled_std

            comparisons.append({
                'group1': str(group1),
                'group2': str(group2),
                'statistic': statistic,
                'p_raw': p_value,
                'mean_diff': data1.mean() - data2.mean(),
                'cohens_d': cohens_d
            })
            p_values.append(p_value)

        # Apply Bonferroni correction
        num_comparisons = len(p_values)
        alpha_corrected = 0.05 / num_comparisons

        for i, comparison in enumerate(comparisons):
            comparison['p_bonferroni'] = min(p_values[i] * num_comparisons, 1.0)
            comparison['significant'] = comparison['p_bonferroni'] <= 0.05
            comparison['alpha_corrected'] = alpha_corrected

        return {
            'method': 'Bonferroni Correction',
            'comparisons': comparisons,
            'num_comparisons': num_comparisons,
            'corrected_alpha': alpha_corrected,
            'family_wise_error_rate': 0.05,
            'interpretation': self._interpret_bonferroni_results(comparisons)
        }

    def _holm_sidak(self, data, dependent_var, independent_var):
        """
        Holm-Šídák sequential correction method.
        """
        # Start with Bonferroni results
        bonferroni_results = self._bonferroni_correction(data, dependent_var, independent_var)
        comparisons = bonferroni_results['comparisons']

        # Sort p-values
        sorted_comparisons = sorted(comparisons, key=lambda x: x['p_raw'])

        # Apply Holm-Šídák correction
        num_comparisons = len(comparisons)
        for i, comparison in enumerate(sorted_comparisons):
            k = i + 1  # Step number (1-indexed)
            alpha_holm = 1 - (1 - 0.05)**(1/(num_comparisons - k + 1))
            comparison['alpha_holm_sidak'] = alpha_holm
            comparison['significant_holm'] = comparison['p_raw'] <= alpha_holm

            # Stop if we fail to reject for the first time (step-down procedure)
            if not comparison['significant_holm']:
                for j in range(i + 1, len(sorted_comparisons)):
                    sorted_comparisons[j]['significant_holm'] = False
                break

        # Restore original order
        comparison_dict = {(c['group1'], c['group2']): c for c in sorted_comparisons}
        for comparison in comparisons:
            key = (comparison['group1'], comparison['group2'])
            if key in comparison_dict:
                comparison.update(comparison_dict[key])

        return {
            'method': 'Holm-Šídák',
            'comparisons': comparisons,
            'num_comparisons': num_comparisons,
            'family_wise_error_rate': 0.05,
            'interpretation': self._interpret_holm_sidak_results(comparisons)
        }

    def _benjamini_hochberg(self, data, dependent_var, independent_var):
        """
        Benjamini-Hochberg False Discovery Rate control.
        """
        # Start with raw p-values from t-tests
        bonferroni_results = self._bonferroni_correction(data, dependent_var, independent_var)
        comparisons = bonferroni_results['comparisons']

        # Extract p-values and sort
        p_values = [c['p_raw'] for c in comparisons]
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]

        # Apply Benjamini-Hochberg correction
        num_comparisons = len(p_values)
        q_values = np.zeros(num_comparisons)

        for i in range(num_comparisons - 1, -1, -1):
            if i == num_comparisons - 1:
                q_values[i] = sorted_p_values[i]
            else:
                q_values[i] = min(sorted_p_values[i] * num_comparisons / (i + 1), q_values[i + 1])

        # Add results to comparisons
        for i, comparison in enumerate(comparisons):
            sorted_idx = sorted_indices[i] if i in sorted_indices else np.where(sorted_indices == i)[0][0]
            comparison['q_value'] = q_values[sorted_idx]
            comparison['significant_bh'] = comparison['q_value'] <= 0.05

        return {
            'method': 'Benjamini-Hochberg FDR',
            'comparisons': comparisons,
            'num_comparisons': num_comparisons,
            'false_discovery_rate': 0.05,
            'interpretation': self._interpret_bh_results(comparisons)
        }

    def _dunn_test(self, data, dependent_var, independent_var):
        """
        Dunn's test for multiple comparisons (non-parametric).
        """
        groups = data[independent_var].unique()
        group_data = {group: data[data[independent_var] == group][dependent_var] for group in groups}

        # Calculate ranks for entire dataset
        all_data = data[dependent_var]
        ranks = stats.rankdata(all_data)

        # Split ranks by group
        group_ranks = {}
        start_idx = 0
        for group in groups:
            group_size = len(group_data[group])
            group_ranks[group] = ranks[start_idx:start_idx + group_size]
            start_idx += group_size

        # Calculate rank sums and mean ranks
        rank_sums = {group: np.sum(group_ranks[group]) for group in groups}
        mean_ranks = {group: np.mean(group_ranks[group]) for group in groups}

        comparisons = []
        n_total = len(all_data)

        # Perform pairwise comparisons
        for group1, group2 in combinations(groups, 2):
            n1, n2 = len(group_data[group1]), len(group_data[group2])
            R1, R2 = rank_sums[group1], rank_sums[group2]

            # Dunn's test statistic
            numerator = abs(R1/n1 - R2/n2)
            denominator = np.sqrt((n_total * (n_total + 1) / 12) * (1/n1 + 1/n2))
            z_statistic = numerator / denominator

            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

            comparisons.append({
                'group1': str(group1),
                'group2': str(group2),
                'z_statistic': z_statistic,
                'p_raw': p_value,
                'mean_rank_1': mean_ranks[group1],
                'mean_rank_2': mean_ranks[group2],
                'rank_diff': mean_ranks[group1] - mean_ranks[group2]
            })

        # Apply Bonferroni correction
        num_comparisons = len(comparisons)
        for comparison in comparisons:
            comparison['p_dunn'] = min(comparison['p_raw'] * num_comparisons, 1.0)
            comparison['significant'] = comparison['p_dunn'] <= 0.05

        return {
            'method': 'Dunn Test',
            'comparisons': comparisons,
            'num_comparisons': num_comparisons,
            'total_n': n_total,
            'interpretation': self._interpret_dunn_results(comparisons)
        }

    def _games_howell(self, data, dependent_var, independent_var):
        """
        Games-Howell test for unequal variances.
        """
        groups = data[independent_var].unique()
        group_data = {group: data[data[independent_var] == group][dependent_var] for group in groups}

        comparisons = []

        for group1, group2 in combinations(groups, 2):
            data1 = group_data[group1]
            data2 = group_data[group2]

            n1, n2 = len(data1), len(data2)
            mean1, mean2 = data1.mean(), data2.mean()
            var1, var2 = data1.var(ddof=1), data2.var(ddof=1)

            # Games-Howell test statistic
            mean_diff = mean1 - mean2
            se_diff = np.sqrt(var1/n1 + var2/n2)
            t_statistic = mean_diff / se_diff

            # Welch-Satterthwaite degrees of freedom
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

            # Two-tailed p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

            # Confidence interval
            t_critical = stats.t.ppf(0.975, df)  # For 95% CI
            margin_error = t_critical * se_diff
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error

            comparisons.append({
                'group1': str(group1),
                'group2': str(group2),
                't_statistic': t_statistic,
                'p_value': p_value,
                'mean_diff': mean_diff,
                'se_diff': se_diff,
                'degrees_freedom': df,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_value <= 0.05
            })

        return {
            'method': 'Games-Howell',
            'comparisons': comparisons,
            'interpretation': self._interpret_games_howell_results(comparisons)
        }

    def _interpret_tukey_results(self, comparisons):
        """Interpret Tukey HSD results."""
        significant_pairs = [c for c in comparisons if c['reject']]

        interpretation = f"Tukey HSD identified {len(significant_pairs)} significant pairwise differences "
        interpretation += f"out of {len(comparisons)} total comparisons.\n\n"

        if significant_pairs:
            interpretation += "Significant differences found between:\n"
            for pair in significant_pairs:
                interpretation += f"- {pair['group1']} vs {pair['group2']} (p = {pair['p_adj']:.4f})\n"
        else:
            interpretation += "No significant pairwise differences were found."

        return interpretation

    def _interpret_bonferroni_results(self, comparisons):
        """Interpret Bonferroni correction results."""
        significant_pairs = [c for c in comparisons if c['significant']]

        interpretation = f"Bonferroni correction identified {len(significant_pairs)} significant pairwise differences "
        interpretation += f"out of {len(comparisons)} total comparisons.\n"
        interpretation += f"Corrected α level: {comparisons[0]['alpha_corrected']:.4f}\n\n"

        if significant_pairs:
            interpretation += "Significant differences found between:\n"
            for pair in significant_pairs:
                interpretation += f"- {pair['group1']} vs {pair['group2']} (p = {pair['p_bonferroni']:.4f})\n"
        else:
            interpretation += "No significant pairwise differences were found after correction."

        return interpretation

    def _interpret_holm_sidak_results(self, comparisons):
        """Interpret Holm-Šídák results."""
        significant_pairs = [c for c in comparisons if c.get('significant_holm', False)]

        interpretation = f"Holm-Šídák method identified {len(significant_pairs)} significant pairwise differences "
        interpretation += f"out of {len(comparisons)} total comparisons.\n\n"

        if significant_pairs:
            interpretation += "Significant differences found between:\n"
            for pair in significant_pairs:
                interpretation += f"- {pair['group1']} vs {pair['group2']} (p = {pair['p_raw']:.4f})\n"
        else:
            interpretation += "No significant pairwise differences were found."

        return interpretation

    def _interpret_bh_results(self, comparisons):
        """Interpret Benjamini-Hochberg results."""
        significant_pairs = [c for c in comparisons if c['significant_bh']]

        interpretation = f"Benjamini-Hochberg FDR control identified {len(significant_pairs)} significant pairwise differences "
        interpretation += f"out of {len(comparisons)} total comparisons.\n\n"

        if significant_pairs:
            interpretation += "Significant differences found between:\n"
            for pair in significant_pairs:
                interpretation += f"- {pair['group1']} vs {pair['group2']} (q = {pair['q_value']:.4f})\n"
        else:
            interpretation += "No significant pairwise differences were found."

        return interpretation

    def _interpret_dunn_results(self, comparisons):
        """Interpret Dunn test results."""
        significant_pairs = [c for c in comparisons if c['significant']]

        interpretation = f"Dunn test identified {len(significant_pairs)} significant pairwise differences "
        interpretation += f"out of {len(comparisons)} total comparisons.\n\n"

        if significant_pairs:
            interpretation += "Significant differences in rank distributions found between:\n"
            for pair in significant_pairs:
                interpretation += f"- {pair['group1']} vs {pair['group2']} (p = {pair['p_dunn']:.4f})\n"
        else:
            interpretation += "No significant pairwise differences in rank distributions were found."

        return interpretation

    def _interpret_games_howell_results(self, comparisons):
        """Interpret Games-Howell results."""
        significant_pairs = [c for c in comparisons if c['significant']]

        interpretation = f"Games-Howell test identified {len(significant_pairs)} significant pairwise differences "
        interpretation += f"out of {len(comparisons)} total comparisons.\n"
        interpretation += "This test accounts for unequal variances between groups.\n\n"

        if significant_pairs:
            interpretation += "Significant differences found between:\n"
            for pair in significant_pairs:
                interpretation += f"- {pair['group1']} vs {pair['group2']} (p = {pair['p_value']:.4f})\n"
        else:
            interpretation += "No significant pairwise differences were found."

        return interpretation

    def recommend_post_hoc_test(self, omnibus_test, assumptions_met=True, equal_variances=True):
        """
        Recommend appropriate post-hoc test based on conditions.

        Parameters:
        -----------
        omnibus_test : str
            The omnibus test that was significant
        assumptions_met : bool
            Whether parametric assumptions are met
        equal_variances : bool
            Whether groups have equal variances

        Returns:
        --------
        dict : Recommendation with rationale
        """
        if omnibus_test in ['One-way ANOVA', 'Repeated measures ANOVA']:
            if assumptions_met and equal_variances:
                return {
                    'recommended': 'tukey_hsd',
                    'alternative': 'bonferroni',
                    'rationale': 'Tukey HSD is optimal for equal variances and normal data'
                }
            elif assumptions_met and not equal_variances:
                return {
                    'recommended': 'games_howell',
                    'alternative': 'welch_anova_post_hoc',
                    'rationale': 'Games-Howell handles unequal variances appropriately'
                }
            else:
                return {
                    'recommended': 'dunn_test',
                    'alternative': 'bonferroni',
                    'rationale': 'Non-parametric post-hoc test for assumption violations'
                }
        elif omnibus_test == 'Kruskal-Wallis test':
            return {
                'recommended': 'dunn_test',
                'alternative': 'nemenyi_test',
                'rationale': 'Dunn test is the standard post-hoc for Kruskal-Wallis'
            }
        elif omnibus_test == 'Friedman test':
            return {
                'recommended': 'nemenyi_test',
                'alternative': 'wilcoxon_with_bonferroni',
                'rationale': 'Nemenyi test is designed for repeated measures non-parametric data'
            }
        else:
            return {
                'recommended': 'bonferroni',
                'alternative': 'holm_sidak',
                'rationale': 'Conservative multiple comparison correction'
            }