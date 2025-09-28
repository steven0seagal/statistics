"""
Assumption Checking Module
=========================

Validates statistical test assumptions and provides recommendations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import levene, bartlett, shapiro
import warnings

class AssumptionChecker:
    """
    Checks statistical test assumptions and provides detailed feedback.
    """

    def __init__(self):
        self.test_assumptions = {
            'Independent t-test': ['normality', 'equal_variances', 'independence'],
            'Welch\'s t-test': ['normality', 'independence'],
            'Paired t-test': ['normality_differences', 'independence'],
            'Mann-Whitney U test': ['independence'],
            'Wilcoxon signed-rank test': ['symmetry_differences', 'independence'],
            'One-way ANOVA': ['normality', 'equal_variances', 'independence'],
            'Kruskal-Wallis test': ['independence'],
            'Repeated measures ANOVA': ['normality', 'sphericity', 'independence'],
            'Friedman test': ['independence'],
            'Chi-squared test': ['expected_frequencies', 'independence'],
            'Fisher\'s exact test': ['independence'],
            'Pearson correlation': ['normality', 'linearity', 'independence'],
            'Spearman correlation': ['independence']
        }

    def check_assumptions(self, data, dependent_var, independent_var=None, test_name=None):
        """
        Check all relevant assumptions for the specified test.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        dependent_var : str
            Dependent variable column name
        independent_var : str, optional
            Independent variable column name
        test_name : str
            Name of the statistical test

        Returns:
        --------
        dict : Assumption check results
        """
        if test_name not in self.test_assumptions:
            return {'error': f'Unknown test: {test_name}'}

        assumptions_to_check = self.test_assumptions[test_name]
        results = {}

        for assumption in assumptions_to_check:
            if assumption == 'normality':
                results[assumption] = self._check_normality(data, dependent_var, independent_var)
            elif assumption == 'normality_differences':
                results[assumption] = self._check_normality_differences(data, dependent_var, independent_var)
            elif assumption == 'equal_variances':
                results[assumption] = self._check_equal_variances(data, dependent_var, independent_var)
            elif assumption == 'independence':
                results[assumption] = self._check_independence(data)
            elif assumption == 'symmetry_differences':
                results[assumption] = self._check_symmetry_differences(data, dependent_var, independent_var)
            elif assumption == 'sphericity':
                results[assumption] = self._check_sphericity(data, dependent_var, independent_var)
            elif assumption == 'expected_frequencies':
                results[assumption] = self._check_expected_frequencies(data, dependent_var, independent_var)
            elif assumption == 'linearity':
                results[assumption] = self._check_linearity(data, dependent_var, independent_var)

        return results

    def _check_normality(self, data, dependent_var, independent_var=None):
        """Check normality assumption"""
        result = {
            'assumption': 'Normality',
            'description': 'Data should be approximately normally distributed',
            'passed': True,
            'message': '',
            'details': {},
            'recommendations': []
        }

        try:
            if independent_var is None:
                # Single group normality
                variable_data = data[dependent_var].dropna()
                sw_stat, sw_p = shapiro(variable_data)

                result['details'] = {
                    'shapiro_wilk_statistic': sw_stat,
                    'shapiro_wilk_p_value': sw_p,
                    'sample_size': len(variable_data)
                }

                if sw_p < 0.05:
                    result['passed'] = False
                    result['message'] = f'Shapiro-Wilk test suggests non-normal distribution (p = {sw_p:.4f})'
                    result['recommendations'].append('Consider non-parametric alternative')
                    if len(variable_data) > 5000:
                        result['recommendations'].append('Large sample size: Shapiro-Wilk may be overly sensitive')
                else:
                    result['message'] = f'Shapiro-Wilk test supports normality assumption (p = {sw_p:.4f})'

            else:
                # Check normality for each group
                groups = data[independent_var].unique()
                group_results = {}

                for group in groups:
                    group_data = data[data[independent_var] == group][dependent_var].dropna()

                    if len(group_data) >= 3:  # Minimum for Shapiro-Wilk
                        sw_stat, sw_p = shapiro(group_data)
                        group_results[str(group)] = {
                            'shapiro_wilk_p': sw_p,
                            'n': len(group_data),
                            'normal': sw_p >= 0.05
                        }

                result['details']['groups'] = group_results

                # Overall assessment
                all_normal = all(group['normal'] for group in group_results.values())
                min_p = min(group['shapiro_wilk_p'] for group in group_results.values())

                if all_normal:
                    result['message'] = 'All groups appear normally distributed'
                else:
                    result['passed'] = False
                    result['message'] = f'One or more groups deviate from normality (min p = {min_p:.4f})'
                    result['recommendations'].append('Consider non-parametric alternative')

                # Check sample sizes
                min_n = min(group['n'] for group in group_results.values())
                if min_n < 30:
                    result['recommendations'].append('Small sample sizes: normality assumption is more critical')

        except Exception as e:
            result['passed'] = False
            result['message'] = f'Error checking normality: {str(e)}'

        return result

    def _check_normality_differences(self, data, dependent_var, independent_var):
        """Check normality of differences for paired tests"""
        result = {
            'assumption': 'Normality of differences',
            'description': 'Differences between paired observations should be normally distributed',
            'passed': True,
            'message': '',
            'details': {},
            'recommendations': []
        }

        try:
            # Assume paired data structure with two conditions
            groups = data[independent_var].unique()
            if len(groups) != 2:
                result['passed'] = False
                result['message'] = 'Cannot check differences: need exactly 2 conditions'
                return result

            group1_data = data[data[independent_var] == groups[0]][dependent_var]
            group2_data = data[data[independent_var] == groups[1]][dependent_var]

            if len(group1_data) != len(group2_data):
                result['passed'] = False
                result['message'] = 'Unequal group sizes: cannot compute paired differences'
                return result

            differences = group1_data.values - group2_data.values
            sw_stat, sw_p = shapiro(differences)

            result['details'] = {
                'shapiro_wilk_statistic': sw_stat,
                'shapiro_wilk_p_value': sw_p,
                'n_pairs': len(differences),
                'mean_difference': np.mean(differences),
                'std_difference': np.std(differences)
            }

            if sw_p < 0.05:
                result['passed'] = False
                result['message'] = f'Differences are not normally distributed (p = {sw_p:.4f})'
                result['recommendations'].append('Consider Wilcoxon signed-rank test')
            else:
                result['message'] = f'Differences appear normally distributed (p = {sw_p:.4f})'

        except Exception as e:
            result['passed'] = False
            result['message'] = f'Error checking normality of differences: {str(e)}'

        return result

    def _check_equal_variances(self, data, dependent_var, independent_var):
        """Check homogeneity of variances"""
        result = {
            'assumption': 'Equal variances (homoscedasticity)',
            'description': 'Groups should have similar variances',
            'passed': True,
            'message': '',
            'details': {},
            'recommendations': []
        }

        try:
            groups = data[independent_var].unique()
            group_data = [data[data[independent_var] == group][dependent_var].dropna() for group in groups]

            # Calculate group variances
            group_vars = [group.var() for group in group_data]
            group_stds = [group.std() for group in group_data]

            # Levene's test (more robust)
            levene_stat, levene_p = levene(*group_data)

            # Bartlett's test (assumes normality)
            bartlett_stat, bartlett_p = bartlett(*group_data)

            result['details'] = {
                'levene_statistic': levene_stat,
                'levene_p_value': levene_p,
                'bartlett_statistic': bartlett_stat,
                'bartlett_p_value': bartlett_p,
                'group_variances': {str(groups[i]): group_vars[i] for i in range(len(groups))},
                'group_std_devs': {str(groups[i]): group_stds[i] for i in range(len(groups))},
                'variance_ratio': max(group_vars) / min(group_vars) if min(group_vars) > 0 else float('inf')
            }

            # Rule of thumb: variance ratio should be < 4
            variance_ratio = result['details']['variance_ratio']

            if levene_p < 0.05:
                result['passed'] = False
                result['message'] = f'Levene\'s test suggests unequal variances (p = {levene_p:.4f})'
                result['recommendations'].append('Consider Welch\'s t-test or non-parametric alternative')
            elif variance_ratio > 4:
                result['passed'] = False
                result['message'] = f'Large variance ratio ({variance_ratio:.2f}) suggests unequal variances'
                result['recommendations'].append('Consider Welch\'s t-test')
            else:
                result['message'] = f'Variances appear equal (Levene p = {levene_p:.4f}, ratio = {variance_ratio:.2f})'

        except Exception as e:
            result['passed'] = False
            result['message'] = f'Error checking equal variances: {str(e)}'

        return result

    def _check_independence(self, data):
        """Check independence assumption"""
        result = {
            'assumption': 'Independence of observations',
            'description': 'Observations should be independent of each other',
            'passed': True,
            'message': 'Independence cannot be statistically tested - depends on study design',
            'details': {},
            'recommendations': [
                'Ensure random sampling',
                'Check for temporal or spatial clustering',
                'Consider experimental design factors'
            ]
        }

        # Look for potential indicators of non-independence
        if 'subject_id' in data.columns or 'id' in data.columns:
            result['recommendations'].append('Found ID column - check for repeated measures')

        if 'time' in data.columns or 'date' in data.columns:
            result['recommendations'].append('Found time variable - check for temporal correlation')

        if 'location' in data.columns or 'site' in data.columns:
            result['recommendations'].append('Found location variable - check for spatial correlation')

        return result

    def _check_symmetry_differences(self, data, dependent_var, independent_var):
        """Check symmetry of differences for Wilcoxon test"""
        result = {
            'assumption': 'Symmetry of differences',
            'description': 'Differences should be symmetrically distributed around zero',
            'passed': True,
            'message': '',
            'details': {},
            'recommendations': []
        }

        try:
            groups = data[independent_var].unique()
            if len(groups) != 2:
                result['passed'] = False
                result['message'] = 'Cannot check differences: need exactly 2 conditions'
                return result

            group1_data = data[data[independent_var] == groups[0]][dependent_var]
            group2_data = data[data[independent_var] == groups[1]][dependent_var]

            if len(group1_data) != len(group2_data):
                result['passed'] = False
                result['message'] = 'Unequal group sizes: cannot compute paired differences'
                return result

            differences = group1_data.values - group2_data.values

            # Calculate skewness
            skewness = stats.skew(differences)

            result['details'] = {
                'skewness': skewness,
                'n_pairs': len(differences),
                'median_difference': np.median(differences)
            }

            if abs(skewness) > 1:
                result['passed'] = False
                result['message'] = f'Differences are highly skewed (skewness = {skewness:.3f})'
                result['recommendations'].append('Consider transformation or different test')
            elif abs(skewness) > 0.5:
                result['message'] = f'Differences are moderately skewed (skewness = {skewness:.3f})'
                result['recommendations'].append('Wilcoxon test is still appropriate but interpret with caution')
            else:
                result['message'] = f'Differences appear reasonably symmetric (skewness = {skewness:.3f})'

        except Exception as e:
            result['passed'] = False
            result['message'] = f'Error checking symmetry: {str(e)}'

        return result

    def _check_sphericity(self, data, dependent_var, independent_var):
        """Check sphericity assumption for repeated measures"""
        result = {
            'assumption': 'Sphericity',
            'description': 'Variances of differences between conditions should be equal',
            'passed': True,
            'message': 'Sphericity testing requires specialized repeated measures design',
            'details': {},
            'recommendations': [
                'Use Greenhouse-Geisser correction if sphericity is violated',
                'Consider multivariate approach (MANOVA) as alternative'
            ]
        }

        # Note: Full sphericity testing requires specialized repeated measures structure
        # This is a simplified check
        return result

    def _check_expected_frequencies(self, data, dependent_var, independent_var):
        """Check expected frequencies for chi-squared test"""
        result = {
            'assumption': 'Adequate expected frequencies',
            'description': 'Expected cell counts should be at least 5',
            'passed': True,
            'message': '',
            'details': {},
            'recommendations': []
        }

        try:
            # Create contingency table
            contingency_table = pd.crosstab(data[independent_var], data[dependent_var])

            # Calculate expected frequencies
            row_totals = contingency_table.sum(axis=1)
            col_totals = contingency_table.sum(axis=0)
            total = contingency_table.sum().sum()

            expected = np.outer(row_totals, col_totals) / total

            result['details'] = {
                'observed_table': contingency_table.to_dict(),
                'expected_frequencies': expected.tolist(),
                'min_expected': expected.min(),
                'cells_below_5': (expected < 5).sum(),
                'total_cells': expected.size
            }

            min_expected = expected.min()
            cells_below_5 = (expected < 5).sum()

            if min_expected < 1:
                result['passed'] = False
                result['message'] = f'Very low expected frequencies (min = {min_expected:.2f})'
                result['recommendations'].append('Use Fisher\'s exact test instead')
            elif cells_below_5 > 0:
                if cells_below_5 / expected.size > 0.2:  # More than 20% of cells
                    result['passed'] = False
                    result['message'] = f'{cells_below_5} cells have expected count < 5'
                    result['recommendations'].append('Consider Fisher\'s exact test or combine categories')
                else:
                    result['message'] = f'{cells_below_5} cells have expected count < 5 (acceptable)'
            else:
                result['message'] = 'All expected frequencies are adequate'

        except Exception as e:
            result['passed'] = False
            result['message'] = f'Error checking expected frequencies: {str(e)}'

        return result

    def _check_linearity(self, data, dependent_var, independent_var):
        """Check linearity assumption for correlation/regression"""
        result = {
            'assumption': 'Linearity',
            'description': 'Relationship between variables should be linear',
            'passed': True,
            'message': '',
            'details': {},
            'recommendations': []
        }

        try:
            # Calculate correlation coefficient
            var1_data = data[dependent_var].dropna()
            var2_data = data[independent_var].dropna()

            # Ensure same length
            min_len = min(len(var1_data), len(var2_data))
            var1_data = var1_data.iloc[:min_len]
            var2_data = var2_data.iloc[:min_len]

            # Pearson vs Spearman correlation comparison
            pearson_r, _ = stats.pearsonr(var1_data, var2_data)
            spearman_r, _ = stats.spearmanr(var1_data, var2_data)

            result['details'] = {
                'pearson_correlation': pearson_r,
                'spearman_correlation': spearman_r,
                'correlation_difference': abs(pearson_r - spearman_r),
                'n': len(var1_data)
            }

            # If Pearson and Spearman are very different, suggests non-linearity
            correlation_diff = abs(pearson_r - spearman_r)

            if correlation_diff > 0.2:
                result['passed'] = False
                result['message'] = f'Large difference between Pearson ({pearson_r:.3f}) and Spearman ({spearman_r:.3f}) correlations'
                result['recommendations'].append('Relationship may be non-linear - consider Spearman correlation')
            else:
                result['message'] = f'Pearson and Spearman correlations are similar (difference = {correlation_diff:.3f})'

        except Exception as e:
            result['passed'] = False
            result['message'] = f'Error checking linearity: {str(e)}'

        return result

    def get_sample_size_recommendations(self, test_name, current_n, effect_size='medium'):
        """
        Provide sample size recommendations for adequate power.

        Parameters:
        -----------
        test_name : str
            Name of the statistical test
        current_n : int
            Current sample size
        effect_size : str
            Expected effect size ('small', 'medium', 'large')

        Returns:
        --------
        dict : Sample size recommendations
        """
        # Sample size recommendations for 80% power, alpha = 0.05
        recommendations = {
            'Independent t-test': {'small': 393, 'medium': 64, 'large': 26},
            'Paired t-test': {'small': 199, 'medium': 34, 'large': 15},
            'One-way ANOVA': {'small': 322, 'medium': 52, 'large': 21},
            'Chi-squared test': {'small': 785, 'medium': 87, 'large': 26},
            'Correlation': {'small': 783, 'medium': 84, 'large': 28}
        }

        # Default recommendations
        default_rec = {'small': 200, 'medium': 50, 'large': 20}

        test_rec = recommendations.get(test_name, default_rec)
        recommended_n = test_rec.get(effect_size, test_rec['medium'])

        result = {
            'test_name': test_name,
            'current_n': current_n,
            'recommended_n': recommended_n,
            'effect_size': effect_size,
            'adequate_power': current_n >= recommended_n
        }

        if current_n < recommended_n:
            result['message'] = f'Consider increasing sample size to {recommended_n} for adequate power'
        else:
            result['message'] = 'Sample size appears adequate for good statistical power'

        return result