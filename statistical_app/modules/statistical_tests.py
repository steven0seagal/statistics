"""
Statistical Tests Module
========================

Implements core statistical tests with comprehensive result reporting.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import warnings

class StatisticalTests:
    """
    Core statistical test implementations with standardized output format.
    """

    def __init__(self):
        self.supported_tests = [
            'Independent t-test',
            'Welch\'s t-test',
            'Paired t-test',
            'Mann-Whitney U test',
            'Wilcoxon signed-rank test',
            'One-way ANOVA',
            'Kruskal-Wallis test',
            'Repeated measures ANOVA',
            'Friedman test',
            'Chi-squared test',
            'Fisher\'s exact test',
            'McNemar\'s test',
            'Cochran\'s Q test',
            'Pearson correlation',
            'Spearman correlation'
        ]

    def perform_test(self, data, dependent_var, independent_var=None, test_name=None, **kwargs):
        """
        Perform the specified statistical test.

        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset
        dependent_var : str
            Name of the dependent variable column
        independent_var : str, optional
            Name of the independent variable column
        test_name : str
            Name of the statistical test to perform
        **kwargs : additional parameters for specific tests

        Returns:
        --------
        dict : Test results including statistic, p-value, effect size, and interpretation
        """

        if test_name not in self.supported_tests:
            raise ValueError(f"Test '{test_name}' not supported. Available tests: {self.supported_tests}")

        # Clean data
        clean_data = self._clean_data(data, dependent_var, independent_var)

        # Route to appropriate test method
        test_methods = {
            'Independent t-test': self._independent_ttest,
            'Welch\'s t-test': self._welch_ttest,
            'Paired t-test': self._paired_ttest,
            'Mann-Whitney U test': self._mann_whitney_test,
            'Wilcoxon signed-rank test': self._wilcoxon_test,
            'One-way ANOVA': self._one_way_anova,
            'Kruskal-Wallis test': self._kruskal_wallis_test,
            'Repeated measures ANOVA': self._repeated_measures_anova,
            'Friedman test': self._friedman_test,
            'Chi-squared test': self._chi_squared_test,
            'Fisher\'s exact test': self._fisher_exact_test,
            'McNemar\'s test': self._mcnemar_test,
            'Pearson correlation': self._pearson_correlation,
            'Spearman correlation': self._spearman_correlation
        }

        test_method = test_methods[test_name]
        return test_method(clean_data, dependent_var, independent_var, **kwargs)

    def _clean_data(self, data, dependent_var, independent_var=None):
        """Remove missing values and prepare data for analysis"""
        if independent_var:
            clean_data = data[[dependent_var, independent_var]].dropna()
        else:
            clean_data = data[[dependent_var]].dropna()
        return clean_data

    def _independent_ttest(self, data, dependent_var, independent_var, **kwargs):
        """Perform independent samples t-test"""
        groups = data[independent_var].unique()
        if len(groups) != 2:
            raise ValueError("Independent t-test requires exactly 2 groups")

        group1_data = data[data[independent_var] == groups[0]][dependent_var]
        group2_data = data[data[independent_var] == groups[1]][dependent_var]

        # Perform test
        statistic, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=True)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() +
                             (len(group2_data) - 1) * group2_data.var()) /
                            (len(group1_data) + len(group2_data) - 2))
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std

        return {
            'test_name': 'Independent t-test',
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': cohens_d,
            'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
            'group1_mean': group1_data.mean(),
            'group1_std': group1_data.std(),
            'group1_n': len(group1_data),
            'group2_mean': group2_data.mean(),
            'group2_std': group2_data.std(),
            'group2_n': len(group2_data),
            'degrees_freedom': len(group1_data) + len(group2_data) - 2,
            'groups': groups.tolist()
        }

    def _welch_ttest(self, data, dependent_var, independent_var, **kwargs):
        """Perform Welch's t-test (unequal variances)"""
        groups = data[independent_var].unique()
        if len(groups) != 2:
            raise ValueError("Welch's t-test requires exactly 2 groups")

        group1_data = data[data[independent_var] == groups[0]][dependent_var]
        group2_data = data[data[independent_var] == groups[1]][dependent_var]

        # Perform test
        statistic, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)

        # Calculate effect size (Cohen's d with pooled standard deviation)
        s1, s2 = group1_data.std(), group2_data.std()
        n1, n2 = len(group1_data), len(group2_data)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std

        # Calculate Welch's degrees of freedom
        df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))

        return {
            'test_name': 'Welch\'s t-test',
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': cohens_d,
            'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
            'group1_mean': group1_data.mean(),
            'group1_std': group1_data.std(),
            'group1_n': len(group1_data),
            'group2_mean': group2_data.mean(),
            'group2_std': group2_data.std(),
            'group2_n': len(group2_data),
            'degrees_freedom': df,
            'groups': groups.tolist()
        }

    def _paired_ttest(self, data, dependent_var, independent_var, **kwargs):
        """Perform paired samples t-test"""
        # Assume data is already in paired format with identifier
        # For now, treat as before/after design
        groups = data[independent_var].unique()
        if len(groups) != 2:
            raise ValueError("Paired t-test requires exactly 2 conditions")

        group1_data = data[data[independent_var] == groups[0]][dependent_var]
        group2_data = data[data[independent_var] == groups[1]][dependent_var]

        if len(group1_data) != len(group2_data):
            raise ValueError("Paired t-test requires equal sample sizes")

        # Perform test
        statistic, p_value = stats.ttest_rel(group1_data, group2_data)

        # Calculate effect size (Cohen's d for paired data)
        differences = group1_data.values - group2_data.values
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)

        return {
            'test_name': 'Paired t-test',
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': cohens_d,
            'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences, ddof=1),
            'n_pairs': len(differences),
            'degrees_freedom': len(differences) - 1,
            'groups': groups.tolist()
        }

    def _mann_whitney_test(self, data, dependent_var, independent_var, **kwargs):
        """Perform Mann-Whitney U test"""
        groups = data[independent_var].unique()
        if len(groups) != 2:
            raise ValueError("Mann-Whitney U test requires exactly 2 groups")

        group1_data = data[data[independent_var] == groups[0]][dependent_var]
        group2_data = data[data[independent_var] == groups[1]][dependent_var]

        # Perform test
        statistic, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')

        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(group1_data), len(group2_data)
        effect_size = 1 - (2 * statistic) / (n1 * n2)

        return {
            'test_name': 'Mann-Whitney U test',
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'effect_size_interpretation': self._interpret_rank_biserial(effect_size),
            'group1_median': group1_data.median(),
            'group1_n': len(group1_data),
            'group2_median': group2_data.median(),
            'group2_n': len(group2_data),
            'groups': groups.tolist()
        }

    def _wilcoxon_test(self, data, dependent_var, independent_var, **kwargs):
        """Perform Wilcoxon signed-rank test"""
        groups = data[independent_var].unique()
        if len(groups) != 2:
            raise ValueError("Wilcoxon test requires exactly 2 conditions")

        group1_data = data[data[independent_var] == groups[0]][dependent_var]
        group2_data = data[data[independent_var] == groups[1]][dependent_var]

        if len(group1_data) != len(group2_data):
            raise ValueError("Wilcoxon test requires equal sample sizes")

        # Perform test
        statistic, p_value = stats.wilcoxon(group1_data, group2_data)

        # Calculate effect size (matched pairs rank-biserial correlation)
        differences = group1_data.values - group2_data.values
        n_pairs = len(differences)
        effect_size = statistic / (n_pairs * (n_pairs + 1) / 4)

        return {
            'test_name': 'Wilcoxon signed-rank test',
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'effect_size_interpretation': self._interpret_rank_biserial(effect_size),
            'median_difference': np.median(differences),
            'n_pairs': n_pairs,
            'groups': groups.tolist()
        }

    def _one_way_anova(self, data, dependent_var, independent_var, **kwargs):
        """Perform one-way ANOVA"""
        groups = data[independent_var].unique()
        group_data = [data[data[independent_var] == group][dependent_var] for group in groups]

        # Perform test
        statistic, p_value = stats.f_oneway(*group_data)

        # Calculate eta-squared (effect size)
        grand_mean = data[dependent_var].mean()
        n_total = len(data)
        k = len(groups)

        ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in group_data)
        ss_total = sum((data[dependent_var] - grand_mean)**2)
        eta_squared = ss_between / ss_total

        # Degrees of freedom
        df_between = k - 1
        df_within = n_total - k

        return {
            'test_name': 'One-way ANOVA',
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': eta_squared,
            'effect_size_interpretation': self._interpret_eta_squared(eta_squared),
            'df_between': df_between,
            'df_within': df_within,
            'group_means': [group.mean() for group in group_data],
            'group_stds': [group.std() for group in group_data],
            'group_ns': [len(group) for group in group_data],
            'groups': groups.tolist()
        }

    def _kruskal_wallis_test(self, data, dependent_var, independent_var, **kwargs):
        """Perform Kruskal-Wallis test"""
        groups = data[independent_var].unique()
        group_data = [data[data[independent_var] == group][dependent_var] for group in groups]

        # Perform test
        statistic, p_value = stats.kruskal(*group_data)

        # Calculate effect size (eta-squared analog for Kruskal-Wallis)
        n_total = len(data)
        eta_squared = (statistic - len(groups) + 1) / (n_total - len(groups))

        return {
            'test_name': 'Kruskal-Wallis test',
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': eta_squared,
            'effect_size_interpretation': self._interpret_eta_squared(eta_squared),
            'degrees_freedom': len(groups) - 1,
            'group_medians': [group.median() for group in group_data],
            'group_ns': [len(group) for group in group_data],
            'groups': groups.tolist()
        }

    def _repeated_measures_anova(self, data, dependent_var, independent_var, **kwargs):
        """Perform repeated measures ANOVA (simplified version)"""
        # This is a simplified implementation
        # For full repeated measures ANOVA, would need subject identifiers
        return self._one_way_anova(data, dependent_var, independent_var, **kwargs)

    def _friedman_test(self, data, dependent_var, independent_var, **kwargs):
        """Perform Friedman test"""
        groups = data[independent_var].unique()
        group_data = [data[data[independent_var] == group][dependent_var] for group in groups]

        # Check if data is properly structured for Friedman test
        if len(set(len(group) for group in group_data)) > 1:
            raise ValueError("Friedman test requires equal sample sizes across all groups")

        # Perform test
        statistic, p_value = stats.friedmanchisquare(*group_data)

        # Calculate effect size (Kendall's W)
        n = len(group_data[0])  # number of subjects
        k = len(groups)  # number of conditions
        kendalls_w = statistic / (n * (k - 1))

        return {
            'test_name': 'Friedman test',
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': kendalls_w,
            'effect_size_interpretation': self._interpret_kendalls_w(kendalls_w),
            'degrees_freedom': len(groups) - 1,
            'group_medians': [group.median() for group in group_data],
            'n_subjects': n,
            'groups': groups.tolist()
        }

    def _chi_squared_test(self, data, dependent_var, independent_var, **kwargs):
        """Perform Chi-squared test of independence"""
        # Create contingency table
        contingency_table = pd.crosstab(data[independent_var], data[dependent_var])

        # Perform test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Calculate effect size (Cramér's V)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

        return {
            'test_name': 'Chi-squared test',
            'statistic': chi2,
            'p_value': p_value,
            'effect_size': cramers_v,
            'effect_size_interpretation': self._interpret_cramers_v(cramers_v),
            'degrees_freedom': dof,
            'contingency_table': contingency_table.to_dict(),
            'expected_frequencies': expected.tolist()
        }

    def _fisher_exact_test(self, data, dependent_var, independent_var, **kwargs):
        """Perform Fisher's exact test"""
        # Create 2x2 contingency table
        contingency_table = pd.crosstab(data[independent_var], data[dependent_var])

        if contingency_table.shape != (2, 2):
            raise ValueError("Fisher's exact test requires a 2x2 contingency table")

        # Perform test
        odds_ratio, p_value = fisher_exact(contingency_table)

        return {
            'test_name': 'Fisher\'s exact test',
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'contingency_table': contingency_table.to_dict()
        }

    def _mcnemar_test(self, data, dependent_var, independent_var, **kwargs):
        """Perform McNemar's test"""
        # This requires paired binary data
        # Implementation would depend on data structure
        raise NotImplementedError("McNemar's test implementation pending")

    def _pearson_correlation(self, data, dependent_var, independent_var, **kwargs):
        """Perform Pearson correlation"""
        # For correlation, both variables are treated equally
        var1_data = data[dependent_var].dropna()
        var2_data = data[independent_var].dropna()

        # Ensure same length
        min_len = min(len(var1_data), len(var2_data))
        var1_data = var1_data.iloc[:min_len]
        var2_data = var2_data.iloc[:min_len]

        # Perform test
        correlation, p_value = stats.pearsonr(var1_data, var2_data)

        return {
            'test_name': 'Pearson correlation',
            'correlation': correlation,
            'p_value': p_value,
            'effect_size': abs(correlation),
            'effect_size_interpretation': self._interpret_correlation(abs(correlation)),
            'n': len(var1_data),
            'r_squared': correlation**2
        }

    def _spearman_correlation(self, data, dependent_var, independent_var, **kwargs):
        """Perform Spearman rank correlation"""
        var1_data = data[dependent_var].dropna()
        var2_data = data[independent_var].dropna()

        # Ensure same length
        min_len = min(len(var1_data), len(var2_data))
        var1_data = var1_data.iloc[:min_len]
        var2_data = var2_data.iloc[:min_len]

        # Perform test
        correlation, p_value = stats.spearmanr(var1_data, var2_data)

        return {
            'test_name': 'Spearman correlation',
            'correlation': correlation,
            'p_value': p_value,
            'effect_size': abs(correlation),
            'effect_size_interpretation': self._interpret_correlation(abs(correlation)),
            'n': len(var1_data)
        }

    # Effect size interpretation methods
    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_eta_squared(self, eta_sq):
        """Interpret eta-squared effect size"""
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"

    def _interpret_correlation(self, r):
        """Interpret correlation coefficient"""
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        else:
            return "large"

    def _interpret_cramers_v(self, v):
        """Interpret Cramér's V effect size"""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"

    def _interpret_rank_biserial(self, r):
        """Interpret rank-biserial correlation"""
        return self._interpret_correlation(abs(r))

    def _interpret_kendalls_w(self, w):
        """Interpret Kendall's W"""
        if w < 0.1:
            return "negligible"
        elif w < 0.3:
            return "small"
        elif w < 0.5:
            return "medium"
        else:
            return "large"