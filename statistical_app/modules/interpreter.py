"""
Results Interpreter Module
=========================

Provides intelligent interpretation of statistical test results.
"""

import numpy as np

class ResultsInterpreter:
    """
    Interprets statistical test results and provides contextual explanations.
    """

    def __init__(self):
        self.significance_levels = {
            0.001: "highly significant",
            0.01: "very significant",
            0.05: "significant",
            0.1: "marginally significant",
            1.0: "not significant"
        }

    def interpret_results(self, results, test_name, alpha=0.05, context=None):
        """
        Provide comprehensive interpretation of test results.

        Parameters:
        -----------
        results : dict
            Test results from statistical_tests module
        test_name : str
            Name of the statistical test
        alpha : float
            Significance level (default 0.05)
        context : str, optional
            Biological or research context

        Returns:
        --------
        str : Detailed interpretation of results
        """

        interpretation_methods = {
            'Independent t-test': self._interpret_two_sample_test,
            'Welch\'s t-test': self._interpret_two_sample_test,
            'Paired t-test': self._interpret_paired_test,
            'Mann-Whitney U test': self._interpret_nonparametric_two_sample,
            'Wilcoxon signed-rank test': self._interpret_nonparametric_paired,
            'One-way ANOVA': self._interpret_anova,
            'Kruskal-Wallis test': self._interpret_nonparametric_multiple,
            'Repeated measures ANOVA': self._interpret_repeated_measures,
            'Friedman test': self._interpret_nonparametric_repeated,
            'Chi-squared test': self._interpret_chi_squared,
            'Fisher\'s exact test': self._interpret_fisher_exact,
            'Pearson correlation': self._interpret_correlation,
            'Spearman correlation': self._interpret_correlation
        }

        interpreter = interpretation_methods.get(test_name, self._interpret_generic)
        return interpreter(results, alpha, context)

    def _get_significance_level(self, p_value):
        """Determine significance level description"""
        for threshold, description in self.significance_levels.items():
            if p_value <= threshold:
                return description
        return "not significant"

    def _interpret_effect_size(self, effect_size, effect_type, test_name):
        """Interpret effect size magnitude"""
        if effect_size is None:
            return ""

        if effect_type in ['cohens_d', 'correlation']:
            if abs(effect_size) < 0.2:
                magnitude = "negligible"
            elif abs(effect_size) < 0.5:
                magnitude = "small"
            elif abs(effect_size) < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"
        elif effect_type == 'eta_squared':
            if effect_size < 0.01:
                magnitude = "negligible"
            elif effect_size < 0.06:
                magnitude = "small"
            elif effect_size < 0.14:
                magnitude = "medium"
            else:
                magnitude = "large"
        else:
            magnitude = "unknown"

        return f" The effect size ({effect_size:.3f}) indicates a {magnitude} effect."

    def _interpret_two_sample_test(self, results, alpha, context):
        """Interpret independent or Welch's t-test results"""
        p_value = results['p_value']
        statistic = results['statistic']
        effect_size = results.get('effect_size')
        groups = results.get('groups', ['Group 1', 'Group 2'])

        significance = self._get_significance_level(p_value)

        if p_value <= alpha:
            conclusion = f"reject the null hypothesis"
            direction = "higher" if statistic > 0 else "lower"
            practical_conclusion = f"There is a statistically {significance} difference between {groups[0]} and {groups[1]} (p = {p_value:.4f}). The mean for {groups[0]} is {direction} than {groups[1]}."
        else:
            conclusion = f"fail to reject the null hypothesis"
            practical_conclusion = f"There is no statistically significant difference between {groups[0]} and {groups[1]} (p = {p_value:.4f})."

        effect_interpretation = self._interpret_effect_size(effect_size, 'cohens_d', results['test_name'])

        interpretation = f"""
        **Statistical Decision:** Based on the p-value ({p_value:.4f}) and significance level (α = {alpha}), we {conclusion}.

        **Practical Interpretation:** {practical_conclusion}{effect_interpretation}

        **Technical Details:**
        - Test statistic (t) = {statistic:.4f}
        - Degrees of freedom = {results.get('degrees_freedom', 'N/A')}
        - Group 1 ({groups[0]}): Mean = {results.get('group1_mean', 'N/A'):.3f}, SD = {results.get('group1_std', 'N/A'):.3f}, n = {results.get('group1_n', 'N/A')}
        - Group 2 ({groups[1]}): Mean = {results.get('group2_mean', 'N/A'):.3f}, SD = {results.get('group2_std', 'N/A'):.3f}, n = {results.get('group2_n', 'N/A')}
        """

        if context:
            interpretation += f"\n\n**Research Context:** {context}"

        return interpretation.strip()

    def _interpret_paired_test(self, results, alpha, context):
        """Interpret paired t-test results"""
        p_value = results['p_value']
        statistic = results['statistic']
        effect_size = results.get('effect_size')
        mean_diff = results.get('mean_difference')

        significance = self._get_significance_level(p_value)

        if p_value <= alpha:
            conclusion = f"reject the null hypothesis"
            direction = "increase" if mean_diff > 0 else "decrease"
            practical_conclusion = f"There is a statistically {significance} {direction} from the first to second measurement (p = {p_value:.4f}). The mean difference is {mean_diff:.3f}."
        else:
            conclusion = f"fail to reject the null hypothesis"
            practical_conclusion = f"There is no statistically significant change between the paired measurements (p = {p_value:.4f})."

        effect_interpretation = self._interpret_effect_size(effect_size, 'cohens_d', results['test_name'])

        interpretation = f"""
        **Statistical Decision:** Based on the p-value ({p_value:.4f}) and significance level (α = {alpha}), we {conclusion}.

        **Practical Interpretation:** {practical_conclusion}{effect_interpretation}

        **Technical Details:**
        - Test statistic (t) = {statistic:.4f}
        - Degrees of freedom = {results.get('degrees_freedom', 'N/A')}
        - Mean difference = {mean_diff:.3f}
        - Standard deviation of differences = {results.get('std_difference', 'N/A'):.3f}
        - Number of pairs = {results.get('n_pairs', 'N/A')}
        """

        return interpretation.strip()

    def _interpret_nonparametric_two_sample(self, results, alpha, context):
        """Interpret Mann-Whitney U test results"""
        p_value = results['p_value']
        statistic = results['statistic']
        effect_size = results.get('effect_size')
        groups = results.get('groups', ['Group 1', 'Group 2'])

        significance = self._get_significance_level(p_value)

        if p_value <= alpha:
            conclusion = f"reject the null hypothesis"
            practical_conclusion = f"There is a statistically {significance} difference in the distribution of ranks between {groups[0]} and {groups[1]} (p = {p_value:.4f})."
        else:
            conclusion = f"fail to reject the null hypothesis"
            practical_conclusion = f"There is no statistically significant difference in the distribution of ranks between {groups[0]} and {groups[1]} (p = {p_value:.4f})."

        effect_interpretation = self._interpret_effect_size(effect_size, 'correlation', results['test_name'])

        interpretation = f"""
        **Statistical Decision:** Based on the p-value ({p_value:.4f}) and significance level (α = {alpha}), we {conclusion}.

        **Practical Interpretation:** {practical_conclusion}{effect_interpretation}

        **Technical Details:**
        - Mann-Whitney U statistic = {statistic}
        - Group 1 ({groups[0]}): Median = {results.get('group1_median', 'N/A')}, n = {results.get('group1_n', 'N/A')}
        - Group 2 ({groups[1]}): Median = {results.get('group2_median', 'N/A')}, n = {results.get('group2_n', 'N/A')}

        **Note:** This non-parametric test compares the distributions rather than just the means, making it appropriate when normality assumptions are violated.
        """

        return interpretation.strip()

    def _interpret_nonparametric_paired(self, results, alpha, context):
        """Interpret Wilcoxon signed-rank test results"""
        p_value = results['p_value']
        statistic = results['statistic']
        effect_size = results.get('effect_size')
        median_diff = results.get('median_difference')

        significance = self._get_significance_level(p_value)

        if p_value <= alpha:
            conclusion = f"reject the null hypothesis"
            direction = "increase" if median_diff > 0 else "decrease"
            practical_conclusion = f"There is a statistically {significance} {direction} in the median from the first to second measurement (p = {p_value:.4f})."
        else:
            conclusion = f"fail to reject the null hypothesis"
            practical_conclusion = f"There is no statistically significant change in the median between paired measurements (p = {p_value:.4f})."

        effect_interpretation = self._interpret_effect_size(effect_size, 'correlation', results['test_name'])

        interpretation = f"""
        **Statistical Decision:** Based on the p-value ({p_value:.4f}) and significance level (α = {alpha}), we {conclusion}.

        **Practical Interpretation:** {practical_conclusion}{effect_interpretation}

        **Technical Details:**
        - Wilcoxon signed-rank statistic = {statistic}
        - Median difference = {median_diff:.3f}
        - Number of pairs = {results.get('n_pairs', 'N/A')}

        **Note:** This non-parametric test is appropriate when the differences are not normally distributed.
        """

        return interpretation.strip()

    def _interpret_anova(self, results, alpha, context):
        """Interpret one-way ANOVA results"""
        p_value = results['p_value']
        statistic = results['statistic']
        effect_size = results.get('effect_size')
        groups = results.get('groups', [])

        significance = self._get_significance_level(p_value)

        if p_value <= alpha:
            conclusion = f"reject the null hypothesis"
            practical_conclusion = f"There is a statistically {significance} difference among the group means (p = {p_value:.4f}). At least one group differs significantly from the others."
            followup = " **Post-hoc tests are needed to determine which specific groups differ.**"
        else:
            conclusion = f"fail to reject the null hypothesis"
            practical_conclusion = f"There is no statistically significant difference among the group means (p = {p_value:.4f})."
            followup = ""

        effect_interpretation = self._interpret_effect_size(effect_size, 'eta_squared', results['test_name'])

        group_summary = ""
        if 'group_means' in results and 'group_stds' in results:
            group_summary = "\n**Group Summary:**\n"
            for i, group in enumerate(groups):
                mean = results['group_means'][i]
                std = results['group_stds'][i]
                n = results['group_ns'][i] if 'group_ns' in results else 'N/A'
                group_summary += f"- {group}: Mean = {mean:.3f}, SD = {std:.3f}, n = {n}\n"

        interpretation = f"""
        **Statistical Decision:** Based on the p-value ({p_value:.4f}) and significance level (α = {alpha}), we {conclusion}.

        **Practical Interpretation:** {practical_conclusion}{effect_interpretation}{followup}

        **Technical Details:**
        - F-statistic = {statistic:.4f}
        - Degrees of freedom = {results.get('df_between', 'N/A')} (between), {results.get('df_within', 'N/A')} (within)
        {group_summary}
        **Note:** ANOVA tests for any difference among groups but doesn't specify which groups differ.
        """

        return interpretation.strip()

    def _interpret_nonparametric_multiple(self, results, alpha, context):
        """Interpret Kruskal-Wallis test results"""
        p_value = results['p_value']
        statistic = results['statistic']
        groups = results.get('groups', [])

        significance = self._get_significance_level(p_value)

        if p_value <= alpha:
            conclusion = f"reject the null hypothesis"
            practical_conclusion = f"There is a statistically {significance} difference among the group distributions (p = {p_value:.4f}). At least one group has a different distribution."
            followup = " **Post-hoc tests (e.g., Dunn's test) are needed to determine which specific groups differ.**"
        else:
            conclusion = f"fail to reject the null hypothesis"
            practical_conclusion = f"There is no statistically significant difference among the group distributions (p = {p_value:.4f})."
            followup = ""

        group_summary = ""
        if 'group_medians' in results:
            group_summary = "\n**Group Summary:**\n"
            for i, group in enumerate(groups):
                median = results['group_medians'][i]
                n = results['group_ns'][i] if 'group_ns' in results else 'N/A'
                group_summary += f"- {group}: Median = {median:.3f}, n = {n}\n"

        interpretation = f"""
        **Statistical Decision:** Based on the p-value ({p_value:.4f}) and significance level (α = {alpha}), we {conclusion}.

        **Practical Interpretation:** {practical_conclusion}{followup}

        **Technical Details:**
        - Kruskal-Wallis H statistic = {statistic:.4f}
        - Degrees of freedom = {results.get('degrees_freedom', 'N/A')}
        {group_summary}
        **Note:** This non-parametric test compares distributions and is appropriate when ANOVA assumptions are violated.
        """

        return interpretation.strip()

    def _interpret_repeated_measures(self, results, alpha, context):
        """Interpret repeated measures ANOVA results"""
        # For now, use regular ANOVA interpretation
        interpretation = self._interpret_anova(results, alpha, context)
        interpretation += "\n\n**Note:** This analysis accounts for the correlation between repeated measurements on the same subjects."
        return interpretation

    def _interpret_nonparametric_repeated(self, results, alpha, context):
        """Interpret Friedman test results"""
        p_value = results['p_value']
        statistic = results['statistic']
        groups = results.get('groups', [])

        significance = self._get_significance_level(p_value)

        if p_value <= alpha:
            conclusion = f"reject the null hypothesis"
            practical_conclusion = f"There is a statistically {significance} difference among the repeated measurements (p = {p_value:.4f})."
            followup = " **Post-hoc tests are needed to determine which specific time points or conditions differ.**"
        else:
            conclusion = f"fail to reject the null hypothesis"
            practical_conclusion = f"There is no statistically significant difference among the repeated measurements (p = {p_value:.4f})."
            followup = ""

        interpretation = f"""
        **Statistical Decision:** Based on the p-value ({p_value:.4f}) and significance level (α = {alpha}), we {conclusion}.

        **Practical Interpretation:** {practical_conclusion}{followup}

        **Technical Details:**
        - Friedman χ² statistic = {statistic:.4f}
        - Degrees of freedom = {results.get('degrees_freedom', 'N/A')}
        - Number of subjects = {results.get('n_subjects', 'N/A')}

        **Note:** This non-parametric test is the repeated measures equivalent of the Kruskal-Wallis test.
        """

        return interpretation.strip()

    def _interpret_chi_squared(self, results, alpha, context):
        """Interpret Chi-squared test results"""
        p_value = results['p_value']
        statistic = results['statistic']
        effect_size = results.get('effect_size')

        significance = self._get_significance_level(p_value)

        if p_value <= alpha:
            conclusion = f"reject the null hypothesis"
            practical_conclusion = f"There is a statistically {significance} association between the two categorical variables (p = {p_value:.4f})."
        else:
            conclusion = f"fail to reject the null hypothesis"
            practical_conclusion = f"There is no statistically significant association between the two categorical variables (p = {p_value:.4f})."

        effect_interpretation = self._interpret_effect_size(effect_size, 'cramers_v', results['test_name'])

        interpretation = f"""
        **Statistical Decision:** Based on the p-value ({p_value:.4f}) and significance level (α = {alpha}), we {conclusion}.

        **Practical Interpretation:** {practical_conclusion}{effect_interpretation}

        **Technical Details:**
        - Chi-squared statistic = {statistic:.4f}
        - Degrees of freedom = {results.get('degrees_freedom', 'N/A')}

        **Note:** This test examines whether the distribution of one categorical variable differs across levels of another categorical variable.
        """

        return interpretation.strip()

    def _interpret_fisher_exact(self, results, alpha, context):
        """Interpret Fisher's exact test results"""
        p_value = results['p_value']
        odds_ratio = results.get('odds_ratio')

        significance = self._get_significance_level(p_value)

        if p_value <= alpha:
            conclusion = f"reject the null hypothesis"
            practical_conclusion = f"There is a statistically {significance} association between the two categorical variables (p = {p_value:.4f})."
        else:
            conclusion = f"fail to reject the null hypothesis"
            practical_conclusion = f"There is no statistically significant association between the two categorical variables (p = {p_value:.4f})."

        or_interpretation = ""
        if odds_ratio:
            if odds_ratio > 1:
                or_interpretation = f" The odds ratio ({odds_ratio:.3f}) suggests the first category is {odds_ratio:.1f} times more likely to be associated with the first outcome."
            elif odds_ratio < 1:
                or_interpretation = f" The odds ratio ({odds_ratio:.3f}) suggests the first category is {1/odds_ratio:.1f} times less likely to be associated with the first outcome."

        interpretation = f"""
        **Statistical Decision:** Based on the p-value ({p_value:.4f}) and significance level (α = {alpha}), we {conclusion}.

        **Practical Interpretation:** {practical_conclusion}{or_interpretation}

        **Technical Details:**
        - Exact p-value = {p_value:.6f}
        - Odds ratio = {odds_ratio:.3f}

        **Note:** Fisher's exact test is used for 2×2 contingency tables, especially when expected frequencies are small.
        """

        return interpretation.strip()

    def _interpret_correlation(self, results, alpha, context):
        """Interpret correlation results"""
        p_value = results['p_value']
        correlation = results['correlation']
        test_name = results['test_name']

        significance = self._get_significance_level(p_value)

        if p_value <= alpha:
            conclusion = f"reject the null hypothesis"
            direction = "positive" if correlation > 0 else "negative"
            practical_conclusion = f"There is a statistically {significance} {direction} correlation between the variables (r = {correlation:.3f}, p = {p_value:.4f})."
        else:
            conclusion = f"fail to reject the null hypothesis"
            practical_conclusion = f"There is no statistically significant correlation between the variables (r = {correlation:.3f}, p = {p_value:.4f})."

        effect_interpretation = self._interpret_effect_size(abs(correlation), 'correlation', test_name)

        correlation_type = "linear" if "Pearson" in test_name else "monotonic"

        interpretation = f"""
        **Statistical Decision:** Based on the p-value ({p_value:.4f}) and significance level (α = {alpha}), we {conclusion}.

        **Practical Interpretation:** {practical_conclusion}{effect_interpretation}

        **Technical Details:**
        - Correlation coefficient = {correlation:.4f}
        - Sample size = {results.get('n', 'N/A')}
        """

        if "Pearson" in test_name:
            r_squared = results.get('r_squared', correlation**2)
            interpretation += f"\n- R-squared = {r_squared:.4f} ({r_squared*100:.1f}% of variance explained)"

        interpretation += f"\n\n**Note:** This test measures the {correlation_type} relationship between two variables."

        return interpretation.strip()

    def _interpret_generic(self, results, alpha, context):
        """Generic interpretation for unsupported tests"""
        p_value = results.get('p_value', 'N/A')
        statistic = results.get('statistic', 'N/A')

        return f"""
        **Test Results:**
        - Test statistic = {statistic}
        - P-value = {p_value}

        **Note:** Detailed interpretation not available for this test type. Please consult statistical literature for guidance on interpreting these results.
        """

    def get_recommendations(self, results, test_name, assumptions_passed=True):
        """
        Provide recommendations based on test results and assumption violations.

        Parameters:
        -----------
        results : dict
            Test results
        test_name : str
            Name of the statistical test
        assumptions_passed : bool
            Whether test assumptions were met

        Returns:
        --------
        list : List of recommendations
        """
        recommendations = []

        p_value = results.get('p_value')

        # Significance-based recommendations
        if p_value is not None:
            if p_value <= 0.05:
                recommendations.append("Consider the practical significance of the result, not just statistical significance")
                if 'effect_size' in results:
                    if results['effect_size'] < 0.2:  # Small effect
                        recommendations.append("Effect size is small - consider whether the difference is practically meaningful")

                # Post-hoc recommendations
                if test_name in ['One-way ANOVA', 'Kruskal-Wallis test', 'Repeated measures ANOVA', 'Friedman test']:
                    recommendations.append("Perform post-hoc tests to identify which specific groups differ")

            elif 0.05 < p_value <= 0.1:
                recommendations.append("Result is marginally significant - consider increasing sample size")

            else:
                recommendations.append("Consider effect size and confidence intervals, not just p-value")
                recommendations.append("Ensure adequate statistical power for your effect size of interest")

        # Assumption-based recommendations
        if not assumptions_passed:
            if test_name == 'Independent t-test':
                recommendations.append("Consider Welch's t-test (unequal variances) or Mann-Whitney U test (non-normal data)")
            elif test_name == 'Paired t-test':
                recommendations.append("Consider Wilcoxon signed-rank test for non-normal differences")
            elif test_name == 'One-way ANOVA':
                recommendations.append("Consider Kruskal-Wallis test for non-normal data or unequal variances")
            elif test_name == 'Repeated measures ANOVA':
                recommendations.append("Consider Friedman test for non-normal data or Greenhouse-Geisser correction for sphericity violations")

        # Sample size recommendations
        if 'group_ns' in results:
            min_n = min(results['group_ns'])
            if min_n < 30:
                recommendations.append("Small sample sizes - interpret results with caution and consider non-parametric alternatives")
        elif 'n' in results and results['n'] < 30:
            recommendations.append("Small sample size - interpret results with caution")

        # General recommendations
        recommendations.append("Report effect sizes and confidence intervals alongside p-values")
        recommendations.append("Consider replication of findings in independent samples")

        return recommendations