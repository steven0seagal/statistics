"""
Test Selection Module
====================

Implements intelligent statistical test selection based on data characteristics
and research objectives.
"""

class TestSelector:
    """
    Intelligent test selection based on research goals and data characteristics.
    """

    def __init__(self):
        self.test_decision_tree = self._build_decision_tree()

    def _build_decision_tree(self):
        """Build the decision tree for test selection"""
        return {
            'compare_groups': {
                'two_groups': {
                    'independent': {
                        'continuous': {
                            'parametric': 'Independent t-test',
                            'non_parametric': 'Mann-Whitney U test',
                            'unequal_variances': 'Welch\'s t-test'
                        },
                        'ordinal': 'Mann-Whitney U test',
                        'binary': 'Chi-squared test'
                    },
                    'paired': {
                        'continuous': {
                            'parametric': 'Paired t-test',
                            'non_parametric': 'Wilcoxon signed-rank test'
                        },
                        'ordinal': 'Wilcoxon signed-rank test',
                        'binary': 'McNemar\'s test'
                    }
                },
                'multiple_groups': {
                    'independent': {
                        'continuous': {
                            'parametric': 'One-way ANOVA',
                            'non_parametric': 'Kruskal-Wallis test'
                        },
                        'ordinal': 'Kruskal-Wallis test',
                        'binary': 'Chi-squared test'
                    },
                    'paired': {
                        'continuous': {
                            'parametric': 'Repeated measures ANOVA',
                            'non_parametric': 'Friedman test'
                        },
                        'ordinal': 'Friedman test',
                        'binary': 'Cochran\'s Q test'
                    }
                }
            },
            'relationships': {
                'two_continuous': {
                    'linear': 'Pearson correlation',
                    'monotonic': 'Spearman correlation',
                    'prediction': 'Linear regression'
                },
                'continuous_categorical': {
                    'binary_outcome': 'Logistic regression',
                    'multiple_predictors': 'Multiple regression'
                },
                'two_categorical': {
                    'independence': 'Chi-squared test',
                    'small_sample': 'Fisher\'s exact test'
                }
            }
        }

    def recommend_test(self, goal, dependent_type, num_groups=None, design=None,
                      sample_size=None, **kwargs):
        """
        Recommend appropriate statistical test based on study characteristics.

        Parameters:
        -----------
        goal : str
            Research goal ('compare_groups', 'relationships', 'prediction')
        dependent_type : str
            Type of dependent variable
        num_groups : str
            Number of groups ('Two groups', 'Three or more groups')
        design : str
            Study design ('Independent groups', 'Paired/repeated measures')
        sample_size : int
            Sample size per group

        Returns:
        --------
        dict : Test recommendation with rationale and alternatives
        """

        # Standardize inputs
        dependent_type_map = {
            'Continuous (numerical)': 'continuous',
            'Ordinal (ranked)': 'ordinal',
            'Binary (yes/no)': 'binary'
        }

        num_groups_map = {
            'Two groups': 'two_groups',
            'Three or more groups': 'multiple_groups'
        }

        design_map = {
            'Independent groups': 'independent',
            'Paired/repeated measures': 'paired'
        }

        # Convert to internal format
        dep_type = dependent_type_map.get(dependent_type, dependent_type.lower())
        groups = num_groups_map.get(num_groups)
        study_design = design_map.get(design)

        if goal == 'compare_groups':
            return self._recommend_group_comparison(dep_type, groups, study_design, sample_size)
        elif goal == 'relationships':
            return self._recommend_relationship_test(dep_type, **kwargs)
        else:
            return {
                'test_name': 'Test selection in progress',
                'rationale': 'This feature is being developed.',
                'alternative': 'Please check back soon!'
            }

    def _recommend_group_comparison(self, dep_type, groups, design, sample_size):
        """Recommend test for group comparisons"""

        try:
            if dep_type == 'continuous':
                # Determine if parametric assumptions likely met
                parametric_suitable = self._assess_parametric_suitability(sample_size)

                if groups == 'two_groups':
                    if design == 'independent':
                        if parametric_suitable:
                            test = 'Independent t-test'
                            rationale = f"Two independent groups with continuous data (n={sample_size} per group). Parametric test appropriate for this sample size."
                            alternative = "If assumptions are violated, use Mann-Whitney U test or Welch's t-test for unequal variances."
                        else:
                            test = 'Mann-Whitney U test'
                            rationale = f"Two independent groups with continuous data (n={sample_size} per group). Non-parametric test recommended for small sample size."
                            alternative = "If data are normally distributed, consider independent t-test."
                    else:  # paired
                        if parametric_suitable:
                            test = 'Paired t-test'
                            rationale = f"Two related groups with continuous data (n={sample_size}). Paired design controls for individual differences."
                            alternative = "If differences are not normally distributed, use Wilcoxon signed-rank test."
                        else:
                            test = 'Wilcoxon signed-rank test'
                            rationale = f"Two related groups with continuous data (n={sample_size}). Non-parametric test for small sample or non-normal differences."
                            alternative = "If differences are normally distributed, consider paired t-test."

                elif groups == 'multiple_groups':
                    if design == 'independent':
                        if parametric_suitable:
                            test = 'One-way ANOVA'
                            rationale = f"Multiple independent groups with continuous data (n={sample_size} per group). ANOVA tests for any group differences."
                            alternative = "If assumptions are violated, use Kruskal-Wallis test. Follow up with post-hoc tests if significant."
                        else:
                            test = 'Kruskal-Wallis test'
                            rationale = f"Multiple independent groups with continuous data (n={sample_size} per group). Non-parametric alternative to ANOVA."
                            alternative = "If assumptions are met, consider one-way ANOVA for increased power."
                    else:  # paired
                        if parametric_suitable:
                            test = 'Repeated measures ANOVA'
                            rationale = f"Multiple related measurements with continuous data (n={sample_size}). Accounts for within-subject correlation."
                            alternative = "If sphericity is violated, use Friedman test or apply corrections (Greenhouse-Geisser)."
                        else:
                            test = 'Friedman test'
                            rationale = f"Multiple related measurements with continuous data (n={sample_size}). Non-parametric alternative to repeated measures ANOVA."
                            alternative = "If assumptions are met, consider repeated measures ANOVA."

            elif dep_type == 'ordinal':
                if groups == 'two_groups':
                    if design == 'independent':
                        test = 'Mann-Whitney U test'
                        rationale = "Two independent groups with ordinal data. Non-parametric test compares rank distributions."
                        alternative = "Consider independent t-test only if data can be treated as interval and n is large."
                    else:  # paired
                        test = 'Wilcoxon signed-rank test'
                        rationale = "Two related groups with ordinal data. Tests for median difference in paired observations."
                        alternative = "Consider paired t-test only if differences can be treated as interval data."

                elif groups == 'multiple_groups':
                    if design == 'independent':
                        test = 'Kruskal-Wallis test'
                        rationale = "Multiple independent groups with ordinal data. Extension of Mann-Whitney U test."
                        alternative = "Follow up with Dunn's test for pairwise comparisons if significant."
                    else:  # paired
                        test = 'Friedman test'
                        rationale = "Multiple related measurements with ordinal data. Non-parametric repeated measures test."
                        alternative = "Follow up with pairwise Wilcoxon tests with Bonferroni correction."

            elif dep_type == 'binary':
                if groups == 'two_groups':
                    if design == 'independent':
                        test = 'Chi-squared test'
                        rationale = "Two independent groups with binary outcome. Tests for association between group and outcome."
                        alternative = "Use Fisher's exact test if expected cell counts < 5."
                    else:  # paired
                        test = 'McNemar\'s test'
                        rationale = "Two related measurements with binary outcome. Tests for change in proportions."
                        alternative = "Ensure adequate discordant pairs for reliable test."

                elif groups == 'multiple_groups':
                    if design == 'independent':
                        test = 'Chi-squared test'
                        rationale = "Multiple independent groups with binary outcome. Tests for association in contingency table."
                        alternative = "Consider logistic regression for additional covariates."
                    else:  # paired
                        test = 'Cochran\'s Q test'
                        rationale = "Multiple related measurements with binary outcome. Extension of McNemar's test."
                        alternative = "Follow up with pairwise McNemar tests if significant."

            else:
                test = 'Test selection error'
                rationale = f"Unknown dependent variable type: {dep_type}"
                alternative = "Please check your variable type selection."

            return {
                'test_name': test,
                'rationale': rationale,
                'alternative': alternative
            }

        except Exception as e:
            return {
                'test_name': 'Selection error',
                'rationale': f"Error in test selection: {str(e)}",
                'alternative': "Please verify your inputs and try again."
            }

    def _recommend_relationship_test(self, dep_type, **kwargs):
        """Recommend test for relationship analysis"""
        # Placeholder for relationship test selection
        return {
            'test_name': 'Relationship analysis',
            'rationale': 'This feature is under development.',
            'alternative': 'Check back soon for relationship analysis options!'
        }

    def _assess_parametric_suitability(self, sample_size):
        """Assess whether parametric tests are likely appropriate"""
        if sample_size is None:
            return True  # Default to parametric

        # Rule of thumb: parametric tests generally appropriate for n >= 30
        # For smaller samples, depends heavily on data distribution
        return sample_size >= 30

    def get_test_assumptions(self, test_name):
        """Get the assumptions for a specific test"""

        assumptions = {
            'Independent t-test': [
                'Independent observations',
                'Normal distribution in each group',
                'Equal variances between groups',
                'Continuous dependent variable'
            ],
            'Welch\'s t-test': [
                'Independent observations',
                'Normal distribution in each group',
                'Continuous dependent variable',
                'Unequal variances allowed'
            ],
            'Paired t-test': [
                'Paired observations',
                'Normal distribution of differences',
                'Continuous dependent variable'
            ],
            'Mann-Whitney U test': [
                'Independent observations',
                'Ordinal or continuous dependent variable',
                'Similar distribution shapes for median comparison'
            ],
            'Wilcoxon signed-rank test': [
                'Paired observations',
                'Continuous dependent variable',
                'Symmetric distribution of differences'
            ],
            'One-way ANOVA': [
                'Independent observations',
                'Normal distribution in each group',
                'Equal variances between groups',
                'Continuous dependent variable'
            ],
            'Kruskal-Wallis test': [
                'Independent observations',
                'Ordinal or continuous dependent variable',
                'Similar distribution shapes for comparison'
            ],
            'Repeated measures ANOVA': [
                'Related observations',
                'Normal distribution within each condition',
                'Sphericity (equal variances of differences)',
                'Continuous dependent variable'
            ],
            'Friedman test': [
                'Related observations',
                'Ordinal or continuous dependent variable',
                'No distributional assumptions'
            ]
        }

        return assumptions.get(test_name, ['Assumptions not available for this test'])

    def get_post_hoc_recommendations(self, test_name, num_groups):
        """Get post-hoc test recommendations for significant omnibus tests"""

        if test_name == 'One-way ANOVA':
            return {
                'recommended': 'Tukey\'s HSD',
                'alternatives': ['Bonferroni', 'Holm-Sidak', 'Dunnett (vs control)'],
                'rationale': 'Tukey\'s HSD controls family-wise error rate while maintaining good power for all pairwise comparisons.'
            }
        elif test_name == 'Kruskal-Wallis test':
            return {
                'recommended': 'Dunn\'s test',
                'alternatives': ['Pairwise Mann-Whitney with Bonferroni correction'],
                'rationale': 'Dunn\'s test is specifically designed for post-hoc analysis after Kruskal-Wallis.'
            }
        elif test_name == 'Repeated measures ANOVA':
            return {
                'recommended': 'Pairwise t-tests with Bonferroni correction',
                'alternatives': ['Tukey\'s HSD for repeated measures', 'Holm-Sidak'],
                'rationale': 'Pairwise comparisons should account for the repeated measures structure.'
            }
        elif test_name == 'Friedman test':
            return {
                'recommended': 'Pairwise Wilcoxon with Bonferroni correction',
                'alternatives': ['Nemenyi test'],
                'rationale': 'Non-parametric pairwise tests appropriate for ordinal data or non-normal distributions.'
            }
        else:
            return None