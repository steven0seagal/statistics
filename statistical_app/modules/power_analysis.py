"""
Power Analysis Module
====================

Implements power analysis and sample size calculations for various statistical tests.
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq
import warnings

class PowerAnalysis:
    """
    Power analysis and sample size calculations for statistical tests.
    """

    def __init__(self):
        self.effect_size_conventions = {
            'cohens_d': {'small': 0.2, 'medium': 0.5, 'large': 0.8},
            'eta_squared': {'small': 0.01, 'medium': 0.06, 'large': 0.14},
            'correlation': {'small': 0.1, 'medium': 0.3, 'large': 0.5},
            'cramers_v': {'small': 0.1, 'medium': 0.3, 'large': 0.5}
        }

    def calculate_power_ttest(self, effect_size, sample_size, alpha=0.05, test_type='two-sample'):
        """
        Calculate statistical power for t-tests.

        Parameters:
        -----------
        effect_size : float
            Cohen's d effect size
        sample_size : int
            Sample size (per group for two-sample tests)
        alpha : float
            Type I error rate (default 0.05)
        test_type : str
            Type of t-test ('one-sample', 'two-sample', 'paired')

        Returns:
        --------
        float : Statistical power (1 - β)
        """
        try:
            if test_type == 'one-sample':
                df = sample_size - 1
                ncp = effect_size * np.sqrt(sample_size)
            elif test_type == 'paired':
                df = sample_size - 1
                ncp = effect_size * np.sqrt(sample_size)
            else:  # two-sample
                df = 2 * sample_size - 2
                ncp = effect_size * np.sqrt(sample_size / 2)

            # Critical t-value
            t_critical = stats.t.ppf(1 - alpha/2, df)

            # Power calculation using non-central t-distribution
            power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)

            return min(power, 1.0)  # Cap at 1.0

        except Exception as e:
            warnings.warn(f"Power calculation failed: {e}")
            return None

    def calculate_sample_size_ttest(self, effect_size, power=0.8, alpha=0.05, test_type='two-sample'):
        """
        Calculate required sample size for t-tests.

        Parameters:
        -----------
        effect_size : float
            Cohen's d effect size
        power : float
            Desired statistical power (default 0.8)
        alpha : float
            Type I error rate (default 0.05)
        test_type : str
            Type of t-test ('one-sample', 'two-sample', 'paired')

        Returns:
        --------
        int : Required sample size (per group)
        """
        try:
            def power_function(n):
                return self.calculate_power_ttest(effect_size, int(n), alpha, test_type) - power

            # Search for sample size that gives desired power
            try:
                n_required = brentq(power_function, 2, 10000)
                return int(np.ceil(n_required))
            except ValueError:
                # If brentq fails, use approximation
                if test_type == 'two-sample':
                    n_approx = 2 * (stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power))**2 / effect_size**2
                else:
                    n_approx = (stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power))**2 / effect_size**2
                return int(np.ceil(n_approx))

        except Exception as e:
            warnings.warn(f"Sample size calculation failed: {e}")
            return None

    def calculate_power_anova(self, effect_size, sample_size_per_group, num_groups, alpha=0.05):
        """
        Calculate statistical power for one-way ANOVA.

        Parameters:
        -----------
        effect_size : float
            Effect size (f = sqrt(eta_squared / (1 - eta_squared)))
        sample_size_per_group : int
            Sample size per group
        num_groups : int
            Number of groups
        alpha : float
            Type I error rate (default 0.05)

        Returns:
        --------
        float : Statistical power (1 - β)
        """
        try:
            # Convert eta-squared to f if needed
            if effect_size < 1:  # Assume eta-squared
                f_effect_size = np.sqrt(effect_size / (1 - effect_size))
            else:
                f_effect_size = effect_size

            # Degrees of freedom
            df_between = num_groups - 1
            df_within = num_groups * (sample_size_per_group - 1)
            total_n = num_groups * sample_size_per_group

            # Non-centrality parameter
            ncp = total_n * f_effect_size**2

            # Critical F-value
            f_critical = stats.f.ppf(1 - alpha, df_between, df_within)

            # Power calculation using non-central F-distribution
            power = 1 - stats.ncf.cdf(f_critical, df_between, df_within, ncp)

            return min(power, 1.0)

        except Exception as e:
            warnings.warn(f"ANOVA power calculation failed: {e}")
            return None

    def calculate_sample_size_anova(self, effect_size, num_groups, power=0.8, alpha=0.05):
        """
        Calculate required sample size per group for one-way ANOVA.

        Parameters:
        -----------
        effect_size : float
            Effect size (eta-squared or f)
        num_groups : int
            Number of groups
        power : float
            Desired statistical power (default 0.8)
        alpha : float
            Type I error rate (default 0.05)

        Returns:
        --------
        int : Required sample size per group
        """
        try:
            def power_function(n):
                return self.calculate_power_anova(effect_size, int(n), num_groups, alpha) - power

            # Search for sample size that gives desired power
            try:
                n_required = brentq(power_function, 2, 1000)
                return int(np.ceil(n_required))
            except ValueError:
                # Use approximation formula
                if effect_size < 1:  # Assume eta-squared
                    f_effect_size = np.sqrt(effect_size / (1 - effect_size))
                else:
                    f_effect_size = effect_size

                # Approximation formula
                n_approx = ((stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power))**2) / (f_effect_size**2 * num_groups)
                return int(np.ceil(n_approx))

        except Exception as e:
            warnings.warn(f"ANOVA sample size calculation failed: {e}")
            return None

    def calculate_power_correlation(self, effect_size, sample_size, alpha=0.05):
        """
        Calculate statistical power for correlation analysis.

        Parameters:
        -----------
        effect_size : float
            Population correlation coefficient (r)
        sample_size : int
            Sample size
        alpha : float
            Type I error rate (default 0.05)

        Returns:
        --------
        float : Statistical power (1 - β)
        """
        try:
            # Fisher's z-transformation
            z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size))

            # Standard error
            se = 1 / np.sqrt(sample_size - 3)

            # Critical z-value
            z_critical = stats.norm.ppf(1 - alpha/2)

            # Power calculation
            z_lower = (z_critical * se - z_r) / se
            z_upper = (-z_critical * se - z_r) / se

            power = stats.norm.cdf(z_lower) + (1 - stats.norm.cdf(z_upper))

            return min(power, 1.0)

        except Exception as e:
            warnings.warn(f"Correlation power calculation failed: {e}")
            return None

    def calculate_sample_size_correlation(self, effect_size, power=0.8, alpha=0.05):
        """
        Calculate required sample size for correlation analysis.

        Parameters:
        -----------
        effect_size : float
            Population correlation coefficient (r)
        power : float
            Desired statistical power (default 0.8)
        alpha : float
            Type I error rate (default 0.05)

        Returns:
        --------
        int : Required sample size
        """
        try:
            def power_function(n):
                return self.calculate_power_correlation(effect_size, int(n), alpha) - power

            # Search for sample size that gives desired power
            try:
                n_required = brentq(power_function, 4, 10000)
                return int(np.ceil(n_required))
            except ValueError:
                # Use approximation formula
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = stats.norm.ppf(power)
                z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size))

                n_approx = ((z_alpha + z_beta) / z_r)**2 + 3
                return int(np.ceil(n_approx))

        except Exception as e:
            warnings.warn(f"Correlation sample size calculation failed: {e}")
            return None

    def calculate_power_chisquare(self, effect_size, sample_size, df, alpha=0.05):
        """
        Calculate statistical power for chi-square tests.

        Parameters:
        -----------
        effect_size : float
            Cramér's V effect size
        sample_size : int
            Total sample size
        df : int
            Degrees of freedom
        alpha : float
            Type I error rate (default 0.05)

        Returns:
        --------
        float : Statistical power (1 - β)
        """
        try:
            # Convert Cramér's V to chi-square effect size
            w = effect_size  # Cramér's V is equivalent to Cohen's w

            # Non-centrality parameter
            ncp = sample_size * w**2

            # Critical chi-square value
            chi2_critical = stats.chi2.ppf(1 - alpha, df)

            # Power calculation using non-central chi-square distribution
            power = 1 - stats.ncx2.cdf(chi2_critical, df, ncp)

            return min(power, 1.0)

        except Exception as e:
            warnings.warn(f"Chi-square power calculation failed: {e}")
            return None

    def calculate_sample_size_chisquare(self, effect_size, df, power=0.8, alpha=0.05):
        """
        Calculate required sample size for chi-square tests.

        Parameters:
        -----------
        effect_size : float
            Cramér's V effect size
        df : int
            Degrees of freedom
        power : float
            Desired statistical power (default 0.8)
        alpha : float
            Type I error rate (default 0.05)

        Returns:
        --------
        int : Required sample size
        """
        try:
            def power_function(n):
                return self.calculate_power_chisquare(effect_size, int(n), df, alpha) - power

            # Search for sample size that gives desired power
            try:
                n_required = brentq(power_function, 10, 100000)
                return int(np.ceil(n_required))
            except ValueError:
                # Use approximation
                chi2_alpha = stats.chi2.ppf(1 - alpha, df)
                chi2_beta = stats.chi2.ppf(power, df)
                n_approx = (chi2_alpha + chi2_beta) / effect_size**2
                return int(np.ceil(n_approx))

        except Exception as e:
            warnings.warn(f"Chi-square sample size calculation failed: {e}")
            return None

    def create_power_curve(self, test_type, effect_size, alpha=0.05, sample_sizes=None):
        """
        Create power curve showing power vs sample size.

        Parameters:
        -----------
        test_type : str
            Type of test ('ttest', 'anova', 'correlation', 'chisquare')
        effect_size : float
            Effect size for the test
        alpha : float
            Type I error rate
        sample_sizes : array-like, optional
            Sample sizes to evaluate (default: range from 5 to 200)

        Returns:
        --------
        dict : Power curve data
        """
        if sample_sizes is None:
            sample_sizes = list(range(5, 201, 5))

        powers = []

        for n in sample_sizes:
            if test_type == 'ttest':
                power = self.calculate_power_ttest(effect_size, n, alpha)
            elif test_type == 'anova':
                power = self.calculate_power_anova(effect_size, n, 3, alpha)  # Assume 3 groups
            elif test_type == 'correlation':
                power = self.calculate_power_correlation(effect_size, n, alpha)
            elif test_type == 'chisquare':
                power = self.calculate_power_chisquare(effect_size, n, 1, alpha)  # Assume df=1
            else:
                power = None

            powers.append(power)

        return {
            'sample_sizes': sample_sizes,
            'powers': powers,
            'effect_size': effect_size,
            'alpha': alpha,
            'test_type': test_type
        }

    def comprehensive_power_analysis(self, test_type, **kwargs):
        """
        Perform comprehensive power analysis for a given test.

        Parameters:
        -----------
        test_type : str
            Type of statistical test
        **kwargs : additional parameters specific to test type

        Returns:
        --------
        dict : Comprehensive power analysis results
        """
        results = {
            'test_type': test_type,
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Get conventional effect sizes
        if test_type in ['ttest', 'anova']:
            effect_sizes = self.effect_size_conventions['cohens_d']
        elif test_type == 'correlation':
            effect_sizes = self.effect_size_conventions['correlation']
        elif test_type == 'chisquare':
            effect_sizes = self.effect_size_conventions['cramers_v']
        else:
            effect_sizes = {'small': 0.2, 'medium': 0.5, 'large': 0.8}

        # Calculate sample sizes for different effect sizes
        sample_size_results = {}
        for size_name, effect_size in effect_sizes.items():
            if test_type == 'ttest':
                n_required = self.calculate_sample_size_ttest(effect_size, **kwargs)
            elif test_type == 'anova':
                n_required = self.calculate_sample_size_anova(effect_size, **kwargs)
            elif test_type == 'correlation':
                n_required = self.calculate_sample_size_correlation(effect_size, **kwargs)
            elif test_type == 'chisquare':
                n_required = self.calculate_sample_size_chisquare(effect_size, df=1, **kwargs)
            else:
                n_required = None

            sample_size_results[size_name] = {
                'effect_size': effect_size,
                'sample_size_required': n_required
            }

        results['sample_size_recommendations'] = sample_size_results

        # Create power curves for different effect sizes
        power_curves = {}
        for size_name, effect_size in effect_sizes.items():
            curve_data = self.create_power_curve(test_type, effect_size, **kwargs)
            power_curves[size_name] = curve_data

        results['power_curves'] = power_curves

        return results

    def interpret_power_analysis(self, power_result):
        """
        Provide interpretation of power analysis results.

        Parameters:
        -----------
        power_result : dict
            Results from power analysis

        Returns:
        --------
        str : Interpretation text
        """
        interpretation = "**Power Analysis Interpretation:**\n\n"

        if 'sample_size_recommendations' in power_result:
            interpretation += "**Sample Size Recommendations (80% power, α = 0.05):**\n"

            for size_name, data in power_result['sample_size_recommendations'].items():
                n_required = data['sample_size_required']
                effect_size = data['effect_size']

                if n_required:
                    interpretation += f"- {size_name.title()} effect (d = {effect_size}): {n_required} participants per group\n"
                else:
                    interpretation += f"- {size_name.title()} effect: Unable to calculate\n"

            interpretation += "\n**Guidelines:**\n"
            interpretation += "- 80% power is the conventional standard (80% chance of detecting a true effect)\n"
            interpretation += "- Larger effect sizes require smaller sample sizes\n"
            interpretation += "- Consider practical constraints and resources when choosing sample size\n"
            interpretation += "- Always consider the minimum meaningful effect size for your research\n"

        return interpretation