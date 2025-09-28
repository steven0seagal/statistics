"""
Advanced Statistical Tests Module
================================

Implements more complex statistical tests including MANOVA, ANCOVA, and Logistic Regression.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings

class AdvancedStatisticalTests:
    """
    Advanced statistical test implementations for complex analyses.
    """

    def __init__(self):
        self.supported_tests = [
            'MANOVA',
            'ANCOVA',
            'Logistic Regression',
            'Multiple Linear Regression',
            'Two-way ANOVA',
            'Mixed-effects ANOVA'
        ]

    def perform_manova(self, data, dependent_vars, independent_var, **kwargs):
        """
        Perform Multivariate Analysis of Variance (MANOVA).

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        dependent_vars : list
            List of dependent variable column names
        independent_var : str
            Independent variable column name

        Returns:
        --------
        dict : MANOVA results
        """
        try:
            # Prepare data
            clean_data = data[dependent_vars + [independent_var]].dropna()

            # Create formula string
            dependent_formula = ' + '.join(dependent_vars)
            formula = f"{dependent_formula} ~ {independent_var}"

            # Perform MANOVA
            manova = MANOVA.from_formula(formula, data=clean_data)
            manova_results = manova.mv_test()

            # Extract key statistics
            wilks_lambda = manova_results.results[independent_var]['stat'].iloc[0]
            p_value = manova_results.results[independent_var]['Pr > F'].iloc[0]
            f_stat = manova_results.results[independent_var]['Value'].iloc[0]

            # Calculate effect size (partial eta-squared approximation)
            effect_size = 1 - wilks_lambda

            # Group statistics
            groups = clean_data[independent_var].unique()
            group_means = {}
            group_stds = {}
            group_ns = {}

            for var in dependent_vars:
                group_means[var] = {}
                group_stds[var] = {}
                for group in groups:
                    group_data = clean_data[clean_data[independent_var] == group][var]
                    group_means[var][str(group)] = group_data.mean()
                    group_stds[var][str(group)] = group_data.std()

            for group in groups:
                group_ns[str(group)] = len(clean_data[clean_data[independent_var] == group])

            return {
                'test_name': 'MANOVA',
                'statistic': f_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'wilks_lambda': wilks_lambda,
                'dependent_variables': dependent_vars,
                'independent_variable': independent_var,
                'group_means': group_means,
                'group_stds': group_stds,
                'group_ns': group_ns,
                'groups': groups.tolist(),
                'degrees_freedom_hypothesis': len(groups) - 1,
                'degrees_freedom_error': len(clean_data) - len(groups),
                'sample_size': len(clean_data)
            }

        except Exception as e:
            return {
                'test_name': 'MANOVA',
                'error': f'MANOVA failed: {str(e)}',
                'statistic': None,
                'p_value': None
            }

    def perform_ancova(self, data, dependent_var, independent_var, covariate, **kwargs):
        """
        Perform Analysis of Covariance (ANCOVA).

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        dependent_var : str
            Dependent variable column name
        independent_var : str
            Independent variable (categorical) column name
        covariate : str
            Covariate (continuous) column name

        Returns:
        --------
        dict : ANCOVA results
        """
        try:
            # Prepare data
            clean_data = data[[dependent_var, independent_var, covariate]].dropna()

            # Create formula
            formula = f"{dependent_var} ~ C({independent_var}) + {covariate}"

            # Fit model
            model = ols(formula, data=clean_data).fit()
            anova_table = anova_lm(model, typ=2)

            # Extract results for main effect (independent variable)
            main_effect_row = f'C({independent_var})'
            f_stat = anova_table.loc[main_effect_row, 'F']
            p_value = anova_table.loc[main_effect_row, 'PR(>F)']

            # Extract covariate results
            covariate_f = anova_table.loc[covariate, 'F']
            covariate_p = anova_table.loc[covariate, 'PR(>F)']

            # Calculate effect size (partial eta-squared)
            ss_effect = anova_table.loc[main_effect_row, 'sum_sq']
            ss_error = anova_table.loc['Residual', 'sum_sq']
            eta_squared = ss_effect / (ss_effect + ss_error)

            # Group statistics adjusted for covariate
            groups = clean_data[independent_var].unique()
            group_stats = {}

            for group in groups:
                group_data = clean_data[clean_data[independent_var] == group]
                group_stats[str(group)] = {
                    'n': len(group_data),
                    'mean_dependent': group_data[dependent_var].mean(),
                    'mean_covariate': group_data[covariate].mean(),
                    'std_dependent': group_data[dependent_var].std(),
                    'std_covariate': group_data[covariate].std()
                }

            return {
                'test_name': 'ANCOVA',
                'statistic': f_stat,
                'p_value': p_value,
                'effect_size': eta_squared,
                'covariate_f': covariate_f,
                'covariate_p': covariate_p,
                'dependent_variable': dependent_var,
                'independent_variable': independent_var,
                'covariate': covariate,
                'group_statistics': group_stats,
                'groups': groups.tolist(),
                'model_summary': model.summary().as_text(),
                'anova_table': anova_table.to_dict(),
                'degrees_freedom_effect': anova_table.loc[main_effect_row, 'df'],
                'degrees_freedom_error': anova_table.loc['Residual', 'df'],
                'sample_size': len(clean_data)
            }

        except Exception as e:
            return {
                'test_name': 'ANCOVA',
                'error': f'ANCOVA failed: {str(e)}',
                'statistic': None,
                'p_value': None
            }

    def perform_logistic_regression(self, data, dependent_var, independent_vars, **kwargs):
        """
        Perform Logistic Regression.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        dependent_var : str
            Binary dependent variable column name
        independent_vars : list
            List of independent variable column names

        Returns:
        --------
        dict : Logistic regression results
        """
        try:
            # Prepare data
            all_vars = [dependent_var] + independent_vars
            clean_data = data[all_vars].dropna()

            # Prepare dependent variable (ensure binary)
            y = clean_data[dependent_var]
            if y.nunique() != 2:
                raise ValueError(f"Dependent variable must be binary, found {y.nunique()} unique values")

            # Encode if necessary
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                classes = le.classes_
            else:
                y_encoded = y
                classes = sorted(y.unique())

            # Prepare independent variables
            X = clean_data[independent_vars].copy()

            # Encode categorical variables
            categorical_encoders = {}
            for col in independent_vars:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    categorical_encoders[col] = le

            # Fit logistic regression
            model = LogisticRegression(random_state=42)
            model.fit(X, y_encoded)

            # Calculate predictions and probabilities
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            # Calculate metrics
            accuracy = (y_pred == y_encoded).mean()

            # Calculate odds ratios
            odds_ratios = np.exp(model.coef_[0])

            # Create results dictionary
            results = {
                'test_name': 'Logistic Regression',
                'accuracy': accuracy,
                'coefficients': dict(zip(independent_vars, model.coef_[0])),
                'odds_ratios': dict(zip(independent_vars, odds_ratios)),
                'intercept': model.intercept_[0],
                'dependent_variable': dependent_var,
                'independent_variables': independent_vars,
                'classes': classes.tolist() if hasattr(classes, 'tolist') else list(classes),
                'sample_size': len(clean_data),
                'categorical_encoders': categorical_encoders,
                'predictions': y_pred.tolist(),
                'probabilities': y_prob.tolist()
            }

            # Calculate classification metrics
            from sklearn.metrics import classification_report, confusion_matrix
            results['classification_report'] = classification_report(y_encoded, y_pred, output_dict=True)
            results['confusion_matrix'] = confusion_matrix(y_encoded, y_pred).tolist()

            # Calculate pseudo R-squared (McFadden's)
            # Note: This is a simplified calculation
            null_accuracy = max((y_encoded == 0).mean(), (y_encoded == 1).mean())
            results['pseudo_r_squared'] = 1 - (1 - accuracy) / (1 - null_accuracy)

            return results

        except Exception as e:
            return {
                'test_name': 'Logistic Regression',
                'error': f'Logistic regression failed: {str(e)}',
                'accuracy': None,
                'coefficients': None
            }

    def perform_two_way_anova(self, data, dependent_var, factor1, factor2, **kwargs):
        """
        Perform Two-way ANOVA.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        dependent_var : str
            Dependent variable column name
        factor1 : str
            First factor column name
        factor2 : str
            Second factor column name

        Returns:
        --------
        dict : Two-way ANOVA results
        """
        try:
            # Prepare data
            clean_data = data[[dependent_var, factor1, factor2]].dropna()

            # Create formula with interaction
            formula = f"{dependent_var} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"

            # Fit model
            model = ols(formula, data=clean_data).fit()
            anova_table = anova_lm(model, typ=2)

            # Extract results
            factor1_f = anova_table.loc[f'C({factor1})', 'F']
            factor1_p = anova_table.loc[f'C({factor1})', 'PR(>F)']

            factor2_f = anova_table.loc[f'C({factor2})', 'F']
            factor2_p = anova_table.loc[f'C({factor2})', 'PR(>F)']

            interaction_f = anova_table.loc[f'C({factor1}):C({factor2})', 'F']
            interaction_p = anova_table.loc[f'C({factor1}):C({factor2})', 'PR(>F)']

            # Calculate effect sizes
            ss_total = anova_table['sum_sq'].sum()
            eta_squared_factor1 = anova_table.loc[f'C({factor1})', 'sum_sq'] / ss_total
            eta_squared_factor2 = anova_table.loc[f'C({factor2})', 'sum_sq'] / ss_total
            eta_squared_interaction = anova_table.loc[f'C({factor1}):C({factor2})', 'sum_sq'] / ss_total

            # Group statistics
            group_stats = {}
            for f1_level in clean_data[factor1].unique():
                for f2_level in clean_data[factor2].unique():
                    group_data = clean_data[
                        (clean_data[factor1] == f1_level) &
                        (clean_data[factor2] == f2_level)
                    ][dependent_var]

                    if len(group_data) > 0:
                        group_stats[f"{f1_level}_{f2_level}"] = {
                            'mean': group_data.mean(),
                            'std': group_data.std(),
                            'n': len(group_data)
                        }

            return {
                'test_name': 'Two-way ANOVA',
                'factor1_name': factor1,
                'factor1_f': factor1_f,
                'factor1_p': factor1_p,
                'factor1_eta_squared': eta_squared_factor1,
                'factor2_name': factor2,
                'factor2_f': factor2_f,
                'factor2_p': factor2_p,
                'factor2_eta_squared': eta_squared_factor2,
                'interaction_f': interaction_f,
                'interaction_p': interaction_p,
                'interaction_eta_squared': eta_squared_interaction,
                'dependent_variable': dependent_var,
                'group_statistics': group_stats,
                'anova_table': anova_table.to_dict(),
                'model_summary': model.summary().as_text(),
                'sample_size': len(clean_data)
            }

        except Exception as e:
            return {
                'test_name': 'Two-way ANOVA',
                'error': f'Two-way ANOVA failed: {str(e)}',
                'factor1_f': None,
                'factor1_p': None
            }

    def perform_multiple_regression(self, data, dependent_var, independent_vars, **kwargs):
        """
        Perform Multiple Linear Regression.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        dependent_var : str
            Dependent variable column name
        independent_vars : list
            List of independent variable column names

        Returns:
        --------
        dict : Multiple regression results
        """
        try:
            from statsmodels.formula.api import ols
            from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad

            # Prepare data
            all_vars = [dependent_var] + independent_vars
            clean_data = data[all_vars].dropna()

            # Create formula
            formula = f"{dependent_var} ~ " + " + ".join(independent_vars)

            # Fit model
            model = ols(formula, data=clean_data).fit()

            # Extract key statistics
            r_squared = model.rsquared
            adj_r_squared = model.rsquared_adj
            f_statistic = model.fvalue
            f_pvalue = model.f_pvalue

            # Individual coefficient tests
            coefficients = {}
            p_values = {}
            confidence_intervals = {}

            for var in independent_vars:
                coefficients[var] = model.params[var]
                p_values[var] = model.pvalues[var]
                ci = model.conf_int().loc[var]
                confidence_intervals[var] = [ci[0], ci[1]]

            # Model diagnostics
            residuals = model.resid
            fitted_values = model.fittedvalues

            # Test for heteroscedasticity
            try:
                bp_test = het_breuschpagan(residuals, model.model.exog)
                heteroscedasticity_p = bp_test[1]
            except:
                heteroscedasticity_p = None

            # Test for normality of residuals
            try:
                normality_test = normal_ad(residuals)
                normality_p = normality_test[1]
            except:
                normality_p = None

            return {
                'test_name': 'Multiple Linear Regression',
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'f_statistic': f_statistic,
                'f_pvalue': f_pvalue,
                'coefficients': coefficients,
                'p_values': p_values,
                'confidence_intervals': confidence_intervals,
                'intercept': model.params['Intercept'],
                'intercept_p': model.pvalues['Intercept'],
                'dependent_variable': dependent_var,
                'independent_variables': independent_vars,
                'sample_size': len(clean_data),
                'degrees_freedom_model': model.df_model,
                'degrees_freedom_residual': model.df_resid,
                'model_summary': model.summary().as_text(),
                'residuals': residuals.tolist(),
                'fitted_values': fitted_values.tolist(),
                'heteroscedasticity_p': heteroscedasticity_p,
                'normality_p': normality_p
            }

        except Exception as e:
            return {
                'test_name': 'Multiple Linear Regression',
                'error': f'Multiple regression failed: {str(e)}',
                'r_squared': None,
                'f_statistic': None
            }