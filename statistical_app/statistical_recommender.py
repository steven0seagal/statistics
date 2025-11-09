"""
Automatic Statistical Test Recommender and Executor
====================================================

This module provides an intelligent statistical test recommendation system that:
- Accepts uploaded data files (CSV, TSV, XLSX)
- Automatically profiles columns (numeric/categorical)
- Tests statistical assumptions (normality, homoscedasticity)
- Recommends appropriate statistical tests
- Executes tests and provides detailed results
"""

import streamlit as st
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import numpy as np
import openpyxl


# ============================================================================
# STEP 12: Session State Clearing Functions
# ============================================================================

def clear_session_state_on_new_file():
    """Reset all session state when a new file is uploaded"""
    keys_to_clear = ['data', 'selected_columns', 'profile', 'assumptions',
                     'recommendation', 'results', 'run_analysis_flag']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def clear_analysis_results():
    """Reset only analysis results (keep loaded data)"""
    keys_to_clear = ['profile', 'assumptions', 'recommendation', 'results', 'run_analysis_flag']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# ============================================================================
# STEP 3: Data Loading Function with Caching
# ============================================================================

@st.cache_data
def load_data(file, delimiter=None, sheet_name=None):
    """
    Load data from uploaded file with caching for performance.

    Parameters:
    -----------
    file : UploadedFile
        The uploaded file object
    delimiter : str, optional
        Delimiter for CSV/TSV files
    sheet_name : str or int, optional
        Sheet name or index for Excel files

    Returns:
    --------
    DataFrame or None
        Loaded data or None if error occurs
    """
    try:
        if file.name.endswith(('.csv', '.tsv')):
            # Use provided delimiter or default to comma
            sep = delimiter if delimiter else ','
            df = pd.read_csv(file, sep=sep)
        elif file.name.endswith('.xlsx'):
            # Use provided sheet name or default to first sheet
            sn = sheet_name if sheet_name else 0
            df = pd.read_excel(file, sheet_name=sn)
        return df
    except Exception as e:
        st.error(f"Błąd podczas ładowania pliku: {e}")
        return None


# ============================================================================
# STEP 6: Column Profiling and Assumption Testing
# ============================================================================

def profile_and_test_columns(df, cols):
    """
    Analyze columns to determine types and test statistical assumptions.

    Parameters:
    -----------
    df : DataFrame
        The data to analyze
    cols : list
        Column names to profile

    Returns:
    --------
    tuple (profile, assumptions)
        - profile: dict with column information
        - assumptions: dict with assumption test results
    """
    profile = {}
    assumptions = {}

    numeric_cols = []
    categorical_cols = []

    # Profile each column
    for col in cols:
        col_info = {}

        # Determine type
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['type'] = 'Numeric'
            numeric_cols.append(col)

            # Test normality
            try:
                data_clean = df[col].dropna()
                if len(data_clean) >= 3:
                    stat, p_value = stats.shapiro(data_clean)
                    col_info['is_normal'] = p_value > 0.05
                    col_info['p_shapiro'] = p_value
                else:
                    col_info['is_normal'] = None
                    col_info['p_shapiro'] = None
            except Exception as e:
                col_info['is_normal'] = None
                col_info['p_shapiro'] = None
                col_info['error'] = str(e)
        else:
            # Categorical or text
            unique_count = df[col].nunique()
            col_info['unique_values'] = unique_count

            if unique_count > 50:
                col_info['type'] = 'Text/ID'
            else:
                col_info['type'] = 'Categorical'
                categorical_cols.append(col)

        profile[col] = col_info

    # Test homoscedasticity (Levene's test) if we have 1 numeric and 1 categorical
    if len(numeric_cols) == 1 and len(categorical_cols) == 1:
        num_col = numeric_cols[0]
        cat_col = categorical_cols[0]

        try:
            groups = df[cat_col].dropna().unique()
            if len(groups) >= 2:
                groups_data = [df[df[cat_col] == g][num_col].dropna() for g in groups]
                # Filter out empty groups
                groups_data = [g for g in groups_data if len(g) > 0]

                if len(groups_data) >= 2:
                    stat, p_value = stats.levene(*groups_data)
                    assumptions['homoscedasticity'] = p_value > 0.05
                    assumptions['p_levene'] = p_value
        except Exception as e:
            assumptions['homoscedasticity'] = None
            assumptions['p_levene'] = None
            assumptions['error'] = str(e)

    return profile, assumptions


# ============================================================================
# STEP 7: Recommendation Engine
# ============================================================================

def get_recommendation(profile, assumptions, selected_cols):
    """
    Recommend appropriate statistical test based on data characteristics.

    Parameters:
    -----------
    profile : dict
        Column profile information
    assumptions : dict
        Assumption test results
    selected_cols : list
        Selected column names

    Returns:
    --------
    dict
        Recommendation with test name, key, and rationale
    """
    # Count variable types
    num_numeric = len([p for p in profile.values() if p['type'] == 'Numeric'])
    num_categorical = len([p for p in profile.values() if p['type'] == 'Categorical'])

    # Scenario 1: 1 Categorical + 1 Numeric
    if num_numeric == 1 and num_categorical == 1:
        # Find which column is which
        cat_col_name = [k for k, v in profile.items() if v['type'] == 'Categorical'][0]
        num_col_name = [k for k, v in profile.items() if v['type'] == 'Numeric'][0]

        num_groups = profile[cat_col_name]['unique_values']
        is_normal = profile[num_col_name].get('is_normal', False)
        is_homoscedastic = assumptions.get('homoscedasticity', False)

        if num_groups == 2:
            # Two groups
            if is_normal and is_homoscedastic:
                return {
                    'name': 'Test t dla prób niezależnych',
                    'key': 'ttest_ind',
                    'reason': f'Dane są normalne (p={profile[num_col_name].get("p_shapiro", "N/A"):.3f}) i wariancje są równe (p={assumptions.get("p_levene", "N/A"):.3f})'
                }
            elif is_normal and not is_homoscedastic:
                return {
                    'name': 'Test t Welcha',
                    'key': 'ttest_welch',
                    'reason': f'Dane są normalne, ale wariancje różnią się (p Levene={assumptions.get("p_levene", "N/A"):.3f})'
                }
            else:
                return {
                    'name': 'Test U Manna-Whitneya',
                    'key': 'mannwhitneyu',
                    'reason': f'Dane nie mają rozkładu normalnego (p Shapiro={profile[num_col_name].get("p_shapiro", "N/A"):.3f})'
                }
        elif num_groups > 2:
            # Three or more groups
            if is_normal:
                # ANOVA is recommended when data are normal
                # ANOVA is robust to violations of homogeneity of variance, especially with balanced designs
                if is_homoscedastic:
                    return {
                        'name': 'Jednoczynnikowa ANOVA',
                        'key': 'anova',
                        'reason': f'Dane są normalne (p={profile[num_col_name].get("p_shapiro", "N/A"):.3f}) i wariancje równe (p Levene={assumptions.get("p_levene", "N/A"):.3f}), porównujemy {num_groups} grup'
                    }
                else:
                    return {
                        'name': 'Jednoczynnikowa ANOVA',
                        'key': 'anova',
                        'reason': f'Dane są normalne (p={profile[num_col_name].get("p_shapiro", "N/A"):.3f}), porównujemy {num_groups} grup. Wariancje nierówne (p Levene={assumptions.get("p_levene", "N/A"):.3f}), ale ANOVA jest odporna przy równolicznych grupach. Użyj testu Games-Howell dla porównań post-hoc.'
                    }
            else:
                return {
                    'name': 'Test Kruskala-Wallisa',
                    'key': 'kruskal',
                    'reason': f'Dane nie mają rozkładu normalnego (p Shapiro={profile[num_col_name].get("p_shapiro", "N/A"):.3f}) przy {num_groups} grupach'
                }
        else:
            return {
                'name': 'Brak testu',
                'key': None,
                'reason': 'Zmienna kategoryczna ma mniej niż 2 grupy'
            }

    # Scenario 2: 2 Numeric variables (correlation)
    elif num_numeric == 2 and num_categorical == 0:
        # Check if both are normal
        numeric_cols = [k for k, v in profile.items() if v['type'] == 'Numeric']
        both_normal = all(profile[col].get('is_normal', False) for col in numeric_cols)

        if both_normal:
            return {
                'name': 'Korelacja Pearsona',
                'key': 'pearsonr',
                'reason': 'Obie zmienne mają rozkład normalny'
            }
        else:
            return {
                'name': 'Korelacja rang Spearmana',
                'key': 'spearmanr',
                'reason': 'Co najmniej jedna zmienna nie ma rozkładu normalnego'
            }

    # Scenario 3: 2 Categorical variables
    elif num_numeric == 0 and num_categorical == 2:
        return {
            'name': 'Test Chi-kwadrat',
            'key': 'chi2',
            'reason': 'Test niezależności dla dwóch zmiennych kategorycznych'
        }

    # Scenario 4: 2 Numeric variables (paired data)
    # Note: This requires user to indicate pairing
    # For now, we'll handle this separately

    else:
        return {
            'name': 'Brak testu',
            'key': None,
            'reason': f'Wybrana kombinacja ({num_numeric} numeryczne, {num_categorical} kategoryczne) nie pasuje do żadnego standardowego testu. Wybierz inną kombinację kolumn.'
        }


# ============================================================================
# STEP 10: Individual Test Functions
# ============================================================================

def run_ttest_ind(df, cols):
    """Independent t-test"""
    # Identify numeric and categorical columns
    num_col = None
    cat_col = None

    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_col = col
        else:
            if df[col].nunique() <= 50:
                cat_col = col

    if not num_col or not cat_col:
        return {'error': 'Nie można zidentyfikować kolumn'}

    groups = df[cat_col].unique()
    if len(groups) != 2:
        return {'error': 'Test t wymaga dokładnie 2 grup'}

    data1 = df[df[cat_col] == groups[0]][num_col].dropna()
    data2 = df[df[cat_col] == groups[1]][num_col].dropna()

    stat, p_value = stats.ttest_ind(data1, data2, equal_var=True)

    # Calculate Cohen's d
    pooled_std = np.sqrt(((len(data1)-1)*data1.std()**2 + (len(data2)-1)*data2.std()**2) / (len(data1)+len(data2)-2))
    cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0

    h0 = f"Średnia wartość '{num_col}' jest taka sama dla grupy '{groups[0]}' i '{groups[1]}'."
    h1 = f"Średnia wartość '{num_col}' różni się istotnie między grupami."

    alpha = 0.05
    if p_value < alpha:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest mniejsza niż {alpha}, odrzucamy hipotezę zerową (H0)."
        conclusion = "Stwierdzono statystycznie istotną różnicę między grupami."
    else:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest większa niż {alpha}, brak podstaw do odrzucenia hipotezy zerowej (H0)."
        conclusion = "Nie stwierdzono statystycznie istotnej różnicy między grupami."

    return {
        'test_name': 'Test t dla prób niezależnych',
        'statistic_name': 't-statistic',
        'statistic_value': stat,
        'p_value': p_value,
        'effect_size': cohens_d,
        'effect_size_name': "Cohen's d",
        'h0': h0,
        'h1': h1,
        'interpretation': interpretation,
        'conclusion': conclusion,
        'group_means': {
            str(groups[0]): data1.mean(),
            str(groups[1]): data2.mean()
        }
    }


def run_ttest_welch(df, cols):
    """Welch's t-test (unequal variances)"""
    # Identify numeric and categorical columns
    num_col = None
    cat_col = None

    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_col = col
        else:
            if df[col].nunique() <= 50:
                cat_col = col

    if not num_col or not cat_col:
        return {'error': 'Nie można zidentyfikować kolumn'}

    groups = df[cat_col].unique()
    if len(groups) != 2:
        return {'error': 'Test t wymaga dokładnie 2 grup'}

    data1 = df[df[cat_col] == groups[0]][num_col].dropna()
    data2 = df[df[cat_col] == groups[1]][num_col].dropna()

    stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)

    # Calculate Cohen's d
    pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
    cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0

    h0 = f"Średnia wartość '{num_col}' jest taka sama dla grupy '{groups[0]}' i '{groups[1]}'."
    h1 = f"Średnia wartość '{num_col}' różni się istotnie między grupami."

    alpha = 0.05
    if p_value < alpha:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest mniejsza niż {alpha}, odrzucamy hipotezę zerową (H0)."
        conclusion = "Stwierdzono statystycznie istotną różnicę między grupami."
    else:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest większa niż {alpha}, brak podstaw do odrzucenia hipotezy zerowej (H0)."
        conclusion = "Nie stwierdzono statystycznie istotnej różnicy między grupami."

    return {
        'test_name': 'Test t Welcha',
        'statistic_name': 't-statistic',
        'statistic_value': stat,
        'p_value': p_value,
        'effect_size': cohens_d,
        'effect_size_name': "Cohen's d",
        'h0': h0,
        'h1': h1,
        'interpretation': interpretation,
        'conclusion': conclusion,
        'group_means': {
            str(groups[0]): data1.mean(),
            str(groups[1]): data2.mean()
        }
    }


def run_mannwhitneyu(df, cols):
    """Mann-Whitney U test"""
    # Identify numeric and categorical columns
    num_col = None
    cat_col = None

    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_col = col
        else:
            if df[col].nunique() <= 50:
                cat_col = col

    if not num_col or not cat_col:
        return {'error': 'Nie można zidentyfikować kolumn'}

    groups = df[cat_col].unique()
    if len(groups) != 2:
        return {'error': 'Test U Manna-Whitneya wymaga dokładnie 2 grup'}

    data1 = df[df[cat_col] == groups[0]][num_col].dropna()
    data2 = df[df[cat_col] == groups[1]][num_col].dropna()

    stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

    # Calculate rank-biserial correlation
    n1, n2 = len(data1), len(data2)
    r = 1 - (2*stat) / (n1 * n2)

    h0 = f"Rozkłady '{num_col}' są takie same dla grupy '{groups[0]}' i '{groups[1]}'."
    h1 = f"Rozkłady '{num_col}' różnią się między grupami."

    alpha = 0.05
    if p_value < alpha:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest mniejsza niż {alpha}, odrzucamy hipotezę zerową (H0)."
        conclusion = "Stwierdzono statystycznie istotną różnicę między grupami."
    else:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest większa niż {alpha}, brak podstaw do odrzucenia hipotezy zerowej (H0)."
        conclusion = "Nie stwierdzono statystycznie istotnej różnicy między grupami."

    return {
        'test_name': 'Test U Manna-Whitneya',
        'statistic_name': 'U-statistic',
        'statistic_value': stat,
        'p_value': p_value,
        'effect_size': r,
        'effect_size_name': 'Rank-biserial correlation',
        'h0': h0,
        'h1': h1,
        'interpretation': interpretation,
        'conclusion': conclusion,
        'group_medians': {
            str(groups[0]): data1.median(),
            str(groups[1]): data2.median()
        }
    }


def run_anova(df, cols):
    """One-way ANOVA"""
    # Identify numeric and categorical columns
    num_col = None
    cat_col = None

    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_col = col
        else:
            if df[col].nunique() <= 50:
                cat_col = col

    if not num_col or not cat_col:
        return {'error': 'Nie można zidentyfikować kolumn'}

    # Clean data
    df_clean = df[[num_col, cat_col]].dropna()

    try:
        # Use statsmodels for full ANOVA table
        model = ols(f'Q("{num_col}") ~ C(Q("{cat_col}"))', data=df_clean).fit()
        anova_table = anova_lm(model, typ=2)

        p_value = anova_table.loc[f'C(Q("{cat_col}"))', 'PR(>F)']
        f_value = anova_table.loc[f'C(Q("{cat_col}"))', 'F']

        # Calculate eta-squared
        ss_effect = anova_table.loc[f'C(Q("{cat_col}"))', 'sum_sq']
        ss_total = anova_table['sum_sq'].sum()
        eta_squared = ss_effect / ss_total

    except Exception as e:
        return {'error': f'Błąd podczas wykonywania ANOVA: {str(e)}'}

    groups = df_clean[cat_col].unique()
    h0 = f"Średnie wartości '{num_col}' są takie same dla wszystkich grup."
    h1 = f"Co najmniej jedna grupa różni się średnią wartością '{num_col}'."

    alpha = 0.05
    if p_value < alpha:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest mniejsza niż {alpha}, odrzucamy hipotezę zerową (H0)."
        conclusion = f"Stwierdzono statystycznie istotne różnice między grupami. Zalecane testy post-hoc (np. Tukey HSD)."
    else:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest większa niż {alpha}, brak podstaw do odrzucenia hipotezy zerowej (H0)."
        conclusion = "Nie stwierdzono statystycznie istotnych różnic między grupami."

    # Calculate group means
    group_means = {}
    for group in groups:
        group_means[str(group)] = df_clean[df_clean[cat_col] == group][num_col].mean()

    return {
        'test_name': 'Jednoczynnikowa ANOVA',
        'statistic_name': 'F-statistic',
        'statistic_value': f_value,
        'p_value': p_value,
        'effect_size': eta_squared,
        'effect_size_name': 'Eta-squared (η²)',
        'h0': h0,
        'h1': h1,
        'interpretation': interpretation,
        'conclusion': conclusion,
        'group_means': group_means
    }


def run_kruskal(df, cols):
    """Kruskal-Wallis test"""
    # Identify numeric and categorical columns
    num_col = None
    cat_col = None

    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_col = col
        else:
            if df[col].nunique() <= 50:
                cat_col = col

    if not num_col or not cat_col:
        return {'error': 'Nie można zidentyfikować kolumn'}

    groups = df[cat_col].unique()
    groups_data = [df[df[cat_col] == g][num_col].dropna() for g in groups]

    # Filter out empty groups
    groups_data = [g for g in groups_data if len(g) > 0]

    if len(groups_data) < 2:
        return {'error': 'Potrzeba co najmniej 2 grup z danymi'}

    stat, p_value = stats.kruskal(*groups_data)

    # Calculate epsilon-squared
    n = sum(len(g) for g in groups_data)
    epsilon_squared = (stat - len(groups_data) + 1) / (n - len(groups_data))

    h0 = f"Rozkłady '{num_col}' są takie same dla wszystkich grup."
    h1 = f"Co najmniej jedna grupa ma inny rozkład '{num_col}'."

    alpha = 0.05
    if p_value < alpha:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest mniejsza niż {alpha}, odrzucamy hipotezę zerową (H0)."
        conclusion = "Stwierdzono statystycznie istotne różnice między grupami. Zalecane testy post-hoc (np. Dunn's test)."
    else:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest większa niż {alpha}, brak podstaw do odrzucenia hipotezy zerowej (H0)."
        conclusion = "Nie stwierdzono statystycznie istotnych różnic między grupami."

    # Calculate group medians
    group_medians = {}
    for i, group in enumerate(groups):
        if i < len(groups_data):
            group_medians[str(group)] = groups_data[i].median()

    return {
        'test_name': 'Test Kruskala-Wallisa',
        'statistic_name': 'H-statistic',
        'statistic_value': stat,
        'p_value': p_value,
        'effect_size': epsilon_squared,
        'effect_size_name': 'Epsilon-squared (ε²)',
        'h0': h0,
        'h1': h1,
        'interpretation': interpretation,
        'conclusion': conclusion,
        'group_medians': group_medians
    }


def run_pearsonr(df, cols):
    """Pearson correlation"""
    if len(cols) != 2:
        return {'error': 'Korelacja wymaga dokładnie 2 kolumn'}

    col1, col2 = cols

    # Clean data
    df_clean = df[[col1, col2]].dropna()

    r, p_value = stats.pearsonr(df_clean[col1], df_clean[col2])

    h0 = f"Nie ma liniowej korelacji między '{col1}' a '{col2}' (r = 0)."
    h1 = f"Istnieje liniowa korelacja między '{col1}' a '{col2}'."

    alpha = 0.05
    if p_value < alpha:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest mniejsza niż {alpha}, odrzucamy hipotezę zerową (H0)."
        if abs(r) < 0.3:
            strength = "słabą"
        elif abs(r) < 0.7:
            strength = "umiarkowaną"
        else:
            strength = "silną"
        direction = "dodatnią" if r > 0 else "ujemną"
        conclusion = f"Stwierdzono statystycznie istotną {strength} korelację {direction} (r = {r:.3f})."
    else:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest większa niż {alpha}, brak podstaw do odrzucenia hipotezy zerowej (H0)."
        conclusion = "Nie stwierdzono statystycznie istotnej korelacji."

    return {
        'test_name': 'Korelacja Pearsona',
        'statistic_name': 'Współczynnik korelacji (r)',
        'statistic_value': r,
        'p_value': p_value,
        'effect_size': r,
        'effect_size_name': 'Współczynnik korelacji Pearsona (r)',
        'h0': h0,
        'h1': h1,
        'interpretation': interpretation,
        'conclusion': conclusion
    }


def run_spearmanr(df, cols):
    """Spearman correlation"""
    if len(cols) != 2:
        return {'error': 'Korelacja wymaga dokładnie 2 kolumn'}

    col1, col2 = cols

    # Clean data
    df_clean = df[[col1, col2]].dropna()

    rho, p_value = stats.spearmanr(df_clean[col1], df_clean[col2])

    h0 = f"Nie ma monotonicznej korelacji między '{col1}' a '{col2}' (ρ = 0)."
    h1 = f"Istnieje monotoniczna korelacja między '{col1}' a '{col2}'."

    alpha = 0.05
    if p_value < alpha:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest mniejsza niż {alpha}, odrzucamy hipotezę zerową (H0)."
        if abs(rho) < 0.3:
            strength = "słabą"
        elif abs(rho) < 0.7:
            strength = "umiarkowaną"
        else:
            strength = "silną"
        direction = "dodatnią" if rho > 0 else "ujemną"
        conclusion = f"Stwierdzono statystycznie istotną {strength} korelację monotoniczną {direction} (ρ = {rho:.3f})."
    else:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest większa niż {alpha}, brak podstaw do odrzucenia hipotezy zerowej (H0)."
        conclusion = "Nie stwierdzono statystycznie istotnej korelacji monotonicznej."

    return {
        'test_name': 'Korelacja Spearmana',
        'statistic_name': "Współczynnik korelacji Spearmana (ρ)",
        'statistic_value': rho,
        'p_value': p_value,
        'effect_size': rho,
        'effect_size_name': "Współczynnik korelacji Spearmana (ρ)",
        'h0': h0,
        'h1': h1,
        'interpretation': interpretation,
        'conclusion': conclusion
    }


def run_chi2(df, cols):
    """Chi-square test of independence"""
    if len(cols) != 2:
        return {'error': 'Test Chi-kwadrat wymaga dokładnie 2 kolumn kategorycznych'}

    col1, col2 = cols

    # Create contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Calculate Cramér's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    h0 = f"Zmienne '{col1}' i '{col2}' są od siebie niezależne."
    h1 = f"Zmienne '{col1}' i '{col2}' są zależne (istnieje związek)."

    alpha = 0.05
    if p_value < alpha:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest mniejsza niż {alpha}, odrzucamy hipotezę zerową (H0)."
        conclusion = f"Stwierdzono statystycznie istotny związek między zmiennymi (V = {cramers_v:.3f})."
    else:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest większa niż {alpha}, brak podstaw do odrzucenia hipotezy zerowej (H0)."
        conclusion = "Nie stwierdzono statystycznie istotnego związku między zmiennymi."

    return {
        'test_name': 'Test Chi-kwadrat niezależności',
        'statistic_name': 'Chi-kwadrat (χ²)',
        'statistic_value': chi2,
        'p_value': p_value,
        'effect_size': cramers_v,
        'effect_size_name': "Cramér's V",
        'h0': h0,
        'h1': h1,
        'interpretation': interpretation,
        'conclusion': conclusion,
        'degrees_of_freedom': dof,
        'contingency_table': contingency_table
    }


def run_fisher_exact(df, cols):
    """Fisher's exact test"""
    if len(cols) != 2:
        return {'error': 'Test Fishera wymaga dokładnie 2 kolumn kategorycznych'}

    col1, col2 = cols

    # Create contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])

    # Fisher's exact test works only for 2x2 tables
    if contingency_table.shape != (2, 2):
        return {'error': 'Test dokładny Fishera działa tylko dla tabel 2x2'}

    oddsratio, p_value = stats.fisher_exact(contingency_table)

    h0 = f"Zmienne '{col1}' i '{col2}' są od siebie niezależne."
    h1 = f"Zmienne '{col1}' i '{col2}' są zależne (istnieje związek)."

    alpha = 0.05
    if p_value < alpha:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest mniejsza niż {alpha}, odrzucamy hipotezę zerową (H0)."
        conclusion = f"Stwierdzono statystycznie istotny związek między zmiennymi (OR = {oddsratio:.3f})."
    else:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest większa niż {alpha}, brak podstaw do odrzucenia hipotezy zerowej (H0)."
        conclusion = "Nie stwierdzono statystycznie istotnego związku między zmiennymi."

    return {
        'test_name': 'Test dokładny Fishera',
        'statistic_name': 'Iloraz szans (Odds Ratio)',
        'statistic_value': oddsratio,
        'p_value': p_value,
        'effect_size': oddsratio,
        'effect_size_name': 'Odds Ratio',
        'h0': h0,
        'h1': h1,
        'interpretation': interpretation,
        'conclusion': conclusion,
        'contingency_table': contingency_table
    }


def run_ttest_rel(df, cols):
    """Paired t-test"""
    if len(cols) != 2:
        return {'error': 'Test t dla prób zależnych wymaga dokładnie 2 kolumn numerycznych'}

    col1, col2 = cols

    # Clean data (paired)
    df_clean = df[[col1, col2]].dropna()

    stat, p_value = stats.ttest_rel(df_clean[col1], df_clean[col2])

    # Calculate Cohen's d for paired data
    diff = df_clean[col1] - df_clean[col2]
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

    h0 = f"Średnia różnica między '{col1}' a '{col2}' wynosi zero."
    h1 = f"Średnia różnica między '{col1}' a '{col2}' jest różna od zera."

    alpha = 0.05
    if p_value < alpha:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest mniejsza niż {alpha}, odrzucamy hipotezę zerową (H0)."
        conclusion = "Stwierdzono statystycznie istotną różnicę między pomiarami."
    else:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest większa niż {alpha}, brak podstaw do odrzucenia hipotezy zerowej (H0)."
        conclusion = "Nie stwierdzono statystycznie istotnej różnicy między pomiarami."

    return {
        'test_name': 'Test t dla prób zależnych (sparowanych)',
        'statistic_name': 't-statistic',
        'statistic_value': stat,
        'p_value': p_value,
        'effect_size': cohens_d,
        'effect_size_name': "Cohen's d (paired)",
        'h0': h0,
        'h1': h1,
        'interpretation': interpretation,
        'conclusion': conclusion,
        'mean_difference': diff.mean()
    }


def run_wilcoxon(df, cols):
    """Wilcoxon signed-rank test"""
    if len(cols) != 2:
        return {'error': 'Test Wilcoxona wymaga dokładnie 2 kolumn numerycznych'}

    col1, col2 = cols

    # Clean data (paired)
    df_clean = df[[col1, col2]].dropna()

    stat, p_value = stats.wilcoxon(df_clean[col1], df_clean[col2])

    # Calculate matched-pairs rank biserial correlation
    diff = df_clean[col1] - df_clean[col2]
    n = len(diff)
    r = stat / (n * (n + 1) / 4) - 1

    h0 = f"Mediana różnicy między '{col1}' a '{col2}' wynosi zero."
    h1 = f"Mediana różnicy między '{col1}' a '{col2}' jest różna od zera."

    alpha = 0.05
    if p_value < alpha:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest mniejsza niż {alpha}, odrzucamy hipotezę zerową (H0)."
        conclusion = "Stwierdzono statystycznie istotną różnicę między pomiarami."
    else:
        interpretation = f"Ponieważ p-wartość ({p_value:.4f}) jest większa niż {alpha}, brak podstaw do odrzucenia hipotezy zerowej (H0)."
        conclusion = "Nie stwierdzono statystycznie istotnej różnicy między pomiarami."

    return {
        'test_name': 'Test rang Wilcoxona',
        'statistic_name': 'W-statistic',
        'statistic_value': stat,
        'p_value': p_value,
        'effect_size': r,
        'effect_size_name': 'Rank-biserial correlation',
        'h0': h0,
        'h1': h1,
        'interpretation': interpretation,
        'conclusion': conclusion,
        'median_difference': diff.median()
    }


# ============================================================================
# STEP 9: Statistical Test Router
# ============================================================================

def run_statistical_test(df, cols, key):
    """
    Route to appropriate statistical test function based on key.

    Parameters:
    -----------
    df : DataFrame
        The data
    cols : list
        Column names to analyze
    key : str
        Test key from recommendation

    Returns:
    --------
    dict
        Test results
    """
    test_functions = {
        'ttest_ind': run_ttest_ind,
        'ttest_welch': run_ttest_welch,
        'mannwhitneyu': run_mannwhitneyu,
        'anova': run_anova,
        'kruskal': run_kruskal,
        'pearsonr': run_pearsonr,
        'spearmanr': run_spearmanr,
        'chi2': run_chi2,
        'fisher_exact': run_fisher_exact,
        'ttest_rel': run_ttest_rel,
        'wilcoxon': run_wilcoxon
    }

    if key in test_functions:
        return test_functions[key](df, cols)
    else:
        return {'error': 'Nie znaleziono funkcji testującej dla klucza: ' + str(key)}


# ============================================================================
# STEP 1: Main Module Function
# ============================================================================

def run_recommender_tool():
    """Main function for the statistical recommender tool"""

    st.title("🤖 Automatyczny Rekomender Testów Statystycznych")
    st.markdown("""
    Ten moduł automatycznie analizuje Twoje dane, sprawdza założenia statystyczne
    i rekomenduje oraz wykonuje odpowiednie testy statystyczne.
    """)

    # STEP 2: File Upload UI
    st.header("Krok 1: Wgraj swój plik z danymi")
    uploaded_file = st.file_uploader(
        "Wybierz plik",
        type=["csv", "tsv", "xlsx"],
        on_change=clear_session_state_on_new_file
    )

    if uploaded_file is not None:
        # STEP 2: Parsing options
        with st.expander("Opcje ładowania danych"):
            if uploaded_file.name.endswith(('.csv', '.tsv')):
                delimiter = st.radio(
                    "Wybierz separator (delimiter):",
                    (',', ';', '\t', ' '),
                    horizontal=True,
                    key='file_delimiter'
                )
            elif uploaded_file.name.endswith('.xlsx'):
                sheet_name = st.text_input(
                    "Podaj nazwę arkusza (zostaw puste dla pierwszego)",
                    key='file_sheet'
                )

        # STEP 3 & 4: Load and display data
        delimiter_opt = st.session_state.get('file_delimiter', ',')
        sheet_opt = st.session_state.get('file_sheet', None)

        df = load_data(uploaded_file, delimiter_opt, sheet_opt)

        if df is not None:
            st.session_state['data'] = df

            st.header("Podgląd Danych (pierwsze 5 wierszy)")
            st.dataframe(df.head())

            st.info(f"**Wczytano:** {df.shape[0]} wierszy, {df.shape[1]} kolumn")

    # Main analysis flow (Steps 5-11)
    if 'data' in st.session_state:
        df = st.session_state['data']

        # STEP 5: Column selection
        st.header("Krok 2: Wybierz kolumny do analizy")
        st.info("Wybierz kolumny, które chcesz porównać. Rekomendacja będzie zależeć od ich typów i liczby.")

        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Wybierz kolumny:",
            all_columns,
            key='selected_columns',
            on_change=clear_analysis_results
        )

        # Add checkbox for paired data
        is_paired = st.checkbox(
            "Dane są sparowane (pomiary przed/po, matched pairs)",
            help="Zaznacz, jeśli wybierasz 2 kolumny numeryczne reprezentujące pomiary na tych samych jednostkach"
        )

        # STEP 6 & 7: Profiling and recommendation
        if len(selected_columns) > 0:
            if 'profile' not in st.session_state or st.session_state.get('selected_columns') != selected_columns:
                profile, assumptions = profile_and_test_columns(df, selected_columns)
                st.session_state['profile'] = profile
                st.session_state['assumptions'] = assumptions

            # Display profile
            st.header("Krok 3: Automatyczna Analiza Założeń")
            with st.expander("Zobacz szczegóły analizy kolumn", expanded=False):
                for col, info in st.session_state['profile'].items():
                    st.subheader(f"Kolumna: {col}")
                    st.write(f"**Typ:** {info['type']}")
                    if info['type'] == 'Numeric':
                        if info.get('is_normal') is not None:
                            normality = "TAK ✓" if info['is_normal'] else "NIE ✗"
                            st.write(f"**Rozkład normalny:** {normality} (p = {info.get('p_shapiro', 'N/A'):.4f})")
                        else:
                            st.write("**Rozkład normalny:** Nie można określić (za mało danych)")
                    elif info['type'] == 'Categorical':
                        st.write(f"**Liczba unikalnych wartości:** {info['unique_values']}")

                if st.session_state['assumptions']:
                    st.subheader("Testy Interakcji")
                    for test, result in st.session_state['assumptions'].items():
                        if test == 'homoscedasticity':
                            homo = "TAK ✓" if result else "NIE ✗"
                            p_lev = st.session_state['assumptions'].get('p_levene', 'N/A')
                            st.write(f"**Równość wariancji (test Levene'a):** {homo} (p = {p_lev:.4f})")

            # Get recommendation (modified for paired data)
            if is_paired and len(selected_columns) == 2:
                # Override recommendation for paired data
                profile = st.session_state['profile']
                both_numeric = all(profile[col]['type'] == 'Numeric' for col in selected_columns)

                if both_numeric:
                    # Check normality of differences
                    col1, col2 = selected_columns
                    df_clean = df[[col1, col2]].dropna()
                    diff = df_clean[col1] - df_clean[col2]

                    try:
                        if len(diff) >= 3:
                            stat, p_val = stats.shapiro(diff)
                            is_normal = p_val > 0.05
                        else:
                            is_normal = False
                    except:
                        is_normal = False

                    if is_normal:
                        rec = {
                            'name': 'Test t dla prób zależnych (sparowanych)',
                            'key': 'ttest_rel',
                            'reason': 'Dane są sparowane i różnice mają rozkład normalny'
                        }
                    else:
                        rec = {
                            'name': 'Test rang Wilcoxona',
                            'key': 'wilcoxon',
                            'reason': 'Dane są sparowane, ale różnice nie mają rozkładu normalnego'
                        }
                else:
                    rec = {
                        'name': 'Brak testu',
                        'key': None,
                        'reason': 'Dla danych sparowanych wybierz 2 kolumny numeryczne'
                    }
            else:
                rec = get_recommendation(st.session_state['profile'], st.session_state['assumptions'], selected_columns)

            st.session_state['recommendation'] = rec

            # Display recommendation
            st.header("Krok 4: Rekomendacja Testu")
            if rec['key'] is not None:
                st.success(f"**Sugerowany test:** {rec['name']}")
            else:
                st.warning(f"**Brak rekomendacji:** {rec['name']}")
            st.info(f"**Uzasadnienie:** {rec['reason']}")

            # STEP 8: Analysis execution button
            if rec['key'] is not None:
                if st.button(f"✅ Tak, wykonaj analizę ({rec['name']})", type="primary"):
                    st.session_state['run_analysis_flag'] = True

        # STEP 9: Execute analysis
        if st.session_state.get('run_analysis_flag', False):
            df = st.session_state['data']
            cols = st.session_state['selected_columns']
            rec_key = st.session_state['recommendation']['key']

            with st.spinner("Przeprowadzanie analizy statystycznej..."):
                results = run_statistical_test(df, cols, rec_key)
                st.session_state['results'] = results

            st.session_state['run_analysis_flag'] = False

        # STEP 11: Display results
        if 'results' in st.session_state:
            results = st.session_state['results']

            if 'error' in results:
                st.error(f"Błąd: {results['error']}")
            else:
                st.header(f"Krok 5: Wyniki Analizy ({results['test_name']})")
                st.markdown("---")

                # Hypotheses
                st.subheader("Sformułowane Hipotezy")
                st.markdown(f"**Hipoteza Zerowa (H0):** {results['h0']}")
                st.markdown(f"**Hipoteza Alternatywna (H1):** {results['h1']}")

                # Test results
                st.subheader("Wyniki Testu")
                col1, col2, col3 = st.columns(3)
                col1.metric(results['statistic_name'], f"{results['statistic_value']:.4f}")
                col2.metric("P-wartość (p-value)", f"{results['p_value']:.4f}")
                if 'effect_size' in results:
                    col3.metric(results.get('effect_size_name', 'Effect Size'), f"{results['effect_size']:.4f}")

                # Interpretation
                st.subheader("Interpretacja i Wnioski (dla poziomu istotności α = 0.05)")
                st.info(f"**Interpretacja:** {results['interpretation']}")

                # Visual indicator for significance
                if results['p_value'] < 0.05:
                    st.success(f"**Wniosek:** {results['conclusion']}")
                else:
                    st.warning(f"**Wniosek:** {results['conclusion']}")

                # Additional information
                if 'group_means' in results:
                    with st.expander("Szczegóły grup"):
                        st.write("**Średnie grup:**")
                        for group, mean in results['group_means'].items():
                            st.write(f"- {group}: {mean:.4f}")

                if 'group_medians' in results:
                    with st.expander("Szczegóły grup"):
                        st.write("**Mediany grup:**")
                        for group, median in results['group_medians'].items():
                            st.write(f"- {group}: {median:.4f}")

                if 'contingency_table' in results:
                    with st.expander("Tabela kontyngencji"):
                        st.dataframe(results['contingency_table'])
