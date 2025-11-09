"""
Generate Demo Datasets for Statistical Tests
==============================================

This script generates synthetic demo datasets for each statistical test type.
"""

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Output directory
output_dir = "demo_data"

# ============================================================================
# 1. Independent T-Test (normal distributions, equal variances)
# ============================================================================
print("Generating: demo_ttest_independent.csv")
n = 50
group_a = np.random.normal(75, 10, n)  # Mean=75, SD=10
group_b = np.random.normal(82, 10, n)  # Mean=82, SD=10

df_ttest = pd.DataFrame({
    'Wynik_Testu': np.concatenate([group_a, group_b]),
    'Grupa': ['Kontrola'] * n + ['Terapia'] * n
})
df_ttest.to_csv(f"{output_dir}/demo_ttest_independent.csv", index=False)


# ============================================================================
# 2. Welch's T-Test (normal distributions, unequal variances)
# ============================================================================
print("Generating: demo_ttest_welch.csv")
n = 50
group_a = np.random.normal(75, 8, n)   # Mean=75, SD=8
group_b = np.random.normal(82, 18, n)  # Mean=82, SD=18 (larger variance)

df_welch = pd.DataFrame({
    'Poziom_Cholesterolu': np.concatenate([group_a, group_b]),
    'Dieta': ['Standardowa'] * n + ['Niskotluszczowa'] * n
})
df_welch.to_csv(f"{output_dir}/demo_ttest_welch.csv", index=False)


# ============================================================================
# 3. Mann-Whitney U Test (non-normal distributions)
# ============================================================================
print("Generating: demo_mannwhitney.csv")
n = 50
# Use exponential distribution (highly skewed, non-normal)
group_a = np.random.exponential(scale=20, size=n)
group_b = np.random.exponential(scale=30, size=n)

df_mw = pd.DataFrame({
    'Czas_Rekonwalescencji_Dni': np.concatenate([group_a, group_b]),
    'Metoda_Leczenia': ['Metoda_A'] * n + ['Metoda_B'] * n
})
df_mw.to_csv(f"{output_dir}/demo_mannwhitney.csv", index=False)


# ============================================================================
# 4. One-Way ANOVA (3+ groups, normal, equal variances)
# ============================================================================
print("Generating: demo_anova.csv")
n = 40
group1 = np.random.normal(100, 15, n)  # Mean=100, SD=15
group2 = np.random.normal(115, 15, n)  # Mean=115, SD=15
group3 = np.random.normal(125, 15, n)  # Mean=125, SD=15

df_anova = pd.DataFrame({
    'Wzrost_Roslin_cm': np.concatenate([group1, group2, group3]),
    'Rodzaj_Nawozu': ['Nawoz_A'] * n + ['Nawoz_B'] * n + ['Nawoz_C'] * n
})
df_anova.to_csv(f"{output_dir}/demo_anova.csv", index=False)


# ============================================================================
# 5. Kruskal-Wallis Test (3+ groups, non-normal)
# ============================================================================
print("Generating: demo_kruskal.csv")
n = 40
# Use gamma distribution (skewed, non-normal)
group1 = np.random.gamma(shape=2, scale=5, size=n)
group2 = np.random.gamma(shape=2, scale=7, size=n)
group3 = np.random.gamma(shape=2, scale=10, size=n)
group4 = np.random.gamma(shape=2, scale=6, size=n)

df_kruskal = pd.DataFrame({
    'Ocena_Zadowolenia': np.concatenate([group1, group2, group3, group4]),
    'Klinika': ['Klinika_A'] * n + ['Klinika_B'] * n + ['Klinika_C'] * n + ['Klinika_D'] * n
})
df_kruskal.to_csv(f"{output_dir}/demo_kruskal.csv", index=False)


# ============================================================================
# 6. Pearson Correlation (both variables normal, linear relationship)
# ============================================================================
print("Generating: demo_pearson.csv")
n = 100
wiek = np.random.normal(45, 12, n)
# Create correlated variable with some noise
cisnienie = 90 + 0.8 * wiek + np.random.normal(0, 8, n)

df_pearson = pd.DataFrame({
    'Wiek_Lat': wiek,
    'Cisnienie_Skurczowe': cisnienie
})
df_pearson.to_csv(f"{output_dir}/demo_pearson.csv", index=False)


# ============================================================================
# 7. Spearman Correlation (non-normal, monotonic relationship)
# ============================================================================
print("Generating: demo_spearman.csv")
n = 100
# Use log-normal distribution (non-normal)
dochod = np.random.lognormal(10, 0.8, n)
# Create monotonic but non-linear relationship
wydatki = 0.3 * dochod ** 0.85 + np.random.lognormal(8, 0.5, n)

df_spearman = pd.DataFrame({
    'Dochod_Roczny': dochod,
    'Wydatki_Luksusowe': wydatki
})
df_spearman.to_csv(f"{output_dir}/demo_spearman.csv", index=False)


# ============================================================================
# 8. Chi-Square Test (two categorical variables)
# ============================================================================
print("Generating: demo_chisquare.csv")
n = 200
# Create association between gender and preference
genders = np.random.choice(['Kobieta', 'Mezczyzna'], size=n, p=[0.5, 0.5])
preferences = []
for gender in genders:
    if gender == 'Kobieta':
        pref = np.random.choice(['Produkt_A', 'Produkt_B', 'Produkt_C'], p=[0.5, 0.3, 0.2])
    else:
        pref = np.random.choice(['Produkt_A', 'Produkt_B', 'Produkt_C'], p=[0.2, 0.5, 0.3])
    preferences.append(pref)

df_chi2 = pd.DataFrame({
    'Plec': genders,
    'Preferowany_Produkt': preferences
})
df_chi2.to_csv(f"{output_dir}/demo_chisquare.csv", index=False)


# ============================================================================
# 9. Fisher's Exact Test (2x2 contingency table, small sample)
# ============================================================================
print("Generating: demo_fisher.csv")
# Small sample with 2x2 table
treatments = ['Placebo'] * 20 + ['Lek'] * 20
outcomes = []
for treatment in treatments:
    if treatment == 'Placebo':
        outcome = np.random.choice(['Poprawa', 'Brak_Poprawy'], p=[0.3, 0.7])
    else:
        outcome = np.random.choice(['Poprawa', 'Brak_Poprawy'], p=[0.7, 0.3])
    outcomes.append(outcome)

df_fisher = pd.DataFrame({
    'Leczenie': treatments,
    'Wynik': outcomes
})
df_fisher.to_csv(f"{output_dir}/demo_fisher.csv", index=False)


# ============================================================================
# 10. Paired T-Test (before-after measurements, normal differences)
# ============================================================================
print("Generating: demo_ttest_paired.csv")
n = 50
# Before measurements
przed = np.random.normal(140, 20, n)
# After measurements (with improvement and some noise)
po = przed - np.random.normal(15, 8, n)  # Average decrease of 15

df_paired = pd.DataFrame({
    'Cisnienie_Przed': przed,
    'Cisnienie_Po': po
})
df_paired.to_csv(f"{output_dir}/demo_ttest_paired.csv", index=False)


# ============================================================================
# 11. Wilcoxon Signed-Rank Test (paired data, non-normal differences)
# ============================================================================
print("Generating: demo_wilcoxon.csv")
n = 50
# Before measurements
przed = np.random.gamma(shape=5, scale=4, size=n)
# After measurements with non-normal differences
po = przed - np.random.exponential(scale=2, size=n)

df_wilcoxon = pd.DataFrame({
    'Poziom_Bolu_Przed': przed,
    'Poziom_Bolu_Po': po
})
df_wilcoxon.to_csv(f"{output_dir}/demo_wilcoxon.csv", index=False)


# ============================================================================
# Create a README file
# ============================================================================
print("Generating: README.md")
readme_content = """# Demo Datasets for Statistical Tests

This directory contains synthetic demo datasets for various statistical tests.

## Files and Recommended Tests:

1. **demo_ttest_independent.csv** - Independent t-test
   - Columns: Wynik_Testu (numeric), Grupa (categorical: Kontrola, Terapia)
   - Use: Compare test scores between control and therapy groups

2. **demo_ttest_welch.csv** - Welch's t-test
   - Columns: Poziom_Cholesterolu (numeric), Dieta (categorical)
   - Use: Compare cholesterol levels between diets (unequal variances)

3. **demo_mannwhitney.csv** - Mann-Whitney U test
   - Columns: Czas_Rekonwalescencji_Dni (numeric, non-normal), Metoda_Leczenia (categorical)
   - Use: Compare recovery times between two treatment methods

4. **demo_anova.csv** - One-way ANOVA
   - Columns: Wzrost_Roslin_cm (numeric), Rodzaj_Nawozu (categorical: 3 groups)
   - Use: Compare plant growth across three fertilizer types

5. **demo_kruskal.csv** - Kruskal-Wallis test
   - Columns: Ocena_Zadowolenia (numeric, non-normal), Klinika (categorical: 4 groups)
   - Use: Compare satisfaction scores across four clinics

6. **demo_pearson.csv** - Pearson correlation
   - Columns: Wiek_Lat (numeric), Cisnienie_Skurczowe (numeric)
   - Use: Examine linear relationship between age and blood pressure

7. **demo_spearman.csv** - Spearman correlation
   - Columns: Dochod_Roczny (numeric, non-normal), Wydatki_Luksusowe (numeric, non-normal)
   - Use: Examine monotonic relationship between income and luxury spending

8. **demo_chisquare.csv** - Chi-square test
   - Columns: Plec (categorical), Preferowany_Produkt (categorical)
   - Use: Test association between gender and product preference

9. **demo_fisher.csv** - Fisher's exact test
   - Columns: Leczenie (categorical, 2 levels), Wynik (categorical, 2 levels)
   - Use: Test association in small 2x2 table

10. **demo_ttest_paired.csv** - Paired t-test
    - Columns: Cisnienie_Przed (numeric), Cisnienie_Po (numeric)
    - Use: Compare blood pressure before and after treatment (paired)

11. **demo_wilcoxon.csv** - Wilcoxon signed-rank test
    - Columns: Poziom_Bolu_Przed (numeric, non-normal), Poziom_Bolu_Po (numeric)
    - Use: Compare pain levels before and after treatment (paired, non-normal)

## How to Use:

1. Upload any of these files to the Automatic Test Recommender
2. Select the appropriate columns
3. The system will automatically recommend the correct test
4. Execute the test and view results

All datasets are synthetic and generated for demonstration purposes only.
"""

with open(f"{output_dir}/README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("\nAll demo datasets generated successfully!")
print(f"Files saved to: {output_dir}/")
