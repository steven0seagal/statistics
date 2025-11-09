# Demo Datasets for Statistical Tests

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
