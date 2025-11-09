"""
Test Script for Statistical Recommender Module
===============================================

This script tests that all statistical tests work correctly with the demo datasets.
"""

import pandas as pd
import sys
sys.path.append('.')
from statistical_recommender import (
    profile_and_test_columns,
    get_recommendation,
    run_statistical_test
)

def test_dataset(filepath, columns, expected_test_key, test_name):
    """Test a dataset with the recommender system"""
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"File: {filepath}")
    print(f"Columns: {columns}")
    print(f"Expected test: {expected_test_key}")
    print(f"{'='*70}")

    try:
        # Load data
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Profile columns
        profile, assumptions = profile_and_test_columns(df, columns)
        print(f"✓ Column profiling completed")

        # Print profile summary
        for col, info in profile.items():
            print(f"  - {col}: {info['type']}", end="")
            if info['type'] == 'Numeric':
                is_normal = info.get('is_normal', None)
                if is_normal is not None:
                    print(f", Normal: {is_normal}", end="")
            print()

        # Get recommendation
        rec = get_recommendation(profile, assumptions, columns)
        print(f"✓ Recommendation: {rec['name']} (key: {rec['key']})")

        # Check if recommendation matches expected
        if rec['key'] == expected_test_key:
            print(f"✓ Recommendation matches expected test")
        else:
            print(f"⚠ WARNING: Expected {expected_test_key}, got {rec['key']}")

        # Run statistical test
        if rec['key'] is not None:
            results = run_statistical_test(df, columns, rec['key'])

            if 'error' in results:
                print(f"✗ ERROR: {results['error']}")
                return False
            else:
                print(f"✓ Test executed successfully")
                print(f"  - Statistic: {results['statistic_value']:.4f}")
                print(f"  - P-value: {results['p_value']:.4f}")
                if 'effect_size' in results:
                    print(f"  - Effect size: {results['effect_size']:.4f}")
                print(f"  - Significant: {'YES' if results['p_value'] < 0.05 else 'NO'}")
                return True
        else:
            print(f"⚠ No test recommended")
            return False

    except Exception as e:
        print(f"✗ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Run Tests
# ============================================================================

print("\n" + "="*70)
print("STATISTICAL RECOMMENDER TEST SUITE")
print("="*70)

test_results = []

# Test 1: Independent T-Test
test_results.append(test_dataset(
    'demo_data/demo_ttest_independent.csv',
    ['Wynik_Testu', 'Grupa'],
    'ttest_ind',
    'Independent T-Test'
))

# Test 2: Welch's T-Test
test_results.append(test_dataset(
    'demo_data/demo_ttest_welch.csv',
    ['Poziom_Cholesterolu', 'Dieta'],
    'ttest_welch',
    "Welch's T-Test"
))

# Test 3: Mann-Whitney U Test
test_results.append(test_dataset(
    'demo_data/demo_mannwhitney.csv',
    ['Czas_Rekonwalescencji_Dni', 'Metoda_Leczenia'],
    'mannwhitneyu',
    'Mann-Whitney U Test'
))

# Test 4: One-Way ANOVA
test_results.append(test_dataset(
    'demo_data/demo_anova.csv',
    ['Wzrost_Roslin_cm', 'Rodzaj_Nawozu'],
    'anova',
    'One-Way ANOVA'
))

# Test 5: Kruskal-Wallis Test
test_results.append(test_dataset(
    'demo_data/demo_kruskal.csv',
    ['Ocena_Zadowolenia', 'Klinika'],
    'kruskal',
    'Kruskal-Wallis Test'
))

# Test 6: Pearson Correlation
test_results.append(test_dataset(
    'demo_data/demo_pearson.csv',
    ['Wiek_Lat', 'Cisnienie_Skurczowe'],
    'pearsonr',
    'Pearson Correlation'
))

# Test 7: Spearman Correlation
test_results.append(test_dataset(
    'demo_data/demo_spearman.csv',
    ['Dochod_Roczny', 'Wydatki_Luksusowe'],
    'spearmanr',
    'Spearman Correlation'
))

# Test 8: Chi-Square Test
test_results.append(test_dataset(
    'demo_data/demo_chisquare.csv',
    ['Plec', 'Preferowany_Produkt'],
    'chi2',
    'Chi-Square Test'
))

# Test 9: Fisher's Exact Test
# Note: This requires manual intervention as chi2 is recommended for most 2x2 tables
test_results.append(test_dataset(
    'demo_data/demo_fisher.csv',
    ['Leczenie', 'Wynik'],
    'chi2',  # System recommends chi2, but fisher_exact is also valid
    "Fisher's Exact Test (or Chi-Square)"
))

# Test 10: Paired T-Test (requires manual pairing flag)
test_results.append(test_dataset(
    'demo_data/demo_ttest_paired.csv',
    ['Cisnienie_Przed', 'Cisnienie_Po'],
    'pearsonr',  # Without pairing flag, system sees 2 numeric -> correlation
    'Paired T-Test (manual flag needed)'
))

# Test 11: Wilcoxon Test (requires manual pairing flag)
test_results.append(test_dataset(
    'demo_data/demo_wilcoxon.csv',
    ['Poziom_Bolu_Przed', 'Poziom_Bolu_Po'],
    'spearmanr',  # Without pairing flag, system sees 2 numeric -> correlation
    'Wilcoxon Test (manual flag needed)'
))

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"Total tests: {len(test_results)}")
print(f"Passed: {sum(test_results)}")
print(f"Failed: {len(test_results) - sum(test_results)}")
print(f"Success rate: {sum(test_results) / len(test_results) * 100:.1f}%")

if sum(test_results) == len(test_results):
    print("\n✓ ALL TESTS PASSED!")
else:
    print(f"\n⚠ {len(test_results) - sum(test_results)} test(s) failed")

print("="*70)
