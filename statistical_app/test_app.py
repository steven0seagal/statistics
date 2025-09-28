#!/usr/bin/env python3
"""
Test script to verify the Streamlit application can run properly
"""

import sys
import traceback

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...")

    try:
        import streamlit as st
        print(f"✓ Streamlit {st.__version__}")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False

    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False

    try:
        import numpy as np
        print(f"✓ Numpy {np.__version__}")
    except ImportError as e:
        print(f"✗ Numpy import failed: {e}")
        return False

    try:
        import plotly
        print(f"✓ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False

    try:
        import scipy
        print(f"✓ Scipy {scipy.__version__}")
    except ImportError as e:
        print(f"✗ Scipy import failed: {e}")
        return False

    return True

def test_app_modules():
    """Test that application modules can be imported"""
    print("\nTesting application modules...")

    modules = [
        'modules.test_selector',
        'modules.data_processor',
        'modules.statistical_tests',
        'modules.assumption_checker',
        'modules.visualizer',
        'modules.interpreter'
    ]

    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module} import failed: {e}")
            return False

    return True

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\nTesting basic functionality...")

    try:
        from modules.test_selector import TestSelector
        from modules.data_processor import DataProcessor
        from modules.statistical_tests import StatisticalTests

        # Test TestSelector
        selector = TestSelector()
        recommendation = selector.recommend_test(
            goal='compare_groups',
            dependent_type='Continuous (numerical)',
            num_groups='Two groups',
            design='Independent groups',
            sample_size=30
        )
        print(f"✓ Test selection: {recommendation['test_name']}")

        # Test DataProcessor
        processor = DataProcessor()
        sample_data = processor.create_sample_datasets()
        print(f"✓ Sample datasets: {len(sample_data)} created")

        # Test StatisticalTests
        data = sample_data['biological_growth']
        tester = StatisticalTests()
        results = tester.perform_test(data, 'growth_rate', 'treatment', 'Independent t-test')
        print(f"✓ Statistical test: p={results['p_value']:.4f}")

        return True

    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_app_startup():
    """Test that the main app module can be imported"""
    print("\nTesting app startup...")

    try:
        import app
        print("✓ Main app module imported successfully")
        return True
    except Exception as e:
        print(f"✗ App startup failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Statistical Testing App - System Check")
    print("=" * 40)

    tests = [
        test_imports,
        test_app_modules,
        test_basic_functionality,
        test_app_startup
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("✓ All tests passed! The application should run correctly.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
        return True
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)