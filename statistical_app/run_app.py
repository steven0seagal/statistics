#!/usr/bin/env python3
"""
Simple startup script for the Statistical Testing Streamlit Application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import scipy
        import plotly
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False

def main():
    print("Statistical Testing Streamlit Application")
    print("=" * 45)

    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("✗ app.py not found. Please run this script from the statistical_app directory.")
        return False

    # Check dependencies
    if not check_dependencies():
        return False

    # Run the application
    print("Starting Streamlit application...")
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, navigate to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application")
    print("-" * 45)

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running application: {e}")
        return False
    except FileNotFoundError:
        print("\n✗ Streamlit not found. Please install with: pip install streamlit")
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)