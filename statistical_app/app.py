"""
Statistical Testing Streamlit Application
=========================================

A comprehensive tool for statistical analysis that provides:
- Intelligent test selection based on data characteristics
- Educational content about statistical tests
- Data upload and preprocessing
- Assumption checking and validation
- Results interpretation and visualization

Author: Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
from modules.test_selector import TestSelector
from modules.data_processor import DataProcessor
from modules.statistical_tests import StatisticalTests
from modules.assumption_checker import AssumptionChecker
from modules.visualizer import Visualizer
from modules.interpreter import ResultsInterpreter

# Configure Streamlit page
st.set_page_config(
    page_title="Statistical Testing Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""

    # Main header
    st.markdown('<h1 class="main-header">📊 Statistical Testing Tool</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Test Selection Wizard", "Data Upload & Analysis", "Educational Content", "Test Library"]
    )

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'test_results' not in st.session_state:
        st.session_state.test_results = None
    if 'selected_test' not in st.session_state:
        st.session_state.selected_test = None

    # Route to different pages
    if page == "Home":
        show_home_page()
    elif page == "Test Selection Wizard":
        show_test_selection_wizard()
    elif page == "Data Upload & Analysis":
        show_data_analysis_page()
    elif page == "Educational Content":
        show_educational_content()
    elif page == "Test Library":
        show_test_library()

def show_home_page():
    """Display the home page with application overview"""

    st.markdown("""
    <div class="info-box">
    <h3>Welcome to the Statistical Testing Tool</h3>
    <p>This comprehensive application helps you select, perform, and interpret statistical tests for biological research.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🧭 Test Selection Wizard
        Get personalized test recommendations based on:
        - Your research question
        - Data characteristics
        - Study design
        - Sample size considerations
        """)

    with col2:
        st.markdown("""
        ### 📈 Data Analysis
        Upload your data and:
        - Perform automatic assumption checking
        - Run statistical tests
        - Generate visualizations
        - Get interpretation guidance
        """)

    with col3:
        st.markdown("""
        ### 📚 Educational Content
        Learn about:
        - Statistical test principles
        - When to use each test
        - Common pitfalls
        - Best practices
        """)

    st.markdown("""
    ---
    ### Quick Start Guide

    1. **New to statistics?** Start with the **Educational Content** to learn the basics
    2. **Have data ready?** Use the **Test Selection Wizard** to find the right test
    3. **Know your test?** Go directly to **Data Upload & Analysis**
    4. **Need reference?** Check the **Test Library** for detailed information
    """)

    # Sample datasets showcase
    st.markdown('<h3 class="section-header">📊 Sample Datasets</h3>', unsafe_allow_html=True)

    sample_datasets = {
        "Biological Growth Study": "Compare growth rates between treatment groups",
        "Gene Expression Analysis": "Analyze differential expression across conditions",
        "Clinical Trial Data": "Compare treatment effectiveness",
        "Ecological Survey": "Species abundance across habitats"
    }

    for dataset, description in sample_datasets.items():
        st.markdown(f"**{dataset}**: {description}")

def show_test_selection_wizard():
    """Display the test selection wizard"""

    st.markdown('<h2 class="section-header">🧭 Test Selection Wizard</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <p>Answer a few questions about your research to get personalized test recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Research question
    st.subheader("1. What is your research goal?")
    research_goal = st.radio(
        "Select your primary objective:",
        [
            "Compare means/medians between groups",
            "Assess relationship between two variables",
            "Predict an outcome from predictors",
            "Analyze frequencies/proportions"
        ]
    )

    # Variable characteristics
    st.subheader("2. Describe your variables")

    if research_goal == "Compare means/medians between groups":
        show_group_comparison_wizard()
    elif research_goal == "Assess relationship between two variables":
        show_relationship_wizard()
    elif research_goal == "Predict an outcome from predictors":
        show_prediction_wizard()
    elif research_goal == "Analyze frequencies/proportions":
        show_categorical_wizard()

def show_group_comparison_wizard():
    """Wizard for group comparison tests"""

    col1, col2 = st.columns(2)

    with col1:
        dependent_var_type = st.selectbox(
            "Dependent variable type:",
            ["Continuous (numerical)", "Ordinal (ranked)", "Binary (yes/no)"]
        )

        num_groups = st.selectbox(
            "Number of groups to compare:",
            ["Two groups", "Three or more groups"]
        )

    with col2:
        study_design = st.selectbox(
            "Study design:",
            ["Independent groups", "Paired/repeated measures"]
        )

        sample_size = st.number_input(
            "Sample size per group:",
            min_value=1, value=30
        )

    # Generate recommendation
    if st.button("Get Test Recommendation"):
        test_selector = TestSelector()
        recommendation = test_selector.recommend_test(
            goal="compare_groups",
            dependent_type=dependent_var_type,
            num_groups=num_groups,
            design=study_design,
            sample_size=sample_size
        )

        st.markdown(f"""
        <div class="success-box">
        <h4>Recommended Test: {recommendation['test_name']}</h4>
        <p><strong>Rationale:</strong> {recommendation['rationale']}</p>
        <p><strong>Alternative:</strong> {recommendation['alternative']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.selected_test = recommendation['test_name']

def show_relationship_wizard():
    """Wizard for relationship analysis"""
    st.write("Relationship analysis wizard - Coming soon!")

def show_prediction_wizard():
    """Wizard for prediction analysis"""
    st.write("Prediction analysis wizard - Coming soon!")

def show_categorical_wizard():
    """Wizard for categorical analysis"""
    st.write("Categorical analysis wizard - Coming soon!")

def show_data_analysis_page():
    """Display the data analysis page"""

    st.markdown('<h2 class="section-header">📈 Data Upload & Analysis</h2>', unsafe_allow_html=True)

    # Data upload section
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV or Excel)",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_file is not None:
        try:
            # Load data
            data_processor = DataProcessor()
            data = data_processor.load_data(uploaded_file)
            st.session_state.data = data

            st.success(f"Data loaded successfully! Shape: {data.shape}")

            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())

            # Data summary
            st.subheader("Data Summary")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Basic Information:**")
                st.write(f"- Rows: {data.shape[0]}")
                st.write(f"- Columns: {data.shape[1]}")
                st.write(f"- Missing values: {data.isnull().sum().sum()}")

            with col2:
                st.write("**Column Types:**")
                for col, dtype in data.dtypes.items():
                    st.write(f"- {col}: {dtype}")

            # Statistical analysis section
            if st.session_state.selected_test:
                st.subheader(f"Perform {st.session_state.selected_test}")
                perform_statistical_test(data)
            else:
                st.info("Use the Test Selection Wizard to choose an appropriate test.")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    else:
        # Sample dataset option
        st.subheader("Or try a sample dataset:")
        sample_data_option = st.selectbox(
            "Choose a sample dataset:",
            ["None", "Biological Growth Study", "Gene Expression", "Clinical Trial"]
        )

        if sample_data_option != "None":
            data = load_sample_dataset(sample_data_option)
            st.session_state.data = data
            st.dataframe(data.head())

def perform_statistical_test(data):
    """Perform the selected statistical test"""

    # Column selection
    columns = data.columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        dependent_var = st.selectbox("Select dependent variable:", columns)
    with col2:
        independent_var = st.selectbox("Select independent variable:", columns)

    if st.button("Run Analysis"):
        try:
            # Initialize test handler
            test_handler = StatisticalTests()
            assumption_checker = AssumptionChecker()
            visualizer = Visualizer()
            interpreter = ResultsInterpreter()

            # Check assumptions
            st.subheader("Assumption Checking")
            assumptions = assumption_checker.check_assumptions(
                data, dependent_var, independent_var, st.session_state.selected_test
            )

            for assumption, result in assumptions.items():
                if result['passed']:
                    st.success(f"✓ {assumption}: {result['message']}")
                else:
                    st.warning(f"⚠️ {assumption}: {result['message']}")

            # Perform test
            st.subheader("Test Results")
            results = test_handler.perform_test(
                data, dependent_var, independent_var, st.session_state.selected_test
            )

            # Display results
            st.write(f"**Test Statistic:** {results['statistic']:.4f}")
            st.write(f"**P-value:** {results['p_value']:.4f}")
            st.write(f"**Effect Size:** {results.get('effect_size', 'N/A')}")

            # Interpretation
            interpretation = interpreter.interpret_results(results, st.session_state.selected_test)
            st.markdown(f"""
            <div class="info-box">
            <h4>Interpretation:</h4>
            <p>{interpretation}</p>
            </div>
            """, unsafe_allow_html=True)

            # Visualization
            st.subheader("Visualization")
            fig = visualizer.create_plot(data, dependent_var, independent_var, st.session_state.selected_test)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error performing analysis: {str(e)}")

def show_educational_content():
    """Display educational content"""

    st.markdown('<h2 class="section-header">📚 Educational Content</h2>', unsafe_allow_html=True)

    topic = st.selectbox(
        "Choose a topic:",
        [
            "Introduction to Statistical Testing",
            "Hypothesis Testing Fundamentals",
            "Choosing the Right Test",
            "Understanding P-values",
            "Effect Sizes and Practical Significance",
            "Common Statistical Mistakes"
        ]
    )

    if topic == "Introduction to Statistical Testing":
        show_intro_content()
    elif topic == "Hypothesis Testing Fundamentals":
        show_hypothesis_content()
    else:
        st.write(f"Content for '{topic}' coming soon!")

def show_intro_content():
    """Show introduction to statistical testing"""

    st.markdown("""
    ### What is Statistical Testing?

    Statistical testing is a formal procedure for making decisions about populations based on sample data.
    It helps us determine whether observed differences or relationships in our data are likely due to real
    effects or just random variation.

    ### Key Concepts:

    **Population vs. Sample**
    - Population: The entire group you want to study
    - Sample: A subset of the population that you actually measure

    **Null Hypothesis (H₀)**
    - A statement of "no effect" or "no difference"
    - What we assume to be true until proven otherwise

    **Alternative Hypothesis (H₁)**
    - A statement that contradicts the null hypothesis
    - What we want to provide evidence for

    **P-value**
    - The probability of observing data as extreme as ours, assuming H₀ is true
    - Helps us decide whether to reject or fail to reject H₀
    """)

def show_hypothesis_content():
    """Show hypothesis testing content"""

    st.markdown("""
    ### The Hypothesis Testing Process

    1. **Formulate Hypotheses**
       - State your null and alternative hypotheses clearly
       - Example: H₀: μ₁ = μ₂ (no difference between group means)

    2. **Choose Significance Level (α)**
       - Usually 0.05 (5%) in biological sciences
       - This is your tolerance for Type I error

    3. **Select Appropriate Test**
       - Based on data type, study design, and assumptions
       - This tool helps you choose!

    4. **Check Assumptions**
       - Most tests have requirements (normality, equal variances, etc.)
       - Use alternative tests if assumptions are violated

    5. **Calculate Test Statistic**
       - Compare your data to what's expected under H₀

    6. **Make Decision**
       - If p ≤ α: Reject H₀ (statistically significant)
       - If p > α: Fail to reject H₀ (not statistically significant)
    """)

def show_test_library():
    """Display the test library"""

    st.markdown('<h2 class="section-header">📖 Test Library</h2>', unsafe_allow_html=True)

    test_category = st.selectbox(
        "Choose a category:",
        [
            "Comparing Two Groups",
            "Comparing Multiple Groups",
            "Relationships Between Variables",
            "Categorical Data Analysis",
            "Advanced Tests"
        ]
    )

    if test_category == "Comparing Two Groups":
        show_two_group_tests()
    else:
        st.write(f"Tests for '{test_category}' coming soon!")

def show_two_group_tests():
    """Show information about two-group tests"""

    tests = {
        "Independent t-test": {
            "purpose": "Compare means of two independent groups",
            "assumptions": ["Normality", "Equal variances", "Independence"],
            "example": "Comparing blood pressure between treatment and control groups"
        },
        "Welch's t-test": {
            "purpose": "Compare means when variances are unequal",
            "assumptions": ["Normality", "Independence"],
            "example": "Comparing gene expression between two cell types with different variability"
        },
        "Mann-Whitney U test": {
            "purpose": "Non-parametric comparison of two independent groups",
            "assumptions": ["Independence", "Ordinal or continuous data"],
            "example": "Comparing pain scores (ordinal scale) between treatments"
        }
    }

    for test_name, info in tests.items():
        with st.expander(test_name):
            st.write(f"**Purpose:** {info['purpose']}")
            st.write(f"**Key Assumptions:** {', '.join(info['assumptions'])}")
            st.write(f"**Example:** {info['example']}")

def load_sample_dataset(dataset_name):
    """Load a sample dataset for demonstration"""

    np.random.seed(42)

    if dataset_name == "Biological Growth Study":
        n = 50
        control = np.random.normal(10, 2, n)
        treatment = np.random.normal(12, 2.5, n)
        data = pd.DataFrame({
            'growth_rate': np.concatenate([control, treatment]),
            'group': ['Control'] * n + ['Treatment'] * n,
            'replicate': list(range(1, n+1)) * 2
        })
    elif dataset_name == "Gene Expression":
        n = 30
        baseline = np.random.lognormal(2, 0.5, n)
        stimulated = np.random.lognormal(2.3, 0.6, n)
        data = pd.DataFrame({
            'expression_level': np.concatenate([baseline, stimulated]),
            'condition': ['Baseline'] * n + ['Stimulated'] * n,
            'subject_id': list(range(1, n+1)) * 2
        })
    else:
        # Default dataset
        data = pd.DataFrame({
            'value': np.random.normal(0, 1, 100),
            'group': np.random.choice(['A', 'B'], 100)
        })

    return data

if __name__ == "__main__":
    main()