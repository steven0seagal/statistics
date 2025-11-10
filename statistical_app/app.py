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
from modules.advanced_tests import AdvancedStatisticalTests
from modules.power_analysis import PowerAnalysis
from modules.report_generator import ReportGenerator
from modules.post_hoc_tests import PostHocTests
from modules.regression_module import run_regression_module
from modules.clustering_module import run_clustering_module
import statistical_recommender

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
        ["Home", "Test Selection Wizard", "Data Upload & Analysis",
         "Automatic Test Recommender", "Advanced Analysis",
         "Power Analysis", "Regression Analysis", "K-Means Clustering", "Educational Content", "Test Library"]
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
    elif page == "Automatic Test Recommender":
        statistical_recommender.run_recommender_tool()
    elif page == "Advanced Analysis":
        show_advanced_analysis_page()
    elif page == "Power Analysis":
        show_power_analysis_page()
    elif page == "Regression Analysis":
        run_regression_module()
    elif page == "K-Means Clustering":
        run_clustering_module()
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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🤖 Automatic Test Recommender (NEW!)
        Upload your data and get:
        - Automatic column profiling
        - Assumption testing (normality, homoscedasticity)
        - Intelligent test recommendations
        - One-click test execution
        - Detailed interpretation
        """)

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
    2. **Want automatic recommendations?** Try the **Automatic Test Recommender** - just upload your data!
    3. **Have data ready?** Use the **Test Selection Wizard** to find the right test
    4. **Know your test?** Go directly to **Data Upload & Analysis**
    5. **Need reference?** Check the **Test Library** for detailed information
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

            # Post-hoc analysis if needed
            if st.session_state.selected_test in ['One-way ANOVA', 'Kruskal-Wallis test'] and results['p_value'] <= 0.05:
                st.subheader("Post-Hoc Analysis")
                st.info("Significant omnibus test detected. Post-hoc analysis recommended.")

                post_hoc_handler = PostHocTests()
                recommendation = post_hoc_handler.recommend_post_hoc_test(
                    st.session_state.selected_test,
                    assumptions_passed=all(a.get('passed', True) for a in assumptions.values())
                )

                st.write(f"**Recommended:** {recommendation['recommended']}")
                st.write(f"**Rationale:** {recommendation['rationale']}")

                if st.button("Perform Post-Hoc Analysis"):
                    post_hoc_results = post_hoc_handler.perform_post_hoc(
                        data, dependent_var, independent_var, recommendation['recommended']
                    )

                    if 'error' not in post_hoc_results:
                        st.success("Post-hoc analysis completed!")
                        st.write(post_hoc_results['interpretation'])

                        # Display results table
                        if 'comparisons' in post_hoc_results:
                            comparisons_df = pd.DataFrame(post_hoc_results['comparisons'])
                            st.dataframe(comparisons_df)
                    else:
                        st.error(post_hoc_results['error'])

            # Report generation
            st.subheader("Generate Report")
            report_generator = ReportGenerator()

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Basic Report"):
                    report = report_generator.generate_report(results, 'basic', interpretation=interpretation)
                    st.session_state.report_content = report['content']

            with col2:
                if st.button("Comprehensive Report"):
                    report = report_generator.generate_report(
                        results, 'comprehensive',
                        assumptions=assumptions,
                        interpretation=interpretation
                    )
                    st.session_state.report_content = report['content']

            with col3:
                if st.button("Publication Report"):
                    report = report_generator.generate_report(results, 'publication')
                    st.session_state.report_content = report['content']

            # Display and download report
            if 'report_content' in st.session_state:
                st.subheader("Generated Report")
                st.markdown(st.session_state.report_content)

                # Download options
                st.subheader("Download Report")
                download_data = report_generator.create_downloadable_report(
                    st.session_state.report_content, 'html'
                )
                st.download_button(
                    label="Download as HTML",
                    data=download_data['content'],
                    file_name=download_data['filename'],
                    mime=download_data['mime_type']
                )

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
            "Common Statistical Mistakes",
            "Regression Analysis: Linear and Logarithmic",
            "Multiple Linear Regression (MLR)",
            "K-Means Clustering"
        ]
    )

    if topic == "Introduction to Statistical Testing":
        show_intro_content()
    elif topic == "Hypothesis Testing Fundamentals":
        show_hypothesis_content()
    elif topic == "Regression Analysis: Linear and Logarithmic":
        show_regression_content()
    elif topic == "Multiple Linear Regression (MLR)":
        show_mlr_content()
    elif topic == "K-Means Clustering":
        show_kmeans_content()
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

def show_regression_content():
    """Show regression analysis educational content"""

    st.markdown("""
    ## 📈 Analiza Regresji: Liniowa i Logarytmiczna

    Analiza regresji to jedna z fundamentalnych technik statystycznych. Jej celem jest **modelowanie zależności**
    między zmiennymi. Najczęściej chcemy zrozumieć, jak jedna zmienna (zależna, $Y$) zmienia się, gdy zmienia
    się inna zmienna (niezależna, $X$).

    ### Czym jest Regresja Liniowa?

    Regresja liniowa jest najprostszym typem regresji. Zakłada ona, że zależność między zmienną $X$ a $Y$
    można opisać za pomocą **linii prostej**.

    #### Wzór

    Model opisany jest prostym równaniem:

    $$
    y = ax + b
    $$

    Gdzie:

    * **$y$** – Wartość prognozowana (zmienna zależna).
    * **$x$** – Wartość zmiennej niezależnej.
    * **$a$** – **Współczynnik nachylenia** (slope). Mówi nam, o ile *średnio* zmieni się $y$, jeśli $x$ wzrośnie o jedną jednostkę.
    * **$b$** – **Wyraz wolny** (intercept). Jest to punkt przecięcia linii z osią $Y$, czyli wartość $y$, gdy $x$ wynosi 0.

    #### Jak się ją liczy?

    Najpopularniejszą metodą jest **Metoda Najmniejszych Kwadratów (Ordinary Least Squares - OLS)**.
    Komputer szuka takiej linii (czyli takich wartości $a$ i $b$), która minimalizuje sumę kwadratów odległości
    (tzw. "błędów" lub "rezyduów") między rzeczywistymi punktami danych a linią regresji.

    #### Kiedy jej używać?

    * Gdy na wykresie punktowym dane wydają się układać wzdłuż linii prostej.
    * Gdy zakładamy stałe tempo zmiany (np. koszt wyprodukowania 10 sztuk jest 10 razy większy niż 1 sztuki).
    * Jako model bazowy do porównania z bardziej skomplikowanymi modelami.

    ---

    ### Czym jest Regresja Logarytmiczna?

    Regresja logarytmiczna jest modelem krzywoliniowym. Jest idealna do opisywania sytuacji, w których
    zależność **szybko rośnie na początku, a następnie zwalnia** (lub nasyca się).

    #### Wzór

    Model opisany jest równaniem:

    $$
    y = a \\cdot \\ln(x) + b
    $$

    Gdzie:

    * **$y$**, **$a$**, **$b$** – Mają podobne znaczenie jak w regresji liniowej.
    * **$\\ln(x)$** – **Logarytm naturalny** ze zmiennej $x$.

    **Ważna uwaga:** Model ten można stosować tylko wtedy, gdy wartości zmiennej niezależnej $X$ są
    **dodatnie** ($x > 0$), ponieważ logarytm z zera lub liczb ujemnych jest niezdefiniowany.

    #### Jak się ją liczy?

    Najczęściej sprowadza się to do prostego triku:

    1. Tworzymy nową zmienną, $X' = \\ln(X)$.
    2. Wykonujemy **regresję liniową** na danych $(X', Y)$.
       Model, który otrzymujemy, to $y = a \\cdot X' + b$, co po podstawieniu $X'$ daje nam $y = a \\cdot \\ln(x) + b$.

    #### Kiedy jej używać?

    * Gdy obserwujemy **prawo malejących przychodów**. Na przykład:
        * Wpływ wydatków na reklamę na sprzedaż (pierwsza wydana złotówka przynosi duży zwrot, ale milionowa już znacznie mniejszy).
        * Wpływ liczby godzin nauki na wynik egzaminu (przejście z 0 na 2 godziny daje duży skok; przejście z 10 na 12 godzin – niewielki).
    * Gdy tempo wzrostu $Y$ maleje wraz ze wzrostem $X$.

    ---

    ### Jak ocenić, który model jest lepszy?

    Po dopasowaniu obu modeli do tych samych danych, musimy zdecydować, który z nich jest lepszy.
    Używamy do tego dwóch kluczowych metryk:

    #### 1. R-kwadrat (R²) – Współczynnik Determinacji

    * **Co mierzy:** Jak dobrze model (linia lub krzywa) **wyjaśnia zmienność** w danych.
    * **Interpretacja:** Jest to wartość od 0 do 1 (lub 0% do 100%).
        * R² = 0.85 oznacza, że 85% zmienności w $Y$ jest wyjaśniane przez zmienną $X$ za pomocą naszego modelu.
    * **Wybór:** **Im wyższy R², tym (zazwyczaj) lepszy model.** Lepsze dopasowanie do danych.

    #### 2. Błąd Średniokwadratowy (Mean Squared Error - MSE)

    * **Co mierzy:** Średnią kwadratów błędów (różnic między wartością rzeczywistą $y$ a prognozowaną $\\hat{y}$).
    * **Interpretacja:** Jest to miara błędu prognozy. Ponieważ błędy są podnoszone do kwadratu, metryka ta
      bardzo **mocno "karze" duże błędy**.
    * **Wybór:** **Im niższy MSE, tym lepszy model.** Oznacza to, że prognozy modelu są średnio bliższe prawdy.

    #### Wnioskowanie:

    Najczęściej szukamy modelu, który ma **jednocześnie wysoki R² i niski MSE**.

    ### Problem Overfittingu (Przeuczenia)

    Overfitting to jeden z największych problemów w modelowaniu.

    * **Definicja:** Zjawisko, w którym model jest **zbyt skomplikowany** i zamiast uczyć się ogólnego trendu
      w danych, zaczyna "wkuwać na pamięć" dane treningowe, łącznie z przypadkowym szumem.
    * **Skutek:** Taki model ma fantastyczne metryki (np. R² bliskie 1.0) na danych, na których był uczony,
      ale kompletnie nie potrafi przewidywać nowych, nieznanych mu danych.
    * **Jak się chronić:** W przypadku prostych modeli (jak regresja liniowa czy logarytmiczna z jedną zmienną)
      ryzyko jest bardzo małe. Problem pojawia się przy modelach bardzo złożonych (np. wielomiany wysokiego stopnia,
      głębokie sieci neuronowe).
    * **Podstawowa zasada:** Zawsze należy dążyć do modelu **możliwie najprostszego**, który wciąż dobrze wyjaśnia
      dane (tzw. "Zasada Oszczędności" lub "Brzytwa Ockhama").

    ---

    ### 💡 Podsumowanie

    * **Regresja liniowa** jest odpowiednia dla zależności liniowych (stałe tempo zmiany).
    * **Regresja logarytmiczna** jest lepsza dla zależności, które zwalniają wraz ze wzrostem X (malejące przychody).
    * Porównujemy modele używając **R²** (wyższe = lepsze) i **MSE** (niższe = lepsze).
    * Proste modele minimalizują ryzyko przeuczenia (overfitting).
    * Zawsze należy rozważyć praktyczne znaczenie wyników, nie tylko statystyczne dopasowanie.

    **Zachęcamy do eksperymentowania z modułem Analizy Regresji, aby zobaczyć te koncepcje w praktyce!**
    """)

def show_mlr_content():
    """Show Multiple Linear Regression educational content"""

    st.markdown("""
    ## 📊 Wielokrotna Regresja Liniowa (Multiple Linear Regression - MLR)

    Wielokrotna Regresja Liniowa (MLR) to rozszerzenie Regresji Liniowej Prostej. Zamiast używać tylko jednej zmiennej $X$ do przewidywania $Y$, MLR pozwala nam używać **wielu zmiennych $X$ jednocześnie**.

    Jest to jedna z najczęściej używanych technik w statystyce i data science, pozwalająca na budowanie bardziej złożonych i dokładniejszych modeli prognozujących.

    ### Wzór

    Podczas gdy regresja prosta miała wzór $y = ax + b$, regresja wielokrotna ma postać:

    $$
    y = b_0 + a_1x_1 + a_2x_2 + \\dots + a_kx_k
    $$

    Gdzie:

    * **$y$** – Wartość prognozowana (zmienna zależna).
    * **$x_1, x_2, \\dots, x_k$** – Zmienne niezależne (predyktory).
    * **$b_0$** – **Wyraz wolny** (intercept). Wartość $y$, gdy wszystkie zmienne $x$ są równe zero.
    * **$a_1, a_2, \\dots, a_k$** – **Współczynniki** regresji.

    ### Jak interpretować współczynniki (to kluczowe!)

    Interpretacja współczynnika $a$ w MLR jest inna niż w regresji prostej.

    > Współczynnik **$a_1$** mówi nam, o ile *średnio* zmieni się $y$, jeśli $x_1$ wzrośnie o jedną jednostkę, **przy założeniu, że wszystkie pozostałe zmienne ($x_2, \\dots, x_k$) pozostają bez zmian**.

    Jest to miara "czystego" wpływu danej zmiennej $x_1$ na $y$, po wyizolowaniu wpływu pozostałych zmiennych uwzględnionych w modelu.

    **Przykład:** Model ceny mieszkania:
    `Cena = 50000 + (7000 * Liczba_Pokoi) + (300 * Powierzchnia_m2)`

    * Współczynnik `7000` oznacza, że (według modelu) dodanie jednego pokoju podnosi cenę o 7000 zł, **przy tej samej powierzchni**.
    * Współczynnik `300` oznacza, że każdy dodatkowy m² podnosi cenę o 300 zł, **przy tej samej liczbie pokoi**.

    ---

    ### Metryki Oceny: R² vs. R² Skorygowany

    W regresji wielokrotnej pojawia się nowy, ważniejszy wskaźnik.

    #### R-kwadrat (R²)

    Mówi, jaki procent zmienności $Y$ jest wyjaśniany przez *wszystkie* zmienne $X$ w modelu.

    * **Problem:** Wartość R² **zawsze rośnie (lub pozostaje taka sama)**, gdy dodajemy do modelu nową zmienną $X$, nawet jeśli ta zmienna jest kompletnie losowa i nie ma żadnego związku z $Y$. To zachęca do budowania przeuczonych (overfitted) modeli.

    #### R-kwadrat Skorygowany (Adjusted R²)

    To jest ulepszona wersja R², która rozwiązuje ten problem.

    * **Co robi:** "Karze" model za posiadanie wielu zmiennych. R² Skorygowany rośnie tylko wtedy, gdy dodana nowa zmienna wnosi do modelu **istotną** moc wyjaśniającą.
    * **Interpretacja:** Jeśli dodasz do modelu nową zmienną $X$ i R² Skorygowany spadnie, oznacza to, że ta zmienna jest zbędna i pogarsza jakość modelu (prawdopodobnie dodaje więcej szumu niż sygnału).

    **Wniosek: Przy Wielokrotnej Regresji Liniowej zawsze używaj R-kwadrat Skorygowanego do oceny i porównywania modeli.**

    ---

    ### Największe Zagrożenia w MLR

    #### 1. Overfitting (Przeuczenie)

    * **Problem:** Dodanie zbyt wielu zmiennych $X$, szczególnie przy małej ilości danych. Model zaczyna "uczyć się na pamięć" szumu w danych, zamiast ogólnego trendu.
    * **Skutek:** Model będzie miał świetny R² na danych treningowych, ale fatalnie poradzi sobie z prognozowaniem nowych, nieznanych danych.
    * **Obrona:** Używaj R² Skorygowanego; stosuj Zasadę Oszczędności (wybieraj najprostszy model, który dobrze działa).

    #### 2. Multikolinearność (Współliniowość)

    * **Problem:** Sytuacja, w której dwie lub więcej zmiennych $X$ są ze sobą silnie skorelowane (np. `waga_w_kg` i `waga_w_funtach`, albo `wzrost` i `długość_nogi`).
    * **Skutek:** Algorytm regresji "głupieje". Nie wie, której ze skorelowanych zmiennych przypisać wpływ na $Y$. Powoduje to, że:
        * Współczynniki ($a_1, a_2$) stają się bardzo niestabilne.
        * Mogą mieć absurdalne wartości lub znaki (np. model mówi, że wzrost *obniża* wagę, bo źle rozdzielił wpływ).
    * **Obrona:** Sprawdź korelację między zmiennymi $X$ *przed* zbudowaniem modelu. Jeśli dwie zmienne mają korelację > 0.8 (lub < -0.8), rozważ usunięcie jednej z nich.

    ---

    ### 💡 Podsumowanie

    * **Wielokrotna Regresja Liniowa** pozwala modelować wpływ wielu zmiennych $X$ na $Y$ jednocześnie.
    * **Współczynniki** pokazują "czysty" wpływ każdej zmiennej, po kontroli pozostałych.
    * **R² Skorygowany** jest lepszą metryką niż zwykły R², ponieważ karze za zbędne zmienne.
    * **Overfitting** jest większym zagrożeniem niż w regresji prostej – uważaj na dodawanie zbyt wielu zmiennych.
    * **Multikolinearność** między predyktorami może prowadzić do niestabilnych i mylących wyników.
    * Zawsze dąż do **najprostszego modelu**, który wystarczająco dobrze wyjaśnia dane (Zasada Oszczędności).

    **Zachęcamy do eksperymentowania z modułem Analizy Regresji, aby zobaczyć te koncepcje w praktyce!**
    """)

def show_kmeans_content():
    """Show K-Means clustering educational content"""

    st.markdown("""
    ## 💠 Klastrowanie K-Means (Grupowanie Metodą K-Średnich)

    K-Means to jeden z najpopularniejszych algorytmów **uczenia nienadzorowanego**. Jego celem jest automatyczne podzielenie zbioru danych na $K$ odrębnych grup (klastrów), gdzie punkty wewnątrz jednego klastra są do siebie jak najbardziej podobne, a punkty między różnymi klastrami – jak najmniej.

    ### Jak Działa Algorytm K-Means?

    Algorytm działa iteracyjnie, próbując znaleźć "środki" (centroidy) dla $K$ grup.

    1.  **Krok 1: Wybór $K$**
        Użytkownik musi na początku **zdecydować**, na ile grup ($K$) chce podzielić dane. (Patrz poniżej, jak wybrać $K$).
    2.  **Krok 2: Inicjalizacja**
        Algorytm losowo (lub "inteligentnie" dzięki metodzie `k-means++`) umieszcza $K$ centroidów (punktów centralnych) w przestrzeni danych.
    3.  **Krok 3: Przypisanie**
        Każdy punkt danych jest przypisywany do **najbliższego** mu centroidu (zazwyczaj na podstawie odległości euklidesowej).
    4.  **Krok 4: Aktualizacja**
        Po przypisaniu wszystkich punktów, centroidy są **przesuwane**. Nową lokalizacją każdego centroidu jest **średnia arytmetyczna** wszystkich punktów, które zostały do niego przypisane.
    5.  **Krok 5: Powtórzenie**
        Kroki 3 i 4 są powtarzane aż do **konwergencji** – czyli do momentu, gdy centroidy przestaną się znacząco przemieszczać (klastry się ustabilizują).

    ---

    ### Metody Statystyczne: "Jak wybrać K?"

    To największe wyzwanie w K-Means. Nie ma jednej "poprawnej" odpowiedzi, ale używamy metod statystycznych, aby oszacować dobrą wartość $K$.

    #### 1. Metoda Łokcia (Elbow Method)

      * **Co mierzy:** **Inercję (WCSS)**, czyli sumę kwadratów odległości każdego punktu od jego centroidu. Mówiąc prościej: jak bardzo "ścisłe" są klastry.
      * **Jak to działa:** Uruchamiamy K-Means dla różnych wartości $K$ (np. od 2 do 10) i liczymy inercję dla każdej z nich.
      * **Interpretacja:**
          * Im więcej klastrów ($K$), tym mniejsza będzie inercja (bo klastry są mniejsze i ciaśniejsze).
          * Rysujemy wykres $K$ vs. Inercja.
          * Szukamy punktu **"załamania" (łokcia)** – miejsca, w którym linia przestaje gwałtownie opadać. Jest to punkt, w którym dodanie kolejnego klastra nie przynosi już dużej korzyści (nie zmniejsza znacząco inercji).

    #### 2. Analiza Sylwetkowa (Silhouette Analysis)

      * **Co mierzy:** **Współczynnik Sylwetkowy (Silhouette Score)**. Jest to miara bardziej zaawansowana, która ocenia dwie rzeczy jednocześnie:
        1.  **Spójność (Cohesion):** Jak blisko punkty są do innych punktów w tym samym klastrze?
        2.  **Separację (Separation):** Jak daleko punkty są od punktów w *innych* klastrach?
      * **Jak to działa:** Wynik jest obliczany dla każdego punktu i mieści się w zakresie od -1 do 1.
          * **+1:** Idealnie. Punkt jest daleko od sąsiednich klastrów i blisko swoich.
          * **0:** Punkt jest na granicy dwóch klastrów (nakładanie się).
          * **-1:** Źle. Punkt jest prawdopodobnie przypisany do złego klastra.
      * **Interpretacja:**
          * Uruchamiamy K-Means dla różnych $K$ i liczymy *średni* Współczynnik Sylwetkowy dla wszystkich punktów.
          * Rysujemy wykres $K$ vs. Średni Silhouette Score.
          * W przeciwieństwie do Metody Łokcia, tutaj **szukamy maksimum (szczytu)**. Najwyższy wynik wskazuje na $K$, które daje najbardziej spójne i najlepiej odseparowane klastry.

    ---

    ### Inne "Metody" i Wymagania K-Means

    Aby K-Means zadziałało poprawnie, nie wystarczy tylko wybrać $K$. Kluczowe są też poniższe "metody" (techniki).

    #### 1. Metoda: Standaryzacja Danych

      * **Problem:** Algorytm K-Means opiera się na **odległości**. Jeśli masz cechy o różnych skalach (np. `Wiek` [0-100] i `Zarobki` [3 000 - 50 000]), cecha o większej skali (`Zarobki`) całkowicie zdominuje obliczenia odległości. Algorytm praktycznie zignoruje `Wiek`.
      * **Rozwiązanie (Metoda):** **Standaryzacja** (np. `StandardScaler` w Scikit-learn).
      * **Co robi:** Przekształca wszystkie cechy tak, aby miały średnią równą 0 i odchylenie standardowe równe 1. To sprawia, że każda cecha ma "równą wagę" w algorytmie.
      * **Wniosek:** **Prawie zawsze powinieneś standaryzować dane przed użyciem K-Means.**

    #### 2. Metoda: Inicjalizacja (Problem Lokalne Minimum)

      * **Problem:** Wynik K-Means może zależeć od tego, gdzie **na początku** zostały umieszczone centroidy (Krok 2). Zły początkowy wybór może uwięzić algorytm w "lokalnym minimum" – znajdzie on klastry, ale nie będą one optymalne.
      * **Rozwiązanie (Metoda 1): `n_init`**
          * Uruchamiamy algorytm K-Means **wiele razy** (np. `n_init=10`) z różnymi losowymi punktami startowymi.
          * Jako ostateczny wynik wybierany jest ten przebieg, który dał najniższą inercję (WCSS).
      * **Rozwiązanie (Metoda 2): `init='k-means++'`**
          * To domyślna metoda inicjalizacji w Scikit-learn.
          * Zamiast umieszczać centroidy w pełni losowo, `k-means++` robi to "inteligentnie" – stara się umieścić początkowe centroidy daleko od siebie. Znacząco zwiększa to szansę na znalezienie optymalnych klastrów i przyspiesza konwergencję.

    ---

    ### 💡 Podsumowanie

    * **K-Means** to algorytm uczenia nienadzorowanego do grupowania danych w klastry.
    * Wymaga wybrania liczby klastrów **K** z góry.
    * **Metoda Łokcia** pomaga znaleźć K poprzez identyfikację punktu, w którym dodanie kolejnego klastra nie przynosi znaczącej redukcji inercji.
    * **Analiza Sylwetkowa** mierzy jakość klastrów (spójność i separację) – szukamy K z najwyższym wynikiem.
    * **Standaryzacja danych** jest kluczowa, aby zapewnić równą wagę wszystkim cechom.
    * Używamy **k-means++** i **n_init** aby uniknąć lokalnych minimów i znaleźć optymalne rozwiązanie.

    **Zachęcamy do eksperymentowania z modułem Klastrowania K-Means, aby zobaczyć te koncepcje w praktyce!**
    """)

def show_test_library():
    """Display the test library"""

    st.markdown('<h2 class="section-header">📖 Test Library</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <p>Comprehensive reference guide for all statistical tests available in the application.</p>
    </div>
    """, unsafe_allow_html=True)

    test_category = st.selectbox(
        "Choose a category:",
        [
            "Comparing Two Groups",
            "Comparing Multiple Groups",
            "Relationships Between Variables",
            "Categorical Data Analysis",
            "Advanced Multivariate Tests",
            "Post-Hoc Tests",
            "Power Analysis"
        ]
    )

    if test_category == "Comparing Two Groups":
        show_two_group_tests()
    elif test_category == "Comparing Multiple Groups":
        show_multiple_group_tests()
    elif test_category == "Relationships Between Variables":
        show_relationship_tests()
    elif test_category == "Categorical Data Analysis":
        show_categorical_tests()
    elif test_category == "Advanced Multivariate Tests":
        show_advanced_tests()
    elif test_category == "Post-Hoc Tests":
        show_posthoc_tests()
    elif test_category == "Power Analysis":
        show_power_analysis_info()

def show_two_group_tests():
    """Show information about two-group tests"""

    st.markdown("### Parametric Tests for Two Groups")

    parametric_tests = {
        "Independent t-test": {
            "purpose": "Compare means of two independent groups when data are normally distributed",
            "assumptions": ["Normality in both groups", "Equal variances (homoscedasticity)", "Independence of observations"],
            "when_to_use": "When you have two separate groups and want to compare their means",
            "example": "Comparing average blood pressure between patients receiving drug A vs. drug B",
            "effect_size": "Cohen's d",
            "interpretation": "Tests if the difference between group means is statistically significant",
            "alternatives": "Use Welch's t-test if variances are unequal, or Mann-Whitney U if normality is violated"
        },
        "Welch's t-test": {
            "purpose": "Compare means of two independent groups when variances are unequal",
            "assumptions": ["Normality in both groups", "Independence of observations", "Unequal variances allowed"],
            "when_to_use": "When groups have significantly different variances (failed Levene's test)",
            "example": "Comparing gene expression between healthy and diseased tissue (different variability)",
            "effect_size": "Cohen's d",
            "interpretation": "Adjusts degrees of freedom to account for unequal variances",
            "alternatives": "Use Mann-Whitney U test if normality assumptions are also violated"
        },
        "Paired t-test": {
            "purpose": "Compare means of two related measurements on the same subjects",
            "assumptions": ["Normality of differences", "Independence of paired differences"],
            "when_to_use": "Before-after studies, matched pairs, or repeated measures on same subjects",
            "example": "Comparing blood pressure before and after treatment in the same patients",
            "effect_size": "Cohen's d for paired data",
            "interpretation": "Tests if the mean difference between paired observations is significantly different from zero",
            "alternatives": "Use Wilcoxon signed-rank test if differences are not normally distributed"
        }
    }

    for test_name, info in parametric_tests.items():
        with st.expander(f"📊 {test_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**When to Use:** {info['when_to_use']}")
                st.write(f"**Effect Size:** {info['effect_size']}")
            with col2:
                st.write(f"**Key Assumptions:**")
                for assumption in info['assumptions']:
                    st.write(f"• {assumption}")

            st.write(f"**Example:** {info['example']}")
            st.write(f"**Interpretation:** {info['interpretation']}")
            st.write(f"**Alternatives:** {info['alternatives']}")

    st.markdown("### Non-Parametric Tests for Two Groups")

    nonparametric_tests = {
        "Mann-Whitney U test": {
            "purpose": "Compare distributions of two independent groups without assuming normality",
            "assumptions": ["Independence of observations", "Ordinal or continuous data", "Similar distribution shapes for median comparison"],
            "when_to_use": "When data are not normally distributed, ordinal, or have outliers",
            "example": "Comparing patient satisfaction scores (1-10 scale) between two hospitals",
            "effect_size": "Rank-biserial correlation (r)",
            "interpretation": "Tests if one group tends to have higher values than the other",
            "alternatives": "Use independent t-test if normality assumptions are met"
        },
        "Wilcoxon signed-rank test": {
            "purpose": "Compare paired observations when differences are not normally distributed",
            "assumptions": ["Paired observations", "Continuous data", "Symmetric distribution of differences"],
            "when_to_use": "Paired data where differences violate normality assumption",
            "example": "Comparing pain scores before and after treatment when data are skewed",
            "effect_size": "Matched pairs rank-biserial correlation",
            "interpretation": "Tests if the median difference between pairs is significantly different from zero",
            "alternatives": "Use paired t-test if differences are normally distributed"
        }
    }

    for test_name, info in nonparametric_tests.items():
        with st.expander(f"📈 {test_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**When to Use:** {info['when_to_use']}")
                st.write(f"**Effect Size:** {info['effect_size']}")
            with col2:
                st.write(f"**Key Assumptions:**")
                for assumption in info['assumptions']:
                    st.write(f"• {assumption}")

            st.write(f"**Example:** {info['example']}")
            st.write(f"**Interpretation:** {info['interpretation']}")
            st.write(f"**Alternatives:** {info['alternatives']}")

def show_multiple_group_tests():
    """Show information about multiple group tests"""

    st.markdown("### Parametric Tests for Multiple Groups")

    parametric_tests = {
        "One-way ANOVA": {
            "purpose": "Compare means across three or more independent groups",
            "assumptions": ["Normality in each group", "Equal variances across groups", "Independence of observations"],
            "when_to_use": "Comparing means of 3+ independent groups",
            "example": "Comparing effectiveness of 4 different fertilizers on plant growth",
            "effect_size": "Eta-squared (η²) or partial eta-squared",
            "interpretation": "Tests if at least one group mean differs from the others (omnibus test)",
            "follow_up": "Post-hoc tests (Tukey's HSD) needed to identify which groups differ",
            "alternatives": "Use Kruskal-Wallis test if assumptions are violated"
        },
        "Repeated measures ANOVA": {
            "purpose": "Compare means across three or more related measurements",
            "assumptions": ["Normality within each condition", "Sphericity (equal variances of differences)", "Independence of subjects"],
            "when_to_use": "Same subjects measured at multiple time points or conditions",
            "example": "Measuring blood pressure at baseline, 1 month, 3 months, and 6 months",
            "effect_size": "Partial eta-squared",
            "interpretation": "Tests if means change significantly across time/conditions",
            "follow_up": "Pairwise comparisons with Bonferroni correction",
            "alternatives": "Use Friedman test if assumptions are violated"
        },
        "Two-way ANOVA": {
            "purpose": "Analyze effects of two factors and their interaction simultaneously",
            "assumptions": ["Normality in each cell", "Equal variances across all cells", "Independence of observations"],
            "when_to_use": "Two categorical independent variables (factorial design)",
            "example": "Testing effects of drug type (A, B, C) and dosage (low, high) on recovery time",
            "effect_size": "Partial eta-squared for each effect",
            "interpretation": "Tests main effects of each factor and their interaction",
            "follow_up": "Simple effects analysis if interaction is significant",
            "alternatives": "Use non-parametric alternatives or transform data"
        }
    }

    for test_name, info in parametric_tests.items():
        with st.expander(f"📊 {test_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**When to Use:** {info['when_to_use']}")
                st.write(f"**Effect Size:** {info['effect_size']}")
            with col2:
                st.write(f"**Key Assumptions:**")
                for assumption in info['assumptions']:
                    st.write(f"• {assumption}")

            st.write(f"**Example:** {info['example']}")
            st.write(f"**Interpretation:** {info['interpretation']}")
            st.write(f"**Follow-up:** {info['follow_up']}")
            st.write(f"**Alternatives:** {info['alternatives']}")

    st.markdown("### Non-Parametric Tests for Multiple Groups")

    nonparametric_tests = {
        "Kruskal-Wallis test": {
            "purpose": "Compare distributions across three or more independent groups",
            "assumptions": ["Independence of observations", "Ordinal or continuous data", "Similar distribution shapes"],
            "when_to_use": "Multiple independent groups with non-normal data or ordinal scales",
            "example": "Comparing patient satisfaction ratings across 5 different clinics",
            "effect_size": "Epsilon-squared (ε²)",
            "interpretation": "Tests if at least one group has a different distribution",
            "follow_up": "Dunn's test for pairwise comparisons",
            "alternatives": "Use one-way ANOVA if normality assumptions are met"
        },
        "Friedman test": {
            "purpose": "Compare distributions across three or more related measurements",
            "assumptions": ["Related observations", "Ordinal or continuous data", "No distributional assumptions"],
            "when_to_use": "Multiple related measurements with non-normal data",
            "example": "Comparing taste ratings for 4 food samples rated by the same judges",
            "effect_size": "Kendall's W (coefficient of concordance)",
            "interpretation": "Tests if distributions differ across repeated measurements",
            "follow_up": "Pairwise Wilcoxon tests with Bonferroni correction",
            "alternatives": "Use repeated measures ANOVA if assumptions are met"
        }
    }

    for test_name, info in nonparametric_tests.items():
        with st.expander(f"📈 {test_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**When to Use:** {info['when_to_use']}")
                st.write(f"**Effect Size:** {info['effect_size']}")
            with col2:
                st.write(f"**Key Assumptions:**")
                for assumption in info['assumptions']:
                    st.write(f"• {assumption}")

            st.write(f"**Example:** {info['example']}")
            st.write(f"**Interpretation:** {info['interpretation']}")
            st.write(f"**Follow-up:** {info['follow_up']}")
            st.write(f"**Alternatives:** {info['alternatives']}")

def show_relationship_tests():
    """Show information about relationship tests"""

    st.markdown("### Correlation Analysis")

    correlation_tests = {
        "Pearson correlation": {
            "purpose": "Measure the strength and direction of linear relationship between two continuous variables",
            "assumptions": ["Linear relationship", "Normality of both variables", "Homoscedasticity", "Independence"],
            "when_to_use": "Two continuous variables with suspected linear relationship",
            "example": "Correlation between height and weight in adults",
            "effect_size": "Correlation coefficient (r)",
            "interpretation": "r = -1 (perfect negative), r = 0 (no linear relationship), r = +1 (perfect positive)",
            "range": "r ranges from -1 to +1",
            "alternatives": "Use Spearman correlation for non-linear monotonic relationships"
        },
        "Spearman correlation": {
            "purpose": "Measure the strength and direction of monotonic relationship between two variables",
            "assumptions": ["Monotonic relationship", "Ordinal or continuous data", "Independence"],
            "when_to_use": "Non-linear but monotonic relationships, ordinal data, or outliers present",
            "example": "Correlation between education level (ordinal) and income",
            "effect_size": "Spearman's rho (ρ)",
            "interpretation": "ρ = -1 (perfect negative monotonic), ρ = 0 (no monotonic relationship), ρ = +1 (perfect positive monotonic)",
            "range": "ρ ranges from -1 to +1",
            "alternatives": "Use Pearson correlation if linear relationship and normality assumptions are met"
        }
    }

    for test_name, info in correlation_tests.items():
        with st.expander(f"📊 {test_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**When to Use:** {info['when_to_use']}")
                st.write(f"**Effect Size:** {info['effect_size']}")
                st.write(f"**Range:** {info['range']}")
            with col2:
                st.write(f"**Key Assumptions:**")
                for assumption in info['assumptions']:
                    st.write(f"• {assumption}")

            st.write(f"**Example:** {info['example']}")
            st.write(f"**Interpretation:** {info['interpretation']}")
            st.write(f"**Alternatives:** {info['alternatives']}")

    st.markdown("### Regression Analysis")

    regression_tests = {
        "Linear regression": {
            "purpose": "Model and predict a continuous outcome from one or more predictor variables",
            "assumptions": ["Linear relationship", "Independence of residuals", "Normality of residuals", "Homoscedasticity of residuals"],
            "when_to_use": "Predicting continuous outcomes, identifying significant predictors",
            "example": "Predicting house prices from size, location, and age",
            "effect_size": "R-squared (R²) - proportion of variance explained",
            "interpretation": "Slope coefficients show change in outcome per unit change in predictor",
            "diagnostics": "Check residual plots, Q-Q plots, and leverage points",
            "alternatives": "Use non-linear regression, transformation, or robust regression"
        },
        "Logistic regression": {
            "purpose": "Model and predict binary outcomes from one or more predictor variables",
            "assumptions": ["Binary outcome", "Independence of observations", "Linear relationship between logit and predictors"],
            "when_to_use": "Predicting binary outcomes (success/failure, disease/healthy)",
            "example": "Predicting disease risk from age, gender, and biomarker levels",
            "effect_size": "Pseudo R-squared, odds ratios",
            "interpretation": "Odds ratios show multiplicative change in odds per unit change in predictor",
            "diagnostics": "Check classification accuracy, ROC curves, and residual deviance",
            "alternatives": "Use discriminant analysis or machine learning methods"
        }
    }

    for test_name, info in regression_tests.items():
        with st.expander(f"📊 {test_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**When to Use:** {info['when_to_use']}")
                st.write(f"**Effect Size:** {info['effect_size']}")
            with col2:
                st.write(f"**Key Assumptions:**")
                for assumption in info['assumptions']:
                    st.write(f"• {assumption}")

            st.write(f"**Example:** {info['example']}")
            st.write(f"**Interpretation:** {info['interpretation']}")
            st.write(f"**Diagnostics:** {info['diagnostics']}")
            st.write(f"**Alternatives:** {info['alternatives']}")

def show_categorical_tests():
    """Show information about categorical data tests"""

    st.markdown("### Tests for Independence")

    independence_tests = {
        "Chi-squared test": {
            "purpose": "Test for association between two categorical variables",
            "assumptions": ["Independence of observations", "Expected frequencies ≥ 5 in each cell", "Mutually exclusive categories"],
            "when_to_use": "Two categorical variables with adequate sample sizes",
            "example": "Testing association between smoking status and lung cancer",
            "effect_size": "Cramér's V",
            "interpretation": "Tests if distribution of one variable differs across levels of another",
            "limitations": "Sensitive to sample size, doesn't indicate direction of association",
            "alternatives": "Use Fisher's exact test for small expected frequencies"
        },
        "Fisher's exact test": {
            "purpose": "Test for association in 2×2 contingency tables with small sample sizes",
            "assumptions": ["Independence of observations", "2×2 contingency table", "Fixed marginal totals"],
            "when_to_use": "Small sample sizes where chi-squared assumptions are violated",
            "example": "Testing association between treatment and outcome in small clinical trial",
            "effect_size": "Odds ratio",
            "interpretation": "Provides exact p-value for association in 2×2 tables",
            "limitations": "Limited to 2×2 tables, computationally intensive for large samples",
            "alternatives": "Use chi-squared test for larger samples"
        }
    }

    for test_name, info in independence_tests.items():
        with st.expander(f"📊 {test_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**When to Use:** {info['when_to_use']}")
                st.write(f"**Effect Size:** {info['effect_size']}")
            with col2:
                st.write(f"**Key Assumptions:**")
                for assumption in info['assumptions']:
                    st.write(f"• {assumption}")

            st.write(f"**Example:** {info['example']}")
            st.write(f"**Interpretation:** {info['interpretation']}")
            st.write(f"**Limitations:** {info['limitations']}")
            st.write(f"**Alternatives:** {info['alternatives']}")

    st.markdown("### Tests for Paired Categorical Data")

    paired_tests = {
        "McNemar's test": {
            "purpose": "Test for change in proportions in paired binary data",
            "assumptions": ["Paired binary observations", "Independence of pairs", "Dichotomous outcome"],
            "when_to_use": "Before-after studies with binary outcomes, matched case-control studies",
            "example": "Testing if training program changes pass/fail rates in same students",
            "effect_size": "Odds ratio for change",
            "interpretation": "Tests if probability of success changes between paired conditions",
            "focus": "Only considers discordant pairs (those who changed)",
            "alternatives": "Use paired t-test for continuous outcomes"
        },
        "Cochran's Q test": {
            "purpose": "Test for change in proportions across multiple related measurements",
            "assumptions": ["Related binary observations", "Independence of subjects", "3+ time points or conditions"],
            "when_to_use": "Multiple repeated binary measurements on same subjects",
            "example": "Testing if success rates change across 4 quarters in same companies",
            "effect_size": "Effect size depends on specific implementation",
            "interpretation": "Tests if proportion of successes differs across repeated measurements",
            "follow_up": "Pairwise McNemar tests with correction for multiple comparisons",
            "alternatives": "Use repeated measures ANOVA for continuous outcomes"
        }
    }

    for test_name, info in paired_tests.items():
        with st.expander(f"📊 {test_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**When to Use:** {info['when_to_use']}")
                st.write(f"**Effect Size:** {info['effect_size']}")
            with col2:
                st.write(f"**Key Assumptions:**")
                for assumption in info['assumptions']:
                    st.write(f"• {assumption}")

            st.write(f"**Example:** {info['example']}")
            st.write(f"**Interpretation:** {info['interpretation']}")
            if 'focus' in info:
                st.write(f"**Focus:** {info['focus']}")
            if 'follow_up' in info:
                st.write(f"**Follow-up:** {info['follow_up']}")
            st.write(f"**Alternatives:** {info['alternatives']}")

def show_advanced_tests():
    """Show information about advanced multivariate tests"""

    st.markdown("### Multivariate Analysis of Variance")

    advanced_tests = {
        "MANOVA": {
            "purpose": "Compare means of multiple dependent variables across groups simultaneously",
            "assumptions": ["Multivariate normality", "Homogeneity of covariance matrices", "Independence", "Linear relationships"],
            "when_to_use": "Multiple related dependent variables, want to control for correlations",
            "example": "Comparing effects of diet on weight, cholesterol, and blood pressure simultaneously",
            "effect_size": "Partial eta-squared, Pillai's trace",
            "interpretation": "Tests overall group differences across all dependent variables",
            "advantages": "Controls Type I error, more powerful when variables are correlated",
            "follow_up": "Univariate ANOVAs or discriminant analysis if significant"
        },
        "ANCOVA": {
            "purpose": "Compare group means while controlling for continuous covariates",
            "assumptions": ["Normality", "Homogeneity of variances", "Independence", "Homogeneity of regression slopes"],
            "when_to_use": "Want to control for confounding variables, increase statistical power",
            "example": "Comparing treatment effects on test scores while controlling for baseline IQ",
            "effect_size": "Partial eta-squared",
            "interpretation": "Tests group differences after adjusting for covariate effects",
            "advantages": "Increases power by reducing error variance, controls for confounders",
            "follow_up": "Adjusted pairwise comparisons if main effect is significant"
        }
    }

    for test_name, info in advanced_tests.items():
        with st.expander(f"🔬 {test_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**When to Use:** {info['when_to_use']}")
                st.write(f"**Effect Size:** {info['effect_size']}")
                st.write(f"**Advantages:** {info['advantages']}")
            with col2:
                st.write(f"**Key Assumptions:**")
                for assumption in info['assumptions']:
                    st.write(f"• {assumption}")

            st.write(f"**Example:** {info['example']}")
            st.write(f"**Interpretation:** {info['interpretation']}")
            st.write(f"**Follow-up:** {info['follow_up']}")

def show_posthoc_tests():
    """Show information about post-hoc tests"""

    st.markdown("### Multiple Comparison Procedures")

    st.markdown("""
    <div class="warning-box">
    <p><strong>Important:</strong> Post-hoc tests are used after a significant omnibus test (ANOVA, Kruskal-Wallis)
    to identify which specific groups differ from each other while controlling for multiple comparisons.</p>
    </div>
    """, unsafe_allow_html=True)

    posthoc_tests = {
        "Tukey's HSD": {
            "purpose": "All pairwise comparisons with optimal balance of power and Type I error control",
            "controls": "Family-wise error rate (FWER)",
            "best_for": "Equal sample sizes, homogeneous variances, all pairwise comparisons",
            "when_to_use": "After significant one-way ANOVA with equal group sizes",
            "advantages": "Optimal power among FWER-controlling methods, controls overall α",
            "disadvantages": "Requires equal variances assumption",
            "recommendation": "First choice for most ANOVA follow-ups"
        },
        "Bonferroni correction": {
            "purpose": "Conservative multiple comparison correction",
            "controls": "Family-wise error rate (FWER)",
            "best_for": "Any design, planned comparisons, small number of tests",
            "when_to_use": "When you need very conservative control, planned contrasts",
            "advantages": "Simple to understand and calculate, very conservative",
            "disadvantages": "Low power with many comparisons, overly conservative",
            "recommendation": "Use for planned comparisons or when being conservative is important"
        },
        "Holm-Šídák method": {
            "purpose": "Sequential testing method more powerful than Bonferroni",
            "controls": "Family-wise error rate (FWER)",
            "best_for": "Multiple comparisons where order of testing doesn't matter",
            "when_to_use": "Want more power than Bonferroni while controlling FWER",
            "advantages": "More powerful than Bonferroni, still controls FWER",
            "disadvantages": "More complex than Bonferroni, sequential nature",
            "recommendation": "Good alternative to Bonferroni for exploratory analysis"
        },
        "Benjamini-Hochberg": {
            "purpose": "Controls false discovery rate instead of family-wise error rate",
            "controls": "False Discovery Rate (FDR)",
            "best_for": "Exploratory research, large number of comparisons",
            "when_to_use": "Many comparisons, exploratory analysis, genomics research",
            "advantages": "Much more powerful than FWER methods with many tests",
            "disadvantages": "Allows some false positives, less familiar to many researchers",
            "recommendation": "Excellent for exploratory research with many comparisons"
        },
        "Dunn's test": {
            "purpose": "Non-parametric post-hoc test for Kruskal-Wallis follow-up",
            "controls": "Family-wise error rate (FWER)",
            "best_for": "Follow-up to significant Kruskal-Wallis test",
            "when_to_use": "After significant Kruskal-Wallis test, non-normal data",
            "advantages": "Designed specifically for Kruskal-Wallis follow-up",
            "disadvantages": "Less powerful than parametric alternatives",
            "recommendation": "Standard choice for Kruskal-Wallis post-hoc analysis"
        },
        "Games-Howell": {
            "purpose": "Post-hoc test that handles unequal variances and sample sizes",
            "controls": "Family-wise error rate (FWER)",
            "best_for": "Unequal variances, unequal sample sizes",
            "when_to_use": "When Levene's test shows unequal variances",
            "advantages": "Robust to unequal variances and sample sizes",
            "disadvantages": "More complex calculations, less familiar",
            "recommendation": "Use when assumption of equal variances is violated"
        }
    }

    for test_name, info in posthoc_tests.items():
        with st.expander(f"📊 {test_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**Controls:** {info['controls']}")
                st.write(f"**Best For:** {info['best_for']}")
                st.write(f"**Advantages:** {info['advantages']}")
            with col2:
                st.write(f"**When to Use:** {info['when_to_use']}")
                st.write(f"**Disadvantages:** {info['disadvantages']}")

            st.markdown(f"**💡 Recommendation:** {info['recommendation']}")

    # Decision guide
    st.markdown("### 🎯 Post-Hoc Test Selection Guide")

    decision_guide = """
    | Situation | Recommended Test | Rationale |
    |-----------|------------------|-----------|
    | Significant one-way ANOVA, equal variances | Tukey's HSD | Optimal balance of power and error control |
    | Significant one-way ANOVA, unequal variances | Games-Howell | Robust to assumption violations |
    | Significant Kruskal-Wallis test | Dunn's test | Designed for non-parametric follow-up |
    | Planned comparisons (few tests) | Bonferroni | Conservative, simple, appropriate for planned tests |
    | Exploratory analysis (many tests) | Benjamini-Hochberg | More powerful for multiple comparisons |
    | Need maximum power while controlling FWER | Holm-Šídák | More powerful than Bonferroni |
    """

    st.markdown(decision_guide)

def show_power_analysis_info():
    """Show information about power analysis"""

    st.markdown("### Statistical Power Concepts")

    power_concepts = {
        "Statistical Power": {
            "definition": "The probability of correctly rejecting a false null hypothesis (1 - β)",
            "typical_value": "0.80 (80%) is the conventional standard",
            "interpretation": "80% power means 80% chance of detecting a true effect if it exists",
            "factors": "Depends on effect size, sample size, alpha level, and test type",
            "importance": "Low power leads to high risk of missing real effects (Type II error)"
        },
        "Effect Size": {
            "definition": "Standardized measure of the magnitude of difference or relationship",
            "examples": "Cohen's d, eta-squared, correlation coefficient",
            "conventions": "Small (d=0.2), Medium (d=0.5), Large (d=0.8) for Cohen's d",
            "importance": "Determines practical significance beyond statistical significance",
            "consideration": "Should be based on practical/clinical importance, not just conventions"
        },
        "Sample Size": {
            "definition": "Number of observations needed to achieve desired power",
            "relationship": "Larger samples provide more power to detect smaller effects",
            "calculation": "Based on desired power, effect size, and alpha level",
            "practical": "Must balance statistical requirements with practical constraints",
            "recommendation": "Calculate before data collection, not after"
        }
    }

    for concept, info in power_concepts.items():
        with st.expander(f"📊 {concept}"):
            for key, value in info.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")

    st.markdown("### 🎯 Power Analysis Applications")

    applications = {
        "Prospective (Planning)": {
            "purpose": "Determine required sample size before data collection",
            "inputs": "Expected effect size, desired power (usually 80%), alpha level (usually 0.05)",
            "outputs": "Minimum sample size needed",
            "benefits": "Ensures adequate power, prevents underpowered studies",
            "example": "Need 64 subjects per group to detect medium effect (d=0.5) with 80% power"
        },
        "Retrospective (Post-hoc)": {
            "purpose": "Calculate achieved power after data collection",
            "inputs": "Observed effect size, actual sample size, alpha level",
            "outputs": "Statistical power achieved",
            "benefits": "Helps interpret non-significant results",
            "example": "Study with n=20 per group achieved only 35% power to detect medium effect"
        },
        "Sensitivity Analysis": {
            "purpose": "Determine minimum detectable effect size for given sample and power",
            "inputs": "Sample size, desired power, alpha level",
            "outputs": "Minimum effect size detectable",
            "benefits": "Shows what effects the study can realistically detect",
            "example": "With n=30 per group, can detect effects of d=0.73 or larger with 80% power"
        }
    }

    for analysis_type, info in applications.items():
        with st.expander(f"⚡ {analysis_type} Power Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**Inputs:** {info['inputs']}")
                st.write(f"**Outputs:** {info['outputs']}")
            with col2:
                st.write(f"**Benefits:** {info['benefits']}")
                st.write(f"**Example:** {info['example']}")

    # Effect size guidelines
    st.markdown("### 📏 Effect Size Guidelines")

    effect_size_table = """
    | Test Type | Effect Size Measure | Small | Medium | Large |
    |-----------|-------------------|--------|---------|--------|
    | t-test | Cohen's d | 0.2 | 0.5 | 0.8 |
    | ANOVA | Eta-squared (η²) | 0.01 | 0.06 | 0.14 |
    | Correlation | Pearson's r | 0.1 | 0.3 | 0.5 |
    | Chi-square | Cramér's V | 0.1 | 0.3 | 0.5 |
    """

    st.markdown(effect_size_table)

    st.markdown("""
    <div class="info-box">
    <p><strong>Note:</strong> These are general conventions. Effect sizes should be interpreted in the context
    of your specific field and research question. A "small" effect might be very important in some contexts.</p>
    </div>
    """, unsafe_allow_html=True)

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

def show_advanced_analysis_page():
    """Display advanced statistical analysis options"""

    st.markdown('<h2 class="section-header">🔬 Advanced Statistical Analysis</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <p>Advanced statistical methods for complex research questions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if data is available
    if st.session_state.data is None:
        st.warning("Please upload data in the 'Data Upload & Analysis' section first.")
        return

    data = st.session_state.data
    st.write(f"**Current Dataset:** {data.shape[0]} rows, {data.shape[1]} columns")

    # Advanced test selection
    advanced_test = st.selectbox(
        "Select Advanced Test:",
        ["MANOVA", "ANCOVA", "Logistic Regression", "Multiple Linear Regression", "Two-way ANOVA"]
    )

    advanced_tests = AdvancedStatisticalTests()

    if advanced_test == "MANOVA":
        st.subheader("Multivariate Analysis of Variance (MANOVA)")

        # Variable selection
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        dependent_vars = st.multiselect("Select dependent variables (continuous):", numeric_columns)
        independent_var = st.selectbox("Select independent variable (categorical):", categorical_columns)

        if len(dependent_vars) >= 2 and independent_var:
            if st.button("Perform MANOVA"):
                with st.spinner("Running MANOVA..."):
                    results = advanced_tests.perform_manova(data, dependent_vars, independent_var)

                    if 'error' not in results:
                        st.success("MANOVA completed!")

                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Wilks' Lambda", f"{results['wilks_lambda']:.4f}")
                            st.metric("F-statistic", f"{results['statistic']:.4f}")
                        with col2:
                            st.metric("P-value", f"{results['p_value']:.6f}")
                            st.metric("Effect Size", f"{results['effect_size']:.4f}")

                        # Group means
                        st.subheader("Group Means")
                        for var in dependent_vars:
                            st.write(f"**{var}:**")
                            means_df = pd.DataFrame(results['group_means'][var], index=['Mean']).T
                            st.dataframe(means_df)
                    else:
                        st.error(results['error'])

    elif advanced_test == "ANCOVA":
        st.subheader("Analysis of Covariance (ANCOVA)")

        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        dependent_var = st.selectbox("Select dependent variable:", numeric_columns)
        independent_var = st.selectbox("Select independent variable (categorical):", categorical_columns)
        covariate = st.selectbox("Select covariate (continuous):", numeric_columns)

        if dependent_var and independent_var and covariate and dependent_var != covariate:
            if st.button("Perform ANCOVA"):
                with st.spinner("Running ANCOVA..."):
                    results = advanced_tests.perform_ancova(data, dependent_var, independent_var, covariate)

                    if 'error' not in results:
                        st.success("ANCOVA completed!")

                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Main Effect F", f"{results['statistic']:.4f}")
                            st.metric("Main Effect p-value", f"{results['p_value']:.6f}")
                        with col2:
                            st.metric("Covariate F", f"{results['covariate_f']:.4f}")
                            st.metric("Covariate p-value", f"{results['covariate_p']:.6f}")

                        st.metric("Effect Size (η²)", f"{results['effect_size']:.4f}")

                        # Group statistics
                        st.subheader("Adjusted Group Statistics")
                        stats_data = []
                        for group, stats in results['group_statistics'].items():
                            stats_data.append([
                                group, stats['n'],
                                f"{stats['mean_dependent']:.3f}",
                                f"{stats['mean_covariate']:.3f}"
                            ])

                        stats_df = pd.DataFrame(stats_data,
                                              columns=['Group', 'N', f'Mean {dependent_var}', f'Mean {covariate}'])
                        st.dataframe(stats_df)
                    else:
                        st.error(results['error'])

    elif advanced_test == "Logistic Regression":
        st.subheader("Logistic Regression")

        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        all_columns = data.columns.tolist()

        # Check for binary variables
        binary_columns = []
        for col in all_columns:
            if data[col].nunique() == 2:
                binary_columns.append(col)

        dependent_var = st.selectbox("Select dependent variable (binary):", binary_columns)
        independent_vars = st.multiselect("Select independent variables:",
                                        [col for col in all_columns if col != dependent_var])

        if dependent_var and independent_vars:
            if st.button("Perform Logistic Regression"):
                with st.spinner("Running Logistic Regression..."):
                    results = advanced_tests.perform_logistic_regression(data, dependent_var, independent_vars)

                    if 'error' not in results:
                        st.success("Logistic Regression completed!")

                        # Display results
                        st.metric("Accuracy", f"{results['accuracy']:.4f}")
                        st.metric("Pseudo R²", f"{results['pseudo_r_squared']:.4f}")

                        # Coefficients and odds ratios
                        st.subheader("Coefficients and Odds Ratios")
                        coef_data = []
                        for var in independent_vars:
                            coef_data.append([
                                var,
                                f"{results['coefficients'][var]:.4f}",
                                f"{results['odds_ratios'][var]:.4f}"
                            ])

                        coef_df = pd.DataFrame(coef_data, columns=['Variable', 'Coefficient', 'Odds Ratio'])
                        st.dataframe(coef_df)

                        # Classification report
                        st.subheader("Classification Performance")
                        class_report = results['classification_report']
                        st.write(f"**Precision (Class 1):** {class_report['1']['precision']:.3f}")
                        st.write(f"**Recall (Class 1):** {class_report['1']['recall']:.3f}")
                        st.write(f"**F1-Score (Class 1):** {class_report['1']['f1-score']:.3f}")
                    else:
                        st.error(results['error'])

def show_power_analysis_page():
    """Display power analysis and sample size calculation page"""

    st.markdown('<h2 class="section-header">⚡ Power Analysis & Sample Size Calculation</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <p>Calculate statistical power and determine appropriate sample sizes for your research.</p>
    </div>
    """, unsafe_allow_html=True)

    power_analyzer = PowerAnalysis()

    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Sample Size Calculation", "Power Calculation", "Comprehensive Analysis"]
    )

    test_type = st.selectbox(
        "Select Test Type:",
        ["t-test", "ANOVA", "Correlation", "Chi-square"]
    )

    if analysis_type == "Sample Size Calculation":
        st.subheader("Calculate Required Sample Size")

        col1, col2 = st.columns(2)
        with col1:
            effect_size = st.slider("Effect Size", 0.1, 2.0, 0.5, 0.1)
            power = st.slider("Desired Power", 0.5, 0.99, 0.8, 0.01)
        with col2:
            alpha = st.slider("Alpha Level", 0.001, 0.1, 0.05, 0.001)

        if test_type == "t-test":
            test_design = st.selectbox("Test Design:", ["two-sample", "one-sample", "paired"])
        elif test_type == "ANOVA":
            num_groups = st.number_input("Number of Groups:", 3, 10, 3)

        if st.button("Calculate Sample Size"):
            with st.spinner("Calculating..."):
                try:
                    if test_type == "t-test":
                        n_required = power_analyzer.calculate_sample_size_ttest(
                            effect_size, power, alpha, test_design
                        )
                    elif test_type == "ANOVA":
                        n_required = power_analyzer.calculate_sample_size_anova(
                            effect_size, num_groups, power, alpha
                        )
                    elif test_type == "Correlation":
                        n_required = power_analyzer.calculate_sample_size_correlation(
                            effect_size, power, alpha
                        )
                    elif test_type == "Chi-square":
                        n_required = power_analyzer.calculate_sample_size_chisquare(
                            effect_size, 1, power, alpha
                        )

                    if n_required:
                        st.success(f"Required sample size: **{n_required}** participants per group")

                        # Additional context
                        st.markdown(f"""
                        **Analysis Parameters:**
                        - Effect size: {effect_size}
                        - Desired power: {power} ({power*100:.0f}%)
                        - Alpha level: {alpha}
                        - Test type: {test_type}
                        """)

                        # Recommendations
                        st.markdown("""
                        **Recommendations:**
                        - Consider adding 10-20% more participants to account for dropouts
                        - Verify that this sample size is feasible for your study
                        - Consider the practical significance of your effect size
                        """)
                    else:
                        st.error("Unable to calculate sample size with these parameters.")

                except Exception as e:
                    st.error(f"Calculation failed: {str(e)}")

    elif analysis_type == "Power Calculation":
        st.subheader("Calculate Statistical Power")

        col1, col2 = st.columns(2)
        with col1:
            effect_size = st.slider("Effect Size", 0.1, 2.0, 0.5, 0.1)
            sample_size = st.number_input("Sample Size per Group:", 5, 1000, 30)
        with col2:
            alpha = st.slider("Alpha Level", 0.001, 0.1, 0.05, 0.001)

        if test_type == "t-test":
            test_design = st.selectbox("Test Design:", ["two-sample", "one-sample", "paired"])
        elif test_type == "ANOVA":
            num_groups = st.number_input("Number of Groups:", 3, 10, 3)

        if st.button("Calculate Power"):
            with st.spinner("Calculating..."):
                try:
                    if test_type == "t-test":
                        power = power_analyzer.calculate_power_ttest(
                            effect_size, sample_size, alpha, test_design
                        )
                    elif test_type == "ANOVA":
                        power = power_analyzer.calculate_power_anova(
                            effect_size, sample_size, num_groups, alpha
                        )
                    elif test_type == "Correlation":
                        power = power_analyzer.calculate_power_correlation(
                            effect_size, sample_size, alpha
                        )
                    elif test_type == "Chi-square":
                        power = power_analyzer.calculate_power_chisquare(
                            effect_size, sample_size, 1, alpha
                        )

                    if power is not None:
                        # Color code power level
                        if power >= 0.8:
                            power_color = "green"
                            power_assessment = "Adequate"
                        elif power >= 0.6:
                            power_color = "orange"
                            power_assessment = "Moderate"
                        else:
                            power_color = "red"
                            power_assessment = "Low"

                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border: 2px solid {power_color}; border-radius: 10px;">
                        <h3 style="color: {power_color};">Statistical Power: {power:.3f} ({power*100:.1f}%)</h3>
                        <p style="color: {power_color}; font-size: 18px;">{power_assessment} Power</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Interpretation
                        st.markdown(f"""
                        **Interpretation:**
                        - There is a {power*100:.1f}% chance of detecting a true effect of size {effect_size}
                        - Risk of Type II error (β): {(1-power)*100:.1f}%
                        """)

                        if power < 0.8:
                            st.warning("⚠️ Power is below the conventional standard of 80%. Consider increasing sample size.")
                    else:
                        st.error("Unable to calculate power with these parameters.")

                except Exception as e:
                    st.error(f"Calculation failed: {str(e)}")

    else:  # Comprehensive Analysis
        st.subheader("Comprehensive Power Analysis")

        if st.button("Generate Comprehensive Analysis"):
            with st.spinner("Generating comprehensive analysis..."):
                try:
                    # Set default parameters based on test type
                    kwargs = {'power': 0.8, 'alpha': 0.05}
                    if test_type == "ANOVA":
                        kwargs['num_groups'] = 3

                    comprehensive_results = power_analyzer.comprehensive_power_analysis(test_type, **kwargs)

                    st.success("Comprehensive analysis completed!")

                    # Display sample size recommendations
                    st.subheader("Sample Size Recommendations")
                    recommendations = comprehensive_results['sample_size_recommendations']

                    rec_data = []
                    for effect_type, data in recommendations.items():
                        rec_data.append([
                            effect_type.title(),
                            f"{data['effect_size']:.2f}",
                            f"{data['sample_size_required']}" if data['sample_size_required'] else "N/A"
                        ])

                    rec_df = pd.DataFrame(rec_data, columns=['Effect Size', 'Value', 'Sample Size Required'])
                    st.dataframe(rec_df)

                    # Power curves
                    st.subheader("Power Curves")
                    st.write("Power vs. Sample Size for Different Effect Sizes")

                    # Create power curve plot
                    import plotly.graph_objects as go
                    fig = go.Figure()

                    colors = ['blue', 'orange', 'green']
                    for i, (effect_name, curve_data) in enumerate(comprehensive_results['power_curves'].items()):
                        fig.add_trace(go.Scatter(
                            x=curve_data['sample_sizes'],
                            y=curve_data['powers'],
                            mode='lines',
                            name=f'{effect_name.title()} Effect (d={curve_data["effect_size"]})',
                            line=dict(color=colors[i % len(colors)])
                        ))

                    # Add horizontal line at 80% power
                    fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                                 annotation_text="80% Power Threshold")

                    fig.update_layout(
                        title=f"Power Analysis for {test_type.title()}",
                        xaxis_title="Sample Size per Group",
                        yaxis_title="Statistical Power",
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Interpretation
                    interpretation = power_analyzer.interpret_power_analysis(comprehensive_results)
                    st.markdown(interpretation)

                except Exception as e:
                    st.error(f"Comprehensive analysis failed: {str(e)}")

if __name__ == "__main__":
    main()