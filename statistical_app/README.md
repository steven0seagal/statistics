# Statistical Testing Streamlit Application

🔬 **A comprehensive, educational statistical analysis platform** that provides intelligent test selection, data analysis capabilities, and results interpretation for researchers, students, and professionals.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This application bridges the gap between statistical theory and practical application, offering an intuitive interface for conducting rigorous statistical analyses. Whether you're a researcher, student, or data analyst, this tool provides expert guidance through the entire statistical analysis workflow.

## ✨ Key Features

### 🤖 Automatic Test Recommender (NEW!)
- **Upload & Analyze**: Simply upload your data and get instant recommendations
- **Intelligent Profiling**: Automatic detection of numeric vs. categorical columns
- **Assumption Testing**: Automated normality (Shapiro-Wilk) and homoscedasticity (Levene's) checks
- **Smart Recommendations**: Decision tree logic recommends the optimal test for your data
- **One-Click Execution**: Run the recommended test with a single button click
- **11 Core Tests Supported**: From t-tests to correlations to ANOVA
- **Demo Datasets Included**: 11 ready-to-use example datasets

### 🧭 Intelligent Test Selection Wizard
- **Smart Recommendations**: AI-powered test selection based on data characteristics
- **Decision Tree Logic**: Step-by-step guidance following statistical best practices
- **Comprehensive Coverage**: Considers research goals, variable types, study design, and sample size
- **Alternative Suggestions**: Recommends alternatives when assumptions are violated

### 📈 Advanced Data Analysis Pipeline
- **Multi-Format Support**: CSV, Excel, and 5 built-in sample datasets
- **Automated Preprocessing**: Data validation, type detection, and cleaning
- **Outlier Detection**: Multiple methods (IQR, Z-score, Modified Z-score)
- **Missing Data Handling**: Intelligent strategies and recommendations

### 🔍 Comprehensive Assumption Checking
- **Automated Validation**: Real-time checking of all statistical test assumptions
- **Visual Diagnostics**: Q-Q plots, residual plots, and distribution plots
- **Multiple Tests**: Shapiro-Wilk, Kolmogorov-Smirnov, Levene's, Bartlett's
- **Alternative Recommendations**: Suggests non-parametric alternatives when needed

### 📊 Complete Statistical Test Library (28+ Tests)
- **Basic Comparisons**: t-tests, Mann-Whitney U, Wilcoxon signed-rank
- **Multiple Groups**: ANOVA, Kruskal-Wallis, Friedman, Repeated Measures ANOVA
- **Categorical Analysis**: Chi-squared, Fisher's exact, McNemar's, Cochran's Q
- **Correlation & Regression**: Pearson, Spearman, Linear, Multiple, Logistic regression
- **Advanced Multivariate**: MANOVA, ANCOVA, Two-way ANOVA
- **Post-Hoc Procedures**: 7 comprehensive multiple comparison methods

### 📈 Publication-Ready Visualizations
- **Test-Specific Plots**: Automatically generated appropriate visualizations
- **Interactive Graphics**: Plotly-powered plots with zoom, pan, and export
- **Diagnostic Plots**: Assumption checking visualizations
- **Customizable Styling**: Professional themes for publications

### 🎓 Comprehensive Educational System
- **Interactive Learning**: Step-by-step tutorials and examples
- **Test Library**: Detailed documentation for all 28+ statistical tests
- **Best Practices**: Common pitfalls and how to avoid them
- **Real Examples**: Biological and research-based use cases

### 🧠 Expert Results Interpretation
- **Automated Analysis**: Intelligent interpretation of statistical outputs
- **Effect Size Calculations**: Cohen's d, eta-squared, Cramer's V, and more
- **Practical Significance**: Beyond p-values to meaningful results
- **Follow-up Recommendations**: Guidance for next analytical steps

### ⚡ Advanced Power Analysis Suite
- **Sample Size Planning**: Calculate required N for desired power
- **Power Assessment**: Determine power for existing studies
- **Interactive Curves**: Visual power analysis with multiple scenarios
- **Multiple Test Support**: t-test, ANOVA, correlation, chi-square, and more

### 📊 Professional Post-Hoc Testing
- **7 Methods Available**: Tukey's HSD, Bonferroni, Holm-Šídák, Benjamini-Hochberg, Dunn's, Games-Howell, Sidak
- **Smart Recommendations**: Automatic selection based on test conditions
- **Family-wise Error Control**: Proper multiple comparison correction
- **Effect Size Integration**: Post-hoc effect sizes and confidence intervals

### 📝 Professional Report Generation
- **Three Report Types**: Basic, Comprehensive, and Publication formats
- **Multiple Export Options**: HTML, Markdown, and plain text
- **APA-Style Formatting**: Professional statistical reporting standards
- **Complete Documentation**: Methods, results, and interpretations

## 🚀 Quick Start

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended for large datasets)
- **Storage**: 500MB free space
- **Browser**: Chrome, Firefox, Safari, or Edge

### Installation Methods

#### Method 1: Standard Installation
```bash
# 1. Navigate to your projects directory
cd /path/to/your/projects

# 2. Clone or download the repository
git clone <repository-url>  # or download and extract ZIP
cd statistical_app

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
```

#### Method 2: Virtual Environment (Recommended)
```bash
# 1. Create virtual environment
python -m venv statistical_app_env

# 2. Activate virtual environment
# On Windows:
statistical_app_env\Scripts\activate
# On macOS/Linux:
source statistical_app_env/bin/activate

# 3. Navigate to project and install
cd statistical_app
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
```

#### Method 3: Using run_app.py (Simplified)
```bash
# Single command to launch
python run_app.py
```

### 📦 Dependencies

**Core Libraries:**
- streamlit >= 1.28.0 (Web framework)
- pandas >= 1.5.0 (Data manipulation)
- numpy >= 1.24.0 (Numerical computing)
- scipy >= 1.10.0 (Statistical functions)

**Visualization:**
- plotly >= 5.15.0 (Interactive plots)
- seaborn >= 0.12.0 (Statistical visualizations)
- matplotlib >= 3.7.0 (Publication plots)

**Statistical Analysis:**
- statsmodels >= 0.14.0 (Advanced statistics)
- pingouin >= 0.5.3 (Statistical tests)
- scikit-learn >= 1.3.0 (Machine learning)

**File Support:**
- openpyxl >= 3.1.0 (Excel files)
- xlrd >= 2.0.0 (Excel reading)
- markdown >= 3.4.0 (Report generation)

### 🔧 Troubleshooting Installation

**Common Issues:**

1. **Python Version Error**
   ```bash
   python --version  # Check version is 3.8+
   ```

2. **Package Installation Fails**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --no-cache-dir
   ```

3. **Permission Errors (macOS/Linux)**
   ```bash
   pip install -r requirements.txt --user
   ```

4. **Windows Long Path Issues**
   - Enable long paths in Windows settings
   - Or use shorter directory paths

## 📖 User Guide

### 🏠 Application Navigation

**8 Main Pages:**
1. **🏠 Home**: Overview, quick start guide, and feature highlights
2. **🧭 Test Selection Wizard**: Interactive test recommendation system
3. **📊 Data Upload & Analysis**: Main analysis workspace
4. **🤖 Automatic Test Recommender**: Upload data and get instant test recommendations
5. **🔬 Advanced Analysis**: MANOVA, ANCOVA, and multivariate methods
6. **⚡ Power Analysis**: Sample size calculation and power assessment
7. **🎓 Educational Content**: Statistical concepts and best practices
8. **📚 Test Library**: Comprehensive reference for all 28+ tests

### 🔄 Complete Analysis Workflow

#### Phase 1: Planning & Preparation
1. **Define Research Question**
   - What are you trying to test or explore?
   - Identify dependent and independent variables
   - Consider study design (experimental vs. observational)

2. **Prepare Your Data**
   - Format: CSV or Excel with clear column headers
   - Structure: One row per observation, one column per variable
   - Clean: Remove empty rows/columns, handle missing values
   - Validate: Ensure consistent data types

#### Phase 2: Test Selection
3. **Use Test Selection Wizard**
   - Answer guided questions about your research
   - Get intelligent test recommendations
   - Review alternative options and rationale

4. **Explore Sample Data** (Optional)
   - Try built-in datasets to understand workflow
   - Practice with different test types
   - Learn from provided examples

#### Phase 3: Data Analysis
5. **Upload and Validate Data**
   - Upload your prepared dataset
   - Review automatic data type detection
   - Check summary statistics and distributions

6. **Check Assumptions**
   - Review automated assumption tests
   - Examine visual diagnostic plots
   - Consider alternative tests if assumptions fail

7. **Execute Analysis**
   - Run recommended statistical test
   - Review detailed output and statistics
   - Examine effect sizes and confidence intervals

#### Phase 4: Interpretation & Reporting
8. **Interpret Results**
   - Read comprehensive result explanations
   - Understand practical vs. statistical significance
   - Review post-hoc tests if applicable

9. **Generate Reports**
   - Choose appropriate report format
   - Export in preferred format (HTML/Markdown/Text)
   - Save visualizations and analysis results

### 📊 Built-in Sample Datasets

**5 Professional Datasets for Learning:**

1. **🧬 Biological Growth Study**
   - Compare growth rates between treatment groups
   - Variables: treatment, growth_rate, time_point
   - Best for: t-tests, ANOVA

2. **🧪 Gene Expression Analysis**
   - Differential expression across conditions
   - Variables: gene_id, expression_level, condition, replicate
   - Best for: Multiple comparisons, post-hoc tests

3. **🏥 Clinical Trial Data**
   - Treatment effectiveness comparison
   - Variables: treatment_group, outcome_score, baseline_score, age
   - Best for: ANCOVA, paired tests

4. **🌿 Ecological Survey**
   - Species abundance across habitats
   - Variables: habitat_type, species_count, environmental_factors
   - Best for: Non-parametric tests, correlation

5. **📈 Paired Measurements**
   - Before-after treatment effects
   - Variables: subject_id, before_score, after_score, treatment
   - Best for: Paired t-test, Wilcoxon signed-rank

### 🎯 Use Case Examples

#### Example 1: Quick Analysis with Automatic Recommender (NEW!)
**Research Question**: I have data - what test should I use?
- **Data**: Any CSV/Excel file with your measurements
- **Recommended Path**: Automatic Test Recommender → Upload → Select columns → Execute
- **Key Features**:
  - Automatic column type detection
  - Normality and variance testing
  - Instant test recommendation with rationale
  - One-click execution
  - Try with demo datasets: `demo_data/demo_ttest_independent.csv`

#### Example 2: Comparing Two Groups
**Research Question**: Does a new drug improve patient outcomes?
- **Data**: Patient scores before/after treatment
- **Recommended Path**: Test Selection Wizard → Paired t-test
- **Key Features**: Assumption checking, effect size, confidence intervals

#### Example 3: Multiple Group Comparison
**Research Question**: Which of 4 treatments is most effective?
- **Data**: Treatment groups and outcome measures
- **Recommended Path**: ANOVA → Tukey's post-hoc tests
- **Key Features**: Multiple comparisons, family-wise error control

#### Example 4: Relationship Analysis
**Research Question**: Is there a relationship between age and response time?
- **Data**: Age and response time measurements
- **Recommended Path**: Correlation analysis → Regression modeling
- **Key Features**: Scatter plots, correlation coefficients, prediction intervals

## 📊 Complete Statistical Test Library (28+ Tests)

### 🔄 Comparing Two Groups

**Parametric Tests:**
- **Independent Samples t-test**: Compare means of two independent groups
- **Welch's t-test**: Independent groups with unequal variances
- **Paired Samples t-test**: Compare related/matched observations

**Non-parametric Tests:**
- **Mann-Whitney U**: Non-parametric alternative to independent t-test
- **Wilcoxon Signed-Rank**: Non-parametric alternative to paired t-test

### 🔄 Comparing Multiple Groups

**Parametric Tests:**
- **One-way ANOVA**: Compare means across 3+ independent groups
- **Repeated Measures ANOVA**: Compare related observations across conditions
- **Two-way ANOVA**: Analyze two factors and their interaction

**Non-parametric Tests:**
- **Kruskal-Wallis**: Non-parametric alternative to one-way ANOVA
- **Friedman Test**: Non-parametric alternative to repeated measures ANOVA

### 📈 Relationships Between Variables

**Correlation:**
- **Pearson Correlation**: Linear relationships between continuous variables
- **Spearman Rank Correlation**: Monotonic relationships, non-parametric

**Regression:**
- **Simple Linear Regression**: Predict one variable from another
- **Multiple Linear Regression**: Predict from multiple variables
- **Logistic Regression**: Predict binary outcomes

### 🎯 Categorical Data Analysis

- **Chi-squared Test of Independence**: Association between categorical variables
- **Fisher's Exact Test**: Small sample alternative to chi-squared
- **McNemar's Test**: Paired categorical data
- **Cochran's Q Test**: Multiple related binary variables

### 🔬 Advanced Multivariate Tests

- **MANOVA**: Compare multiple dependent variables simultaneously
- **ANCOVA**: Compare groups while controlling for covariates
- **Two-way ANOVA**: Factorial designs with interaction effects

### 📊 Post-Hoc Multiple Comparison Procedures

**Family-wise Error Control:**
- **Tukey's HSD**: Balanced designs, equal variances
- **Games-Howell**: Unequal variances and sample sizes
- **Bonferroni**: Conservative, any design
- **Holm-Šídák**: Sequential method, more powerful than Bonferroni
- **Sidak**: Similar to Bonferroni with independence assumption

**False Discovery Rate Control:**
- **Benjamini-Hochberg**: Exploratory analysis, multiple testing

**Non-parametric:**
- **Dunn's Test**: Follow-up to Kruskal-Wallis test

### ⚡ Power Analysis & Sample Size

- **t-test Power Analysis**: One-sample, two-sample, paired designs
- **ANOVA Power Analysis**: One-way and factorial designs
- **Correlation Power Analysis**: Pearson and Spearman correlations
- **Chi-square Power Analysis**: Independence and goodness-of-fit tests
- **Custom Effect Size Calculations**: Cohen's conventions and user-defined

## 🤖 Automatic Test Recommender - Quick Start Guide

The **Automatic Test Recommender** is the fastest way to analyze your data. Simply upload a file and let the intelligent system guide you through the entire process.

### 🚀 How It Works (5 Simple Steps)

1. **Upload Data**: Drop your CSV, TSV, or Excel file
2. **Select Columns**: Pick which columns you want to analyze
3. **Review Profile**: See automatic type detection and assumption tests
4. **Get Recommendation**: Receive intelligent test suggestion with rationale
5. **Execute & View Results**: One-click analysis with detailed interpretation

### 📊 Supported Test Types (11 Tests)

**Comparing Groups:**
- Independent t-test (2 groups, normal, equal variances)
- Welch's t-test (2 groups, normal, unequal variances)
- Mann-Whitney U test (2 groups, non-normal)
- One-way ANOVA (3+ groups, normal, equal variances)
- Kruskal-Wallis test (3+ groups, non-normal)

**Relationships:**
- Pearson correlation (2 numeric, both normal)
- Spearman correlation (2 numeric, non-normal)

**Categorical Analysis:**
- Chi-square test (2 categorical variables)
- Fisher's exact test (2x2 table, small samples)

**Paired Data:**
- Paired t-test (before/after, normal differences)
- Wilcoxon signed-rank test (before/after, non-normal)

### 📁 Demo Datasets (`demo_data/` directory)

**11 Ready-to-Use Example Datasets:**

1. **demo_ttest_independent.csv** - Independent t-test
   - Columns: `Wynik_Testu`, `Grupa`
   - Example: Test scores between control and therapy groups

2. **demo_ttest_welch.csv** - Welch's t-test
   - Columns: `Poziom_Cholesterolu`, `Dieta`
   - Example: Cholesterol levels across different diets

3. **demo_mannwhitney.csv** - Mann-Whitney U test
   - Columns: `Czas_Rekonwalescencji_Dni`, `Metoda_Leczenia`
   - Example: Recovery times for different treatments

4. **demo_anova.csv** - One-way ANOVA
   - Columns: `Wzrost_Roslin_cm`, `Rodzaj_Nawozu`
   - Example: Plant growth across fertilizer types

5. **demo_kruskal.csv** - Kruskal-Wallis test
   - Columns: `Ocena_Zadowolenia`, `Klinika`
   - Example: Satisfaction scores across clinics

6. **demo_pearson.csv** - Pearson correlation
   - Columns: `Wiek_Lat`, `Cisnienie_Skurczowe`
   - Example: Age vs. blood pressure relationship

7. **demo_spearman.csv** - Spearman correlation
   - Columns: `Dochod_Roczny`, `Wydatki_Luksusowe`
   - Example: Income vs. luxury spending

8. **demo_chisquare.csv** - Chi-square test
   - Columns: `Plec`, `Preferowany_Produkt`
   - Example: Gender and product preference association

9. **demo_fisher.csv** - Fisher's exact test
   - Columns: `Leczenie`, `Wynik`
   - Example: Treatment vs. outcome (small sample)

10. **demo_ttest_paired.csv** - Paired t-test
    - Columns: `Cisnienie_Przed`, `Cisnienie_Po`
    - Example: Blood pressure before/after treatment

11. **demo_wilcoxon.csv** - Wilcoxon signed-rank test
    - Columns: `Poziom_Bolu_Przed`, `Poziom_Bolu_Po`
    - Example: Pain levels before/after intervention

### 🎯 Workflow Example with Demo Data

```bash
# Step 1: Navigate to Automatic Test Recommender page
# Step 2: Upload demo_ttest_independent.csv
# Step 3: Select both columns: Wynik_Testu and Grupa
# Step 4: Review automatic analysis:
#   - Wynik_Testu: Numeric, Normal distribution ✓
#   - Grupa: Categorical, 2 groups
#   - Levene's test: Equal variances ✓
# Step 5: Recommendation: "Independent t-test"
#   Rationale: "Data are normal and variances are equal"
# Step 6: Click "Execute Analysis"
# Step 7: View results with hypotheses, p-value, and interpretation
```

### ⚙️ Technical Features

**Automatic Column Profiling:**
- Numeric vs. categorical detection
- Unique value counting
- Distribution analysis

**Statistical Assumption Testing:**
- **Shapiro-Wilk test**: Tests normality of numeric columns
- **Levene's test**: Tests homogeneity of variances across groups

**Intelligent Decision Tree:**
- Analyzes column types and counts
- Checks assumption test results
- Recommends optimal test with clear rationale
- Provides alternatives when assumptions are violated

**Session State Management:**
- Preserves data across interactions
- Clears state when new file is uploaded
- Resets analysis when columns change

## 🎓 Educational System

### 📚 Comprehensive Learning Resources

**Statistical Concepts Covered:**
- **Hypothesis Testing**: Null/alternative hypotheses, Type I/II errors
- **P-values & Significance**: Interpretation, multiple testing, alpha levels
- **Effect Sizes**: Cohen's d, eta-squared, Cramer's V, practical significance
- **Assumptions**: Normality, homogeneity, independence, and their violations
- **Test Selection**: Decision trees, parametric vs. non-parametric choices

**Interactive Learning Features:**
- **Step-by-step Tutorials**: Guided walkthroughs for each test type
- **Real Data Examples**: Biological and research-based scenarios
- **Visual Demonstrations**: Animated plots showing statistical concepts
- **Common Pitfalls**: Mistakes to avoid and best practices
- **Decision Support**: When to use which test and why

### 🔧 Technical Implementation

#### Application Architecture
```
statistical_app/
├── 📱 app.py                        # Main Streamlit application (8 pages)
├── 🤖 statistical_recommender.py    # NEW: Automatic test recommender
├── 🧪 test_recommender.py          # NEW: Test suite for recommender
├── 📊 generate_demo_data.py        # NEW: Demo dataset generator
├── 🚀 run_app.py                   # Easy startup script
├── ⚙️  requirements.txt              # Python dependencies
├── 📖 README.md                    # This documentation
├── 📊 ADVANCED_FEATURES.md         # Advanced features guide
├── 🧪 test_app.py                  # Comprehensive testing suite
├── 📂 modules/                     # Core application modules
│   ├── 🎯 test_selector.py          # Intelligent test selection
│   ├── 📊 statistical_tests.py      # 15 basic statistical tests
│   ├── 🔬 advanced_tests.py         # 6 advanced statistical tests
│   ├── 🔄 post_hoc_tests.py         # 7 post-hoc procedures
│   ├── 💾 data_processor.py         # Data upload and processing
│   ├── ✅ assumption_checker.py     # Assumption validation
│   ├── 📈 visualizer.py            # Interactive visualizations
│   ├── 🧠 interpreter.py           # Results interpretation
│   ├── ⚡ power_analysis.py        # Power analysis & sample size
│   └── 📝 report_generator.py      # Professional report generation
├── 📊 demo_data/                   # NEW: 11 demo datasets
│   ├── demo_ttest_independent.csv  # Independent t-test example
│   ├── demo_ttest_welch.csv       # Welch's t-test example
│   ├── demo_mannwhitney.csv       # Mann-Whitney U example
│   ├── demo_anova.csv             # One-way ANOVA example
│   ├── demo_kruskal.csv           # Kruskal-Wallis example
│   ├── demo_pearson.csv           # Pearson correlation example
│   ├── demo_spearman.csv          # Spearman correlation example
│   ├── demo_chisquare.csv         # Chi-square test example
│   ├── demo_fisher.csv            # Fisher's exact test example
│   ├── demo_ttest_paired.csv      # Paired t-test example
│   ├── demo_wilcoxon.csv          # Wilcoxon test example
│   └── README.md                  # Demo data documentation
├── 📊 data/                        # Built-in datasets
│   └── example_datasets/           # 5 sample datasets
└── 📚 docs/                        # Documentation
    └── additional resources
```

#### Core Technologies
- **Frontend**: Streamlit (Interactive web interface)
- **Data Processing**: pandas, numpy (Data manipulation)
- **Statistics**: scipy, statsmodels, pingouin (Statistical computations)
- **Visualization**: plotly, seaborn, matplotlib (Interactive plots)
- **Machine Learning**: scikit-learn (Advanced analyses)
- **Reports**: markdown (Professional documentation)

## 🛠️ Development & Contributing

### 🔧 Development Setup

```bash
# Clone repository
git clone <repository-url>
cd statistical_app

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_app.py

# Start development server
streamlit run app.py
```

### 📝 Adding New Statistical Tests

**Step-by-step Process:**

1. **Test Implementation** (`modules/statistical_tests.py` or `modules/advanced_tests.py`)
   ```python
   def new_test(data, **kwargs):
       # Implement test logic
       return {
           'test_statistic': value,
           'p_value': p_val,
           'effect_size': effect,
           'interpretation': text
       }
   ```

2. **Update Test Selector** (`modules/test_selector.py`)
   - Add decision logic for when to recommend the new test
   - Include in appropriate test categories

3. **Assumption Checking** (`modules/assumption_checker.py`)
   - Add specific assumption tests for the new method
   - Include visual diagnostic plots

4. **Visualization** (`modules/visualizer.py`)
   - Create appropriate plots for the test type
   - Add to visualization dispatch system

5. **Interpretation** (`modules/interpreter.py`)
   - Add intelligent result interpretation
   - Include effect size guidelines and practical significance

6. **Documentation** (Test Library)
   - Add comprehensive test description
   - Include assumptions, examples, and best practices

### 🎯 Contribution Guidelines

**Code Standards:**
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add type hints where appropriate
- Write unit tests for new functionality

**Statistical Standards:**
- Validate against established statistical packages (R, SPSS)
- Include worked examples from statistical literature
- Properly handle edge cases and missing data
- Provide appropriate effect size measures

**Documentation Requirements:**
- Clear explanations suitable for non-statisticians
- Biological/research examples when possible
- Assumption explanations with practical implications
- References to statistical literature

## 🔍 Testing & Quality Assurance

### Automated Testing
```bash
# Run comprehensive test suite
python test_app.py

# Test specific modules
python -m pytest modules/test_statistical_tests.py
```

### Manual Testing Checklist
- [ ] All statistical tests produce correct results
- [ ] Assumption checking works properly
- [ ] Visualizations render correctly
- [ ] Report generation functions properly
- [ ] Educational content is accurate

## 🚨 Troubleshooting Guide

### 🔧 Installation Issues

| Problem | Solution |
|---------|----------|
| Python version error | Ensure Python 3.8+ installed |
| Package conflicts | Use virtual environment |
| Permission errors | Use `pip install --user` |
| Memory errors | Increase system RAM or use smaller datasets |

### 📊 Data Upload Problems

| Problem | Solution |
|---------|----------|
| File not recognized | Ensure CSV/Excel format with proper encoding |
| Column issues | Use clear headers, avoid special characters |
| Missing data | Check data preprocessing recommendations |
| Large file slowness | Consider data sampling or chunking |

### 🔍 Analysis Issues

| Problem | Solution |
|---------|----------|
| Test assumptions failed | Review alternative non-parametric tests |
| Sample size warnings | Use power analysis for planning |
| Unexpected results | Verify data structure and variable types |
| Performance slowdown | Restart application, close other browser tabs |

### 📞 Getting Support

1. **Built-in Help**: Use Educational Content and Test Library
2. **Sample Data**: Practice with provided datasets
3. **Documentation**: Review ADVANCED_FEATURES.md
4. **Statistical Consultation**: Consult with domain experts for complex analyses

## 📚 References & Citation

### Academic Citation
```
Statistical Testing Streamlit Application (2024).
A comprehensive educational platform for statistical analysis.
GitHub: [repository-url]
```

### Key Statistical References
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Field, A. (2013). Discovering Statistics Using IBM SPSS Statistics
- Zar, J. H. (2010). Biostatistical Analysis

## 📄 License & Legal

**MIT License** - Free for academic, research, and commercial use.

See LICENSE file for complete terms and conditions.

## 🙏 Acknowledgments

**Development Team**: Statistical analysis experts and software developers
**Scientific Community**: Open-source statistical computing community
**Libraries**: scipy, statsmodels, plotly, streamlit, and all dependencies
**Users**: Researchers, students, and educators providing feedback

---

## ⚠️ Important Disclaimer

**This application is designed to assist with statistical analysis but does not replace professional statistical expertise.**

- Always verify results with domain experts
- Understand limitations of statistical methods
- Consider consulting statisticians for complex analyses
- Validate findings with independent approaches when possible

**For questions about appropriate statistical methods for your specific research, consult with qualified statisticians or methodologists.**