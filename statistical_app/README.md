# Statistical Testing Streamlit Application

A comprehensive tool for statistical analysis that provides intelligent test selection, educational content, data analysis capabilities, and results interpretation for biological research.

## Features

### 🧭 Test Selection Wizard
- Intelligent test recommendation based on data characteristics
- Decision tree approach following statistical best practices
- Considers research goals, variable types, study design, and sample size

### 📈 Data Analysis Pipeline
- Support for CSV and Excel file uploads
- Automatic data validation and preprocessing
- Built-in sample datasets for demonstration
- Outlier detection and handling

### 🔍 Assumption Checking
- Automated validation of statistical test assumptions
- Normality testing (Shapiro-Wilk)
- Homogeneity of variance testing (Levene's, Bartlett's)
- Visual diagnostic plots

### 📊 Comprehensive Statistical Tests
- **Two-group comparisons**: t-tests, Mann-Whitney U, Wilcoxon signed-rank
- **Multiple group comparisons**: ANOVA, Kruskal-Wallis, Friedman test
- **Categorical data**: Chi-squared, Fisher's exact, McNemar's test
- **Relationships**: Pearson/Spearman correlation, regression
- **Advanced tests**: MANOVA, ANCOVA (planned)

### 📈 Interactive Visualizations
- Appropriate plots for each test type
- Box plots, violin plots, scatter plots, histograms
- Q-Q plots for assumption checking
- Publication-ready figures

### 🎓 Educational Content
- Detailed explanations of statistical concepts
- Test assumptions and when to use each test
- Common pitfalls and best practices
- Biological examples and interpretations

### 🧠 Intelligent Results Interpretation
- Comprehensive explanation of test results
- Effect size calculations and interpretations
- Practical significance assessment
- Recommendations for follow-up analyses

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the repository:**
   ```bash
   cd /path/to/your/projects
   git clone <repository-url>  # or download and extract
   cd statistical_app
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your web browser:**
   The application will automatically open in your default browser, or navigate to `http://localhost:8501`

### Required Packages
- streamlit >= 1.28.0
- pandas >= 1.5.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- plotly >= 5.15.0
- seaborn >= 0.12.0
- matplotlib >= 3.7.0
- statsmodels >= 0.14.0
- pingouin >= 0.5.3
- scikit-learn >= 1.3.0
- openpyxl >= 3.1.0 (for Excel file support)

## Usage Guide

### Getting Started

1. **Home Page**: Overview of application capabilities and quick start guide
2. **Test Selection Wizard**: Answer guided questions to get test recommendations
3. **Data Upload & Analysis**: Upload your data or use sample datasets
4. **Educational Content**: Learn about statistical concepts and best practices
5. **Test Library**: Reference guide for all available tests

### Workflow Example

1. **Prepare your data**: Ensure data is in CSV or Excel format with clear column headers
2. **Use the Test Selection Wizard**: Answer questions about your research goals and data
3. **Upload your data**: Use the Data Upload & Analysis page
4. **Review recommendations**: Check suggested data types and test selection
5. **Check assumptions**: Review automated assumption checks
6. **Run analysis**: Execute the recommended statistical test
7. **Interpret results**: Review comprehensive results interpretation
8. **Export findings**: Save visualizations and results

### Sample Datasets

The application includes several built-in datasets for demonstration:

- **Biological Growth Study**: Compare growth rates between treatment groups
- **Gene Expression Analysis**: Differential expression across conditions
- **Clinical Trial Data**: Treatment effectiveness comparison
- **Ecological Survey**: Species abundance across habitats
- **Paired Measurements**: Before-after treatment effects

## Statistical Tests Supported

### Parametric Tests
- Independent samples t-test
- Welch's t-test (unequal variances)
- Paired samples t-test
- One-way ANOVA
- Repeated measures ANOVA

### Non-parametric Tests
- Mann-Whitney U test
- Wilcoxon signed-rank test
- Kruskal-Wallis test
- Friedman test

### Categorical Data Tests
- Chi-squared test of independence
- Fisher's exact test
- McNemar's test
- Cochran's Q test

### Correlation and Regression
- Pearson correlation
- Spearman rank correlation
- Linear regression
- Logistic regression (planned)

## Educational Features

### Key Concepts Covered
- Hypothesis testing fundamentals
- P-values and significance levels
- Effect sizes and practical significance
- Test assumptions and when they matter
- Choosing between parametric and non-parametric tests

### Learning Resources
- Interactive examples with sample data
- Visual demonstrations of statistical concepts
- Common mistakes and how to avoid them
- Best practices for biological research

## Application Structure

```
statistical_app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── modules/
│   ├── __init__.py
│   ├── test_selector.py      # Test selection algorithm
│   ├── statistical_tests.py  # Test implementations
│   ├── data_processor.py     # Data handling
│   ├── assumption_checker.py # Assumption validation
│   ├── visualizer.py        # Plot generation
│   └── interpreter.py       # Results interpretation
├── data/
│   └── example_datasets/     # Sample datasets
└── docs/
    └── test_descriptions.json # Test metadata
```

## Development

### Adding New Tests
1. Add test logic to `modules/statistical_tests.py`
2. Update test selection algorithm in `modules/test_selector.py`
3. Add assumption checks in `modules/assumption_checker.py`
4. Update visualization logic in `modules/visualizer.py`
5. Add interpretation logic in `modules/interpreter.py`

### Contributing
Contributions are welcome! Please ensure:
- Tests are well-documented with biological examples
- Assumptions are clearly stated and checked
- Visualizations are appropriate for the test type
- Results include effect sizes and practical interpretation

## Troubleshooting

### Common Issues

**Installation Problems:**
- Ensure Python 3.8+ is installed
- Use `pip install --upgrade pip` if package installation fails
- Try installing packages individually if bulk installation fails

**Data Upload Issues:**
- Ensure file is in CSV or Excel format
- Check that column headers are in the first row
- Remove any special characters from column names
- Ensure data doesn't have completely empty rows/columns

**Test Selection Problems:**
- Verify data types are correctly identified
- Check that dependent/independent variables are properly selected
- Ensure sample sizes meet minimum requirements for chosen test

**Performance Issues:**
- Large datasets (>10,000 rows) may be slow
- Close other browser tabs to free memory
- Restart the application if it becomes unresponsive

### Getting Help

1. Check the Educational Content section for statistical guidance
2. Review test assumptions in the Test Library
3. Use sample datasets to verify functionality
4. Consult statistical textbooks for advanced interpretation

## Citation

If you use this application in your research, please cite:

```
Statistical Testing Streamlit Application. (2024).
A comprehensive tool for statistical analysis in biological research.
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Acknowledgments

This application was developed based on comprehensive statistical guidelines and best practices for biological research. Special thanks to the open-source community for the excellent Python packages that make this application possible.

---

**Note**: This tool is designed to assist with statistical analysis but does not replace the need for statistical expertise. Always consult with a statistician for complex analyses or when in doubt about the appropriate method for your research question.