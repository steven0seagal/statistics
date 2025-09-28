# Statistical Testing Application - User Guide

## 📋 Table of Contents

1. [Getting Started](#-getting-started)
2. [Application Overview](#-application-overview)
3. [Step-by-Step Workflows](#-step-by-step-workflows)
4. [Sample Dataset Tutorials](#-sample-dataset-tutorials)
5. [Advanced Features](#-advanced-features)
6. [Best Practices](#-best-practices)
7. [Common Scenarios](#-common-scenarios)
8. [Tips and Tricks](#-tips-and-tricks)

## 🚀 Getting Started

### First Time Setup

1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```
   The application will open in your browser at `http://localhost:8501`

2. **Navigate the Home Page**
   - Read the feature overview
   - Understand the 7 main sections
   - Review the quick start guide

3. **Try Sample Data First**
   - Start with built-in datasets before using your own data
   - This helps you understand the workflow and features

### Understanding the Interface

**7 Main Pages:**
- **🏠 Home**: Welcome and overview
- **🧭 Test Selection Wizard**: Intelligent test recommendations
- **📊 Data Upload & Analysis**: Main analysis workspace
- **🔬 Advanced Analysis**: Multivariate and complex tests
- **⚡ Power Analysis**: Sample size and power calculations
- **🎓 Educational Content**: Learning resources
- **📚 Test Library**: Comprehensive test reference

## 🔍 Application Overview

### Core Philosophy

This application follows a **guided, educational approach** to statistical analysis:

1. **Understand before doing**: Learn about tests before running them
2. **Check assumptions**: Always validate test requirements
3. **Interpret meaningfully**: Focus on practical significance, not just p-values
4. **Document thoroughly**: Generate professional reports

### Key Features Summary

| Feature | Purpose | Location |
|---------|---------|----------|
| Test Selection Wizard | Get test recommendations | Page 2 |
| Assumption Checking | Validate test requirements | All analysis pages |
| Effect Size Calculation | Measure practical significance | All test results |
| Interactive Visualizations | Explore data patterns | All analysis pages |
| Post-hoc Testing | Multiple comparisons | Automatic with ANOVA |
| Power Analysis | Plan sample sizes | Page 5 |
| Report Generation | Document findings | All analysis pages |

## 🔄 Step-by-Step Workflows

### Workflow 1: Basic Comparison (Two Groups)

**Scenario**: Compare treatment vs. control group outcomes

1. **Plan Your Analysis**
   - Research question: "Is treatment A better than control?"
   - Variables: treatment_group (categorical), outcome_score (continuous)
   - Design: Independent samples

2. **Get Test Recommendation**
   - Go to **Test Selection Wizard**
   - Select "Compare groups"
   - Choose "Two groups"
   - Select "Independent samples"
   - Choose "Continuous outcome"
   - **Result**: Independent t-test or Mann-Whitney U

3. **Upload Your Data**
   - Go to **Data Upload & Analysis**
   - Upload CSV/Excel file OR use "Clinical Trial Data" sample
   - Verify column types are correctly detected
   - Review summary statistics

4. **Check Assumptions**
   - Review normality tests (Shapiro-Wilk)
   - Check homogeneity of variance (Levene's test)
   - Examine Q-Q plots and histograms
   - **Decision**: Use t-test if assumptions met, Mann-Whitney U if not

5. **Run Analysis**
   - Select appropriate test
   - Choose your grouping variable
   - Choose your outcome variable
   - Click "Run Analysis"

6. **Interpret Results**
   - Read the automated interpretation
   - Note effect size (Cohen's d)
   - Check confidence intervals
   - Consider practical significance

7. **Generate Report**
   - Choose report format (Basic/Comprehensive/Publication)
   - Download in preferred format (HTML/Markdown/Text)

### Workflow 2: Multiple Group Comparison

**Scenario**: Compare effectiveness of 4 different treatments

1. **Plan Your Analysis**
   - Research question: "Which treatment is most effective?"
   - Variables: treatment (4 levels), effectiveness_score (continuous)
   - Design: Independent groups

2. **Get Test Recommendation**
   - **Test Selection Wizard** → "Compare groups" → "Three or more groups"
   - **Result**: One-way ANOVA or Kruskal-Wallis

3. **Upload and Validate Data**
   - Use "Gene Expression Analysis" sample or your data
   - Ensure treatment variable has 4 levels
   - Check for balanced/unbalanced design

4. **Check Assumptions**
   - Normality: Each group should be approximately normal
   - Homogeneity: Equal variances across groups
   - Independence: Observations should be independent

5. **Run One-way ANOVA**
   - Select test and variables
   - Review omnibus F-test results
   - **If significant**: Automatic post-hoc testing

6. **Review Post-hoc Results**
   - Application automatically suggests appropriate method
   - Options: Tukey's HSD, Games-Howell, Bonferroni
   - Identify which groups differ significantly

7. **Visualize and Report**
   - Box plots showing group differences
   - Post-hoc comparison plots
   - Comprehensive report with all comparisons

### Workflow 3: Relationship Analysis

**Scenario**: Examine relationship between age and reaction time

1. **Plan Your Analysis**
   - Research question: "Does reaction time increase with age?"
   - Variables: age (continuous), reaction_time (continuous)
   - Design: Correlational

2. **Get Test Recommendation**
   - **Test Selection Wizard** → "Examine relationships"
   - **Result**: Pearson or Spearman correlation

3. **Upload Data**
   - Use "Ecological Survey" sample or your data
   - Check for outliers that might affect correlation

4. **Check Assumptions**
   - Linearity: Examine scatter plot
   - Normality: Both variables should be normal for Pearson
   - Homoscedasticity: Equal variance across range

5. **Run Correlation Analysis**
   - Choose Pearson (parametric) or Spearman (non-parametric)
   - Examine scatter plot with regression line
   - Note correlation coefficient and significance

6. **Consider Regression** (Optional)
   - If strong correlation, consider predictive modeling
   - Go to **Advanced Analysis** for regression options
   - Examine R-squared and prediction intervals

## 📊 Sample Dataset Tutorials

### Tutorial 1: Biological Growth Study

**Learning Goals**: Basic t-test, assumption checking, effect sizes

1. **Load Sample Data**
   - Go to **Data Upload & Analysis**
   - Select "Biological Growth Study"
   - Examine data structure: treatment, growth_rate, time_point

2. **Explore the Data**
   - Review summary statistics
   - Note sample sizes for each treatment
   - Check for outliers in growth_rate

3. **Select Test**
   - Independent samples t-test (treatment vs. control)
   - Dependent variable: growth_rate
   - Grouping variable: treatment

4. **Check Assumptions**
   - Normality: Both groups approximately normal?
   - Variance: Equal variances between groups?
   - Independence: Each observation independent?

5. **Interpret Results**
   - P-value: Statistical significance
   - Cohen's d: Effect size magnitude
   - Confidence interval: Range of plausible differences
   - Practical significance: Is the difference meaningful?

6. **Key Learning Points**
   - When to use t-test vs. Mann-Whitney U
   - Importance of effect size beyond p-values
   - How to interpret confidence intervals

### Tutorial 2: Gene Expression Analysis

**Learning Goals**: ANOVA, post-hoc testing, multiple comparisons

1. **Load and Explore Data**
   - Multiple conditions (control, treatment1, treatment2, treatment3)
   - Multiple replicates per condition
   - Continuous outcome: expression_level

2. **Run One-way ANOVA**
   - Check assumptions carefully
   - Interpret omnibus F-test
   - Understand what "significant ANOVA" means

3. **Post-hoc Analysis**
   - Automatic post-hoc testing triggered
   - Compare different correction methods
   - Understand family-wise error rate

4. **Advanced Visualization**
   - Box plots with individual points
   - Post-hoc comparison visualization
   - Effect size plots

5. **Key Learning Points**
   - Multiple comparison problem
   - When to use different post-hoc methods
   - Interpreting complex ANOVA results

### Tutorial 3: Clinical Trial Data

**Learning Goals**: ANCOVA, controlling for covariates

1. **Understanding ANCOVA**
   - Compare treatment groups while controlling for baseline
   - More powerful than simple group comparison
   - Adjusts for pre-existing differences

2. **Go to Advanced Analysis**
   - Select ANCOVA
   - Dependent: outcome_score
   - Factor: treatment_group
   - Covariate: baseline_score

3. **Interpret ANCOVA Results**
   - Main effect of treatment (adjusted)
   - Effect of covariate
   - Adjusted group means
   - Partial eta-squared effect size

4. **Key Learning Points**
   - When to use ANCOVA vs. ANOVA
   - Interpreting adjusted means
   - Understanding partial effect sizes

## 🔬 Advanced Features

### Power Analysis Deep Dive

**Purpose**: Plan studies and interpret existing results

1. **Sample Size Planning**
   - Go to **Power Analysis** page
   - Choose your test type
   - Set desired power (usually 80%)
   - Set expected effect size
   - Get sample size recommendation

2. **Power Assessment**
   - For completed studies
   - Enter actual sample size and observed effect
   - Determine if study was adequately powered
   - Understand Type II error risk

3. **Interactive Power Curves**
   - Visualize power across different scenarios
   - Understand trade-offs between sample size and power
   - Plan for different effect sizes

### Advanced Multivariate Analysis

**When to Use Advanced Tests:**

- **MANOVA**: Multiple dependent variables
- **ANCOVA**: Control for covariates
- **Two-way ANOVA**: Factorial designs
- **Logistic Regression**: Binary outcomes
- **Multiple Regression**: Multiple predictors

**Workflow for Advanced Tests:**

1. **Go to Advanced Analysis Page**
2. **Select Appropriate Test**
3. **Configure Variables Carefully**
4. **Interpret Complex Output**
5. **Use Specialized Visualizations**

### Professional Report Generation

**Three Report Types:**

1. **Basic Report**
   - Essential results only
   - Suitable for quick summaries
   - Minimal technical detail

2. **Comprehensive Report**
   - Complete statistical documentation
   - Assumption checking results
   - Methodology section
   - Detailed interpretation

3. **Publication Report**
   - APA-style formatting
   - Concise, professional presentation
   - Ready for academic papers

**Export Options:**
- **HTML**: Styled, interactive format
- **Markdown**: Plain text with formatting
- **Text**: Universal compatibility

## ✅ Best Practices

### Data Preparation

1. **File Format**
   - Use CSV or Excel files
   - Clear column headers in first row
   - One row per observation
   - One column per variable

2. **Data Quality**
   - Remove completely empty rows/columns
   - Handle missing data appropriately
   - Check for data entry errors
   - Ensure consistent formatting

3. **Variable Naming**
   - Use descriptive names
   - Avoid special characters
   - Use underscores instead of spaces
   - Be consistent across variables

### Statistical Best Practices

1. **Plan Before Analyzing**
   - Define research questions clearly
   - Choose appropriate study design
   - Plan sample size using power analysis
   - Decide on analysis strategy beforehand

2. **Check Assumptions Always**
   - Use automated assumption checking
   - Examine diagnostic plots carefully
   - Consider alternatives when assumptions fail
   - Don't ignore assumption violations

3. **Interpret Meaningfully**
   - Focus on effect sizes, not just p-values
   - Consider practical significance
   - Use confidence intervals
   - Discuss limitations and alternative explanations

4. **Handle Multiple Comparisons**
   - Use appropriate correction methods
   - Understand family-wise error rate
   - Consider false discovery rate when appropriate
   - Plan comparisons, don't fish for significance

### Reporting Best Practices

1. **Complete Documentation**
   - Report all analyses conducted
   - Include assumption checking results
   - Provide complete test statistics
   - Discuss effect sizes and confidence intervals

2. **Transparent Methods**
   - Describe data collection procedures
   - Explain analysis decisions
   - Report any deviations from planned analysis
   - Provide data and code when possible

## 🎯 Common Scenarios

### Scenario 1: Small Sample Sizes

**Challenge**: Limited data, low power

**Solutions**:
- Use exact tests when available (Fisher's exact)
- Consider non-parametric alternatives
- Report confidence intervals and effect sizes
- Discuss power limitations
- Use power analysis for future planning

**Workflow**:
1. Check if assumptions can be met with small n
2. Consider exact or non-parametric tests
3. Focus on effect sizes and confidence intervals
4. Use comprehensive power analysis

### Scenario 2: Non-Normal Data

**Challenge**: Assumption violations

**Solutions**:
- Try data transformations
- Use non-parametric alternatives
- Use robust statistical methods
- Check for outliers

**Workflow**:
1. Examine distribution plots carefully
2. Try log, square root, or inverse transformations
3. If transformation doesn't help, use non-parametric tests
4. Consider removing outliers (with justification)

### Scenario 3: Unequal Sample Sizes

**Challenge**: Imbalanced groups

**Solutions**:
- Use Welch's t-test instead of standard t-test
- Use Games-Howell for post-hoc testing
- Check homogeneity of variance carefully
- Consider reasons for imbalance

**Workflow**:
1. Application automatically detects unequal samples
2. Uses appropriate tests (Welch's t-test)
3. Recommends appropriate post-hoc methods
4. Provides warnings about power implications

### Scenario 4: Multiple Testing

**Challenge**: Family-wise error inflation

**Solutions**:
- Plan comparisons in advance
- Use appropriate correction methods
- Consider false discovery rate approaches
- Report both corrected and uncorrected results

**Workflow**:
1. Application automatically detects multiple testing
2. Provides correction method recommendations
3. Offers multiple correction options
4. Reports both corrected and uncorrected results

## 💡 Tips and Tricks

### Efficiency Tips

1. **Start with Sample Data**
   - Learn the interface without your data
   - Practice interpreting results
   - Understand feature locations

2. **Use Test Selection Wizard**
   - Even if you think you know the right test
   - May suggest better alternatives
   - Provides rationale for recommendations

3. **Bookmark Key Pages**
   - Save frequently used analysis pages
   - Educational content for quick reference
   - Test library for assumption checking

### Interpretation Tips

1. **Read the Automated Interpretation**
   - Comprehensive explanation of results
   - Practical significance assessment
   - Recommendations for follow-up

2. **Focus on Effect Sizes**
   - More informative than p-values alone
   - Consider practical importance
   - Use confidence intervals

3. **Use Visualizations**
   - Pictures often tell the story better
   - Check assumptions visually
   - Communicate results effectively

### Advanced Usage Tips

1. **Combine Multiple Analyses**
   - Start with exploratory analysis
   - Move to confirmatory testing
   - Use power analysis for planning

2. **Export Everything**
   - Save plots as images
   - Download reports in multiple formats
   - Keep analysis documentation

3. **Learn Progressively**
   - Start with basic tests
   - Gradually try advanced features
   - Use educational content extensively

### Troubleshooting Tips

1. **Data Issues**
   - Check file encoding (UTF-8 recommended)
   - Verify column headers are in first row
   - Ensure no merged cells in Excel files

2. **Performance Issues**
   - Close other browser tabs
   - Restart application if sluggish
   - Consider data sampling for very large datasets

3. **Analysis Problems**
   - Review assumption checking carefully
   - Try alternative tests when assumptions fail
   - Consult educational content for guidance

---

## 📞 Getting Additional Help

1. **Built-in Resources**
   - Educational Content page
   - Test Library reference
   - Automated interpretations

2. **Practice Datasets**
   - 5 comprehensive sample datasets
   - Varied difficulty levels
   - Real-world scenarios

3. **Documentation**
   - README.md for overview
   - ADVANCED_FEATURES.md for complex analyses
   - This USER_GUIDE.md for detailed workflows

4. **Statistical Consultation**
   - For complex research questions
   - When in doubt about test selection
   - For interpretation of unusual results

**Remember**: This tool assists with statistical analysis but doesn't replace statistical expertise. When in doubt, consult with qualified statisticians!