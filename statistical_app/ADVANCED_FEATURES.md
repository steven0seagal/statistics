# Advanced Features Documentation

This document describes the advanced statistical analysis features available in the Statistical Testing Application.

## 🔬 Advanced Statistical Tests

### MANOVA (Multivariate Analysis of Variance)
**Purpose:** Compare means of multiple dependent variables across groups simultaneously.

**When to Use:**
- Multiple related dependent variables
- Want to test overall group differences while controlling for correlations between variables
- More powerful than separate ANOVAs when variables are correlated

**Implementation:**
- Uses statsmodels MANOVA implementation
- Provides Wilks' Lambda test statistic
- Calculates multivariate effect size
- Shows group means for each dependent variable

**Example Application:**
Comparing the effect of different treatments on multiple health outcomes (blood pressure, cholesterol, heart rate) simultaneously.

### ANCOVA (Analysis of Covariance)
**Purpose:** Compare group means while controlling for a continuous covariate.

**When to Use:**
- Want to compare groups while accounting for a confounding variable
- Increase statistical power by reducing error variance
- Adjust for baseline differences between groups

**Implementation:**
- Uses statsmodels OLS with categorical and continuous predictors
- Tests both main effect and covariate significance
- Provides adjusted group means
- Calculates partial eta-squared effect size

**Example Application:**
Comparing treatment effects on test scores while controlling for baseline IQ scores.

### Logistic Regression
**Purpose:** Predict binary outcomes from one or more predictor variables.

**When to Use:**
- Binary dependent variable (success/failure, yes/no)
- Want to identify significant predictors
- Need probability estimates for outcomes

**Implementation:**
- Uses scikit-learn LogisticRegression
- Provides odds ratios for interpretation
- Includes classification metrics (precision, recall, F1-score)
- Calculates pseudo R-squared

**Example Application:**
Predicting disease occurrence based on age, gender, and biomarker levels.

### Multiple Linear Regression
**Purpose:** Predict continuous outcomes from multiple predictor variables.

**When to Use:**
- Continuous dependent variable
- Multiple potential predictors
- Want to identify most important predictors

**Implementation:**
- Uses statsmodels OLS
- Provides coefficient significance tests
- Includes model diagnostics (residual analysis)
- Tests for heteroscedasticity and normality

### Two-way ANOVA
**Purpose:** Analyze effects of two factors and their interaction.

**When to Use:**
- Two categorical independent variables
- Want to test main effects and interaction
- Factorial experimental design

**Implementation:**
- Uses statsmodels OLS with interaction terms
- Tests main effects and interaction separately
- Provides effect sizes for each component
- Includes group statistics for all combinations

## ⚡ Power Analysis & Sample Size Calculation

### Sample Size Calculation
**Purpose:** Determine required sample size for desired statistical power.

**Features:**
- Multiple test types (t-test, ANOVA, correlation, chi-square)
- Adjustable power levels (typically 80%)
- Variable effect sizes
- Different alpha levels

**Implementation:**
- Uses statistical power formulas and numerical optimization
- Provides recommendations for different effect sizes
- Includes safety margins for dropouts

### Power Calculation
**Purpose:** Calculate statistical power for given sample size and effect size.

**Features:**
- Visual power assessment with color coding
- Risk assessment (Type II error rates)
- Recommendations for improvement

**Implementation:**
- Uses non-central distributions for accurate calculations
- Handles different test designs (one-sample, two-sample, paired)

### Comprehensive Power Analysis
**Purpose:** Generate complete power analysis report with curves and recommendations.

**Features:**
- Power curves showing relationship between sample size and power
- Multiple effect size scenarios
- Publication-ready visualizations

## 📊 Post-Hoc Testing

### Available Methods

#### Tukey's HSD (Honestly Significant Difference)
- **Best for:** Equal sample sizes and homogeneous variances
- **Controls:** Family-wise error rate
- **Advantage:** Optimal balance of power and Type I error control

#### Bonferroni Correction
- **Best for:** Conservative approach, any design
- **Controls:** Family-wise error rate
- **Advantage:** Simple and widely accepted

#### Holm-Šídák Sequential Method
- **Best for:** More powerful than Bonferroni
- **Controls:** Family-wise error rate
- **Advantage:** Sequential testing with early stopping

#### Benjamini-Hochberg FDR
- **Best for:** Exploratory analysis
- **Controls:** False discovery rate
- **Advantage:** More powerful when many comparisons

#### Dunn's Test
- **Best for:** Follow-up to Kruskal-Wallis test
- **Controls:** Family-wise error rate
- **Advantage:** Non-parametric, rank-based

#### Games-Howell Test
- **Best for:** Unequal variances between groups
- **Controls:** Family-wise error rate
- **Advantage:** Robust to assumption violations

### Automatic Recommendations
The system automatically recommends appropriate post-hoc tests based on:
- Omnibus test type (ANOVA vs. Kruskal-Wallis)
- Assumption violations detected
- Variance equality assessment

## 📝 Report Generation

### Report Types

#### Basic Report
- Essential test statistics and interpretation
- Suitable for quick analysis summary
- Minimal technical details

#### Comprehensive Report
- Complete statistical analysis documentation
- Assumption checking results
- Detailed interpretation with recommendations
- Technical methodology section

#### Publication Report
- APA-style statistical reporting
- Concise results presentation
- Formatted for academic papers

### Export Options
- **HTML:** Styled web format with CSS
- **Markdown:** Plain text with formatting
- **Text:** Plain text for universal compatibility

### Report Components
1. **Executive Summary**
2. **Methodology and Test Selection**
3. **Assumption Checking Results**
4. **Statistical Results**
5. **Effect Size and Practical Significance**
6. **Interpretation and Recommendations**
7. **Limitations and Future Directions**

## 🔧 Implementation Details

### Data Processing Enhancements
- **Robust CSV/Excel loading** with encoding detection
- **Automatic data type suggestions** based on content analysis
- **Missing data pattern analysis**
- **Outlier detection** with multiple methods (IQR, z-score, modified z-score)

### Assumption Checking Improvements
- **Comprehensive normality testing** with sample size considerations
- **Variance homogeneity tests** (Levene's and Bartlett's)
- **Visual diagnostic plots** for assumption validation
- **Alternative test recommendations** when assumptions fail

### Visualization Enhancements
- **Test-specific plot types** optimized for each analysis
- **Interactive plots** with Plotly for exploration
- **Assumption diagnostic plots** (Q-Q plots, residual plots)
- **Publication-ready styling** with customizable themes

## 🎯 Best Practices Integration

### Effect Size Reporting
- **Standardized effect size measures** for each test type
- **Interpretation guidelines** (small, medium, large effects)
- **Practical significance** considerations alongside statistical significance

### Multiple Comparison Awareness
- **Automatic detection** of multiple comparison situations
- **Appropriate correction methods** based on research context
- **Family-wise error rate** vs. **False discovery rate** guidance

### Sample Size Consciousness
- **Power analysis integration** throughout the workflow
- **Sample size recommendations** based on detected effects
- **Post-hoc power calculations** for completed studies

### Reproducibility Features
- **Complete methodology documentation** in reports
- **Parameter logging** for all analyses
- **Version tracking** of statistical methods used

## 🚀 Usage Workflow

### For Advanced Users
1. **Upload Data** → Use robust data processing features
2. **Explore Data** → Utilize advanced visualization options
3. **Select Test** → Choose from expanded test library
4. **Check Assumptions** → Review comprehensive diagnostics
5. **Run Analysis** → Execute with automatic post-hoc testing
6. **Interpret Results** → Use enhanced interpretation features
7. **Generate Report** → Create publication-ready documentation

### For Researchers
1. **Plan Study** → Use power analysis for sample size determination
2. **Collect Data** → Follow design recommendations
3. **Analyze Data** → Apply appropriate advanced methods
4. **Validate Results** → Check assumptions and effect sizes
5. **Report Findings** → Use standardized reporting formats

## 📚 Educational Integration

### Interactive Learning
- **Guided tutorials** for each advanced method
- **Real-world examples** with biological data
- **Common mistakes** and how to avoid them
- **Decision trees** for method selection

### Reference Materials
- **Statistical background** for each method
- **Assumption explanations** with practical implications
- **Effect size interpretation** guidelines
- **Power analysis** concepts and applications

## ⚠️ Limitations and Considerations

### Statistical Limitations
- **Advanced methods require larger sample sizes** for reliable results
- **Multiple testing** increases complexity of interpretation
- **Model assumptions** become more critical with complex analyses
- **Effect size interpretation** may vary by field

### Technical Limitations
- **Computational intensity** of some advanced methods
- **Memory requirements** for large datasets
- **Dependency requirements** for specialized packages

### Recommendations
- **Always check assumptions** before interpreting results
- **Consider practical significance** alongside statistical significance
- **Use appropriate corrections** for multiple comparisons
- **Validate findings** with independent datasets when possible

---

*This documentation covers the advanced features implemented in version 2.0 of the Statistical Testing Application. For basic features, see the main README.md file.*