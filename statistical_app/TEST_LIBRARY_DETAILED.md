# Detailed Statistical Test Library

This comprehensive guide provides detailed documentation for each of the 11 core statistical tests supported by the Automatic Test Recommender.

---

## I. Tests for Comparing Groups

These tests are used to determine if there are statistically significant differences in numerical values between two or more categorical groups.

### 1. Independent T-test (Independent Samples t-test)

* **Goal:** 🎯 Compares the **means** of two **independent** groups to determine if the difference between them is statistically significant.
* **Application Scenario:** 🧑‍🤝‍🧑 User selected **1 numerical column** (e.g., 'Score') and **1 categorical column** (e.g., 'Group') that has exactly **2 unique values** (e.g., 'A' and 'B').
* **Key Assumptions (to check):**
  1. **Normality:** Data in the numerical column have a normal distribution (checked with **Shapiro-Wilk Test**, p > 0.05).
  2. **Homogeneity of Variance:** Variances (variability) in both groups are equal (checked with **Levene's Test**, p > 0.05).
* **Hypotheses (for report):**
  * **H0 (Null Hypothesis):** The mean value in group A equals the mean value in group B.
  * **H1 (Alternative Hypothesis):** The mean value in group A differs from the mean value in group B.
* **Implementation (Python):** 🐍
  ```python
  # data1, data2 = group data from Pandas
  from scipy.stats import ttest_ind
  stat, p_value = ttest_ind(data1, data2, equal_var=True)
  ```

### 2. Welch's T-test

* **Goal:** 🎯 Compares **means** of two **independent** groups when their **variances are different**. This is a modification of the standard t-test.
* **Application Scenario:** 🧑‍🤝‍🧑 User selected **1 numerical column** and **1 categorical column** (with 2 groups).
* **Key Assumptions (to check):**
  1. **Normality:** Data in the numerical column have a normal distribution (Shapiro-Wilk Test, p > 0.05).
  2. **Lack of Homogeneity of Variance:** Variances in both groups are **different** (Levene's Test, p <= 0.05).
* **Hypotheses (for report):**
  * **H0:** The mean value in group A equals the mean value in group B.
  * **H1:** The mean value in group A differs from the mean value in group B.
* **Implementation (Python):** 🐍
  ```python
  from scipy.stats import ttest_ind
  stat, p_value = ttest_ind(data1, data2, equal_var=False)  # Key change
  ```

### 3. Mann-Whitney U Test

* **Goal:** 🎯 **Non-parametric** equivalent of the t-test. Compares **medians** (or generally distributions) of two **independent** groups.
* **Application Scenario:** 🧑‍🤝‍🧑 User selected **1 numerical column** and **1 categorical column** (with 2 groups).
* **Key Assumptions (to check):**
  1. **Lack of Normality:** Data in the numerical column **do not have** a normal distribution (Shapiro-Wilk Test, p <= 0.05).
* **Hypotheses (for report):**
  * **H0:** The distributions of values in both groups are the same (or: the median in group A equals the median in group B).
  * **H1:** The distributions of values in both groups differ (or: the median in group A differs from the median in group B).
* **Implementation (Python):** 🐍
  ```python
  from scipy.stats import mannwhitneyu
  stat, p_value = mannwhitneyu(data1, data2)
  ```

### 4. One-Way ANOVA (One-Way Analysis of Variance)

* **Goal:** 🎯 Compares **means** in **three or more** independent groups to check if at least one group differs from the others.
* **Application Scenario:** 👨‍👩‍👧‍👦 User selected **1 numerical column** and **1 categorical column** that has **3 or more** unique values (e.g., 'Group A', 'Group B', 'Group C').
* **Key Assumptions (to check):**
  1. **Normality:** Data in the numerical column have a normal distribution (Shapiro-Wilk Test, p > 0.05).
  2. **Homogeneity of Variance:** Variances in all groups are equal (Levene's Test, p > 0.05).
* **Hypotheses (for report):**
  * **H0:** Mean values in all groups are equal.
  * **H1:** At least one group has a mean value different from the others.
* **Implementation (Python):** 🐍
  ```python
  # groups_data = [data_A, data_B, data_C, ...]
  from scipy.stats import f_oneway
  stat, p_value = f_oneway(*groups_data)
  ```

### 5. Kruskal-Wallis H Test

* **Goal:** 🎯 **Non-parametric** equivalent of ANOVA. Compares **medians** (or distributions) in **three or more** independent groups.
* **Application Scenario:** 👨‍👩‍👧‍👦 User selected **1 numerical column** and **1 categorical column** (with 3+ groups).
* **Key Assumptions (to check):**
  1. **Lack of Normality:** Data in the numerical column **do not have** a normal distribution (Shapiro-Wilk Test, p <= 0.05) OR
  2. **Lack of Homogeneity of Variance:** Variances in groups are **different** (Levene's Test, p <= 0.05).
* **Hypotheses (for report):**
  * **H0:** Distributions of values in all groups are the same (or: medians in all groups are equal).
  * **H1:** At least one group has a distribution of values different from the others (or: at least one median is different).
* **Implementation (Python):** 🐍
  ```python
  # groups_data = [data_A, data_B, data_C, ...]
  from scipy.stats import kruskal
  stat, p_value = kruskal(*groups_data)
  ```

---

## II. Dependency Tests (Correlation and Association)

These tests are used to check if two variables are related (whether a change in one is associated with a change in the other).

### 6. Pearson Correlation Coefficient (r)

* **Goal:** 📈 Measures the strength and direction of **linear** relationship between **two numerical variables**.
* **Application Scenario:** 🔢 User selected **2 numerical columns** (e.g., 'Height' and 'Weight').
* **Key Assumptions (to check):**
  1. **Normality:** Both numerical variables have a normal distribution (Shapiro-Wilk Test, p > 0.05 for both).
  2. **Linearity:** The relationship between variables is (visually) linear (difficult to automate, but normality is a strong indicator).
* **Hypotheses (for report):**
  * **H0:** There is no linear correlation between variable X and variable Y (correlation coefficient = 0).
  * **H1:** There is a linear correlation between variable X and variable Y (correlation coefficient ≠ 0).
* **Implementation (Python):** 🐍
  ```python
  # x, y = two data columns from Pandas
  from scipy.stats import pearsonr
  correlation_coefficient, p_value = pearsonr(x, y)
  ```

### 7. Spearman Rank Correlation Coefficient (rho)

* **Goal:** 📉 Measures the strength and direction of **monotonic** relationship between two variables (can be numerical or ordinal).
* **Application Scenario:** 🔢 User selected **2 numerical columns**.
* **Key Assumptions (to check):**
  1. **Lack of Normality:** At least one variable **does not have** a normal distribution (Shapiro-Wilk Test, p <= 0.05).
  2. **Non-linear Relationship:** Used when the relationship is not linear but is monotonic (one variable increases as the other increases/decreases, but not necessarily at a constant rate).
* **Hypotheses (for report):**
  * **H0:** There is no monotonic correlation between variable X and variable Y.
  * **H1:** There is a monotonic correlation between variable X and variable Y.
* **Implementation (Python):** 🐍
  ```python
  from scipy.stats import spearmanr
  correlation_coefficient, p_value = spearmanr(x, y)
  ```

### 8. Chi-Square (χ²) Test of Independence

* **Goal:** 📊 Checks if there is an **association (relationship)** between **two categorical variables**.
* **Application Scenario:** 🔠 User selected **2 categorical columns** (e.g., 'Gender' and 'Education').
* **Key Assumptions (to check):**
  1. **Sample Size:** Expected values in each cell of the contingency table (cross-tabulation) are sufficiently large (typically > 5).
* **Hypotheses (for report):**
  * **H0:** Variable A and variable B are **independent** (there is no relationship between them).
  * **H1:** Variable A and variable B are **dependent** (there is a relationship between them).
* **Implementation (Python):** 🐍
  ```python
  import pandas as pd
  from scipy.stats import chi2_contingency
  contingency_table = pd.crosstab(df['cat_var_1'], df['cat_var_2'])
  chi2_stat, p_value, dof, expected_freqs = chi2_contingency(contingency_table)
  ```

### 9. Fisher's Exact Test

* **Goal:** 📊 Alternative to the Chi-square test, used when the sample size assumption is not met. Especially for small samples and 2x2 tables.
* **Application Scenario:** 🔠 User selected **2 categorical columns** (especially 2x2, e.g., 'Smoker: Yes/No' vs 'Disease: Yes/No').
* **Key Assumptions (to check):**
  1. **Small Sample:** Used when the Chi-square test is inappropriate because **expected values in cells are too small** (e.g., < 5).
* **Hypotheses (for report):**
  * **H0:** Variable A and variable B are **independent**.
  * **H1:** Variable A and variable B are **dependent**.
* **Implementation (Python):** 🐍
  ```python
  from scipy.stats import fisher_exact
  # Works only for 2x2 tables
  odds_ratio, p_value = fisher_exact(table_2x2)
  ```

---

## III. Tests for Paired (Dependent) Data

These tests are used to compare two measurements made on the same group (e.g., "before" and "after").

### 10. Paired T-test (Paired Samples t-test)

* **Goal:** 🔬 Compares **means** of two **related** measurements (e.g., the same patient's result before treatment and after treatment).
* **Application Scenario:** 🔁 User selected **2 numerical columns** AND additionally checked the option **"Data are paired"**.
* **Key Assumptions (to check):**
  1. **Normality of Differences:** The **differences** between pairs of measurements (e.g., `after - before`) have a normal distribution (Shapiro-Wilk Test on the differences column, p > 0.05).
* **Hypotheses (for report):**
  * **H0:** The mean difference between measurement 1 and measurement 2 is zero (no change).
  * **H1:** The mean difference between measurements is different from zero (a change occurred).
* **Implementation (Python):** 🐍
  ```python
  # data_before, data_after = two paired columns
  from scipy.stats import ttest_rel
  stat, p_value = ttest_rel(data_before, data_after)
  ```

### 11. Wilcoxon Signed-Rank Test

* **Goal:** 🔬 **Non-parametric** equivalent of the paired t-test. Compares **medians** of two **related** measurements.
* **Application Scenario:** 🔁 User selected **2 numerical columns** AND checked the option **"Data are paired"**.
* **Key Assumptions (to check):**
  1. **Lack of Normality of Differences:** The **differences** between pairs of measurements **do not have** a normal distribution (Shapiro-Wilk Test on differences, p <= 0.05).
* **Hypotheses (for report):**
  * **H0:** The median of differences between measurement 1 and measurement 2 is zero (no change).
  * **H1:** The median of differences between measurements is different from zero (a change occurred).
* **Implementation (Python):** 🐍
  ```python
  from scipy.stats import wilcoxon
  stat, p_value = wilcoxon(data_before, data_after)
  ```

---

## Enhanced Workflow with Visualization and Descriptive Statistics

### Updated Implementation Plan: Including Visualization and Descriptive Statistics

The application automatically generates **descriptive statistics** and **visualizations** immediately after column selection, before the user decides to perform the actual statistical test.

#### Modified Application Flow:

1. Data upload and loading
2. Column selection
3. **(NEW): Display descriptive statistics and automatically generated plots**
4. Check test assumptions and provide recommendation
5. Execute statistical test

---

### Step 5 (Modified): UI for Column Selection and Displaying Descriptive Data

**Goal:** Enable column selection and immediately generate descriptive statistics and plots.

| Action | Description | Implementation Elements |
| :--- | :--- | :--- |
| **A. Column Selection (UI)** | After data upload, display `st.multiselect` for columns. This selection triggers descriptive analysis. | `st.multiselect("Select columns...", key='selected_columns', on_change=clear_analysis_results)` |
| **B. Trigger Analysis** | When `len(selected_columns) > 0`, call a new function that calculates descriptive statistics and generates plots. | `if len(st.session_state.get('selected_columns', [])) > 0: display_exploratory_data(df, selected_columns)` |
| **C. Descriptive Function (Backend)** | Create function `def get_descriptive_stats(df, col):` that returns a dictionary or DataFrame with key measures. | Use **Pandas**: `df[col].describe()`. Additionally calculate mode (`df[col].mode()`), interquartile range (IQR), and variance (`df[col].var()`). |
| **D. Save and Display** | Save descriptive statistics in `st.session_state` (e.g., `'descriptive_stats'`). Display them in a readable table for each selected column. | `st.subheader("Descriptive Statistics")`; `st.dataframe(...)` |

---

### Step 6 (New): Automatic Plot Generation

**Goal:** Automatically generate **the most informative plots** (distribution plots) depending on the selected column type, so the user can visually assess the distribution (key assumption for parametric tests).

| Data Type | Suggested Plot | Purpose | Implementation (Libraries) |
| :--- | :--- | :--- | :--- |
| **Numerical** (single column) | **Histogram** (with density/KDE function) | Visualize data distribution and **normality** (check if it's bell-shaped). | `plotly.express` (`px.histogram`) OR `seaborn` (`sns.histplot`) |
| **Numerical** (for each group) | **Box Plot** | Compare medians, quartiles, range, and **deviations and symmetry** between categorical groups. | `plotly.express` (`px.box`) OR `seaborn` (`sns.boxplot`) |
| **Categorical** (single column) | **Bar Chart** | Show frequency of occurrence for each category. | `plotly.express` (`px.bar`) |
| **2 Numerical** (for correlation) | **Scatter Plot** | Visualize **linearity** or **monotonicity** of the relationship. | `plotly.express` (`px.scatter`) |

**Detailed Description (Plot Generation Logic):**

1. Create function `def generate_plots(df, selected_cols):`.
2. **Logic:** Check how many columns are selected:
   * **One numerical column:** Generate **Histogram** for that column.
   * **One categorical and one numerical column:** Generate **Box Plot** (numerical variable on Y-axis, categorical on X-axis). Use `plotly.express` for interactivity.
   * **Two numerical columns:** Generate **Scatter Plot**.
   * **Two categorical columns:** Generate **Mosaic bar chart** (frequencies).
3. Display generated plots using `st.plotly_chart()` (requires `plotly` in `requirements.txt`).

---

### Step 7 (Modified): Checking Assumptions and Test Recommendation

**Goal:** Run assumption checking logic *after* displaying visualizations, ending with a button to execute the test.

| Action | Description | Implementation Elements |
| :--- | :--- | :--- |
| **A. Display Assumptions** | Display assumption checking summary (normality, Levene's) in `st.expander` so the user can review them. | Use `'assumptions'` dictionary saved in session state. |
| **B. Test Recommendation** | Display the recommended test name (`'recommendation'`). | `st.success(...)` |
| **C. Wait for Execution (UI)** | **Remove** automatic analysis execution. Introduce one clear button. | `st.button("Execute Statistical Analysis and Report", key='run_test_final', type="primary")` |
| **D. Execution Flag** | Button click sets a flag in session state to `True`, which triggers Step 8. | `if st.button(...): st.session_state['run_analysis_flag'] = True` |

---

### Step 8: Statistical Function Router and Report

**Goal:** This step remains unchanged but is triggered by the **prepared button** (Step 7C), not automatically.

1. Logic checks the `'run_analysis_flag'`.
2. Calls router `run_statistical_test(df, cols, rec_key)`.
3. Saves results to `'results'`.
4. Displays full report (hypotheses, p-value, interpretation).

---

### Summary of Changes

| Plan Section | Change | Key Challenge |
| :--- | :--- | :--- |
| **Step 5 (UI)** | Added logic to call descriptive statistics. | Implementing **Pandas** logic (`.describe()`, `.var()`, `.mode()`) and displaying them in `st.dataframe`. |
| **Step 6 (New)** | Introduced **Automatic Visualization** module (Histograms, Box Plots, Scatter Plots). | Proper use of `plotly.express` or `seaborn` and **dynamic** plot generation based on **type and number** of selected columns. |
| **Step 7 (UI/Flow)** | Removed automatic analysis execution in favor of **one button** to generate report after visual data assessment. | Ensuring all descriptive data and plots are visible *before* this button. |

---

## Integration with Application Features

This detailed test library integrates with:
- **Automatic Test Recommender** (`statistical_recommender.py`)
- **Test Selection Wizard** for guided test selection
- **Educational Content** for learning statistical concepts
- **Report Generation** for documenting analysis results

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Field, A. (2013). Discovering Statistics Using IBM SPSS Statistics
- Zar, J. H. (2010). Biostatistical Analysis
- Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality
- Levene, H. (1960). Robust tests for equality of variances

---

*This documentation provides comprehensive information about the 11 core statistical tests implemented in the Automatic Test Recommender. For additional tests and advanced features, see ADVANCED_FEATURES.md.*
