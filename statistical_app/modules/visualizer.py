"""
Visualization Module
===================

Creates appropriate statistical plots and visualizations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats

class Visualizer:
    """
    Creates appropriate visualizations for different statistical tests and data types.
    """

    def __init__(self):
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def create_plot(self, data, dependent_var, independent_var=None, test_name=None, **kwargs):
        """
        Create appropriate visualization based on test type and data characteristics.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        dependent_var : str
            Dependent variable column name
        independent_var : str, optional
            Independent variable column name
        test_name : str, optional
            Name of the statistical test
        **kwargs : additional parameters

        Returns:
        --------
        plotly.graph_objects.Figure : Interactive plot
        """

        if test_name is None:
            return self._create_exploratory_plot(data, dependent_var, independent_var)

        # Route to appropriate visualization method
        plot_methods = {
            'Independent t-test': self._create_two_group_plot,
            'Welch\'s t-test': self._create_two_group_plot,
            'Paired t-test': self._create_paired_plot,
            'Mann-Whitney U test': self._create_two_group_plot,
            'Wilcoxon signed-rank test': self._create_paired_plot,
            'One-way ANOVA': self._create_multiple_group_plot,
            'Kruskal-Wallis test': self._create_multiple_group_plot,
            'Repeated measures ANOVA': self._create_multiple_group_plot,
            'Friedman test': self._create_multiple_group_plot,
            'Chi-squared test': self._create_contingency_plot,
            'Fisher\'s exact test': self._create_contingency_plot,
            'Pearson correlation': self._create_correlation_plot,
            'Spearman correlation': self._create_correlation_plot
        }

        plot_method = plot_methods.get(test_name, self._create_exploratory_plot)
        return plot_method(data, dependent_var, independent_var, **kwargs)

    def _create_two_group_plot(self, data, dependent_var, independent_var, **kwargs):
        """Create visualization for two-group comparisons"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Box Plot', 'Violin Plot', 'Histogram', 'Q-Q Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        groups = data[independent_var].unique()
        colors = self.color_palette[:len(groups)]

        # Box plot
        for i, group in enumerate(groups):
            group_data = data[data[independent_var] == group][dependent_var]
            fig.add_trace(
                go.Box(y=group_data, name=str(group), marker_color=colors[i]),
                row=1, col=1
            )

        # Violin plot
        for i, group in enumerate(groups):
            group_data = data[data[independent_var] == group][dependent_var]
            fig.add_trace(
                go.Violin(y=group_data, name=str(group), marker_color=colors[i],
                         showlegend=False),
                row=1, col=2
            )

        # Histograms
        for i, group in enumerate(groups):
            group_data = data[data[independent_var] == group][dependent_var]
            fig.add_trace(
                go.Histogram(x=group_data, name=str(group), marker_color=colors[i],
                           opacity=0.7, showlegend=False),
                row=2, col=1
            )

        # Q-Q plot for normality check
        combined_data = data[dependent_var].dropna()
        qq_data = stats.probplot(combined_data, dist="norm")
        fig.add_trace(
            go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers',
                      name='Q-Q Plot', marker=dict(color='blue'), showlegend=False),
            row=2, col=2
        )
        # Add diagonal line
        line_min, line_max = min(qq_data[0][0]), max(qq_data[0][0])
        fig.add_trace(
            go.Scatter(x=[line_min, line_max], y=[line_min, line_max],
                      mode='lines', name='Normal Distribution',
                      line=dict(color='red', dash='dash'), showlegend=False),
            row=2, col=2
        )

        fig.update_layout(
            title=f'{dependent_var} by {independent_var}',
            height=600,
            showlegend=True
        )

        return fig

    def _create_paired_plot(self, data, dependent_var, independent_var, **kwargs):
        """Create visualization for paired comparisons"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Before-After Plot', 'Difference Distribution',
                           'Paired Box Plot', 'Change by Subject'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        groups = data[independent_var].unique()
        if len(groups) != 2:
            return self._create_multiple_group_plot(data, dependent_var, independent_var)

        # Assume data structure with repeated subject measurements
        group1_data = data[data[independent_var] == groups[0]][dependent_var]
        group2_data = data[data[independent_var] == groups[1]][dependent_var]

        if len(group1_data) == len(group2_data):
            # Before-after scatter plot
            fig.add_trace(
                go.Scatter(x=group1_data, y=group2_data, mode='markers',
                          name='Paired Data', marker=dict(size=8)),
                row=1, col=1
            )
            # Add diagonal line
            min_val, max_val = min(min(group1_data), min(group2_data)), max(max(group1_data), max(group2_data))
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          mode='lines', name='No Change',
                          line=dict(color='red', dash='dash')),
                row=1, col=1
            )

            # Difference distribution
            differences = group2_data.values - group1_data.values
            fig.add_trace(
                go.Histogram(x=differences, name='Differences',
                           marker_color='lightblue'),
                row=1, col=2
            )

            # Box plots for each condition
            fig.add_trace(
                go.Box(y=group1_data, name=str(groups[0]), marker_color=self.color_palette[0]),
                row=2, col=1
            )
            fig.add_trace(
                go.Box(y=group2_data, name=str(groups[1]), marker_color=self.color_palette[1]),
                row=2, col=1
            )

            # Individual change plot
            subject_ids = range(len(group1_data))
            for i in subject_ids:
                fig.add_trace(
                    go.Scatter(x=[groups[0], groups[1]],
                             y=[group1_data.iloc[i], group2_data.iloc[i]],
                             mode='lines+markers',
                             line=dict(color='gray', width=1),
                             marker=dict(size=4),
                             showlegend=False),
                    row=2, col=2
                )

        fig.update_layout(
            title=f'Paired Analysis: {dependent_var}',
            height=600
        )

        return fig

    def _create_multiple_group_plot(self, data, dependent_var, independent_var, **kwargs):
        """Create visualization for multiple group comparisons"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Box Plots', 'Mean ± SE', 'Distribution by Group', 'Summary Statistics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        groups = data[independent_var].unique()
        colors = self.color_palette[:len(groups)]

        # Box plots
        for i, group in enumerate(groups):
            group_data = data[data[independent_var] == group][dependent_var]
            fig.add_trace(
                go.Box(y=group_data, name=str(group), marker_color=colors[i]),
                row=1, col=1
            )

        # Mean ± Standard Error plot
        means = []
        sems = []
        group_names = []
        for group in groups:
            group_data = data[data[independent_var] == group][dependent_var]
            means.append(group_data.mean())
            sems.append(group_data.sem())
            group_names.append(str(group))

        fig.add_trace(
            go.Scatter(x=group_names, y=means,
                      error_y=dict(type='data', array=sems),
                      mode='markers+lines', name='Mean ± SE',
                      marker=dict(size=10, color='red')),
            row=1, col=2
        )

        # Overlaid histograms
        for i, group in enumerate(groups):
            group_data = data[data[independent_var] == group][dependent_var]
            fig.add_trace(
                go.Histogram(x=group_data, name=str(group),
                           marker_color=colors[i], opacity=0.7),
                row=2, col=1
            )

        # Summary statistics table
        summary_stats = []
        for group in groups:
            group_data = data[data[independent_var] == group][dependent_var]
            summary_stats.append([
                str(group),
                f"{len(group_data)}",
                f"{group_data.mean():.2f}",
                f"{group_data.std():.2f}",
                f"{group_data.median():.2f}"
            ])

        fig.add_trace(
            go.Table(
                header=dict(values=['Group', 'N', 'Mean', 'SD', 'Median'],
                           fill_color='lightblue'),
                cells=dict(values=list(zip(*summary_stats)),
                          fill_color='white')
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=f'{dependent_var} by {independent_var}',
            height=700,
            barmode='overlay'
        )

        return fig

    def _create_contingency_plot(self, data, dependent_var, independent_var, **kwargs):
        """Create visualization for categorical data analysis"""
        # Create contingency table
        contingency_table = pd.crosstab(data[independent_var], data[dependent_var])

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Stacked Bar Chart', 'Grouped Bar Chart',
                           'Contingency Table', 'Proportions'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Stacked bar chart
        for i, col in enumerate(contingency_table.columns):
            fig.add_trace(
                go.Bar(x=contingency_table.index, y=contingency_table[col],
                      name=str(col), marker_color=self.color_palette[i % len(self.color_palette)]),
                row=1, col=1
            )

        # Grouped bar chart
        for i, col in enumerate(contingency_table.columns):
            fig.add_trace(
                go.Bar(x=contingency_table.index, y=contingency_table[col],
                      name=str(col), marker_color=self.color_palette[i % len(self.color_palette)],
                      showlegend=False, offsetgroup=i),
                row=1, col=2
            )

        # Contingency table
        fig.add_trace(
            go.Table(
                header=dict(values=[''] + list(contingency_table.columns),
                           fill_color='lightblue'),
                cells=dict(values=[contingency_table.index] + [contingency_table[col] for col in contingency_table.columns],
                          fill_color='white')
            ),
            row=2, col=1
        )

        # Proportions
        proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)
        for i, col in enumerate(proportions.columns):
            fig.add_trace(
                go.Bar(x=proportions.index, y=proportions[col],
                      name=f'{col} (prop)', marker_color=self.color_palette[i % len(self.color_palette)],
                      showlegend=False),
                row=2, col=2
            )

        fig.update_layout(
            title=f'{dependent_var} by {independent_var}',
            height=600,
            barmode='stack'
        )

        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Proportion", row=2, col=2)

        return fig

    def _create_correlation_plot(self, data, dependent_var, independent_var, **kwargs):
        """Create visualization for correlation analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Scatter Plot', 'Residuals vs Fitted',
                           'Distribution of X', 'Distribution of Y'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Clean data
        clean_data = data[[dependent_var, independent_var]].dropna()
        x = clean_data[independent_var]
        y = clean_data[dependent_var]

        # Scatter plot with trend line
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='markers', name='Data',
                      marker=dict(size=6, opacity=0.7)),
            row=1, col=1
        )

        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(x=x, y=p(x), mode='lines', name='Trend Line',
                      line=dict(color='red')),
            row=1, col=1
        )

        # Calculate correlation
        correlation = np.corrcoef(x, y)[0, 1]

        # Residuals plot
        residuals = y - p(x)
        fitted = p(x)
        fig.add_trace(
            go.Scatter(x=fitted, y=residuals, mode='markers',
                      name='Residuals', marker=dict(size=6)),
            row=1, col=2
        )
        # Add horizontal line at 0
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

        # Distribution of X
        fig.add_trace(
            go.Histogram(x=x, name=independent_var, marker_color='lightblue'),
            row=2, col=1
        )

        # Distribution of Y
        fig.add_trace(
            go.Histogram(x=y, name=dependent_var, marker_color='lightgreen'),
            row=2, col=2
        )

        fig.update_layout(
            title=f'Correlation Analysis: {dependent_var} vs {independent_var}<br>r = {correlation:.3f}',
            height=600
        )

        fig.update_xaxes(title_text=independent_var, row=1, col=1)
        fig.update_yaxes(title_text=dependent_var, row=1, col=1)
        fig.update_xaxes(title_text="Fitted Values", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)

        return fig

    def _create_exploratory_plot(self, data, dependent_var, independent_var=None):
        """Create exploratory data visualization"""
        if independent_var is None:
            # Single variable exploration
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram', 'Box Plot', 'Q-Q Plot', 'Summary Statistics')
            )

            var_data = data[dependent_var].dropna()

            # Histogram
            fig.add_trace(
                go.Histogram(x=var_data, name=dependent_var, marker_color='lightblue'),
                row=1, col=1
            )

            # Box plot
            fig.add_trace(
                go.Box(y=var_data, name=dependent_var, marker_color='lightgreen'),
                row=1, col=2
            )

            # Q-Q plot
            qq_data = stats.probplot(var_data, dist="norm")
            fig.add_trace(
                go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers',
                          name='Q-Q Plot', marker=dict(color='blue')),
                row=2, col=1
            )
            line_min, line_max = min(qq_data[0][0]), max(qq_data[0][0])
            fig.add_trace(
                go.Scatter(x=[line_min, line_max], y=[line_min, line_max],
                          mode='lines', name='Normal Distribution',
                          line=dict(color='red', dash='dash')),
                row=2, col=1
            )

            # Summary statistics
            stats_data = [
                ['Count', f'{len(var_data)}'],
                ['Mean', f'{var_data.mean():.3f}'],
                ['Std Dev', f'{var_data.std():.3f}'],
                ['Min', f'{var_data.min():.3f}'],
                ['25%', f'{var_data.quantile(0.25):.3f}'],
                ['50%', f'{var_data.median():.3f}'],
                ['75%', f'{var_data.quantile(0.75):.3f}'],
                ['Max', f'{var_data.max():.3f}']
            ]

            fig.add_trace(
                go.Table(
                    header=dict(values=['Statistic', 'Value']),
                    cells=dict(values=list(zip(*stats_data)))
                ),
                row=2, col=2
            )

        else:
            # Two variable exploration - determine types and create appropriate plot
            if pd.api.types.is_numeric_dtype(data[dependent_var]) and pd.api.types.is_numeric_dtype(data[independent_var]):
                return self._create_correlation_plot(data, dependent_var, independent_var)
            elif pd.api.types.is_numeric_dtype(data[dependent_var]):
                return self._create_multiple_group_plot(data, dependent_var, independent_var)
            else:
                return self._create_contingency_plot(data, dependent_var, independent_var)

        fig.update_layout(
            title=f'Exploratory Analysis: {dependent_var}',
            height=600
        )

        return fig

    def create_assumption_plots(self, data, dependent_var, independent_var, test_name):
        """Create plots to check test assumptions"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Normality Check', 'Equal Variances',
                           'Outlier Detection', 'Sample Sizes'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        if independent_var:
            groups = data[independent_var].unique()
            colors = self.color_palette[:len(groups)]

            # Q-Q plots for each group
            for i, group in enumerate(groups):
                group_data = data[data[independent_var] == group][dependent_var].dropna()
                if len(group_data) > 2:
                    qq_data = stats.probplot(group_data, dist="norm")
                    fig.add_trace(
                        go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers',
                                  name=f'{group} Q-Q', marker=dict(color=colors[i])),
                        row=1, col=1
                    )

            # Variance comparison
            for i, group in enumerate(groups):
                group_data = data[data[independent_var] == group][dependent_var]
                fig.add_trace(
                    go.Box(y=group_data, name=str(group), marker_color=colors[i]),
                    row=1, col=2
                )

            # Outlier detection
            combined_data = data[dependent_var].dropna()
            Q1 = combined_data.quantile(0.25)
            Q3 = combined_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = combined_data[(combined_data < lower_bound) | (combined_data > upper_bound)]

            fig.add_trace(
                go.Scatter(x=range(len(combined_data)), y=combined_data,
                          mode='markers', name='All Data',
                          marker=dict(color='lightblue')),
                row=2, col=1
            )

            if len(outliers) > 0:
                outlier_indices = outliers.index
                fig.add_trace(
                    go.Scatter(x=outlier_indices, y=outliers,
                              mode='markers', name='Outliers',
                              marker=dict(color='red', size=8)),
                    row=2, col=1
                )

            # Sample sizes
            sample_sizes = [len(data[data[independent_var] == group]) for group in groups]
            fig.add_trace(
                go.Bar(x=[str(group) for group in groups], y=sample_sizes,
                      marker_color=colors),
                row=2, col=2
            )

        fig.update_layout(
            title=f'Assumption Checks for {test_name}',
            height=600
        )

        return fig