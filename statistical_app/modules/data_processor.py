"""
Data Processing Module
=====================

Handles data upload, validation, preprocessing, and basic exploratory analysis.
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import warnings
from scipy import stats

class DataProcessor:
    """
    Handles all data-related operations including upload, validation, and preprocessing.
    """

    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls']

    def load_data(self, uploaded_file):
        """
        Load data from uploaded file.

        Parameters:
        -----------
        uploaded_file : UploadedFile
            Streamlit uploaded file object

        Returns:
        --------
        pandas.DataFrame : Loaded data
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'csv':
                # Try different encodings and separators
                data = self._load_csv_robust(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                data = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Basic validation
            self._validate_data(data)

            return data

        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def _load_csv_robust(self, uploaded_file):
        """
        Robustly load CSV file with different encodings and separators.
        """
        # Reset file pointer
        uploaded_file.seek(0)

        # Try different combinations of encoding and separator
        encodings = ['utf-8', 'latin-1', 'cp1252']
        separators = [',', ';', '\t']

        for encoding in encodings:
            for separator in separators:
                try:
                    uploaded_file.seek(0)
                    # Read as string first
                    content = uploaded_file.read().decode(encoding)
                    data = pd.read_csv(StringIO(content), sep=separator)

                    # Check if it looks reasonable (at least 2 columns)
                    if data.shape[1] >= 2:
                        return data
                except:
                    continue

        # If all else fails, try default pandas read_csv
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)

    def _validate_data(self, data):
        """
        Validate loaded data for basic requirements.
        """
        if data.empty:
            raise ValueError("Data file is empty")

        if data.shape[0] < 2:
            raise ValueError("Data must have at least 2 rows")

        if data.shape[1] < 1:
            raise ValueError("Data must have at least 1 column")

        # Check for completely empty columns
        empty_columns = data.columns[data.isnull().all()].tolist()
        if empty_columns:
            warnings.warn(f"Found completely empty columns: {empty_columns}")

    def get_data_summary(self, data):
        """
        Generate comprehensive data summary.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset

        Returns:
        --------
        dict : Summary statistics and information
        """
        summary = {
            'basic_info': {
                'n_rows': data.shape[0],
                'n_columns': data.shape[1],
                'total_missing': data.isnull().sum().sum(),
                'memory_usage': data.memory_usage(deep=True).sum()
            },
            'column_info': {},
            'missing_data': {},
            'data_types': {}
        }

        # Analyze each column
        for col in data.columns:
            col_data = data[col]

            # Basic column info
            summary['column_info'][col] = {
                'dtype': str(col_data.dtype),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
                'unique_count': col_data.nunique()
            }

            # Determine data type category
            if pd.api.types.is_numeric_dtype(col_data):
                summary['column_info'][col]['category'] = 'numeric'
                summary['column_info'][col]['min'] = col_data.min()
                summary['column_info'][col]['max'] = col_data.max()
                summary['column_info'][col]['mean'] = col_data.mean()
                summary['column_info'][col]['std'] = col_data.std()
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                summary['column_info'][col]['category'] = 'datetime'
                summary['column_info'][col]['min_date'] = col_data.min()
                summary['column_info'][col]['max_date'] = col_data.max()
            else:
                summary['column_info'][col]['category'] = 'categorical'
                value_counts = col_data.value_counts()
                summary['column_info'][col]['top_values'] = value_counts.head().to_dict()

        # Missing data pattern
        if data.isnull().sum().sum() > 0:
            summary['missing_data'] = {
                'columns_with_missing': data.columns[data.isnull().any()].tolist(),
                'missing_by_column': data.isnull().sum().to_dict(),
                'rows_with_missing': data.isnull().any(axis=1).sum()
            }

        return summary

    def detect_outliers(self, data, column, method='iqr'):
        """
        Detect outliers in a numeric column.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        column : str
            Column name to analyze
        method : str
            Method for outlier detection ('iqr', 'zscore', 'modified_zscore')

        Returns:
        --------
        dict : Outlier information
        """
        if not pd.api.types.is_numeric_dtype(data[column]):
            return {'error': f'Column {column} is not numeric'}

        col_data = data[column].dropna()

        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(col_data))
            outliers = col_data[z_scores > 3]

        elif method == 'modified_zscore':
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))
            modified_z_scores = 0.6745 * (col_data - median) / mad
            outliers = col_data[np.abs(modified_z_scores) > 3.5]

        else:
            return {'error': f'Unknown outlier detection method: {method}'}

        return {
            'method': method,
            'n_outliers': len(outliers),
            'outlier_percentage': (len(outliers) / len(col_data)) * 100,
            'outlier_values': outliers.tolist(),
            'outlier_indices': outliers.index.tolist()
        }

    def suggest_data_types(self, data):
        """
        Suggest appropriate data types for statistical analysis.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset

        Returns:
        --------
        dict : Suggested data types for each column
        """
        suggestions = {}

        for col in data.columns:
            col_data = data[col].dropna()

            if len(col_data) == 0:
                suggestions[col] = 'empty'
                continue

            # Check if numeric
            if pd.api.types.is_numeric_dtype(col_data):
                unique_count = col_data.nunique()
                total_count = len(col_data)

                # Check if it might be categorical despite being numeric
                if unique_count <= 10 and unique_count / total_count < 0.5:
                    suggestions[col] = 'categorical (nominal)'
                else:
                    suggestions[col] = 'continuous'

            # Check if datetime
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                suggestions[col] = 'datetime'

            # Check if boolean
            elif pd.api.types.is_bool_dtype(col_data):
                suggestions[col] = 'binary'

            # String/object type
            else:
                unique_count = col_data.nunique()
                total_count = len(col_data)

                # Check if binary
                if unique_count == 2:
                    suggestions[col] = 'binary'

                # Check if ordinal (look for patterns that suggest ordering)
                elif self._might_be_ordinal(col_data):
                    suggestions[col] = 'ordinal'

                # Check if nominal categorical
                elif unique_count <= 20:  # Arbitrary threshold
                    suggestions[col] = 'categorical (nominal)'

                # High cardinality - might be identifier or text
                else:
                    if unique_count / total_count > 0.95:
                        suggestions[col] = 'identifier'
                    else:
                        suggestions[col] = 'categorical (high cardinality)'

        return suggestions

    def _might_be_ordinal(self, series):
        """
        Check if a categorical series might be ordinal.
        """
        # Convert to string and check for common ordinal patterns
        str_values = series.astype(str).str.lower().unique()

        # Common ordinal patterns
        ordinal_patterns = [
            ['low', 'medium', 'high'],
            ['small', 'medium', 'large'],
            ['poor', 'fair', 'good', 'excellent'],
            ['never', 'rarely', 'sometimes', 'often', 'always'],
            ['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree'],
            ['first', 'second', 'third'],
            ['elementary', 'middle', 'high', 'college'],
            ['mild', 'moderate', 'severe']
        ]

        # Check if any pattern matches
        for pattern in ordinal_patterns:
            if all(val in str_values for val in pattern):
                return True

        # Check for numeric-like ordering (1st, 2nd, 3rd, etc.)
        if any(val.endswith(('st', 'nd', 'rd', 'th')) for val in str_values):
            return True

        return False

    def prepare_for_analysis(self, data, dependent_var, independent_var=None,
                           remove_outliers=False, outlier_method='iqr'):
        """
        Prepare data for statistical analysis.

        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        dependent_var : str
            Dependent variable column name
        independent_var : str, optional
            Independent variable column name
        remove_outliers : bool
            Whether to remove outliers
        outlier_method : str
            Method for outlier removal

        Returns:
        --------
        pandas.DataFrame : Prepared dataset
        dict : Information about preprocessing steps
        """
        preparation_info = {
            'original_shape': data.shape,
            'steps_performed': []
        }

        # Start with copy of data
        processed_data = data.copy()

        # Select relevant columns
        if independent_var:
            relevant_columns = [dependent_var, independent_var]
        else:
            relevant_columns = [dependent_var]

        # Add any additional columns that might be needed (subject IDs, etc.)
        processed_data = processed_data[relevant_columns]
        preparation_info['steps_performed'].append(f'Selected columns: {relevant_columns}')

        # Remove rows with missing values in key columns
        initial_rows = len(processed_data)
        processed_data = processed_data.dropna(subset=relevant_columns)
        rows_removed = initial_rows - len(processed_data)

        if rows_removed > 0:
            preparation_info['steps_performed'].append(f'Removed {rows_removed} rows with missing values')

        # Remove outliers if requested
        if remove_outliers and pd.api.types.is_numeric_dtype(processed_data[dependent_var]):
            outlier_info = self.detect_outliers(processed_data, dependent_var, outlier_method)

            if 'outlier_indices' in outlier_info:
                processed_data = processed_data.drop(outlier_info['outlier_indices'])
                preparation_info['steps_performed'].append(
                    f'Removed {outlier_info["n_outliers"]} outliers using {outlier_method} method'
                )

        preparation_info['final_shape'] = processed_data.shape
        preparation_info['data_reduction'] = (
            (preparation_info['original_shape'][0] - preparation_info['final_shape'][0]) /
            preparation_info['original_shape'][0] * 100
        )

        return processed_data, preparation_info

    def create_sample_datasets(self):
        """
        Create sample datasets for demonstration purposes.

        Returns:
        --------
        dict : Dictionary of sample datasets
        """
        np.random.seed(42)

        datasets = {}

        # 1. Biological Growth Study
        n = 50
        control_growth = np.random.normal(10, 2, n)
        treatment_growth = np.random.normal(12, 2.5, n)

        datasets['biological_growth'] = pd.DataFrame({
            'growth_rate': np.concatenate([control_growth, treatment_growth]),
            'treatment': ['Control'] * n + ['Treatment'] * n,
            'replicate': list(range(1, n+1)) * 2,
            'initial_size': np.random.normal(5, 1, 2*n),
            'temperature': np.random.choice([20, 25, 30], 2*n)
        })

        # 2. Gene Expression Study
        n = 30
        baseline_expr = np.random.lognormal(2, 0.5, n)
        stimulated_expr = np.random.lognormal(2.3, 0.6, n)

        datasets['gene_expression'] = pd.DataFrame({
            'expression_level': np.concatenate([baseline_expr, stimulated_expr]),
            'condition': ['Baseline'] * n + ['Stimulated'] * n,
            'subject_id': list(range(1, n+1)) * 2,
            'age': np.random.randint(18, 65, 2*n),
            'gender': np.random.choice(['M', 'F'], 2*n)
        })

        # 3. Clinical Trial Data
        n = 100
        placebo_response = np.random.binomial(1, 0.3, n)
        drug_response = np.random.binomial(1, 0.6, n)

        datasets['clinical_trial'] = pd.DataFrame({
            'response': np.concatenate([placebo_response, drug_response]),
            'treatment': ['Placebo'] * n + ['Drug'] * n,
            'age': np.random.normal(55, 15, 2*n),
            'severity': np.random.choice(['Mild', 'Moderate', 'Severe'], 2*n),
            'duration_days': np.random.gamma(2, 10, 2*n)
        })

        # 4. Ecological Survey
        n = 75
        habitat_a_abundance = np.random.poisson(8, n)
        habitat_b_abundance = np.random.poisson(12, n)
        habitat_c_abundance = np.random.poisson(6, n)

        datasets['ecological_survey'] = pd.DataFrame({
            'species_abundance': np.concatenate([habitat_a_abundance, habitat_b_abundance, habitat_c_abundance]),
            'habitat_type': ['Forest'] * n + ['Grassland'] * n + ['Wetland'] * n,
            'season': np.random.choice(['Spring', 'Summer', 'Fall'], 3*n),
            'temperature': np.random.normal(20, 8, 3*n),
            'precipitation': np.random.exponential(2, 3*n)
        })

        # 5. Paired Measurements Study
        n = 40
        before_values = np.random.normal(100, 15, n)
        after_values = before_values + np.random.normal(5, 8, n)  # Some improvement with noise

        datasets['paired_measurements'] = pd.DataFrame({
            'measurement': np.concatenate([before_values, after_values]),
            'time_point': ['Before'] * n + ['After'] * n,
            'subject_id': list(range(1, n+1)) * 2,
            'baseline_score': np.repeat(np.random.normal(50, 10, n), 2)
        })

        return datasets