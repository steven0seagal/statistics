"""
Regression Analysis Module
==========================

A comprehensive module for performing and comparing linear and logarithmic regression analysis.
Includes demo data generation, model fitting, visualization, and interpretation.

Author: Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go


def get_demo_data_linear():
    """Generate demo data with linear relationship"""
    X = np.linspace(1, 50, 100)
    y = 2.5 * X + 4 + np.random.randn(100) * 8  # Strong linear correlation + noise
    return pd.DataFrame({'X_lin': X, 'Y_lin': y})


def get_demo_data_log():
    """Generate demo data with logarithmic relationship"""
    X = np.linspace(1, 50, 100)
    y = 3 * np.log(X) + 2 + np.random.randn(100) * 0.5  # Logarithmic correlation + noise
    return pd.DataFrame({'X_log': X, 'Y_log': y})


def run_regression_module():
    """Main function for the regression analysis module"""

    st.title("📈 Moduł Analizy Regresji")
    st.markdown("Analizuj i porównuj modele regresji liniowej i logarytmicznej.")

    # === SECTION 1: DATA SOURCE ===
    st.sidebar.header("1. Wprowadź Dane")
    data_source = st.sidebar.radio(
        "Wybierz źródło danych:",
        ("Użyj danych demo", "Wprowadź własne dane")
    )

    df = None  # Initialize DataFrame

    # --- Demo Data ---
    if data_source == "Użyj danych demo":
        demo_type = st.sidebar.selectbox("Wybierz typ danych demo:", ["Dane Liniowe", "Dane Logarytmiczne"])
        if demo_type == "Dane Liniowe":
            df = get_demo_data_linear()
        else:
            df = get_demo_data_log()

    # --- Custom Data ---
    else:
        uploaded_file = st.sidebar.file_uploader("Wgraj plik CSV lub Excel", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Błąd podczas wczytywania pliku: {e}")

    # If data not loaded, stop here
    if df is None or df.empty:
        st.info("Proszę wczytać dane lub wybrać zestaw demo, aby rozpocząć analizę.")
        st.stop()

    st.subheader("Podgląd Danych")
    st.dataframe(df.head())

    # === SECTION 2: VARIABLE SELECTION ===
    st.sidebar.header("2. Konfiguracja Modelu")

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Twoje dane muszą zawierać co najmniej dwie kolumny numeryczne.")
        st.stop()

    col_x = st.sidebar.selectbox("Wybierz zmienną niezależną (X):", numeric_cols)
    # Remove selected X column from available Y columns
    available_y_cols = [col for col in numeric_cols if col != col_x]
    col_y = st.sidebar.selectbox("Wybierz zmienną zależną (Y):", available_y_cols)

    # === SECTION 3: ANALYSIS AND RESULTS ===

    # Prepare data (remove missing values)
    analysis_df = df[[col_x, col_y]].dropna()
    X_raw = analysis_df[[col_x]]
    y = analysis_df[col_y]

    if X_raw.empty or y.empty:
        st.error("Wybrane kolumny nie zawierają danych po usunięciu braków (NaN).")
        st.stop()

    st.header("Wyniki Analizy Regresji")

    # Use columns for side-by-side results
    col1, col2 = st.columns(2)

    # --- Linear Model ---
    with col1:
        st.subheader("Model Liniowy")
        st.markdown("`y = a * x + b`")

        model_lin = LinearRegression()
        model_lin.fit(X_raw, y)
        y_pred_lin = model_lin.predict(X_raw)

        a_lin = model_lin.coef_[0]
        b_lin = model_lin.intercept_
        r2_lin = r2_score(y, y_pred_lin)
        mse_lin = mean_squared_error(y, y_pred_lin)

        st.metric("R-kwadrat (R²)", f"{r2_lin:.4f}")
        st.metric("Błąd Średniokw. (MSE)", f"{mse_lin:.4f}")
        st.write(f"**Wzór:** `y = {a_lin:.3f} * x + {b_lin:.3f}`")

    # --- Logarithmic Model ---
    with col2:
        st.subheader("Model Logarytmiczny")
        st.markdown("`y = a * ln(x) + b`")

        # Logarithmic model requires X > 0
        log_df = analysis_df[analysis_df[col_x] > 0]
        if log_df.empty:
            st.warning("Model logarytmiczny wymaga wartości X > 0. Brak danych do analizy.")
            r2_log, mse_log, a_log, b_log = (np.nan,) * 4  # Set as NaN
        else:
            X_log_transformed = np.log(log_df[[col_x]])  # Transform X -> ln(X)
            y_log_target = log_df[col_y]

            model_log = LinearRegression()
            model_log.fit(X_log_transformed, y_log_target)
            y_pred_log = model_log.predict(X_log_transformed)

            a_log = model_log.coef_[0]
            b_log = model_log.intercept_
            r2_log = r2_score(y_log_target, y_pred_log)
            mse_log = mean_squared_error(y_log_target, y_pred_log)

            st.metric("R-kwadrat (R²)", f"{r2_log:.4f}")
            st.metric("Błąd Średniokw. (MSE)", f"{mse_log:.4f}")
            st.write(f"**Wzór:** `y = {a_log:.3f} * ln(x) + {b_log:.3f}`")
            st.caption(f"Analiza przeprowadzona na {len(log_df)} wierszach (gdzie {col_x} > 0).")

    # --- Visualization ---
    st.subheader("Wizualizacja Dopasowania")

    # Create scatter plot
    fig = px.scatter(
        analysis_df,
        x=col_x,
        y=col_y,
        title="Dopasowanie Modeli Regresji",
        labels={col_x: f"X: {col_x}", col_y: f"Y: {col_y}"}
    )

    # Add linear regression line
    fig.add_trace(
        go.Scatter(
            x=X_raw[col_x],
            y=y_pred_lin,
            mode='lines',
            name=f"Liniowa (R²={r2_lin:.3f})",
            line=dict(color='red', width=3)
        )
    )

    # Add logarithmic regression line (if calculated)
    if not np.isnan(r2_log):
        # Sort values for smooth line
        log_df_sorted = log_df.sort_values(by=col_x)
        X_log_plot = np.log(log_df_sorted[[col_x]])
        y_pred_log_plot = model_log.predict(X_log_plot)

        fig.add_trace(
            go.Scatter(
                x=log_df_sorted[col_x],
                y=y_pred_log_plot,
                mode='lines',
                name=f"Logarytm. (R²={r2_log:.3f})",
                line=dict(color='green', width=3)
            )
        )

    st.plotly_chart(fig, use_container_width=True)

    # === SECTION 4: INTERPRETATION AND OVERFITTING ===
    st.header("Wybór Modelu i Interpretacja")

    # Comparison
    if np.isnan(r2_log) and not np.isnan(r2_lin):
        st.info("Tylko model liniowy mógł zostać obliczony (model logarytmiczny wymaga X > 0).")
    elif np.isnan(r2_lin) and np.isnan(r2_log):
        st.error("Nie można było obliczyć żadnego modelu.")
    else:
        # Compare R²
        if r2_lin > r2_log:
            st.success(
                f"**Model Liniowy wydaje się lepiej dopasowany.** Posiada wyższy współczynnik R² "
                f"({r2_lin:.4f}) w porównaniu do modelu logarytmicznego ({r2_log:.4f})."
            )
        elif r2_log > r2_lin:
            st.success(
                f"**Model Logarytmiczny wydaje się lepiej dopasowany.** Posiada wyższy współczynnik R² "
                f"({r2_log:.4f}) w porównaniu do modelu liniowego ({r2_lin:.4f})."
            )
        else:
            st.info(f"Oba modele mają identyczny współczynnik R² ({r2_lin:.4f}).")

        # Compare MSE
        if mse_lin < mse_log:
            st.success(
                f"Model Liniowy ma również **mniejszy błąd średniokwadratowy (MSE)** "
                f"({mse_lin:.4f} vs {mse_log:.4f}), co oznacza, że jego prognozy są średnio bliższe "
                f"rzeczywistym wartościom."
            )
        elif mse_log < mse_lin:
            st.success(
                f"Model Logarytmiczny ma również **mniejszy błąd średniokwadratowy (MSE)** "
                f"({mse_log:.4f} vs {mse_lin:.4f}), co oznacza, że jego prognozy są średnio bliższe "
                f"rzeczywistym wartościom."
            )

    st.subheader("Uwaga na Overfitting (Przeuczenie)")
    st.warning(
        """
        **Czym jest Overfitting?**
        Przeuczenie (overfitting) ma miejsce, gdy model jest zbyt skomplikowany i "uczy się na pamięć"
        danych treningowych, w tym szumu, zamiast wychwytywać ogólny trend. Taki model będzie miał
        świetne wyniki na danych, na których trenował, ale fatalne na nowych, nieznanych danych.

        **Jak to się ma do nas?**
        Modele, których tu użyliśmy (liniowy i logarytmiczny z jedną zmienną), są modelami **bardzo prostymi**.
        Ryzyko przeuczenia jest tu **minimalne**.

        Dużo większym ryzykiem jest **underfitting** (niedouczenie), czyli sytuacja, w której model jest
        zbyt prosty, aby uchwycić prawdziwą zależność w danych (np. próba dopasowania linii prostej do danych,
        które układają się w parabolę).

        **Jak rzetelnie ocenić model?**
        W prawdziwej analizie predykcyjnej, dane dzieli się na **zbiór treningowy** (do nauki modelu)
        i **zbiór testowy** (do jego oceny). Model ocenia się na podstawie jego wyników (np. R² lub MSE)
        na zbiorze testowym. W tym module, dla celów demonstracyjnych, dopasowujemy model do wszystkich
        dostępnych danych.
        """
    )
