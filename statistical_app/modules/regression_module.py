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

    # KROK 1: Najpierw wybierz zmienną ZALEŻNĄ (Y)
    col_y = st.sidebar.selectbox("Wybierz zmienną zależną (Y - prognozowaną):", numeric_cols)

    # KROK 2: Reszta kolumn jest dostępna jako predyktory (X)
    available_x_cols = [col for col in numeric_cols if col != col_y]

    if not available_x_cols:
        st.warning(f"Brak dostępnych zmiennych (X) do prognozowania {col_y}.")
        st.stop()

    # KROK 3: Użyj MULTISELECT dla zmiennych X
    cols_x = st.sidebar.multiselect(
        "Wybierz zmienne niezależne (X - predyktory):",
        available_x_cols,
        default=available_x_cols[0] if available_x_cols else None,
        help="Wybierz JEDNĄ zmienną dla regresji prostej/logarytmicznej lub WIELE zmiennych dla regresji wielokrotnej."
    )

    if not cols_x:
        st.info("Proszę wybrać co najmniej jedną zmienną niezależną (X), aby rozpocząć analizę.")
        st.stop()

    # === SECTION 3: ANALYSIS AND RESULTS ===

    # Prepare data (remove missing values)
    all_cols = [col_y] + cols_x
    analysis_df = df[all_cols].dropna()
    X_raw = analysis_df[cols_x]
    y = analysis_df[col_y]

    if X_raw.empty or y.empty:
        st.error("Wybrane kolumny nie zawierają danych po usunięciu braków (NaN).")
        st.stop()

    st.header("Wyniki Analizy Regresji")

    # --- ROZGAŁĘZIENIE LOGIKI ---

    # ===== PRZYPADEK 1: REGRESJA PROSTA I LOGARYTMICZNA (1 predyktor X) =====
    if len(cols_x) == 1:
        st.info("Wykryto 1 predyktor (X). Przeprowadzam analizę regresji prostej i logarytmicznej.")

        # Zmień nazwę X_raw na X_simple, aby było jasne
        col_x_name = cols_x[0]
        X_simple = X_raw[[col_x_name]]  # sklearn wymaga 2D DataFrame

        col1, col2 = st.columns(2)

        # --- Model Liniowy Prosty ---
        with col1:
            st.subheader("Model Liniowy Prosty")
            st.markdown("`y = a * x + b`")

            model_lin = LinearRegression()
            model_lin.fit(X_simple, y)
            y_pred_lin = model_lin.predict(X_simple)

            a_lin = model_lin.coef_[0]
            b_lin = model_lin.intercept_
            r2_lin = r2_score(y, y_pred_lin)
            mse_lin = mean_squared_error(y, y_pred_lin)

            st.metric("R-kwadrat (R²)", f"{r2_lin:.4f}")
            st.metric("Błąd Średniokw. (MSE)", f"{mse_lin:.4f}")
            st.write(f"**Wzór:** `y = {a_lin:.3f} * x + {b_lin:.3f}`")
            st.caption(f"Używasz: {col_x_name}")

        # --- Model Logarytmiczny ---
        with col2:
            st.subheader("Model Logarytmiczny")
            st.markdown("`y = a * ln(x) + b`")

            # Logarithmic model requires X > 0
            log_df = analysis_df[analysis_df[col_x_name] > 0]
            if log_df.empty:
                st.warning("Model logarytmiczny wymaga wartości X > 0. Brak danych do analizy.")
                r2_log, mse_log, a_log, b_log = (np.nan,) * 4  # Set as NaN
            else:
                X_log_transformed = np.log(log_df[[col_x_name]])  # Transform X -> ln(X)
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
                st.caption(f"Używasz: ln({col_x_name}). Analiza na {len(log_df)} wierszach (gdzie {col_x_name} > 0).")

        # --- Wizualizacja 2D ---
        st.subheader("Wizualizacja Dopasowania")

        # Create scatter plot
        fig = px.scatter(
            analysis_df,
            x=col_x_name,
            y=col_y,
            title="Dopasowanie Modeli Regresji",
            labels={col_x_name: f"X: {col_x_name}", col_y: f"Y: {col_y}"}
        )

        # Add linear regression line
        fig.add_trace(
            go.Scatter(
                x=X_simple[col_x_name],
                y=y_pred_lin,
                mode='lines',
                name=f"Liniowa (R²={r2_lin:.3f})",
                line=dict(color='red', width=3)
            )
        )

        # Add logarithmic regression line (if calculated)
        if not np.isnan(r2_log):
            # Sort values for smooth line
            log_df_sorted = log_df.sort_values(by=col_x_name)
            X_log_plot = np.log(log_df_sorted[[col_x_name]])
            y_pred_log_plot = model_log.predict(X_log_plot)

            fig.add_trace(
                go.Scatter(
                    x=log_df_sorted[col_x_name],
                    y=y_pred_log_plot,
                    mode='lines',
                    name=f"Logarytm. (R²={r2_log:.3f})",
                    line=dict(color='green', width=3)
                )
            )

        st.plotly_chart(fig, use_container_width=True)


    # ===== PRZYPADEK 2: WIELOKROTNA REGRESJA LINIOWA (>1 predyktor X) =====
    elif len(cols_x) > 1:
        st.info(f"Wykryto {len(cols_x)} predyktory (X). Przeprowadzam wielokrotną regresję liniową (MLR).")

        st.subheader("Wielokrotna Regresja Liniowa (MLR)")

        # Model jest ten sam, ale karmimy go wieloma kolumnami X
        model_mlr = LinearRegression()
        model_mlr.fit(X_raw, y)
        y_pred_mlr = model_mlr.predict(X_raw)

        r2_mlr = r2_score(y, y_pred_mlr)
        mse_mlr = mean_squared_error(y, y_pred_mlr)

        # Obliczanie R-kwadrat Skorygowanego (Adjusted R²)
        n = len(y)  # liczba obserwacji
        p = len(cols_x)  # liczba predyktorów
        r2_adj = 0.0
        if n - p - 1 > 0:
            r2_adj = 1 - (1 - r2_mlr) * (n - 1) / (n - p - 1)

        # Wyświetlanie metryk
        col1, col2, col3 = st.columns(3)
        col1.metric("R-kwadrat (R²)", f"{r2_mlr:.4f}")
        col2.metric("R-kwadrat Skorygowany (Adjusted R²)", f"{r2_adj:.4f}",
                   help="Ważniejszy niż R² przy wielu zmiennych. Karze za dodawanie nieistotnych predyktorów.")
        col3.metric("Błąd Średniokw. (MSE)", f"{mse_mlr:.4f}")

        # Wyświetlanie wzoru i współczynników
        st.markdown("**Wzór Modelu:**")
        intercept_b = model_mlr.intercept_
        coefficients_a = model_mlr.coef_

        formula = f"`{col_y} = {intercept_b:.3f}`"
        for i, (col, coef) in enumerate(zip(cols_x, coefficients_a)):
            sign = "+" if coef >= 0 else "-"
            formula += f" `{sign} ({abs(coef):.3f} * {col})`"
        st.code(formula, language='text')

        # Tabela współczynników
        st.markdown("**Współczynniki:**")
        coef_df = pd.DataFrame(
            {'Cecha (X)': ['Wyraz Wolny (Intercept)'] + cols_x,
             'Współczynnik (Waga)': [intercept_b] + list(coefficients_a)}
        )
        st.dataframe(coef_df.set_index('Cecha (X)').style.format("{:.4f}"))
        st.caption("Współczynnik pokazuje, o ile średnio zmieni się Y, gdy dana Cecha (X) wzrośnie o 1 jednostkę, **przy założeniu, że pozostałe cechy X się nie zmieniają**.")

        # === Wizualizacja dla MLR ===
        st.subheader("Wizualizacja Wyników MLR")
        st.markdown("Ponieważ mamy wiele wymiarów (X), nie możemy narysować prostej linii. Poniższy wykres pokazuje, jak **wartości prognozowane** przez model mają się do **wartości rzeczywistych**.")

        plot_df = pd.DataFrame({'Rzeczywiste (Y)': y, 'Prognozowane (Y_pred)': y_pred_mlr})
        fig_mlr = px.scatter(plot_df, x='Rzeczywiste (Y)', y='Prognozowane (Y_pred)',
                             title='Rzeczywiste vs. Prognozowane',
                             labels={'Rzeczywiste (Y)': f'Rzeczywiste {col_y}', 'Prognozowane (Y_pred)': 'Prognozowane przez Model'},
                             hover_data=plot_df.columns)

        # Dodaj linię idealnego dopasowania (y=x)
        fig_mlr.add_shape(type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color='red', dash='dash', width=2), name="Idealne dopasowanie")
        st.plotly_chart(fig_mlr, use_container_width=True)

    # === SECTION 4: INTERPRETATION AND OVERFITTING ===
    st.header("Wybór Modelu i Interpretacja")

    # ===== PRZYPADEK 1: REGRESJA PROSTA I LOGARYTMICZNA =====
    if len(cols_x) == 1:
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
            danych treningowych, w tym szumu, zamiast wychwytywać ogólny trend.

            **Jak to się ma do nas?**
            Modele regresji prostej (liniowy i logarytmiczny) są modelami **bardzo prostymi**.
            Ryzyko przeuczenia jest tu **minimalne**.
            """
        )

    # ===== PRZYPADEK 2: WIELOKROTNA REGRESJA LINIOWA =====
    elif len(cols_x) > 1:
        st.info(f"**Interpretacja R-kwadrat Skorygowanego (Adjusted R²):** Używaj tej metryki do oceny modelu. Wynik {r2_adj:.4f} oznacza, że model wyjaśnia ok. {r2_adj*100:.1f}% zmienności w {col_y}, biorąc pod uwagę liczbę użytych predyktorów.")

        st.subheader("Uwaga na Overfitting (Przeuczenie)")
        st.warning(
            """
            **Ryzyko w MLR:** W Wielokrotnej Regresji Liniowej ryzyko przeuczenia **rośnie wraz z liczbą dodanych zmiennych (X)**.

            **Jak się chronić?**
            1.  **Patrz na R-kwadrat Skorygowany:** Zwykłe R² **zawsze** rośnie, gdy dodajesz nowe zmienne, nawet jeśli są bez sensu. R-kwadrat Skorygowany "karze" model za dodawanie nieistotnych zmiennych. Jeśli dodasz zmienną, a R² Skorygowany spadnie, prawdopodobnie ta zmienna jest zbędna.
            2.  **Multikolinearność (Współliniowość):** Uważaj, jeśli twoje zmienne X są ze sobą silnie skorelowane (np. `wzrost_w_cm` i `wzrost_w_calach`). Może to prowadzić do niestabilnych wyników i błędnych współczynników.
            3.  **Zasada Oszczędności:** Wybieraj najprostszy model (najmniej zmiennych X), który daje *wystarczająco dobre* wyniki.
            """
        )
