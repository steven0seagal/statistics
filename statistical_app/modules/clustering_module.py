"""
K-Means Clustering Module
=========================

A comprehensive module for performing K-Means clustering analysis.
Includes demo data generation, optimal K determination (Elbow Method, Silhouette Analysis),
model fitting, visualization, and interpretation.

Author: Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import plotly.express as px
import plotly.graph_objects as go


def get_demo_data_clustering():
    """Generate demo data with clear cluster structure"""
    X, y = make_blobs(n_samples=300, centers=3, n_features=2,
                      cluster_std=1.0, random_state=42)
    demo_df = pd.DataFrame(X, columns=['Cecha_1', 'Cecha_2'])
    demo_df['Prawdziwy_Klaster'] = y  # For validation, though model doesn't see this
    return demo_df


def run_clustering_module():
    """Main function for the K-Means clustering module"""

    st.title("💠 Moduł Klastrowania K-Means")
    st.markdown("Grupuj dane i odkrywaj ukryte struktury za pomocą algorytmu K-Means.")

    # === SECTION 1: DATA SOURCE ===
    st.sidebar.header("1. Wprowadź Dane")
    data_source = st.sidebar.radio(
        "Wybierz źródło danych:",
        ("Użyj danych demo (3 klastry)", "Wprowadź własne dane")
    )

    df = None

    if data_source == "Użyj danych demo (3 klastry)":
        df = get_demo_data_clustering()
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

    if df is None or df.empty:
        st.info("Proszę wczytać dane lub wybrać zestaw demo, aby rozpocząć analizę.")
        st.stop()

    st.subheader("Podgląd Danych")
    st.dataframe(df.head())

    # === SECTION 2: MODEL CONFIGURATION ===
    st.sidebar.header("2. Konfiguracja Modelu")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Remove 'Prawdziwy_Klaster' column from demo data if present
    if 'Prawdziwy_Klaster' in numeric_cols:
        numeric_cols.remove('Prawdziwy_Klaster')

    if len(numeric_cols) == 0:
        st.warning("Twoje dane muszą zawierać kolumny numeryczne.")
        st.stop()

    selected_features = st.sidebar.multiselect(
        "Wybierz cechy (zmienne) do klastrowania:",
        numeric_cols,
        default=numeric_cols[0:min(len(numeric_cols), 2)]  # Default to first two
    )

    if len(selected_features) == 0:
        st.warning("Proszę wybrać co najmniej jedną cechę do analizy.")
        st.stop()

    # --- Method: Data Standardization ---
    do_scale = st.sidebar.checkbox(
        "Standaryzuj dane (Zalecane!)",
        value=True,
        help="K-Means jest wrażliwy na skalę danych (np. 'Wiek' vs 'Zarobki'). Standaryzacja (X-średnia)/odch. std. sprawia, że wszystkie cechy mają równą wagę."
    )

    # Prepare data for analysis
    X_raw = df[selected_features].dropna()

    if X_raw.empty:
        st.error("Wybrane kolumny nie zawierają danych po usunięciu braków (NaN).")
        st.stop()

    X_scaled = X_raw.copy()
    if do_scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = X_raw.values  # Use .values for consistent type (numpy array)

    # === SECTION 3: FINDING OPTIMAL K ===
    st.header("Analiza Optymalnej Liczby Klastrów (K)")
    st.markdown("""
    Algorytm K-Means wymaga podania z góry liczby klastrów (K). Poniższe metody statystyczne pomogą Ci dokonać świadomego wyboru.
    """)

    max_k = st.sidebar.slider("Maksymalna liczba 'K' do analizy:", 2, 15, 10)

    # Storage for results
    inertia_values = []  # For Elbow Method
    silhouette_values = []  # For Silhouette Analysis
    k_range = range(2, max_k + 1)

    # Use progress bar
    progress_bar = st.progress(0, text="Analizowanie K...")

    for i, k in enumerate(k_range):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X_scaled)

        # Elbow Method
        inertia_values.append(kmeans.inertia_)

        # Silhouette Method
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_values.append(score)

        progress_bar.progress((i + 1) / len(k_range), text=f"Analizowanie K={k}...")
    progress_bar.empty()

    # Display results in tabs
    tab1, tab2 = st.tabs(["Metoda Łokcia (Inertia)", "Analiza Sylwetkowa (Silhouette Score)"])

    with tab1:
        st.subheader("Metoda Łokcia (Elbow Method)")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertia_values, mode='lines+markers'))
        fig_elbow.update_layout(
            title="Suma Kwadratów Odległości wewnątrz Klastrów (Inertia)",
            xaxis_title="Liczba Klastrów (K)",
            yaxis_title="Inertia (WCSS)"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.markdown("**Jak interpretować:** Szukaj 'załamania' (łokcia) na wykresie. Jest to punkt, w którym dodanie kolejnego klastra nie przynosi już znaczącej redukcji sumy błędów. To sugeruje optymalne K.")

    with tab2:
        st.subheader("Analiza Sylwetkowa (Silhouette Analysis)")
        fig_silhouette = go.Figure()
        fig_silhouette.add_trace(go.Scatter(x=list(k_range), y=silhouette_values, mode='lines+markers'))
        fig_silhouette.update_layout(
            title="Średni Współczynnik Sylwetkowy",
            xaxis_title="Liczba Klastrów (K)",
            yaxis_title="Silhouette Score"
        )
        st.plotly_chart(fig_silhouette, use_container_width=True)
        st.markdown("**Jak interpretować:** Wynik bliski +1 oznacza, że klastry są gęste i dobrze odseparowane. Wynik bliski 0 oznacza nakładanie się klastrów. **Szukaj wartości 'K', która daje najwyższy (maksymalny) wynik.**")

    # === SECTION 4: MODEL EXECUTION AND RESULTS ===
    st.header("Uruchomienie Modelu K-Means")

    # User selects final K
    st.sidebar.header("3. Uruchom Model")
    final_k = st.sidebar.number_input(
        "Wybierz ostateczną liczbę klastrów (K):",
        min_value=2,
        max_value=max_k,
        value=silhouette_values.index(max(silhouette_values)) + 2,  # Suggest K with best Silhouette
        help="Wybierz K na podstawie analizy z Metody Łokcia i Analizy Sylwetkowej."
    )

    # Run final model
    final_kmeans = KMeans(n_clusters=final_k, init='k-means++', n_init=10, random_state=42)
    final_kmeans.fit(X_scaled)
    cluster_labels = final_kmeans.labels_

    # Add results to original DataFrame
    df_results = X_raw.copy()
    df_results['cluster'] = cluster_labels

    st.subheader(f"Wizualizacja Klastrów (K={final_k})")

    # Visualization 2D/3D
    if len(selected_features) == 2:
        fig_clusters = px.scatter(
            df_results,
            x=selected_features[0],
            y=selected_features[1],
            color='cluster',
            color_continuous_scale=px.colors.qualitative.Vivid,
            title="Wyniki Klastrowania (Dane Oryginalne)"
        )
        st.plotly_chart(fig_clusters, use_container_width=True)
    elif len(selected_features) == 3:
        st.info("Tworzysz klastry w 3D. Możesz obracać poniższy wykres.")
        fig_clusters = px.scatter_3d(
            df_results,
            x=selected_features[0],
            y=selected_features[1],
            z=selected_features[2],
            color='cluster',
            color_continuous_scale=px.colors.qualitative.Vivid,
            title="Wyniki Klastrowania (Dane Oryginalne)"
        )
        st.plotly_chart(fig_clusters, use_container_width=True)
    else:
        st.warning(f"Klastrowanie przeprowadzono na {len(selected_features)} cechach. Wizualizacja jest możliwa tylko dla 2 lub 3 cech. Pokazuję pierwsze dwie:")
        fig_clusters = px.scatter(
            df_results,
            x=selected_features[0],
            y=selected_features[1],
            color='cluster',
            color_continuous_scale=px.colors.qualitative.Vivid,
            title=f"Wizualizacja 2D (użyto {len(selected_features)} cech)"
        )
        st.plotly_chart(fig_clusters, use_container_width=True)

    # === SECTION 5: INTERPRETATION ===
    st.header("Charakterystyka Odkrytych Klastrów")
    st.markdown("Poniższa tabela pokazuje **średnie wartości** każdej cechy dla każdego klastra. Pomaga to zrozumieć i 'nazwać' (stworzyć persony) dla każdej z grup.")

    cluster_summary = df_results.groupby('cluster')[selected_features].mean()
    st.dataframe(cluster_summary.style.format("{:.2f}"))

    st.subheader("Podgląd Danych z Przypisanymi Klastrami")
    st.dataframe(df_results.head(20))
