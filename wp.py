import numpy as np
import pandas as pd
import streamlit as st

def wp_method(matrix, weights):
    weighted_matrix = np.power(matrix, weights)
    scores = np.prod(weighted_matrix, axis=1)
    return scores

def run_wp():
    st.header("Weighted Product (WP)")
    num_criteria = st.number_input(
        "Number of criteria", min_value=2, max_value=10, value=3, key="wp_num_criteria")
    num_alternatives = st.number_input(
        "Number of alternatives", min_value=2, max_value=10, value=3, key="wp_num_alternatives")

    if num_criteria and num_alternatives:
        st.subheader("Decision Matrix")
        decision_matrix = pd.DataFrame(np.zeros((num_alternatives, num_criteria)),
                                       columns=[f"Criterion {i+1}" for i in range(num_criteria)])
        decision_matrix = st.data_editor(decision_matrix, key="wp_decision_matrix")

        st.subheader("Weights")
        weights = pd.DataFrame(np.ones((1, num_criteria)),
                               columns=decision_matrix.columns)
        weights = st.data_editor(weights, key="wp_weights")

        st.subheader("Criteria Types")

        # Menyimpan tipe kriteria 1 Benefit, -1 Cost
        criteria_types = []

        for i in range(num_criteria):
            criteria_type = st.selectbox(
                f"Select for Criterion {i+1}", ["Benefit", "Cost"], key=f"wp_criteria_type_{i}")
            criteria_types.append(1 if criteria_type == "Benefit" else -1)

        if st.button("Calculate WP", key="wp_calculate"):
            try:
                decision_matrix_values = decision_matrix.to_numpy()
                weights_values = weights.to_numpy().flatten()

                if np.all(decision_matrix_values == 0):
                    st.error("Inputs must be non-zero")
                elif np.all(weights_values == 1):
                    st.error("Please readjust the weights")
                else:
                    # Menyesuaikan bobot berdasarkan input Cost/Benefit
                    for i in range(num_criteria):
                        if criteria_types[i] == -1: 
                            weights_values[i] = -abs(weights_values[i])

                    if len(weights_values) != num_criteria:
                        st.error("Number of weights must match number of criteria")
                    else:
                        # Langkah 1: Hitung skor WP (vektor S)
                        scores = wp_method(decision_matrix_values, weights_values)

                        # Langkah 2: Hitung vektor V
                        total_sum = np.sum(scores)
                        V = scores / total_sum

                        # Langkah 3: Peringkat berdasarkan vektor V
                        rankings = np.argsort(-V) + 1 

                        # Menampilkan hasill
                        results_df = pd.DataFrame({
                            'Alternative': [f'Alternative {i+1}' for i in range(num_alternatives)],
                            'S (WP Score)': scores,
                            'V (Normalized Score)': V,
                            'Ranking': rankings
                        }).sort_values(by='Ranking')

                        st.write(results_df)

                        # Alternatif terbaik
                        best_alternative = results_df.iloc[0]["Alternative"]
                        st.markdown(f"**Conclusion: The best alternative is {best_alternative}**")

            except Exception as e:
                st.error(f"Error: {e}")
