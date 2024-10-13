import numpy as np
import pandas as pd
import streamlit as st

def saw_method(decision_matrix, weights, criteria_types):
    normalized_matrix = np.zeros_like(decision_matrix)
    for i in range(decision_matrix.shape[1]):
        if criteria_types[i] == 1: 
            normalized_matrix[:, i] = decision_matrix[:, i] / np.max(decision_matrix[:, i])
        else: 
            normalized_matrix[:, i] = np.min(decision_matrix[:, i]) / decision_matrix[:, i]
    
    weighted_matrix = normalized_matrix * weights
    scores = np.sum(weighted_matrix, axis=1)
    
    return normalized_matrix, weighted_matrix, scores

def run_saw():
    st.header("Simple Additive Weighting (SAW)")
    num_criteria = st.number_input("Number of criteria", min_value=2, max_value=10, value=3, key="saw_num_criteria")
    num_alternatives = st.number_input("Number of alternatives", min_value=2, max_value=10, value=3, key="saw_num_alternatives")

    if num_criteria and num_alternatives:
        st.subheader("Decision Matrix")
        decision_matrix = pd.DataFrame(np.zeros((num_alternatives, num_criteria)), 
                                    columns=[f"Criterion {i+1}" for i in range(num_criteria)])
        decision_matrix = st.data_editor(decision_matrix, key="saw_decision_matrix")

        st.subheader("Weights")
        weights = pd.DataFrame(np.ones((1, num_criteria)), columns=decision_matrix.columns)
        weights = st.data_editor(weights, key="saw_weights")

        st.subheader("Criteria Types")
        criteria_types = []
        for i in range(num_criteria):
            criteria_type = st.selectbox(f"Select for Criterion {i+1}", ["Benefit", "Cost"], key=f"saw_criteria_type_{i}")
            criteria_types.append(1 if criteria_type == "Benefit" else 0)

        if st.button("Calculate SAW", key="saw_calculate"):
            try:
                decision_matrix_values = decision_matrix.to_numpy()
                weights_values = weights.to_numpy().flatten()
                criteria_types_values = np.array(criteria_types)

                if np.all(decision_matrix_values == 0):
                    st.error("Inputs must be non-zero")
                elif np.all(weights_values == 1):
                    st.error("Please readjust the weights")
                else:
                    # Normalisasi
                    normalized_matrix, weighted_matrix, scores = saw_method(decision_matrix_values, weights_values, criteria_types_values)
                    
                    # Menampilkan langkah-langkah perhitungan
                    st.subheader("Step 1: Normalized Decision Matrix")
                    st.write(pd.DataFrame(normalized_matrix, columns=decision_matrix.columns))

                    st.subheader("Step 2: Weighted Normalized Decision Matrix")
                    st.write(pd.DataFrame(weighted_matrix, columns=decision_matrix.columns))
                    
                    # Menambahkan kolom rank berdasarkan score dan sorting
                    results_df = pd.DataFrame({"Alternatives": [f"Alternative {i+1}" for i in range(num_alternatives)], 
                                               "Scores": scores})
                    results_df["Rank"] = results_df["Scores"].rank(ascending=False, method='min')
                    results_df = results_df.sort_values(by="Rank")  # Sorting berdasarkan Rank
                    
                    st.subheader("SAW Final Scores with Ranking (Sorted)")
                    st.write(results_df)

                    # Alternatif terbaik
                    best_alternative = results_df.iloc[0]["Alternatives"]
                    st.markdown(f"**Conclusion: The best alternative is {best_alternative}**")

            except Exception as e:
                st.error(f"Error: {e}")