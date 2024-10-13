import numpy as np
import pandas as pd
import streamlit as st

def topsis_method(matrix, weights, criteria_types):
    normalized_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))

    weighted_matrix = normalized_matrix * weights

    ideal_best = np.zeros(weighted_matrix.shape[1])
    ideal_worst = np.zeros(weighted_matrix.shape[1])

    for i in range(weighted_matrix.shape[1]):
        if criteria_types[i] == 1:  
            ideal_best[i] = np.max(weighted_matrix[:, i])
            ideal_worst[i] = np.min(weighted_matrix[:, i])
        else:  
            ideal_best[i] = np.min(weighted_matrix[:, i])
            ideal_worst[i] = np.max(weighted_matrix[:, i])

    distance_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))  
    distance_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))  

    scores = distance_worst / (distance_best + distance_worst)
    
    return normalized_matrix, weighted_matrix, ideal_best, ideal_worst, distance_best, distance_worst, scores

def run_topsis():
    st.header("Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS)")
    num_criteria = st.number_input("Number of criteria", min_value=2, max_value=10, value=3, key="topsis_num_criteria")
    num_alternatives = st.number_input("Number of alternatives", min_value=2, max_value=10, value=3, key="topsis_num_alternatives")

    if num_criteria and num_alternatives:
        st.subheader("Decision Matrix")
        decision_matrix = pd.DataFrame(np.zeros((num_alternatives, num_criteria)), 
                                       columns=[f"Criterion {i+1}" for i in range(num_criteria)])
        decision_matrix = st.data_editor(decision_matrix, key="topsis_decision_matrix")

        st.subheader("Weights")
        weights = pd.DataFrame(np.ones((1, num_criteria)), columns=decision_matrix.columns)
        weights = st.data_editor(weights, key="topsis_weights")

        st.subheader("Criteria Types")
        criteria_types = []
        for i in range(num_criteria):
            criteria_type = st.selectbox(f"Select for Criterion {i+1}", ["Benefit", "Cost"], key=f"topsis_criteria_type_{i}")
            criteria_types.append(1 if criteria_type == "Benefit" else 0)

        if st.button("Calculate TOPSIS", key="topsis_calculate"):
            try:
                decision_matrix_values = decision_matrix.to_numpy()
                weights_values = weights.to_numpy().flatten()
                criteria_types_values = np.array(criteria_types)

                if np.all(decision_matrix_values == 0):
                    st.error("Inputs must be non-zero")
                elif np.all(weights_values == 1):
                    st.error("Please readjust the weights")
                else:
                    normalized_matrix, weighted_matrix, ideal_best, ideal_worst, distance_best, distance_worst, scores = topsis_method(
                        decision_matrix_values, weights_values, criteria_types_values
                    )

                    # Step 1: Tabel Keputusan Ternormalisasi
                    st.subheader("Step 1: Normalized Decision Matrix")
                    st.write(pd.DataFrame(normalized_matrix, columns=decision_matrix.columns))

                    # Step 2: Tabel Keputusan Ternormalisasi dan Terbobot
                    st.subheader("Step 2: Normalized and Weighted Decision Matrix")
                    st.write(pd.DataFrame(weighted_matrix, columns=decision_matrix.columns))

                    # Step 3: Tabel Solusi Ideal Positif dan Solusi Ideal Negatif
                    st.subheader("Step 3: Positive and Negative Ideal Solutions")
                    ideal_df = pd.DataFrame({
                        "Positive Ideal Solution": ideal_best,
                        "Negative Ideal Solution": ideal_worst
                    }, index=[f"Criterion {i+1}" for i in range(num_criteria)])
                    st.write(ideal_df)

                    # Step 4: Tabel Nilai Separation Measure D+ dan D-
                    st.subheader("Step 4: Separation Measure Values D+ and D-")
                    separation_df = pd.DataFrame({
                        "D+": distance_best,
                        "D-": distance_worst
                    }, index=[f"Alternative {i+1}" for i in range(num_alternatives)])
                    st.write(separation_df)

                    # Step 5: Hasil Perhitungan Kedekatan dengan Nilai Preferensi dan Ranking
                    st.subheader("Step 5: Preference Values and Ranking")
                    preference_df = pd.DataFrame({
                        "Alternatives": [f"Alternative {i+1}" for i in range(num_alternatives)],
                        "Scores": scores
                    })
                    
                    preference_df['Rank'] = preference_df['Scores'].rank(ascending=False, method='min')
                    preference_df = preference_df.sort_values(by='Scores', ascending=False)

                    st.write(preference_df)

                    # Alternatif terbaik
                    best_alternative = preference_df.iloc[0]["Alternatives"]
                    st.markdown(f"**Conclusion: The best alternative is {best_alternative}**")

            except Exception as e:
                st.error(f"Error: {e}")
