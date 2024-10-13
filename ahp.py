import numpy as np
import pandas as pd
import streamlit as st

def ahp_method(criteria_matrix):
    criteria_sum = criteria_matrix.sum(axis=0)
    normalized_matrix = criteria_matrix / criteria_sum
    priority_vector = normalized_matrix.mean(axis=1)

    return priority_vector, normalized_matrix, criteria_sum

def calculate_consistency(criteria_matrix, priority_vector):
    # Menghitung λ_max (t)
    weighted_sum_vector = np.dot(criteria_matrix, priority_vector)
    lambda_max = np.mean(weighted_sum_vector / priority_vector)

    # Menghitung Nilai CI (Consistency Index)
    n = len(criteria_matrix)
    CI = (lambda_max - n) / (n - 1)

    # Menghitung Nilai CR (Consistency Ratio)
    RI_values = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_values.get(n)
    CR = CI / RI

    return lambda_max, CI, CR

def run_ahp():
    st.header("Analytic Hierarchy Process (AHP)")

    st.write(""" 
    | Kode | Nilai                                       |
    |------|---------------------------------------------|
    | 1    | Sama penting dengan                         |
    | 2    | Mendekati sedikit lebih penting dari        |
    | 3    | Sedikit lebih penting dari                  |
    | 4    | Mendekati lebih penting dari                |
    | 5    | Lebih penting dari                          |
    | 6    | Mendekati sangat penting dari               |
    | 7    | Sangat penting dari                         |
    | 8    | Mendekati mutlak dari                       |
    | 9    | Mutlak sangat penting dari                  |
    """)

    # Input jumlah kriteria dan alternatif
    num_criteria = st.number_input("Number of criteria", min_value=2, max_value=10, value=3, key="ahp_num_criteria")
    num_alternatives = st.number_input("Number of alternatives", min_value=2, max_value=10, value=3, key="ahp_num_alternatives")

    if num_criteria and num_alternatives:
        # Matriks perbandingan untuk kriteria
        st.subheader("Pairwise Comparison Matrix (Criteria)")
        criteria_matrix = pd.DataFrame(np.ones((num_criteria, num_criteria)),
                                    columns=[f"Criterion {i + 1}" for i in range(num_criteria)],
                                    index=[f"Criterion {i + 1}" for i in range(num_criteria)])
        criteria_matrix = st.data_editor(criteria_matrix, key="ahp_criteria_matrix")

        # Matriks perbandingan untuk alternatif
        st.subheader("Pairwise Comparison Matrices for Alternatives")
        alternative_results = []
        
        for k in range(num_criteria):
            st.write(f"Comparison Matrix for Criterion {k + 1}")
            alt_matrix = pd.DataFrame(np.ones((num_alternatives, num_alternatives)),
                                    columns=[f"Alternative {i + 1}" for i in range(num_alternatives)],
                                    index=[f"Alternative {i + 1}" for i in range(num_alternatives)])
            alt_matrix = st.data_editor(alt_matrix, key=f"alt_matrix_{k}")
            alternative_results.append(alt_matrix)

        if st.button("Calculate AHP", key="ahp_calculate"):
            try:
                criteria_matrix_values = criteria_matrix.to_numpy()

                if np.all(criteria_matrix_values == 1):
                    st.error("Please consider changing the values")
                else:
                    # Menghitung vektor prioritas kriteria
                    priority_vector, normalized_matrix, criteria_sum = ahp_method(criteria_matrix_values)

                    results_df = pd.DataFrame(index=[f'Kriteria {i + 1}' for i in range(num_criteria)],
                                            columns=['Kode'] + [f'A0{i + 1}' for i in range(num_alternatives)] + ['Rata-rata'])

                    for i in range(num_criteria):
                        results_df.loc[f'Kriteria {i + 1}', 'Kode'] = 1 
                        alt_matrix_values = alternative_results[i].to_numpy()  
                        # Menghitung vektor prioritas alternatif
                        alt_priority_vector, normalized_alt_matrix, _ = ahp_method(alt_matrix_values)  
                        
                        current_criterion_df = pd.DataFrame(normalized_alt_matrix, index=[f'A0{j + 1}' for j in range(num_alternatives)],
                                                            columns=[f'A0{j + 1}' for j in range(num_alternatives)])
                        current_criterion_df['Rata-rata'] = np.mean(normalized_alt_matrix, axis=1)  # Menghitung rata-rata

                        # Menampilkan hasil normalisasi untuk kriteria
                        st.write(f"Kriteria {i + 1}")
                        st.write(current_criterion_df)

                        for j in range(num_alternatives):
                            results_df.loc[f'Kriteria {i + 1}', f'A0{j + 1}'] = round(alt_priority_vector[j], 10)  
                        
                        results_df.loc[f'Kriteria {i + 1}', 'Rata-rata'] = round(priority_vector[i], 10)  

                    results_df.loc['Jumlah'] = ['Jumlah'] + [1] * (num_alternatives + 1)

                    normalized_priority_df = pd.DataFrame(index=[f'C0{i + 1}' for i in range(num_criteria)],
                                                        columns=[f'C0{j + 1}' for j in range(num_criteria)] + ['Rata-rata (W)'])

                    for i in range(num_criteria):
                        for j in range(num_criteria):
                            normalized_priority_df.loc[f'C0{i + 1}', f'C0{j + 1}'] = normalized_matrix[i, j]  
                        normalized_priority_df.loc[f'C0{i + 1}', 'Rata-rata (W)'] = round(priority_vector[i], 10)

                    # Menampilkan vektor prioritas normalisasi
                    st.subheader("Normalized Priority Vector")
                    st.write(normalized_priority_df)

                    # Menghitung λ_max, CI, CR
                    lambda_max, CI, CR = calculate_consistency(criteria_matrix_values, priority_vector)

                    st.subheader("Step 2: λ_max")
                    st.write(lambda_max)

                    st.subheader("Step 3: Consistency Index (CI)")
                    st.write(CI)

                    st.subheader("Step 4: Consistency Ratio (CR)")
                    st.write(f"Consistency Ratio (CR) = {CR:.4f}")
                    if CR < 0.1:
                        st.success("CR is acceptable (CR < 0.1).")
                    else:
                        st.warning("CR is not acceptable (CR ≥ 0.1). Consider reviewing your matrix.")

                    st.subheader("Step 5: Final Ranking")

                    final_scores = np.zeros(num_alternatives)

                    for i in range(num_criteria):
                        final_scores += np.array(results_df.iloc[i, 1:num_alternatives + 1].astype(float)) * priority_vector[i]

                    rankings = np.argsort(-final_scores) + 1  # Mengurutkan dari yang tertinggi ke terendah

                    final_results_df = pd.DataFrame({
                        'Alternative': [f'Alternative {i + 1}' for i in range(num_alternatives)],
                        'Final Score': final_scores,
                        'Ranking': rankings
                    }).sort_values(by='Ranking')

                    st.write(final_results_df)

                    # Alternatif terbaik
                    best_alternative = final_results_df.iloc[0]["Alternative"]
                    st.markdown(f"**Conclusion: The best alternative is {best_alternative}**")

            except Exception as e:
                st.error(f"Error: {e}")
