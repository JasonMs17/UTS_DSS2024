import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

markdown_styling = """
<style>

[data-testid="stMain"] {
    background-image: url("https://wallpapercave.com/wp/wp2197012.jpg");
}

[data-testid="stSidebarContent"] {
    background-image: url("https://i.pinimg.com/736x/d7/a4/cd/d7a4cdb7da730fbd335a8627bb863f60.jpg");
}

[data-testid="stVerticalBlock"] {
    color: #fff !important;
    padding: 30px;
    border-radius: 10px;
}

h1, h2, h3, p {
    color: #fff;
}

[data-testid="stButton"] p {
    color: #949494 !important;
}

[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}

[data-testid="stNumberInputContainer"] {
    width: 90%;
}

[data-testid="stSelectbox"] {
    width: 90% !important;
}

[data-testid="stAlertContainer"] {
    width: 90% !important;
}
</style>
"""

st.markdown(markdown_styling, unsafe_allow_html=True)

# 1. SAW (Simple Additive Weighting)
def saw_method(decision_matrix, weights, criteria_types):
    # Step 1: Normalisasi Matriks Keputusan
    normalized_matrix = np.zeros_like(decision_matrix)
    for i in range(decision_matrix.shape[1]):
        if criteria_types[i] == 1:  # Benefit
            normalized_matrix[:, i] = decision_matrix[:, i] / np.max(decision_matrix[:, i])
        else:  # Cost
            normalized_matrix[:, i] = np.min(decision_matrix[:, i]) / decision_matrix[:, i]
    
    # Step 2: Mengalikan normalisasi dengan bobot
    weighted_matrix = normalized_matrix * weights
    
    # Step 3: Menghitung skor akhir dengan menjumlahkan setiap baris
    scores = np.sum(weighted_matrix, axis=1)
    
    return normalized_matrix, weighted_matrix, scores

# 2. WP (Weighted Product)
def wp_method(matrix, weights):
    weighted_matrix = np.power(matrix, weights)
    scores = np.prod(weighted_matrix, axis=1)
    return scores

# 3. TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
def topsis_method(matrix, weights, criteria_types):
    # Step 1: Normalisasi Matriks Keputusan
    normalized_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))

    # Step 2: Matriks Ternormalisasi dan Terbobot
    weighted_matrix = normalized_matrix * weights

    # Step 3: Menentukan Solusi Ideal Positif dan Negatif (Berdasarkan tipe kriteria Benefit/Cost)
    ideal_best = np.zeros(weighted_matrix.shape[1])
    ideal_worst = np.zeros(weighted_matrix.shape[1])

    for i in range(weighted_matrix.shape[1]):
        if criteria_types[i] == 1:  # Benefit: nilai terbaik adalah maksimum, terburuk adalah minimum
            ideal_best[i] = np.max(weighted_matrix[:, i])
            ideal_worst[i] = np.min(weighted_matrix[:, i])
        else:  # Cost: nilai terbaik adalah minimum, terburuk adalah maksimum
            ideal_best[i] = np.min(weighted_matrix[:, i])
            ideal_worst[i] = np.max(weighted_matrix[:, i])

    # Step 4: Menghitung jarak ke solusi ideal positif (D+) dan negatif (D-)
    distance_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))  # D+
    distance_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))  # D-

    # Step 5: Menghitung skor akhir (Kedekatan dengan solusi ideal negatif)
    scores = distance_worst / (distance_best + distance_worst)
    
    return normalized_matrix, weighted_matrix, ideal_best, ideal_worst, distance_best, distance_worst, scores

# 4. AHP (Analytic Hierarchy Process)
def ahp_method(criteria_matrix):
    # Step 1: Normalize the matrix
    criteria_sum = criteria_matrix.sum(axis=0)
    normalized_matrix = criteria_matrix / criteria_sum

    # Step 2: Calculate the priority vector (mean of normalized rows)
    priority_vector = normalized_matrix.mean(axis=1)

    return priority_vector, normalized_matrix, criteria_sum

def calculate_consistency(criteria_matrix, priority_vector):
    # Step 3: Calculate λ_max
    weighted_sum_vector = np.dot(criteria_matrix, priority_vector)
    lambda_max = np.mean(weighted_sum_vector / priority_vector)

    # Step 4: Calculate Consistency Index (CI)
    n = len(criteria_matrix)
    CI = (lambda_max - n) / (n - 1)

    # Step 5: Calculate Consistency Ratio (CR)
    RI_values = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_values.get(n, 1.12)  # RI for n = 5 is 1.12
    CR = CI / RI

    return lambda_max, CI, CR

# Streamlit layout
st.title("Decision Support System (DSS) Calculator")

# Tab layout
with st.sidebar.expander("Choose a Method"):
    menu = st.radio("Methods", ["SAW", "WP", "TOPSIS", "AHP"])

# 1. SAW Tab
if menu == "SAW":
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

        st.subheader("Criteria Types (Biaya = Cost, Keuntungan = Benefit)")
        criteria_types = []
        for i in range(num_criteria):
            criteria_type = st.selectbox(f"Select for Criterion {i+1}", ["Keuntungan", "Biaya"], key=f"saw_criteria_type_{i}")
            criteria_types.append(1 if criteria_type == "Keuntungan" else 0)

        if st.button("Calculate SAW", key="saw_calculate"):
            try:
                decision_matrix_values = decision_matrix.to_numpy()
                weights_values = weights.to_numpy().flatten()
                criteria_types_values = np.array(criteria_types)

                # Validate if the decision matrix has non-zero values and weights are not default
                if np.all(decision_matrix_values == 0):
                    st.error("Please fill in the Decision Matrix with meaningful values (non-zero).")
                elif np.all(weights_values == 1):
                    st.error("Please adjust the weights to reflect their importance, not all weights should be 1.")
                else:
                    # Step 1: Normalisasi
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

                    # Kesimpulan alternatif terbaik
                    best_alternative = results_df.iloc[0]["Alternatives"]
                    st.markdown(f"**Conclusion: The best alternative is {best_alternative}**")

            except Exception as e:
                st.error(f"Error: {e}")

if menu == "WP":
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

        st.subheader("Tipe Kriteria (Keuntungan/Biaya)")

        # List to store criteria types: 1 for Keuntungan, -1 for Biaya
        criteria_types = []

        # Loop through criteria and create a selectbox for each
        for i in range(num_criteria):
            criteria_type = st.selectbox(
                f"Pilih untuk Kriteria {i+1}", ["Keuntungan", "Biaya"], key=f"wp_criteria_type_{i}")
            criteria_types.append(1 if criteria_type == "Keuntungan" else -1)

        if st.button("Calculate WP", key="wp_calculate"):
            try:
                decision_matrix_values = decision_matrix.to_numpy()
                weights_values = weights.to_numpy().flatten()

                # Check if the decision matrix has non-zero values and weights are not default
                if np.all(decision_matrix_values == 0):
                    st.error("Please fill in the Decision Matrix with meaningful values (non-zero).")
                elif np.all(weights_values == 1):
                    st.error("Please adjust the weights to reflect their importance, not all weights should be 1.")
                else:
                    # Adjust weights based on Keuntungan/Biaya
                    for i in range(num_criteria):
                        if criteria_types[i] == -1:  # If "Biaya", convert weight to negative
                            weights_values[i] = -abs(weights_values[i])

                    if len(weights_values) != num_criteria:
                        st.error("Number of weights must match number of criteria")
                    else:
                        # Step 1: Calculate WP scores (S vector)
                        scores = wp_method(decision_matrix_values, weights_values)

                        # Step 2: Calculate V vector (normalized scores)
                        total_sum = np.sum(scores)
                        V = scores / total_sum

                        # Step 3: Ranking based on V vector
                        rankings = np.argsort(-V) + 1  # Sort in descending order, +1 for ranking 1-based

                        # Display results in a combined table
                        results_df = pd.DataFrame({
                            'Alternative': [f'Alternative {i+1}' for i in range(num_alternatives)],
                            'S (WP Score)': scores,
                            'V (Normalized Score)': V,
                            'Ranking': rankings
                        }).sort_values(by='Ranking')

                        # Show combined results in one table
                        st.write(results_df)

                        # Conclusion for the best alternative
                        best_alternative = results_df.iloc[0]["Alternative"]
                        st.markdown(f"**Conclusion: The best alternative is {best_alternative}**")

            except Exception as e:
                st.error(f"Error: {e}")

# TOPSIS Tab
if menu == "TOPSIS":
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

        st.subheader("Criteria Types (Biaya = Cost, Keuntungan = Benefit)")
        criteria_types = []
        for i in range(num_criteria):
            criteria_type = st.selectbox(f"Select for Criterion {i+1}", ["Keuntungan", "Biaya"], key=f"topsis_criteria_type_{i}")
            criteria_types.append(1 if criteria_type == "Keuntungan" else 0)

        if st.button("Calculate TOPSIS", key="topsis_calculate"):
            try:
                decision_matrix_values = decision_matrix.to_numpy()
                weights_values = weights.to_numpy().flatten()
                criteria_types_values = np.array(criteria_types)

                # Check if decision matrix and weights are properly filled
                if np.all(decision_matrix_values == 0):
                    st.error("Please fill in the Decision Matrix with meaningful values (non-zero).")
                elif np.all(weights_values == 1):
                    st.error("Please adjust the weights to reflect their importance, not all weights should be 1.")
                else:
                    # Step by step calculations
                    normalized_matrix, weighted_matrix, ideal_best, ideal_worst, distance_best, distance_worst, scores = topsis_method(
                        decision_matrix_values, weights_values, criteria_types_values
                    )

                    # Step 1: Tabel Keputusan Ternormalisasi
                    st.subheader("Step 1: Tabel Keputusan Ternormalisasi")
                    st.write(pd.DataFrame(normalized_matrix, columns=decision_matrix.columns))

                    # Step 2: Tabel Keputusan Ternormalisasi dan Terbobot
                    st.subheader("Step 2: Tabel Keputusan Ternormalisasi dan Terbobot")
                    st.write(pd.DataFrame(weighted_matrix, columns=decision_matrix.columns))

                    # Step 3: Tabel Solusi Ideal Positif dan Solusi Ideal Negatif
                    st.subheader("Step 3: Solusi Ideal Positif dan Solusi Ideal Negatif")
                    ideal_df = pd.DataFrame({
                        "Solusi Ideal Positif": ideal_best,
                        "Solusi Ideal Negatif": ideal_worst
                    }, index=[f"Criterion {i+1}" for i in range(num_criteria)])
                    st.write(ideal_df)

                    # Step 4: Tabel Nilai Separation Measure D+ dan D-
                    st.subheader("Step 4: Nilai Separation Measure D+ dan D-")
                    separation_df = pd.DataFrame({
                        "D+": distance_best,
                        "D-": distance_worst
                    }, index=[f"Alternative {i+1}" for i in range(num_alternatives)])
                    st.write(separation_df)

                    # Step 5: Hasil Perhitungan Kedekatan dengan Nilai Preferensi dan Ranking
                    st.subheader("Step 5: Nilai Preferensi dan Ranking")
                    preference_df = pd.DataFrame({
                        "Alternatives": [f"Alternative {i+1}" for i in range(num_alternatives)],
                        "Scores": scores
                    })
                    
                    # Sorting based on scores in descending order
                    preference_df['Rank'] = preference_df['Scores'].rank(ascending=False, method='min')
                    preference_df = preference_df.sort_values(by='Scores', ascending=False)

                    # Display the sorted table
                    st.write(preference_df)

                    # Conclusion for the best alternative
                    best_alternative = preference_df.iloc[0]["Alternatives"]
                    st.markdown(f"**Conclusion: The best alternative is {best_alternative}**")

            except Exception as e:
                st.error(f"Error: {e}")

# AHP Tab
if menu == "AHP":
    st.header("Analytic Hierarchy Process (AHP)")

    # AHP Comparison Values Guide
    st.write("""
    ### Panduan Nilai Perbandingan AHP
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

    num_criteria = st.number_input("Number of criteria", min_value=2, max_value=10, value=5, key="ahp_num_criteria")
    num_alternatives = st.number_input("Number of alternatives", min_value=2, max_value=10, value=3, key="ahp_num_alternatives")

    if num_criteria and num_alternatives:
        # Pairwise comparison matrix for criteria
        st.subheader("Pairwise Comparison Matrix (Criteria)")
        criteria_matrix = pd.DataFrame(np.ones((num_criteria, num_criteria)),
                                    columns=[f"Criterion {i + 1}" for i in range(num_criteria)],
                                    index=[f"Criterion {i + 1}" for i in range(num_criteria)])
        criteria_matrix = st.data_editor(criteria_matrix, key="ahp_criteria_matrix")

        # Pairwise comparison matrices for alternatives
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
                # Ensure the matrices are filled in properly
                criteria_matrix_values = criteria_matrix.to_numpy()

                if np.all(criteria_matrix_values == 1):
                    st.error("Please fill in the Pairwise Comparison Matrix with meaningful values.")
                else:
                    # Calculate priority vector for criteria
                    priority_vector, normalized_matrix, criteria_sum = ahp_method(criteria_matrix_values)

                    # Prepare DataFrame for display
                    results_df = pd.DataFrame(index=[f'Kriteria {i + 1}' for i in range(num_criteria)],
                                            columns=['Kode'] + [f'A0{i + 1}' for i in range(num_alternatives)] + ['Rata-rata'])

                    # Fill in the DataFrame with calculated values for alternatives
                    for i in range(num_criteria):
                        results_df.loc[f'Kriteria {i + 1}', 'Kode'] = 1  # Assuming Kode is 1 for each criterion
                        alt_matrix_values = alternative_results[i].to_numpy()  # Fetch the corresponding alt_matrix
                        alt_priority_vector, normalized_alt_matrix, _ = ahp_method(alt_matrix_values)  # Calculate priority vector for alternatives
                        
                        # Create a DataFrame for the current criterion's normalized values
                        current_criterion_df = pd.DataFrame(normalized_alt_matrix, index=[f'A0{j + 1}' for j in range(num_alternatives)],
                                                            columns=[f'A0{j + 1}' for j in range(num_alternatives)])
                        current_criterion_df['Rata-rata'] = np.mean(normalized_alt_matrix, axis=1)  # Calculate average

                        # Display each criterion's normalized results
                        st.write(f"Kriteria {i + 1}")
                        st.write(current_criterion_df)

                        for j in range(num_alternatives):
                            results_df.loc[f'Kriteria {i + 1}', f'A0{j + 1}'] = round(alt_priority_vector[j], 10)  # Normalized value
                        
                        results_df.loc[f'Kriteria {i + 1}', 'Rata-rata'] = round(priority_vector[i], 10)  # Average of the priority vector for the criterion

                    # Add 'Jumlah' row
                    results_df.loc['Jumlah'] = ['Jumlah'] + [1] * (num_alternatives + 1)

                    # Create a new DataFrame for Normalized Priority Vector
                    normalized_priority_df = pd.DataFrame(index=[f'C0{i + 1}' for i in range(num_criteria)],
                                                        columns=[f'C0{j + 1}' for j in range(num_criteria)] + ['Rata-rata (W)'])

                    # Fill the normalized priority vector DataFrame
                    for i in range(num_criteria):
                        for j in range(num_criteria):
                            normalized_priority_df.loc[f'C0{i + 1}', f'C0{j + 1}'] = normalized_matrix[i, j]  # Using normalized_matrix
                        normalized_priority_df.loc[f'C0{i + 1}', 'Rata-rata (W)'] = round(priority_vector[i], 10)

                    # Display the normalized priority vector
                    st.subheader("Normalized Priority Vector")
                    st.write(normalized_priority_df)

                    # Calculate λ_max, CI, CR
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

                    # Step 5: Final Ranking (average across criteria)
                    st.subheader("Step 5: Final Ranking")

                    # Calculate the final scores using the average priority vector for criteria
                    final_scores = np.zeros(num_alternatives)

                    for i in range(num_criteria):
                        # Use priority_vector[i] to weight the alternatives for criterion i
                        final_scores += np.array(results_df.iloc[i, 1:num_alternatives + 1].astype(float)) * priority_vector[i]

                    # Create DataFrame for final results
                    rankings = np.argsort(-final_scores) + 1  # Rank from highest to lowest

                    final_results_df = pd.DataFrame({
                        'Alternative': [f'Alternative {i + 1}' for i in range(num_alternatives)],
                        'Final Score': final_scores,
                        'Ranking': rankings
                    }).sort_values(by='Ranking')

                    st.write(final_results_df)

                    # Conclusion for the best alternative
                    best_alternative = final_results_df.iloc[0]["Alternative"]
                    st.markdown(f"**Conclusion: The best alternative is {best_alternative}**")

            except Exception as e:
                st.error(f"Error: {e}")