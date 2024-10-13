import streamlit as st
from saw import run_saw
from wp import run_wp
from topsis import run_topsis
from ahp import run_ahp

from style import markdown_styling

st.markdown(markdown_styling, unsafe_allow_html=True)

def main():
    st.title("Decision Support System (DSS) Calculator")

    with st.sidebar.expander("Choose a Method", expanded=True):
        menu = st.radio("Methods", ["SAW", "WP", "TOPSIS", "AHP"])

    if menu == "SAW":
        run_saw()
    elif menu == "WP":
        run_wp()
    elif menu == "TOPSIS":
        run_topsis()
    elif menu == "AHP":
        run_ahp()

if __name__ == "__main__":
    main()
