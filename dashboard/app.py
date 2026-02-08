import streamlit as st

st.set_page_config(page_title="EvasionBench Demo")
st.title("EvasionBench — Evasion Detector Demo")

st.markdown("Enter a question and answer to get a prediction (demo stub).")

question = st.text_area("Question")
answer = st.text_area("Answer")

if st.button("Analyze"):
    # Placeholder prediction
    st.success("Prediction: direct (confidence 0.9)")
    st.write("Explanation: demo placeholder")
