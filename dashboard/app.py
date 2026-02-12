import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import load_model

st.set_page_config(page_title="EvasionBench Demo", page_icon="🔍")


# Load model at startup (cached)
@st.cache_resource
def get_predictor():
    return load_model("boosting")


predictor = get_predictor()

st.title("EvasionBench — Evasion Detector Demo")

st.markdown(
    """
Enter a question and answer pair to detect evasion patterns.

**Labels:**
- **direct**: Answer directly addresses the question
- **intermediate**: Partial answer or some evasion
- **fully_evasive**: Answer completely evades the question
"""
)

question = st.text_area("Question", placeholder="e.g., What is the capital of France?")
answer = st.text_area("Answer", placeholder="e.g., Paris is the capital of France.")

if st.button("Analyze", type="primary"):
    if not question or not answer:
        st.warning("Please enter both a question and an answer.")
    else:
        with st.spinner("Analyzing..."):
            result = predictor.predict_single(question, answer)

        # Display prediction with color coding
        prediction = result["prediction"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        # Color code based on prediction
        if prediction == "direct":
            st.success(f"✅ **Prediction:** {prediction} (confidence: {confidence:.2%})")
        elif prediction == "fully_evasive":
            st.error(f"🚫 **Prediction:** {prediction} (confidence: {confidence:.2%})")
        else:
            st.warning(
                f"⚠️ **Prediction:** {prediction} (confidence: {confidence:.2%})"
            )

        # Show probability breakdown
        st.markdown("### Probability Breakdown")
        for label in ["direct", "intermediate", "fully_evasive"]:
            prob = probabilities.get(label, 0)
            st.progress(prob, text=f"{label}: {prob:.2%}")

st.markdown("---")
st.caption("Powered by tree boosting model (F1: 0.45) | EvasionBench Research Project")
