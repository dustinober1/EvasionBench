"""Results Explorer - Visualize EvasionBench analysis artifacts."""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Results Explorer", page_icon="📊", layout="wide")

st.title("📊 Results Explorer")
st.markdown("Explore model performance, explainability, and label diagnostics.")

# Define paths
ARTIFACTS_ROOT = ROOT / "artifacts"
MODELS_PATH = ARTIFACTS_ROOT / "models" / "phase5"
XAI_PATH = ARTIFACTS_ROOT / "explainability" / "phase6"
DIAGNOSTICS_PATH = ARTIFACTS_ROOT / "diagnostics" / "phase6"

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Model Comparison",
        "🔍 Explainability (SHAP)",
        "🎯 Sample Predictions",
        "⚠️ Label Diagnostics",
    ]
)

# Tab 1: Model Comparison
with tab1:
    st.header("Model Performance Comparison")

    # Load metrics from run summary
    summary_path = MODELS_PATH / "run_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        # Convert to DataFrame
        df_metrics = pd.DataFrame(summary).T
        df_metrics = df_metrics.round(4)
        df_metrics = df_metrics.sort_values("f1_macro", ascending=False)

        # Display metrics table
        st.subheader("Overall Metrics")
        st.dataframe(
            df_metrics.style.highlight_max(axis=0, color="lightgreen"),
            use_container_width=True,
        )

        # Create bar charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("F1 Score Comparison")
            st.bar_chart(df_metrics[["f1_macro"]])

        with col2:
            st.subheader("Accuracy Comparison")
            st.bar_chart(df_metrics[["accuracy"]])

        # Show best model
        best_model = df_metrics.index[0]
        best_f1 = df_metrics.loc[best_model, "f1_macro"]
        st.success(f"🏆 **Best Model:** {best_model} (F1: {best_f1:.4f})")

        # Per-class comparison if available
        per_class_path = (
            MODELS_PATH / "model_comparison" / "per_class_f1_comparison.csv"
        )
        if per_class_path.exists():
            st.subheader("Per-Class F1 Scores")
            df_per_class = pd.read_csv(per_class_path, index_col=0)
            st.dataframe(
                df_per_class.style.highlight_max(axis=1, color="lightgreen"),
                use_container_width=True,
            )
    else:
        st.warning("Model comparison summary not found.")

# Tab 2: SHAP Explainability
with tab2:
    st.header("SHAP Explainability Analysis")

    # Model selector
    xai_models = [
        d.name for d in XAI_PATH.iterdir() if d.is_dir() and d.name != "transformer"
    ]
    if xai_models:
        selected_model = st.selectbox("Select Model", xai_models, index=0)

        model_xai_path = XAI_PATH / selected_model

        # Display SHAP summary image
        shap_image = model_xai_path / "shap_summary.png"
        if shap_image.exists():
            st.subheader("SHAP Summary Plot")
            st.image(str(shap_image), use_container_width=True)

        # Load SHAP summary JSON
        shap_summary_json = model_xai_path / "shap_summary.json"
        if shap_summary_json.exists():
            with open(shap_summary_json) as f:
                shap_data = json.load(f)

            st.subheader("Top Features by Importance")

            # Convert to DataFrame
            if "feature_importance" in shap_data:
                df_importance = pd.DataFrame(shap_data["feature_importance"]).T
                df_importance = df_importance.sort_values(
                    "mean_abs_shap", ascending=False
                ).head(20)
                st.bar_chart(df_importance["mean_abs_shap"])

        # Load sample explanations
        shap_samples = model_xai_path / "shap_samples.json"
        if shap_samples.exists():
            with open(shap_samples) as f:
                samples = json.load(f)

            st.subheader("Sample Explanations")
            st.markdown("Top features influencing predictions for sample instances.")

            for i, sample in enumerate(samples[:3], 1):
                with st.expander(
                    f"Sample {i}: {sample.get('true_label', 'unknown')} → {sample.get('predicted_label', 'unknown')}"
                ):
                    top_features = sample.get("top_features", [])
                    if top_features:
                        df_sample = pd.DataFrame(top_features)
                        st.dataframe(df_sample, use_container_width=True)
    else:
        st.warning("No explainability artifacts found.")

# Tab 3: Sample Predictions
with tab3:
    st.header("Sample Predictions Analysis")

    st.markdown(
        """
    This section would show example predictions from the test set, including:
    - Correctly classified examples (by class)
    - Misclassified examples with explanations
    - High-confidence vs low-confidence predictions
    """
    )

    # Load classification reports
    available_models = [
        d.name
        for d in MODELS_PATH.iterdir()
        if d.is_dir() and (d / "classification_report.json").exists()
    ]

    if available_models:
        selected_model = st.selectbox(
            "Select Model", available_models, key="sample_model"
        )

        report_path = MODELS_PATH / selected_model / "classification_report.json"
        with open(report_path) as f:
            report = json.load(f)

        st.subheader("Classification Report")

        # Show per-class metrics
        classes = ["direct", "intermediate", "fully_evasive"]
        class_data = []
        for cls in classes:
            if cls in report:
                class_data.append(
                    {
                        "Class": cls,
                        "Precision": report[cls]["precision"],
                        "Recall": report[cls]["recall"],
                        "F1-Score": report[cls]["f1-score"],
                        "Support": report[cls]["support"],
                    }
                )

        if class_data:
            df_report = pd.DataFrame(class_data)
            st.dataframe(df_report, use_container_width=True)

        # Load confusion matrix
        confusion_path = MODELS_PATH / selected_model / "confusion_matrix.json"
        if confusion_path.exists():
            with open(confusion_path) as f:
                cm_data = json.load(f)

            st.subheader("Confusion Matrix")
            df_cm = pd.DataFrame(
                cm_data["matrix"], index=cm_data["labels"], columns=cm_data["labels"]
            )
            st.dataframe(df_cm, use_container_width=True)
    else:
        st.info("No sample predictions available.")

# Tab 4: Label Diagnostics
with tab4:
    st.header("Label Quality Diagnostics")

    st.markdown(
        """
    Analysis of potential label quality issues in the dataset.
    """
    )

    # Load diagnostics summary
    diag_summary_path = DIAGNOSTICS_PATH / "label_diagnostics_summary.json"
    if diag_summary_path.exists():
        with open(diag_summary_path) as f:
            diag_summary = json.load(f)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Examples", diag_summary.get("total_examples", "N/A"))
        with col2:
            st.metric("Suspect Examples", diag_summary.get("suspect_count", "N/A"))
        with col3:
            st.metric(
                "Near-Duplicate Pairs", diag_summary.get("near_duplicate_pairs", "N/A")
            )

    # Load suspect examples
    suspect_path = DIAGNOSTICS_PATH / "suspect_examples.csv"
    if suspect_path.exists() and suspect_path.stat().st_size > 50:
        st.subheader("Suspect Examples")
        df_suspect = pd.read_csv(suspect_path)
        if not df_suspect.empty:
            st.dataframe(df_suspect, use_container_width=True)
        else:
            st.success("✅ No suspect examples found!")
    else:
        st.success("✅ No suspect examples found!")

    # Load near-duplicates
    near_dup_path = DIAGNOSTICS_PATH / "near_duplicate_pairs.csv"
    if near_dup_path.exists():
        st.subheader("Near-Duplicate Pairs")
        df_near_dup = pd.read_csv(near_dup_path)
        if not df_near_dup.empty:
            st.dataframe(df_near_dup.head(20), use_container_width=True)
            st.caption(f"Showing 20 of {len(df_near_dup)} pairs")
        else:
            st.success("✅ No near-duplicate pairs found!")

    # Load diagnostics report
    report_path = DIAGNOSTICS_PATH / "label_diagnostics_report.md"
    if report_path.exists():
        st.subheader("Diagnostics Report")
        with open(report_path) as f:
            report_content = f.read()
        st.markdown(report_content)

st.markdown("---")
st.caption("EvasionBench Research Project — Results Explorer")
