# EvasionBench Portfolio Improvement Plan

## Two Ideas to Improve EvasionBench Portfolio Project

### 1. **Connect the Dashboard and API to Real Models**

Currently both [`dashboard/app.py`](../dashboard/app.py:11) and [`api/main.py`](../api/main.py:27) are placeholder stubs returning hardcoded predictions. This significantly limits the portfolio's impact.

**Implementation path:**
- Load the best-performing model (currently [`tree_boosting`](../artifacts/models/phase5/tree_boosting/metrics.json) with F1=0.44) at startup
- Create a model loading utility in `src/inference.py` that handles:
  - Loading serialized model artifacts
  - Tokenization/feature extraction
  - Prediction with confidence scores
- Wire both the Streamlit dashboard and FastAPI endpoints to use real predictions
- Add model version selection (let users compare LogReg vs Tree vs Transformer)

**Why this matters:** A working demo where users can input real Q&A pairs and see actual evasion predictions transforms this from a code repository into a compelling, interactive portfolio piece.

---

### 2. **Add a "Results Explorer" Dashboard Page**

The project generates rich artifacts (SHAP explanations, confusion matrices, topic models) but they're buried in JSON files. Create a second Streamlit page to visualize:

**Suggested tabs:**
- **Model Comparison**: Interactive table/charts comparing accuracy, F1, precision across all models
- **Explainability Viewer**: Display SHAP summaries from [`artifacts/explainability/phase6/`](../artifacts/explainability/phase6/) - let users explore which features drive evasion predictions
- **Sample Predictions**: Show correctly/incorrectly classified examples with explanations
- **Label Diagnostics**: Visualize the suspect examples and near-duplicates from [`artifacts/diagnostics/phase6/`](../artifacts/diagnostics/phase6/)

**Why this matters:** This showcases the depth of your ML work (explainability, label quality analysis, model comparison) in an accessible format. Recruiters and hiring managers can explore your analysis without reading code or JSON files.

---

## Implementation Architecture

```
Current State
├── Idea 1: Real Inference
│   ├── Create src/inference.py
│   ├── Load tree_boosting model
│   └── Wire API + Dashboard
│
└── Idea 2: Results Explorer
    ├── Model Comparison Tab
    ├── SHAP Explorer Tab
    ├── Sample Predictions Tab
    └── Label Diagnostics Tab

Both paths lead to: Compelling Demo
```

## Priority Recommendation

**Start with Idea 1** - It's foundational. A working prediction system is required before Idea 2's results explorer makes sense. The inference module (`src/inference.py`) will be reused by both the API and dashboard.

## Files to Create/Modify

### New Files
- `src/inference.py` - Model loading and prediction utilities
- `dashboard/pages/2_Results_Explorer.py` - Streamlit multi-page app

### Modified Files
- `dashboard/app.py` - Connect to real model
- `api/main.py` - Connect to real model
- `requirements.txt` - Add any missing dependencies

## Estimated Effort

| Idea | Complexity | Time Estimate |
|------|------------|---------------|
| Idea 1 | Medium | 4-6 hours |
| Idea 2 | Medium | 6-8 hours |

Both ideas leverage existing artifacts and would significantly increase the portfolio's demonstrable value.
