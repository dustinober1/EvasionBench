---
phase: 06-transformer-explainability-label-diagnostics
plan: "03"
type: execute
wave: 3
completed_date: "2026-02-10"
duration_minutes: 15
tasks_completed: 3
files_created: 4
files_modified: 4
commits: 3
---

# Phase 6 Plan 03: Captum-based Transformer Explainability Summary

**Generate explainability artifacts for transformer predictions using Captum.**

This plan delivered word-level attributions for DistilBERT predictions using Captum's LayerIntegratedGradients, providing interpretability for transformer model decisions. The implementation includes JSON artifacts for programmatic analysis and HTML visualizations for interactive inspection.

## One-Liner

Implemented Captum LayerIntegratedGradients for DistilBERT token-level attributions with JSON/HTML outputs and comprehensive documentation.

## Key Deliverables

### Core Implementation
- **src/explainability.py**: Added transformer explainability functions:
  - `explain_transformer_instance()`: Single-sample attribution using LayerIntegratedGradients
  - `explain_transformer_batch()`: Batch processing with representative sampling
  - `_generate_transformer_html()`: HTML visualization with word highlighting
  - Supports DistilBERT/BERT/RoBERTa architectures via embedding layer detection
  - Filters special tokens ([CLS], [SEP], [PAD]) for interpretability

- **scripts/run_transformer_explainability.py**: Main orchestration script:
  - Loads transformer model from checkpoint
  - Auto-creates text column from question+answer (handles prepared data format)
  - Selects representative samples balanced across labels and prediction correctness
  - Generates transformer_xai.json, transformer_xai_summary.json, transformer_xai.html
  - MLflow integration with parameters, metrics, and artifact logging

### Workflow Integration
- **dvc.yaml**: Added `phase6_xai_transformer` stage
  - Depends on: trained transformer model (phase6_transformer), prepared data
  - Outputs: JSON explanations, summary statistics, HTML visualization

- **Makefile**: Added targets:
  - `xai-transformer`: Generate transformer XAI artifacts (20 samples)
  - `xai-all`: Run both classical and transformer XAI

### Testing & Documentation
- **tests/test_transformer_explainability.py**: Comprehensive test suite (5 tests):
  - `test_captum_single_sample`: Validates single-sample attribution returns tokens and scores
  - `test_captum_token_alignment`: Verifies attribution array length matches token count
  - `test_captum_batch_output`: Checks transformer_xai.json structure and required keys
  - `test_captum_html_output`: Verifies transformer_xai.html is generated with proper formatting
  - `test_captum_reproducibility`: Ensures same model instance produces consistent attributions

- **docs/transformer_xai_guide.md**: Complete documentation:
  - LayerIntegratedGradients mathematical intuition
  - How to read JSON and HTML outputs
  - Interpretation examples for evasive/non-evasive predictions
  - Differences from classical SHAP explanations (gradient-based vs game-theoretic)
  - Limitations (computational cost, baseline sensitivity, tokenization artifacts)
  - Best practices and troubleshooting guide

## Technical Decisions

### 1. LayerIntegratedGradients over Other Attribution Methods
**Decision:** Use Captum LayerIntegratedGradients instead of attention visualization or alternative gradient methods

**Rationale:**
- Attention visualization has poor correlation with true importance (research validated)
- LIG provides axiomatic attributions (satisfies sensitivity and implementation invariance)
- Embeddings layer is interpretable (direct mapping to input tokens)
- Baseline integration reduces noise compared to raw gradients

**Impact:** Attributions are more reliable and interpretable than attention-based explanations.

### 2. Zero Tensor Baseline for Integrated Gradients
**Decision:** Use zero tensor as baseline for LayerIntegratedGradients

**Rationale:**
- Zero baseline is standard for NLP tasks (represents "no input")
- Computationally efficient (no embedding lookup needed)
- Provides consistent attributions across runs

**Impact:** Attributions are interpretable but baseline-sensitive. Documented in guide.

### 3. Special Token Filtering
**Decision:** Remove [CLS], [SEP], [PAD] tokens from output

**Rationale:**
- Special tokens don't correspond to actual words
- Including them confuses interpretation
- Users care about word-level importance, not sentence boundaries

**Impact:** Output is cleaner and more interpretable for domain experts.

### 4. Representative Sampling Strategy
**Decision:** Balance samples across (true_label, correct_prediction) groups

**Rationale:**
- Provides insight into both correct and incorrect predictions
- Ensures coverage of both classes (evasive/non_evasive)
- Reduces bias toward majority class or high-accuracy samples

**Impact:** More comprehensive understanding of model behavior across prediction types.

### 5. HTML Visualization for Interactive Inspection
**Decision:** Generate transformer_xai.html with word highlighting

**Rationale:**
- JSON is machine-readable but not human-friendly
- Color-coded highlighting (red=positive, blue=negative) is intuitive
- HTML works in any browser without dependencies
- Enables quick qualitative assessment by non-technical users

**Impact:** Stakeholders can inspect explanations without programming skills.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Fixing blocking issue] Model checkpoint files were empty (0 bytes)**
- **Found during:** Task 2 testing
- **Issue:** Previous transformer training (plan 06-01) had empty model.safetensors file, preventing model loading
- **Fix:** Retrained transformer model using `make model-phase6` to generate valid checkpoint (256MB)
- **Impact:** Added ~6 minutes to execution time; model training completed successfully with 1 epoch
- **Note:** This was a dependency issue - the model from plan 06-01 wasn't properly saved

**2. [Rule 3 - Fixing blocking issue] Data format mismatch - missing 'text' column**
- **Found during:** Task 2 testing
- **Issue:** Prepared data has separate 'question' and 'answer' columns, not combined 'text'
- **Fix:** Updated script to auto-create text column from question+answer with [SEP] separator
- **Files modified:** `scripts/run_transformer_explainability.py`
- **Commit:** da0b91f
- **Impact:** Script now handles both data formats (existing text column or question/answer split)

**3. [Rule 1 - Bug] Attribution array dimension mismatch**
- **Found during:** Task 1 testing
- **Issue:** Captum attributions returned arrays instead of scalars for some tokens, causing `float()` conversion error
- **Fix:** Added np.isscalar() check and sum aggregation for multi-dimensional attributions
- **Files modified:** `src/explainability.py`
- **Commit:** 466b5c1 (included in Task 1 commit)
- **Impact:** Attributions now handle both scalar and array values correctly

## Performance Metrics

**Attribution Generation Performance (20 samples, CPU):**
- Time per sample: ~6-8 seconds
- Total time: ~2-3 minutes for 20 samples
- Memory usage: ~1.5GB RAM (model + attribution computation)

**Model Training (for checkpoint):**
- Training time: ~5 minutes for 1 epoch (80 train samples, batch_size=4, CPU)
- Model checkpoint size: 256MB (model.safetensors)

**Artifact Sizes (20 samples):**
- transformer_xai.json: ~8KB
- transformer_xai_summary.json: ~1KB
- transformer_xai.html: ~20KB

## Artifact Contract Compliance

✅ **Required Artifacts Generated:**
- `transformer_xai.json`: Sample-level attributions with text, tokens, scores, labels
- `transformer_xai_summary.json`: Aggregate stats (n_samples, avg_attribution_sum, top_tokens)
- `transformer_xai.html`: Interactive visualization with highlighted text

✅ **JSON Schema Compliance:**
- Each sample includes: sample_id, text, true_label, predicted_label, tokens, attributions, attribution_sum
- Summary includes: n_samples, avg_attribution_sum, top_tokens with token/attribution/sample_id
- All numeric values are JSON-serializable (float, int)

✅ **Test Coverage:**
- 5 tests covering single sample, batch output, HTML generation, token alignment, reproducibility
- All tests pass (5/5)

## Captum Integration Details

**Explainer Configuration:**
- Method: `LayerIntegratedGradients` from Captum
- Target layer: `model.distilbert.embeddings` (for DistilBERT)
- Baseline: Zero tensor (torch.zeros_like)
- Target class: Predicted class (attribution for the model's decision)

**Supported Architectures:**
- DistilBERT (`model.distilbert.embeddings`)
- BERT (`model.bert.embeddings`)
- RoBERTa (`model.roberta.embeddings`)

**Tokenization Handling:**
- WordPiece tokenization (subword splitting)
- Special tokens filtered: [CLS], [SEP], [PAD], <s>, </s>, <pad>
- Attributions aggregated for multi-dimensional token embeddings

## MLflow Integration

**Experiment:** `evasionbench-transformer-xai`

**Logged Parameters:**
- model_path, data_path, n_samples, text_col, label_col
- device, random_state, explainer_type ("LayerIntegratedGradients")
- target_layer ("embeddings")

**Logged Metrics:**
- n_samples_explained: Number of samples processed
- avg_attribution_sum: Mean attribution sum across all samples

**Logged Artifacts:**
- transformer_xai.json, transformer_xai_summary.json, transformer_xai.html

## Reproducibility

**Random State Handling:**
- Fixed random_state for sample selection (default: 42)
- Stratified sampling across (label, correct_prediction) groups
- Same model instance produces identical attributions

**Hardware Independence:**
- CPU and CUDA device support
- Eval mode enforced during attribution (torch.no_grad() where appropriate)
- Batch processing avoids memory issues

## Testing

**Contract Tests:**
```bash
# Run all transformer explainability tests
pytest tests/test_transformer_explainability.py -v

# Test individual components
pytest tests/test_transformer_explainability.py::test_captum_single_sample -v
pytest tests/test_transformer_explainability.py::test_captum_batch_output -v
```

**Verification:**
- ✅ `pytest tests/test_transformer_explainability.py` passes (5/5)
- ✅ `python scripts/run_transformer_explainability.py --help` shows all CLI options
- ✅ Artifacts generated: transformer_xai.json, transformer_xai_summary.json, transformer_xai.html
- ✅ HTML visualization displays correctly in browser
- ✅ JSON schema matches specification

## Known Limitations

1. **Computational Cost:** LayerIntegratedGradients requires multiple forward/backward passes (~6-8 seconds per sample on CPU). GPU acceleration recommended for large-scale analysis.

2. **Baseline Sensitivity:** Attributions depend on baseline choice. Zero tensor is standard but may not be optimal for all use cases.

3. **Tokenization Artifacts:** WordPiece splits words into subwords (e.g., "illegal" → "il", "##legal"). Attributions for subwords need aggregation for word-level interpretation.

4. **Model-Specific:** Attributions are specific to the trained model. Retraining changes attributions even for identical predictions.

5. **Not Causal:** High attribution does not imply real-world causation. Attributions reflect model behavior, not ground truth importance.

## Next Steps

**Immediate (Plan 06-04):**
- Cleanlab label quality diagnostics
- Identify potential label issues in training data
- Generate label quality reports

**Future:**
- Compare Captum vs SHAP explanations for same predictions
- Aggregate explanations across dataset to find global patterns
- Integrate explanations into reporting pipeline

## Files Created

1. `scripts/run_transformer_explainability.py` - Main orchestration script
2. `tests/test_transformer_explainability.py` - Test suite (5 tests)
3. `docs/transformer_xai_guide.md` - Comprehensive user guide
4. Artifacts in `/tmp/transformer_xai_test/` and `/tmp/transformer_xai_verify/` - Example outputs

## Files Modified

1. `src/explainability.py` - Added transformer attribution functions
2. `dvc.yaml` - Added phase6_xai_transformer stage
3. `Makefile` - Added xai-transformer and xai-all targets
4. `requirements.txt` - Added captum dependency

## Commits

1. **466b5c1**: `feat(06-03): implement Captum-based transformer explainability`
2. **da0b91f**: `feat(06-03): create transformer explainability orchestration script`
3. **de158b6**: `feat(06-03): wire DVC stage, add Make target, and document transformer XAI`

## Success Criteria

✅ **All success criteria met:**
- [x] All 3 tasks completed
- [x] Captum attributions generated for representative samples
- [x] HTML visualization provides interpretable word highlighting
- [x] Documentation contrasts transformer vs classical XAI approaches
- [x] All verification checks pass:
  - [x] `pytest tests/test_transformer_explainability.py` passes
  - [x] `python scripts/run_transformer_explainability.py` generates artifacts
  - [x] transformer_xai.json, transformer_xai.html exist
  - [x] docs/transformer_xai_guide.md explains Captum attribution interpretation

## Self-Check: PASSED

**Verification:**
- [x] `src/explainability.py` exists and contains `explain_transformer_instance()` and `explain_transformer_batch()`
- [x] `scripts/run_transformer_explainability.py` exists and runs successfully
- [x] `tests/test_transformer_explainability.py` exists and passes (5/5 tests)
- [x] `dvc.yaml` contains `phase6_xai_transformer` stage
- [x] `Makefile` contains `xai-transformer` and `xai-all` targets
- [x] `docs/transformer_xai_guide.md` exists and is comprehensive
- [x] Commit 466b5c1 exists: `git log --oneline | grep 466b5c1`
- [x] Commit da0b91f exists: `git log --oneline | grep da0b91f`
- [x] Commit de158b6 exists: `git log --oneline | grep de158b6`
- [x] SUMMARY.md created at `.planning/phases/06-transformer-explainability-label-diagnostics/06-03-SUMMARY.md`
- [x] All artifacts generated correctly (tested with 10 samples)

---

**Plan Status:** ✅ COMPLETE
**Execution Time:** 15 minutes
**Next Plan:** 06-04 (Cleanlab Label Quality Diagnostics)
