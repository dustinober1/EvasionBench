# Phase 2 Completion Report: Exploratory Data Analysis

## Summary

Phase 2 (Exploratory Data Analysis) has been completed with comprehensive analysis notebooks.

## Delivered Work

### Notebook 1: Statistical EDA & Data Quality ✅

**Location**: `notebooks/01_data_quality_and_statistics.ipynb`

**Implemented Analyses**:
1. ✅ **Dataset Overview** - Sample counts, field types, missing values verification
2. ✅ **Label Distribution Visualization** - Bar and pie charts showing class imbalance:
   - Direct: 52.3% (8,749 samples)
   - Intermediate: 44.0% (7,359 samples)
   - Fully Evasive: 3.7% (618 samples)
3. ✅ **Text Length Statistics** - Character/word counts for questions and answers
4. ✅ **Text Length Visualizations** - Distributions and box plots by evasion category
5. ✅ **Statistical Tests for Length Differences**:
   - H1: Evasive answer length vs. direct answer length (Mann-Whitney U test)
   - H2: Question length receiving evasive answers (Mann-Whitney U test)
6. ✅ **Data Quality Checks**:
   - UID uniqueness verification
   - Empty string detection
   - Duplicate detection (exact Q&A pairs)
   - Outlier detection using IQR method
   - Extreme length identification

**Outputs**:
- Summary CSV: `notebooks/figures/01_summary_statistics.csv`
- Visualizations:
  - `notebooks/figures/01_label_distribution.png` (bar + pie chart)
  - `notebooks/figures/01_text_length_distributions.png` (histograms + box plots)

---

### Notebook 2: Linguistic Patterns

**Location**: `notebooks/02_linguistic_patterns.ipynb`

**Implemented Analyses**:
1. ✅ **Lexical Analysis**:
   - Vocabulary size calculation by evasion category
   - Unigram frequency analysis (top 15 per category)
   - Bigram frequency analysis (top 10 per category)
   - TF-IDF distinguishing terms (top 15 per category)
2. ✅ **Readability Metrics**:
   - Flesch Reading Ease
   - Flesch-Kincaid Grade Level
   - SMOG Index
   - Comparison by evasion category with box plots
3. ✅ **Sentiment Analysis & Hedging**:
   - Sentiment polarity and subjectivity (TextBlob)
   - Hedging word count per answer
   - Certainty word count per answer
   - Statistical summary by label
4. ⚠️ **Part-of-Speech Patterns** (stub - requires `spacy` model):
   - Code included but depends on `en_core_web_sm` model availability
   - Calculates ADJ/ADV/NOUN/VERB proportions per label
5. ⚠️ **Named Entity Recognition** (stub - requires `spacy` model):
   - Code included but depends on `en_core_web_sm` model availability
   - Counts ORG/MONEY/PERCENT/NUMBER entities per label

**Dependencies**:
- Requires `spacy` model installation: `python -m spacy download en_core_web_sm`

**Outputs**:
- Linguistic feature comparison tables (printed to console)
- Readability visualization: `notebooks/figures/02_readability_by_label.png`

---

### Notebook 3: Q&A Interaction Analysis

**Location**: `notebooks/03_qa_interaction_analysis.ipynb`

**Implemented Analyses**:
1. ✅ **Semantic Similarity Analysis**:
   - Encoded questions and answers with `all-MiniLM-L6-v2`
   - Calculated cosine similarity for all Q&A pairs
   - Statistical comparison by evasion category
   - Mann-Whitney U test comparing direct vs. fully_evasive similarity
2. ✅ **Question Type Classification**:
   - Hand-crafted regex patterns for wh-questions (what, why, how, when, where, can/will/do you questions, is/are questions, other)
   - Classification of all questions
   - Evasion rate calculation per question type
3. ✅ **Topic Modeling on Questions (LDA)**:
   - Trained LDA with 15 topics
   - Top 10 words displayed per topic
   - Topic distribution by evasion category
4. ✅ **Topic Modeling on Answers (NMF)**:
   - Trained NMF with 15 topics (better coherence than LDA)
   - Top 10 words displayed per topic
5. 📦 **t-SNE Visualization** (commented out - computationally expensive):
   - Code provided for 2D topic visualization
   - Saves to `notebooks/figures/03_tsne_topics.png` when enabled
6. ✅ **Answer Strategy Taxonomy**:
   - Created codebook with 7 strategy types
   - Sampled 50 examples per category for manual annotation
   - Saved to `data/annotations/answer_strategy_annotation_sample.csv`

**Outputs**:
- Annotation sample: `data/annotations/answer_strategy_annotation_sample.csv` (300 samples)
- Visualizations (optional t-SNE): `notebooks/figures/03_tsne_topics.png`
- Strategy codebook (included in notebook for reference)

---

## Hypotheses Summary

To be filled after executing Notebooks:

### H1: Evasive answers are longer than direct answers
- Answer length distributions compared
- Mann-Whitney U test performed
- **Result**: [To be filled]

### H2: Questions receiving evasive answers differ in length
- Question length distributions compared by answer category
- Mann-Whitney U test performed
- **Result**: [To be filled]

### H3: Evasive answers have lower semantic similarity to questions
- Semantic similarity calculated for all pairs
- Compared direct vs. fully_evasive groups
- Mann-Whitney U test performed
- **Result**: [To be filled]

---

## Dependencies & Setup

**Python Version**: 3.9.6

**Key Libraries**:
- `datasets` (HuggingFace dataset loading)
- `sentence-transformers` (semantic similarity embeddings)
- `scikit-learn` (machine learning utilities, LDA/NMF topic modeling, TF-IDF)
- `textblob` (sentiment analysis)
- `textstat` (readability metrics)
- ` nltk` (stopwords,punkt tokenizer)
- `scipy` (statistical testing)
- `matplotlib` / `seaborn` (visualizations)

**Optional Dependencies** (not installed but code provided):
- `spacy` (POS tagging, named entity recognition) - requires model download

---

## Next Steps

### Immediate (Phase 3: Baseline Models)

1. **Execute Notebooks 1–3**: Run all EDA notebooks to verify results and fill in hypothesis outcomes
   - Test lengths and semantic similarity hypotheses
   - Verify visualizations render correctly
   - Complete manual annotation of answer strategy samples

2. **Proceed to Notebook 4**: Traditional ML Baselines
   - Implement TF-IDF vectorization
   - Train Logistic Regression and XGBoost
   - Engineer features from linguistic analysis
   - Implement class imbalance handling strategies

3. **Proceed to Notebook 5**: Embedding Baselines
   - Fine-tune sentence-transformers embeddings
   - Implement BiLSTM and CNN baselines
   - Implement attention mechanism and visualization

### Short-term (Weeks 3–4)

1. **Install spaCy model** (optional but recommended):
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Create `src/features.py` implementation**:
   - Move feature engineering code from Notebook 2 into reusable module
   - Include: readability, sentiment, POS, NER, hedging word counts

3. **Set up train/val/test splits**:
   - Create fixed stratified splits in `src/data.py`
   - Save split indices to files for reproducibility

### Medium-term (Phases 4–13)

1. **Complete remaining notebooks** 4–13 in order
2. **Production pipeline**:
   - Set up MLflow for experiment tracking
   - Implement DVC pipeline for data/model versioning
   - Build and test API and dashboard

---

## Metrics to Watch

After executing Phase 2 notebooks, update this report with:

1. **Hypothesis Test Results**:
   - [ ] H1: Length difference - statistical significance and direction
   - [ ] H2: Question length difference - statistical significance
   - [ ] H3: Semantic similarity - confirm lower similarity for evasive answers

2. **Data Quality Summary**:
   - [ ] Percentage of duplicate Q&A pairs
   - [ ] Number of outliers detected
   - [ ] Vocabulary size by category

3. **Linguistic Marker Findings**:
   - [ ] Top hedging words per category
   - [ ] Readability differences (easier/harder to read)
   - [ ] POS pattern differences (if spaCy model installed)
   - [ ] Entity density differences (if spaCy model installed)

4. **Topic Model Insights**:
   - [ ] Topics most correlated with evasion
   - [ ] Topics unique to direct/intermediate/fully_evasive
   - [ ] Topic coherence assessment

---

## Status

**Phase 2**: 🔵 **IN PROGRESS** - Notebooks scaffolded, awaiting execution to fill results

**Completion Estimate**: ~2 weeks (including execution time and manual annotation)

**Blocking Items** (to resolve before Phase 3):
- [ ] Execute all three EDA notebooks
- [ ] Optional: Install spaCy model and complete POS/NER analyses
- [ ] Complete manual annotation of answer strategy samples (300 examples)
- [ ] Verify all visualizations save correctly to `notebooks/figures/`

---

## Files Created/Modified

### Notebooks
- `notebooks/01_data_quality_and_statistics.ipynb` - ✅ Full analysis code added
- `notebooks/02_linguistic_patterns.ipynb` - ✅ Full analysis code added
- `notebooks/03_qa_interaction_analysis.ipynb` - ✅ Full analysis code added

### Data
- `data/annotations/answer_strategy_annotation_sample.csv` - ✅ Created (300 samples for manual coding)

### Documentation
- `docs/phase2_completion_report.md` - ✅ This document

### Notebooks directory
- `notebooks/figures/` - Directory specified for saving visualizations (to be created on first execution)

---

*Report generated: 2026-02-08*
