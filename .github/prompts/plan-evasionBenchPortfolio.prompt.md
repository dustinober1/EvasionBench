## Plan: EvasionBench Financial NLP Research Portfolio

**TL;DR**: Build a comprehensive data science portfolio analyzing the EvasionBench dataset (16.7k corporate earnings call Q&As labeled for evasiveness). This plan includes 10+ Jupyter notebooks covering EDA, traditional ML, transformer fine-tuning, novel research hypotheses, explainability analysis, label quality investigation, and production deployment with complete MLOps infrastructure. The research angle emphasizes hypothesis testing (including "interesting failures"), reproducibility, and methodological rigor while showcasing ML engineering skills through a production-ready evasion detector API with CI/CD pipeline.

---

## Phase 1: Project Foundation & Setup

**Steps:**

1. **Initialize project structure** with research-oriented organization:

```
EvasionBench/
├── notebooks/           # Jupyter notebooks (numbered)
├── src/                 # Reusable Python modules
├── data/                # Data storage (gitignored)
├── models/              # Saved models (DVC tracked)
├── experiments/         # MLflow tracking
├── api/                 # FastAPI application
├── dashboard/           # Streamlit app
├── tests/               # Pytest unit tests
├── docker/              # Containerization
├── .github/workflows/   # CI/CD pipelines
├── scripts/             # Utility scripts
├── papers/              # Research writeup (LaTeX)
├── requirements.txt
├── environment.yml
├── pyproject.toml       # Poetry config
├── .dvcignore
├── .gitignore
└── README.md
```

2. **Create `src/` module structure** for code reuse across notebooks:
   - `src/data.py`: Dataset loading, preprocessing, splits
   - `src/features.py`: Feature engineering utilities
   - `src/models.py`: Model definitions and wrappers
   - `src/evaluation.py`: Metrics, confusion matrices, reports
   - `src/visualization.py`: Consistent plotting functions
   - `src/explainability.py`: SHAP/LIME utilities
   - `src/utils.py`: General helpers (seed setting, logging)

3. **Set up environment management**:
   - Create `requirements.txt` with pinned versions (transformers, datasets, torch, sklearn, shap, mlflow, fastapi, streamlit, pytest, black, nbstripout)
   - Create `environment.yml` for conda users
   - Add `.gitattributes` for notebook diffing
   - Configure `nbstripout` to auto-clear outputs pre-commit

4. **Initialize tooling**:
   - Git repository with `.gitignore` for data/, models/, experiments/
   - DVC init for data and model versioning
   - MLflow tracking server setup (local or cloud)
   - Pre-commit hooks (black, flake8, nbstripout)
   - README template with badges and structure

5. **Download and verify dataset**:
   - Create `scripts/download_data.py` using HuggingFace `datasets` library
   - Load `FutureMa/EvasionBench` and save to `data/raw/evasionbench.parquet`
   - Verify 16,726 samples with 4 fields (uid, question, answer, eva4b_label)
   - Generate MD5 checksum for reproducibility
   - DVC track the raw data file

---

## Phase 2: Exploratory Data Analysis (3 Notebooks)

### Notebook 1: Statistical EDA & Data Quality

**File**: `notebooks/01_data_quality_and_statistics.ipynb`

**Objectives:**
- Comprehensive statistical overview
- Data quality assessment
- Class distribution analysis
- Baseline understanding

**Analyses:**
1. **Dataset overview**: Sample counts, field types, missing values
2. **Label distribution**: Exact counts and percentages by class (direct: 52.3%, intermediate: 44%, fully_evasive: 3.7%)
3. **Class imbalance visualization**: Bar charts, pie charts with annotations
4. **Text length distributions**:
   - Question length (characters, words, sentences)
   - Answer length by evasion category
   - Box plots comparing distributions
   - Statistical tests (Kruskal-Wallis for non-parametric comparison)
5. **Data quality checks**:
   - Duplicate detection (exact and near-duplicate using MinHash)
   - Empty string checks
   - Anomalous lengths (outliers)
   - UID uniqueness verification
6. **Preliminary hypothesis testing**:
   - H1: "Evasive answers are longer than direct answers" (test with statistical significance)
   - H2: "Questions receiving evasive answers are different in structure/length"

**Outputs:**
- Summary statistics table
- Publication-quality visualizations (save to `notebooks/figures/`)
- Initial hypothesis test results
- Data quality report

---

### Notebook 2: Linguistic Analysis

**File**: `notebooks/02_linguistic_patterns.ipynb`

**Objectives:**
- Deep linguistic feature exploration
- Identify markers of evasiveness
- NLP preprocessing decisions

**Analyses:**
1. **Lexical analysis**:
   - Vocabulary size by category
   - Word clouds for each evasion level (questions and answers separately)
   - Most frequent unigrams, bigrams, trigrams per category
   - TF-IDF top terms distinguishing each class
   - Stopword impact analysis
2. **Readability metrics**:
   - Flesch Reading Ease
   - Flesch-Kincaid Grade Level
   - SMOG index
   - Compare across evasion categories
3. **Syntactic complexity**:
   - Average sentence length
   - Parse tree depth (using spaCy)
   - Dependency complexity
4. **Part-of-speech patterns**:
   - POS tag distributions by class
   - Hypothesis: Evasive answers use more adjectives/adverbs (hedging)
5. **Sentiment analysis**:
   - Sentiment scores (VADER, TextBlob) across categories
   - Emotional tone differences
6. **Named entity recognition**:
   - Entity types (ORG, MONEY, PERCENT) by evasion level
   - Do direct answers mention more concrete entities?
7. **Discourse markers**:
   - Hedging words ("might", "could", "possibly", "approximately")
   - Certainty markers ("definitely", "certainly", "exactly")
   - Deflection phrases ("not sure", "hard to say", "depends")
   - Count occurrences by category

**Outputs:**
- Linguistic feature comparison tables
- Visualization dashboard showing all metrics
- Hypothesis: identification of linguistic evasion markers
- Preprocessed text decisions documented

---

### Notebook 3: Question-Answer Interaction Analysis

**File**: `notebooks/03_qa_interaction_analysis.ipynb`

**Objectives:**
- Analyze Q&A as paired sequences
- Topic modeling
- Semantic similarity

**Analyses:**
1. **Topic modeling on questions**:
   - LDA with 10-20 topics
   - NMF topic extraction
   - Visualize with pyLDAvis
   - Topic distribution by evasion category (do certain topics elicit more evasion?)
2. **Topic modeling on answers**:
   - Separate topic modeling for answers
   - Compare answer topics vs. question topics
   - Topic drift: when do answers shift topics?
3. **Semantic similarity**:
   - Encode questions and answers with sentence-transformers (all-MiniLM-L6-v2)
   - Cosine similarity between Q&A pairs
   - Hypothesis: Evasive answers have lower semantic similarity to questions
4. **Question type classification**:
   - Extract question types (wh-questions: what, why, how, when, can you)
   - Evasion rate by question type
   - Are "why" questions more likely to be evaded than "how many"?
5. **Answer strategies taxonomy**:
   - Manual annotation of 50 samples per category
   - Identify patterns: deflection, partial answer, question reframing, refusal
   - Codebook for intermediate category nuances
6. **Temporal patterns** (if extractable):
   - Parse UIDs to extract company/date info
   - Evasion trends over time or by company sector
   - Requires UID format investigation

**Outputs:**
- Topic model visualizations
- Similarity distribution plots
- Question type analysis table
- Answer strategy taxonomy documented
- Research insight: "What types of questions get evaded?"

---

## Phase 3: Baseline Models (2 Notebooks)

### Notebook 4: Traditional ML Baselines

**File**: `notebooks/04_traditional_ml_baselines.ipynb`

**Objectives:**
- Establish non-transformer baselines
- Feature engineering exploration
- Interpretable models

**Models & Features:**

1. **TF-IDF + Logistic Regression**:
   - Answer-only baseline
   - Question-only baseline
   - Combined Q+A features
   - Hyperparameter tuning (C, max_features, ngram_range)
   - Compare macro-F1, per-class F1

2. **TF-IDF + XGBoost**:
   - Gradient boosting baseline
   - Feature importance analysis
   - Hyperparameter optimization (max_depth, learning_rate, subsample)

3. **Engineered features + models**:
   - Create feature set from linguistic analysis:
     - Text lengths (char, word, sentence counts)
     - Readability scores
     - Sentiment scores
     - POS tag ratios
     - Hedge word counts
     - Semantic similarity (Q-A)
     - Entity counts
   - Random Forest on engineered features
   - Logistic Regression on engineered features
   - Feature ablation study

4. **Hybrid models**:
   - Combine TF-IDF + engineered features
   - Stacking ensemble

5. **Class imbalance handling**:
   - Baseline (no handling)
   - Class weights
   - SMOTE (Synthetic Minority Over-sampling)
   - Random under-sampling of majority
   - Compare impact on minority class (fully_evasive) F1

**Experimental setup:**
- Stratified 80/10/10 train/val/test splits (fixed seed)
- All experiments logged to MLflow
- Cross-validation on train+val
- Final test set evaluation

**Outputs:**
- Baseline performance table (all models compared)
- Feature importance visualizations (XGBoost, Logistic Regression coefficients)
- Class imbalance strategy comparison
- Best traditional baseline model saved to `models/baseline_traditional/`
- MLflow experiment tracking

---

### Notebook 5: Sentence Embeddings & Simple Neural Baselines

**File**: `notebooks/05_embedding_baselines.ipynb`

**Objectives:**
- Pre-trained embedding baselines
- Simple neural architectures
- Bridge to transformer fine-tuning

**Approaches:**

1. **Sentence-BERT embeddings**:
   - all-MiniLM-L6-v2 (lightweight)
   - all-mpnet-base-v2 (higher quality)
   - Encode questions and answers separately, concatenate
   - Logistic Regression on embeddings
   - MLP classifier (2-3 hidden layers)

2. **Simple BiLSTM**:
   - GloVe embeddings (300d) or fastText
   - BiLSTM encoder for answers
   - Classification head
   - Compare answer-only vs. Q+A concatenation

3. **CNN for text classification**:
   - Multiple filter sizes (2,3,4,5-grams)
   - Max pooling
   - Baseline inspired by Kim (2014)

4. **Attention-based baseline**:
   - BiLSTM + self-attention mechanism
   - Attention weight visualization on evasive samples

**Class imbalance**:
- Apply focal loss to handle 3.7% minority class
- Compare with weighted cross-entropy

**Experimental setup:**
- Same train/val/test splits as Notebook 4
- Early stopping on validation macro-F1
- MLflow tracking for all experiments
- Training time and compute cost tracking

**Outputs:**
- Embedding baseline performance table
- Simple neural baseline results
- Attention visualization on sample predictions
- Best embedding model saved to `models/baseline_embeddings/`

---

## Phase 4: Transformer Fine-Tuning (2 Notebooks)

### Notebook 6: BERT Family Fine-Tuning

**File**: `notebooks/06_transformer_finetuning.ipynb`

**Objectives:**
- Fine-tune transformer models
- Approach Eva-4B-V2 SOTA (84.9% macro-F1)
- Efficient model selection for moderate GPU

**Models to fine-tune:**

1. **DistilBERT** (efficient, 66M params):
   - distilbert-base-uncased
   - Answer-only input
   - Q+A concatenated with [SEP] token
   - Learning rate search, epochs tuning

2. **RoBERTa-base** (125M params):
   - roberta-base
   - Known for strong performance on classification
   - Compare with DistilBERT (efficiency vs. accuracy)

3. **DeBERTa-v3-base** (184M params):
   - microsoft/deberta-v3-base
   - Current SOTA for many text classification tasks
   - Disentangled attention mechanism

4. **Domain-adapted models** (financial):
   - ProsusAI/finbert (BERT for finance)
   - yiyanghkust/finbert-tone
   - Hypothesis: financial domain pre-training helps

5. **Input formatting experiments**:
   - Answer only: `[CLS] answer [SEP]`
   - Q+A: `[CLS] question [SEP] answer [SEP]`
   - Structured: `[CLS] Q: question [SEP] A: answer [SEP]`
   - Compare which works best

**Training strategy:**
- HuggingFace Trainer API
- Stratified splits (same as baselines)
- Hyperparameter search:
  - Learning rate: [1e-5, 2e-5, 3e-5, 5e-5]
  - Batch size: [8, 16, 32] (memory-dependent)
  - Epochs: early stopping on val macro-F1
  - Warmup steps: 10% of total
- Class imbalance:
  - Weighted cross-entropy loss
  - Focal loss implementation
  - Compare impact on minority class

**Advanced techniques:**
- Gradient accumulation for effective larger batch sizes
- Mixed precision training (fp16) for efficiency
- Learning rate scheduling (linear decay with warmup)
- Gradient clipping

**Evaluation:**
- Macro-F1 (primary metric to match SOTA)
- Per-class precision, recall, F1
- Confusion matrix analysis
- Error analysis on validation set
- Compare to Eva-4B-V2's 84.9% macro-F1

**Outputs:**
- Comprehensive model comparison table
- Training curves (loss, macro-F1 over epochs)
- Confusion matrices for best models
- Best model saved to `models/transformers/`
- MLflow experiment tracking with hyperparameters
- Prediction analysis: where do models fail?

---

### Notebook 7: Advanced Transformer Techniques

**File**: `notebooks/07_advanced_transformer_methods.ipynb`

**Objectives:**
- Parameter-efficient fine-tuning
- Model compression
- Ensemble methods

**Approaches:**

1. **LoRA (Low-Rank Adaptation)**:
   - Fine-tune larger models with PEFT library
   - deberta-v3-large with LoRA (4-bit quantization)
   - Reduce trainable parameters by >99%
   - Enable larger model fine-tuning on moderate GPU
   - Compare full fine-tuning vs. LoRA performance

2. **Adapter layers**:
   - adapter-transformers library
   - Add small bottleneck adapters to frozen BERT
   - Compare with LoRA

3. **Multi-sample dropout**:
   - Technique for better uncertainty estimation
   - Average predictions across multiple dropout samples
   - Improve calibration on uncertain examples

4. **Knowledge distillation**:
   - Use best large model (DeBERTa-v3-base) as teacher
   - Distill to DistilBERT (student) with soft labels
   - Maintain performance with smaller deployment size

5. **Ensemble methods**:
   - Ensemble top 3-5 models from Notebook 6
   - Voting (soft and hard)
   - Stacking with meta-learner
   - Diversity analysis: when do models disagree?

6. **Continued pre-training**:
   - Masked language modeling on earnings call corpus
   - Domain adaptation before fine-tuning
   - Hypothesis: additional financial discourse exposure helps
   - Requires gathering additional earnings call data (unsupervised)

**Outputs:**
- LoRA vs. full fine-tuning comparison
- Distillation performance and model size reduction
- Ensemble results (potentially best overall performance)
- Best advanced model saved
- Deployment recommendation: performance vs. size/speed trade-offs

---

## Phase 5: Novel Research Hypotheses (3 Notebooks)

### Notebook 8: Multi-Task Learning

**File**: `notebooks/08_multitask_learning.ipynb`

**Objectives:**
- Test if auxiliary tasks improve evasion detection
- Focus on minority class improvement
- Novel research contribution

**Hypothesis**: Joint training on related tasks will improve representation learning and boost performance on the minority (fully_evasive) class.

**Experimental design:**

1. **Task 1: Evasion Classification** (primary)
   - 3-class classification (direct, intermediate, fully_evasive)

2. **Task 2: Binary Directness** (auxiliary)
   - Simplify to binary: direct vs. not-direct
   - Easier task may provide useful gradients

3. **Task 3: Answer Sentiment** (auxiliary)
   - Predict sentiment of answer (positive, neutral, negative)
   - Use VADER scores as proxy labels
   - Hypothesis: evasive answers correlate with neutral/negative sentiment

4. **Task 4: Question Topic** (auxiliary)
   - Cluster questions into K topics from Notebook 3
   - Multi-label topic classification
   - Hypothesis: topic-aware representations help

5. **Task 5: Semantic Similarity Regression** (auxiliary)
   - Predict Q-A semantic similarity score
   - Continuous regression task
   - Hypothesis: evasive answers have lower similarity

**Architecture:**
- Shared encoder (RoBERTa or DeBERTa)
- Task-specific heads for each task
- Implement with PyTorch Lightning
- Loss weighting strategies:
  - Equal weights
  - Uncertainty weighting (learned)
  - Manual tuning
  - GradNorm (gradient normalization)

**Experiments:**
1. Single-task baseline (evasion only)
2. MTL with each auxiliary task individually
3. MTL with all tasks combined
4. Ablation: compare loss weighting strategies

**Evaluation focus:**
- Primary: Macro-F1 on evasion classification
- Critical: F1 on fully_evasive class (minority)
- Analysis: Does MTL help minority class specifically?

**Outputs:**
- MTL architecture diagram
- Task performance table (all combinations)
- Minority class improvement analysis
- Loss curve comparisons
- Best MTL model saved
- Research insight: "Does MTL help with extreme class imbalance?"

---

### Notebook 9: Explainability & Interpretability

**File**: `notebooks/09_explainability_analysis.ipynb`

**Objectives:**
- Understand what models learn
- Identify linguistic markers programmatically
- Generate publication-quality explanations

**Analyses:**

1. **SHAP analysis on best transformer model**:
   - Deep learning SHAP explainer
   - Per-class SHAP values
   - Sample 200 examples per class (600 total) for efficiency
   - Identify top tokens/words driving each prediction
   - Aggregate SHAP values to find global patterns
   - Visualization: SHAP summary plots, waterfall plots for individual predictions

2. **LIME explanations**:
   - Local Interpretable Model-agnostic Explanations
   - Sample 50 high-confidence predictions per class
   - Text-specific LIME (perturbs words)
   - Compare LIME and SHAP agreement
   - Highlight words contributing to evasion classification

3. **Attention visualization**:
   - Extract attention weights from best transformer
   - Visualize attention on Q-A pairs
   - Questions:
     - Where does the model attend in evasive answers?
     - Does it focus on deflection phrases?
     - Question-answer cross-attention patterns

4. **Probing classifiers**:
   - Freeze best model, probe hidden representations
   - Train linear probes on intermediate layers
   - What linguistic info is encoded at each layer?
   - Tasks: POS tagging, sentiment, hedge word detection

5. **Saliency-based feature extraction**:
   - Extract top SHAP features as "evasion markers"
   - Build interpretable rule-based classifier using these markers
   - Compare rule-based performance to black box
   - Example rules: "If answer contains >3 hedge words AND low semantic similarity → evasive"

6. **Error analysis with explanations**:
   - Analyze false positives and false negatives
   - Use SHAP to understand why model failed
   - Categorize error types
   - Hypothesis generation for future improvements

**Outputs:**
- SHAP and LIME visualizations (saved as high-res images)
- Attention heatmaps for sample Q-A pairs
- Evasion marker lexicon extracted from SHAP
- Rule-based interpretable classifier
- Error taxonomy with explanations
- Research insight: "What linguistic patterns signal evasion?"
- Potential publication: "Interpreting Financial Discourse Evasion Detection"

---

### Notebook 10: Label Quality Investigation

**File**: `notebooks/10_label_quality_study.ipynb`

**Objectives:**
- Assess Eva-4B-V2 label quality
- Conduct manual annotation study
- Quantify label noise impact
- Honest research on dataset limitations

**Hypothesis**: Since labels are model-generated (not human-annotated), there may be label noise. Understanding this is critical for interpreting results.

**Study design:**

1. **Manual annotation**:
   - Randomly sample 300 examples stratified by predicted label:
     - 100 direct
     - 100 intermediate
     - 100 fully_evasive
   - Author (or collaborators) manually annotate with guidelines
   - Create annotation codebook based on Notebook 3 taxonomy
   - Measure inter-annotator agreement if multiple annotators

2. **Agreement analysis**:
   - Cohen's kappa between manual annotations and Eva-4B-V2 labels
   - Confusion matrix: where do human and model disagree?
   - Per-class agreement rates
   - Identify systematic biases in model labels

3. **Label noise impact estimation**:
   - Train model on clean (human-verified) subset
   - Compare to model trained on full dataset
   - Performance delta estimates noise impact
   - Confident Learning (cleanlab library):
     - Identify likely mislabeled examples in full dataset
     - Estimate true label distribution

4. **Uncertainty correlation**:
   - Model prediction uncertainty (softmax entropy)
   - Hypothesis: low agreement samples have high uncertainty
   - Identify "hard cases" where evasion is subjective

5. **Subjectivity analysis**:
   - Calculate inter-annotator agreement on manual annotations
   - If multiple annotators: identify samples with disagreement
   - Conclusion: Is evasion detection objective or inherently subjective?

6. **Refined dataset creation**:
   - Use cleanlab to identify and remove likely mislabeled samples
   - Create "EvasionBench-Cleaned" subset
   - Retrain best model on cleaned data
   - Compare performance: does cleaning help?

**Outputs:**
- Annotation guidelines document
- Human-annotated dataset (300 samples) saved to `data/annotations/`
- Agreement statistics (kappa, confusion matrix)
- Label noise report with visualizations
- Cleanlab-identified mislabeled examples
- Refined dataset and re-trained model results
- Research insight: "How reliable are model-generated labels for financial discourse?"
- Potential contribution: Release human-annotated subset to community

---

## Phase 6: "Interesting Failures" Research (2 Notebooks)

### Notebook 11: Failed Hypotheses & Negative Results

**File**: `notebooks/11_failed_hypotheses.ipynb`

**Objectives:**
- Document hypotheses that didn't pan out
- Demonstrate scientific rigor and honesty
- Provide insights from failures

**Failed hypotheses to test:**

1. **"Syntactic complexity predicts evasion"**:
   - Test: Parse tree depth, dependency distance correlate with evasion
   - Expectation: Evasive answers use more complex syntax
   - Likely result: Weak or no correlation
   - Analysis: Why this intuition was wrong

2. **"Entity density indicates directness"**:
   - Test: Direct answers mention more named entities (numbers, orgs)
   - Expectation: Strong predictor
   - Likely result: Noisy signal, not strong enough
   - Explanation: Evasive answers can mention entities while deflecting

3. **"Question length correlates with evasion rate"**:
   - Test: Longer questions elicit more evasion
   - Expectation: Complex questions → more evasion
   - Analysis: Null or opposite result

4. **"Back-translation augmentation helps minority class"**:
   - Experiment: Augment fully_evasive class with back-translation (en→de→en)
   - Expectation: Synthetic data boosts F1
   - Likely result: Performance degradation (synthetic data hurts)
   - Analysis: Augmentation quality issues

5. **"GPT-4 zero-shot matches fine-tuned models"**:
   - Experiment: GPT-4 with careful prompt engineering
   - Expectation: Closes gap with Eva-4B-V2
   - Likely result: Falls short by 10-15% macro-F1
   - Insight: Domain-specific fine-tuning still valuable

6. **"Simple rule-based heuristics are competitive"**:
   - Build hand-crafted rules from linguistic analysis
   - Expectation: Rule-based achieves >70% F1
   - Likely result: ~50-60% F1, insufficient
   - Value: Demonstrates complexity of task

**Approach:**
- Pre-register hypotheses (document before testing)
- Test rigorously with statistical controls
- Visualize null results
- Explain why failure occurred
- Extract lessons learned

**Outputs:**
- Failed hypothesis report (transparent results)
- Null result visualizations
- Lessons learned section
- Research insight: "What doesn't work for evasion detection?"
- Demonstrates scientific maturity and honesty

---

### Notebook 12: Cross-Domain Transfer & Generalization

**File**: `notebooks/12_cross_domain_transfer.ipynb`

**Objectives:**
- Test model generalization beyond earnings calls
- Explore transfer learning
- Likely to show interesting failures

**Hypothesis**: Models trained on earnings call Q&As will generalize poorly to other domains due to domain-specific linguistic patterns.

**Experiments:**

1. **Domain shift analysis**:
   - Collect small datasets from other Q&A domains:
     - Political debates (Presidential debates, parliamentary questions)
     - Customer service chats
     - Academic office hours
     - Reddit AMA threads
   - Manual annotation: 50-100 samples per domain for evasiveness
   - Evaluate best EvasionBench model on out-of-domain data
   - Expected result: Performance drop

2. **Feature transferability**:
   - Compare domain-specific vs. universal features
   - Which features transfer (hedge words, semantic similarity)?
   - Which are domain-specific (financial terminology)?

3. **Domain adaptation attempts**:
   - Few-shot fine-tuning on small out-of-domain samples
   - Does adapt, ation recover performance?
   - How many samples needed?

4. **Adversarial domain testing**:
   - Create adversarial examples:
     - Direct answers with evasion markers
     - Evasive answers without typical markers
   - Test model robustness
   - Expected result: Model relies on spurious correlations

5. **Cross-domain transfer learning**:
   - Train on political debates, test on earnings calls (reverse)
   - Quantify domain gap
   - Analysis: Is "evasion" universal or domain-specific?

**Outputs:**
- Cross-domain evaluation results (expected to be poor)
- Domain transfer analysis
- Small annotated datasets for other domains (contribution)
- Research insight: "How domain-specific is evasion detection?"
- Honest conclusion: limitations of current approach
- Pre-registration for potential publication on negative results

---

## Phase 7: Production Deployment (Full MLOps)

### Notebook 13: Model Selection & Optimization for Production

**File**: `notebooks/13_production_model_selection.ipynb`

**Objectives:**
- Select deployment model (accuracy vs. latency vs. cost)
- Optimize for inference
- Benchmark performance

**Analyses:**

1. **Model comparison matrix**:
   - Top 5-7 models from previous notebooks
   - Metrics: macro-F1, inference latency (ms), model size (MB), memory usage
   - Cost estimation: GPU vs. CPU inference
   - Throughput: samples/second

2. **Quantization experiments**:
   - Dynamic quantization (PyTorch)
   - Post-training quantization (INT8)
   - Compare accuracy degradation vs. speedup
   - ONNX Runtime optimization

3. **Model distillation for deployment**:
   - If best model is large (DeBERTa), distill to DistilBERT
   - Quality vs. efficiency trade-off
   - Deployment recommendation

4. **Batching strategies**:
   - Single vs. batch inference
   - Optimal batch size for throughput

5. **Hardware profiling**:
   - CPU inference (Intel, Apple Silicon)
   - GPU inference (T4, A100)
   - Edge deployment feasibility (mobile, browser)

**Selection criteria:**
- Research/demo: Best performing model (max accuracy)
- Production API: Best efficiency for <100ms latency at 99th percentile
- Edge: Best compressed model for CPU inference

**Outputs:**
- Model comparison dashboard
- Production model selection justification
- Optimized model artifacts (ONNX, quantized)
- Deployment recommendation document

---

### Artifact: FastAPI Application

**File**: `api/main.py` (and supporting files)

**Structure:**
```
api/
├── main.py              # FastAPI app
├── models.py            # Pydantic schemas
├── inference.py         # Model loading & prediction
├── config.py            # Configuration
├── Dockerfile           # Container image
├── requirements.txt     # API dependencies
└── tests/
    └── test_api.py      # Pytest API tests
```

**Features:**

1. **Endpoints**:
   - `POST /predict`: Single Q&A prediction
   - `POST /batch_predict`: Batch predictions
   - `GET /health`: Health check
   - `GET /model_info`: Model metadata (version, macro-F1, date)
   - `GET /explain`: SHAP-based explanation for prediction

2. **Request/Response schemas** (Pydantic):
   ```python
   class QAPair(BaseModel):
       question: str
       answer: str

   class PredictionResponse(BaseModel):
       prediction: str  # direct, intermediate, fully_evasive
       confidence: float
       probabilities: dict[str, float]
       explanation: Optional[list[dict]]  # SHAP tokens
   ```

3. **Model serving**:
   - Load optimized model on startup
   - In-memory caching
   - Async inference if needed

4. **Validation & error handling**:
   - Input validation (max length, required fields)
   - Graceful error responses
   - Logging (structured JSON logs)

5. **Monitoring instrumentation**:
   - Prometheus metrics (request count, latency, error rate)
   - OpenTelemetry tracing

6. **Security**:
   - Rate limiting (if public-facing)
   - Input sanitization
   - API key authentication (optional)

**Containerization**:
- Multi-stage Docker build (small final image)
- Environment variable configuration
- Docker Compose for local development (app + monitoring)

**Testing**:
- Unit tests for inference logic
- API integration tests with pytest
- Load testing with Locust

---

### Artifact: Streamlit Dashboard

**File**: `dashboard/app.py`

**Structure:**
```
dashboard/
├── app.py               # Main Streamlit app
├── utils.py             # Helper functions
├── config.py            # Dashboard config
└── requirements.txt     # Dashboard dependencies
```

**Features:**

1. **Evasion Detector Interface**:
   - Text areas for question and answer input
   - "Analyze" button
   - Prediction display with confidence scores
   - Color-coded result (green=direct, yellow=intermediate, red=fully_evasive)

2. **Explanation View**:
   - Highlighted text showing SHAP contributions
   - Word-level saliency (red=evasive signal, green=direct signal)
   - Attention visualization overlay

3. **Batch Upload**:
   - CSV/Excel upload (question, answer columns)
   - Batch processing
   - Downloadable results with predictions

4. **Dataset Explorer**:
   - Browse EvasionBench examples
   - Filter by label
   - Search functionality
   - Display model predictions vs. ground truth

5. **Model Performance Dashboard**:
   - Embed visualizations from notebooks
   - Confusion matrix
   - Performance metrics table
   - Model comparison charts

6. **About/Documentation**:
   - Project overview
   - Methodology summary
   - Links to notebooks and GitHub

**Deployment**:
- Streamlit Cloud deployment (free tier)
- or Docker container for self-hosting

---

### Artifact: CI/CD Pipeline

**File**: `.github/workflows/ci.yml`

**Pipeline stages:**

1. **Linting & Formatting**:
   - Black code formatting check
   - Flake8 linting
   - isort import sorting
   - mypy type checking

2. **Unit Tests**:
   - Pytest on `src/` modules
   - Coverage report (aim for >80%)
   - Upload coverage to Codecov

3. **Notebook Tests**:
   - Execute notebooks with papermill (smoke tests)
   - Verify notebooks run without errors
   - Cache data to avoid repeated downloads
   - Only on changed notebooks (optimization)

4. **API Tests**:
   - Start API in test mode
   - Run integration tests
   - Validate response schemas

5. **Docker Build**:
   - Build API Docker image
   - Tag with commit SHA and version
   - Push to registry (Docker Hub, GitHub Container Registry)

6. **Model Validation**:
   - Load production model
   - Run validation suite on test set
   - Assert macro-F1 >= threshold (e.g., 80%)
   - Prevent regression

**Continuous Deployment (optional):**
- On successful PR merge to main:
  - Deploy to staging environment
  - Run smoke tests
  - Manual approval for production
  - Deploy to production (Cloud Run, AWS Lambda, etc.)

**Additional workflows:**

`.github/workflows/model_training.yml`:
- Scheduled retraining (monthly)
- Trigger on new data
- MLflow logging
- Automated model evaluation
- Slack notification on completion

`.github/workflows/data_validation.yml`:
- Validate data quality on updates
- Great Expectations checks
- Alert on data drift

---

### Artifact: Experiment Tracking & Model Registry

**Setup:**

1. **MLflow**:
   - Local MLflow server or cloud (Databricks Community Edition)
   - Track all experiments from notebooks
   - Log: hyperparameters, metrics, artifacts (models, plots)
   - Model registry for versioning
   - Tag production model

2. **DVC (Data Version Control)**:
   - Track `data/` and `models/` directories
   - Remote storage (S3, Google Cloud Storage, DVC remote)
   - Enable reproducibility: `dvc pull` to get exact data/model versions
   - Pipeline definition (`dvc.yaml`):
     ```yaml
     stages:
       download_data:
         cmd: python scripts/download_data.py
         outs:
           - data/raw/evasionbench.parquet

       train_baseline:
         cmd: jupyter nbconvert --execute notebooks/04_traditional_ml_baselines.ipynb
         deps:
           - data/raw/evasionbench.parquet
           - notebooks/04_traditional_ml_baselines.ipynb
         outs:
           - models/baseline_traditional/
     ```
   - Reproducible pipeline: `dvc repro`

3. **Weights & Biases (W&B)** (optional, alternative to MLflow):
   - Sweep for hyperparameter optimization
   - Beautiful dashboards
   - Model artifact versioning

**Documentation:**
- `docs/mlflow_guide.md`: How to view experiments
- `docs/dvc_guide.md`: How to reproduce results

---

### Artifact: Monitoring & Observability

**Setup:**

1. **Prometheus + Grafana**:
   - Prometheus scrapes API metrics
   - Grafana dashboards:
     - Request rate, latency (p50, p95, p99)
     - Error rate by endpoint
     - Prediction distribution over time
     - Model confidence distribution

2. **Logging**:
   - Structured JSON logs (Python `logging`)
   - Log all predictions (question hash, prediction, confidence, timestamp)
   - Centralized logging (ELK stack, Loki, or CloudWatch)

3. **Data drift detection**:
   - Log input features (text length, similarity, etc.)
   - Evidently AI for drift detection
   - Alert if input distribution shifts significantly
   - Dashboard showing drift metrics

4. **Model performance monitoring**:
   - If labels become available (manual review), log ground truth
   - Track online performance (actual macro-F1)
   - Alert if performance degrades below threshold

**Implementation**:
- `docker-compose.prod.yml`: API + Prometheus + Grafana
- `monitoring/grafana_dashboard.json`: Pre-configured dashboard
- `monitoring/alert_rules.yml`: Prometheus alert rules

---

## Phase 8: Research Publication & Documentation

### Artifact: Research Paper (LaTeX)

**File**: `papers/evasionbench_analysis.tex`

**Structure:**

1. **Abstract**: Summary of comprehensive EvasionBench analysis
2. **Introduction**: Financial discourse analysis motivation, transparency in corporate communications
3. **Related Work**: Evasion detection, financial NLP, earnings call analysis
4. **Dataset Analysis**: EvasionBench overview, statistics, distributions (from Notebooks 1-3)
5. **Baseline Methods**: Traditional ML and neural baselines (Notebooks 4-5)
6. **Transformer Fine-Tuning**: Results from BERT family models (Notebook 6-7)
7. **Novel Contributions**:
   - Multi-task learning results (Notebook 8)
   - Explainability analysis (Notebook 9)
   - Label quality investigation (Notebook 10)
8. **Negative Results**: Failed hypotheses (Notebook 11), cross-domain limits (Notebook 12)
9. **Production System**: MLOps implementation overview
10. **Conclusion**: Findings, limitations, future work
11. **Appendices**: Detailed tables, additional visualizations

**Target venues:**
- arXiv pre-print (immediate)
- ACL Workshop on Economics and NLP
- EMNLP (main conference or workshop)
- Financial Application conferences (e.g., ECONLP, FinNLP)

**Outputs:**
- PDF paper (`papers/evasionbench_analysis.pdf`)
- Camera-ready submission materials
- Supplementary materials (notebooks, code, data)

---

### Artifact: Comprehensive README

**File**: `README.md`

**Sections:**

1. **Banner**: Project title, badges (build status, coverage, license)
2. **Overview**: What is EvasionBench portfolio, key findings
3. **Key Results Table**: Best model performance summary
4. **Quick Start**: Installation, run notebooks, try API
5. **Project Structure**: Directory tree with descriptions
6. **Notebooks Guide**:
   - Index of all 13 notebooks with summaries
   - Recommended reading order
   - Execution time estimates
7. **Reproducibility**:
   - Environment setup (conda, pip)
   - Data download instructions
   - DVC pull for exact data/model versions
   - Seed setting for deterministic results
8. **Deployment**:
   - How to run API locally
   - How to deploy to cloud
   - Dashboard usage
9. **Citation**: BibTeX for EvasionBench and this work
10. **License**: MIT or Apache 2.0
11. **Contact**: Links to author profiles, Discord community

**Visual elements:**
- Architecture diagram
- Sample prediction screenshots
- Key visualizations (confusion matrix, SHAP plot)

---

### Artifact: Blog Post / Portfolio Page

**Medium Article or Personal Site**:

**Title**: "Detecting Corporate Evasion: A Deep Dive into Financial NLP with EvasionBench"

**Sections**:
1. **Hook**: "When asked about revenue targets, how often do executives actually answer?"
2. **Dataset Introduction**: EvasionBench overview with examples
3. **Key Findings**: Highlight most interesting results
4. **Journey**: From EDA to SOTA model to production
5. **Interesting Failures**: What didn't work and why (vulnerability as strength)
6. **Live Demo**: Embedded Streamlit app or link
7. **Call to Action**: "Try the API", "Explore the notebooks", "Read the paper"

**Visuals**:
- Interactive charts (Plotly embedded)
- GIFs of dashboard in action
- Code snippets (key implementations)

**SEO**: Tag with #DataScience, #NLP, #MachineLearning, #FinTech

---

## Phase 9: Community & Open Source

1. **GitHub Repository**:
   - Open source under permissive license
   - CONTRIBUTING.md guidelines
   - Issues template for bugs/features
   - Discussions enabled for Q&A

2. **HuggingFace Integration**:
   - Upload best model to HuggingFace Hub: `{username}/evasion-detector`
   - Model card with performance, usage examples
   - Integration with Spaces for live demo

3. **Kaggle Notebook**:
   - Condensed version of best notebooks
   - Enable community to fork and experiment
   - Participate in discussions

4. **LinkedIn/Twitter Announcement**:
   - Project launch posts
   - Technical thread explaining approach
   - Engage with data science community

5. **Community Contributions**:
   - Release human-annotated dataset (Notebook 10) to improve upon Eva-4B-V2 labels
   - Encourage extensions (other languages, domains)

---

## Verification

**Quality Checks:**

1. **All notebooks execute successfully**:
   - `pytest --nbmake notebooks/*.ipynb`
   - Execution time <30 min per notebook (with caching)

2. **Code quality**:
   - `black .` (formatting)
   - `flake8 .` (linting, <10 issues)
   - `mypy src/` (type checking, strict mode)

3. **Test coverage**:
   - `pytest tests/ --cov=src --cov-report=html`
   - Coverage >80% for `src/` modules

4. **API functional tests**:
   - All endpoints return 200 for valid input
   - Prediction time <100ms p99 on CPU
   - Load test: 100 req/s without errors

5. **Reproducibility**:
   - Fresh environment setup from `requirements.txt`
   - `dvc pull` retrieves all artifacts
   - Re-run notebooks produces same results (seed-fixed)
   - CI/CD pipeline passes all checks

6. **Documentation completeness**:
   - README has all sections
   - All notebooks have markdown narratives
   - Code comments for complex sections
   - API has OpenAPI documentation

7. **Model performance validation**:
   - Best model achieves ≥80% macro-F1 on test set
   - Within 5% of reported Eva-4B-V2 performance (allowing for different model sizes)
   - Minority class (fully_evasive) F1 >60%

8. **Deployment validation**:
   - Docker image builds successfully
   - API container runs and responds
   - Streamlit app loads and functions
   - Grafana dashboard displays metrics

**Portfolio Review Checklist:**

- [ ] All 13 notebooks completed and documented
- [ ] At least 3 novel hypotheses tested (Notebooks 8, 11, 12)
- [ ] Publication-quality visualizations (high-res, clear labels)
- [ ] API deployed and accessible
- [ ] Dashboard publicly available
- [ ] GitHub repository public and polished
- [ ] README professional and comprehensive
- [ ] Research paper drafted (even if not submitted)
- [ ] Blog post/portfolio page published
- [ ] LinkedIn announcement posted
- [ ] HuggingFace model uploaded
- [ ] All code passes linting and tests
- [ ] DVC tracking in place
- [ ] MLflow experiments logged

---

## Timeline Estimate

Assuming **20 hours/week** dedicated work:

| Phase | Notebooks/Artifacts | Time Estimate |
|-------|---------------------|---------------|
| **Phase 1**: Foundation & Setup | Project structure, environment, data download | 1 week |
| **Phase 2**: EDA | Notebooks 1-3 | 2 weeks |
| **Phase 3**: Baselines | Notebooks 4-5 | 2 weeks |
| **Phase 4**: Transformers | Notebooks 6-7 | 2-3 weeks |
| **Phase 5**: Novel Research | Notebooks 8-10 | 3 weeks |
| **Phase 6**: Failures | Notebooks 11-12 | 1-2 weeks |
| **Phase 7**: Production | Notebook 13, API, Dashboard, CI/CD, Monitoring | 3-4 weeks |
| **Phase 8**: Documentation | Paper, README, Blog | 1-2 weeks |
| **Phase 9**: Community | Open source, HuggingFace, Launch | 1 week |
| **Total** | | **16-20 weeks** (~4-5 months) |

**Optimization**: Phases can overlap (e.g., draft README while building notebooks, set up CI/CD early).

---

## Decisions

**Research vs. Engineering Balance**:
- Chose comprehensive approach: deep research + production deployment
- Both demonstrate rigor and practical skills for academic/industry roles

**Compute Strategy**:
- Focus on moderate-sized models (BERT, DeBERTa-base) trainable on Colab Pro/Kaggle
- Use LoRA for efficient large model experimentation
- Avoid competing directly with Eva-4B-V2 (4B params) on size, focus on efficiency

**Honesty About Failures**:
- Dedicated notebooks (11-12) for negative results
- Demonstrates scientific maturity and critical thinking
- Differentiates from "only positive results" portfolios

**Class Imbalance Philosophy**:
- Treat as feature, not bug: opportunity to showcase advanced techniques
- Focus on macro-F1 and minority class F1 (aligned with SOTA metric)

**Deployment Scope**:
- Full MLOps stack shows engineering depth
- Aligns with research/academic goal (reproducibility emphasis)
- CI/CD and monitoring demonstrate production-readiness

**Open Source Release**:
- Maximize portfolio visibility and community impact
- Human-annotated subset is unique contribution

---

## Next Steps

When ready to begin implementation:

1. Create project directory structure as specified in Phase 1
2. Set up Git repository and initialize DVC
3. Create virtual environment and install base dependencies
4. Run data download script and verify dataset
5. Begin Notebook 1 (Statistical EDA)
6. Iterate through phases sequentially (or parallelize EDA notebooks)

---

**DRAFT Plan Complete** — This is a comprehensive roadmap for building an exhaustive EvasionBench data science portfolio with research depth and production deployment. Please review and let me know if you'd like to:

- Adjust scope (add/remove notebooks or components)
- Modify research hypotheses
- Change deployment targets
- Refine any section

Once approved, I can scaffold the repo and create the first notebooks. Thank you!
