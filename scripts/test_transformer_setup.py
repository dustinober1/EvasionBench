"""Quick test to verify transformer setup works."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Test imports
print("Testing imports...")
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"✓ torch {torch.__version__}")
    print(f"✓ transformers {transformers.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test model loading
print("\nTesting model loading...")
try:
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print(f"✓ Model loaded: {model_name}")
    print(f"✓ Vocabulary size: {len(tokenizer)}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)

# Test tokenization
print("\nTesting tokenization...")
try:
    text = "What is the revenue? [SEP] We made $1M in revenue."
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    print(f"✓ Input IDs shape: {inputs['input_ids'].shape}")
    print(f"✓ Attention mask shape: {inputs['attention_mask'].shape}")
except Exception as e:
    print(f"✗ Tokenization failed: {e}")
    sys.exit(1)

# Test model forward pass
print("\nTesting model forward pass...")
try:
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"✓ Logits shape: {outputs.logits.shape}")
    print(f"✓ Prediction: {outputs.logits.argmax().item()}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

print("\n✓ All tests passed!")
