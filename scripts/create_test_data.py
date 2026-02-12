"""Create minimal test data for transformer training validation."""

import pandas as pd

# Create minimal test dataset
test_data = {
    "question": [
        "What is your revenue?",
        "Can you disclose your earnings?",
        "Tell me about your financial performance.",
        "What are your sales figures?",
        "How much profit did you make?",
        "Is your business profitable?",
        "What's your quarterly income?",
        "Can you share your annual report?",
        "What are your operating costs?",
        "How's your company doing financially?",
        # Non-evasive examples
        "What products do you sell?",
        "How many employees do you have?",
        "When was your company founded?",
        "Who is your CEO?",
        "Where are your headquarters?",
        "What industry are you in?",
        "Do you have international offices?",
        "What's your company mission?",
        "How long have you been in business?",
        "What are your core values?",
    ]
    * 5,  # Repeat to get 100 rows
    "answer": [
        "We cannot disclose financial information.",
        "That's confidential.",
        "I'm not at liberty to discuss earnings.",
        "Financial details are private.",
        "We don't share profit information.",
        "That's not public information.",
        "I can't discuss revenue figures.",
        "Our financial reports are confidential.",
        "We keep cost data private.",
        "Financial performance is not disclosed.",
        # Non-evasive answers
        "We sell software products.",
        "We have 500 employees.",
        "We were founded in 2010.",
        "Jane Doe is our CEO.",
        "We're based in San Francisco.",
        "We're in the technology sector.",
        "Yes, we have offices in Europe.",
        "To innovate and serve customers.",
        "For over 10 years.",
        "Integrity and excellence.",
    ]
    * 5,
    "label": [
        "evasive",
        "evasive",
        "evasive",
        "evasive",
        "evasive",
        "evasive",
        "evasive",
        "evasive",
        "evasive",
        "evasive",
        "non_evasive",
        "non_evasive",
        "non_evasive",
        "non_evasive",
        "non_evasive",
        "non_evasive",
        "non_evasive",
        "non_evasive",
        "non_evasive",
        "non_evasive",
    ]
    * 5,
}

df = pd.DataFrame(test_data)
df.to_parquet("data/processed/evasionbench_prepared.parquet", index=False)
print(f"Created test data with {len(df)} rows")
print(f"Label distribution:\n{df['label'].value_counts()}")
