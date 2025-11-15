# text_cleaner.py

import json
import re
import nltk
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
from pathlib import Path

# ---------------------------
# Step 1: Setup resources
# ---------------------------
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm', disable=["ner", "parser"])

# ---------------------------
# Step 2: Text cleaning helper
# ---------------------------
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)             # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)     # remove special chars
    text = re.sub(r"\s+", " ", text)                # normalize spaces
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# ---------------------------
# Step 3: Full preprocessing pipeline
# ---------------------------
def preprocess_dataset(input_path="data/api_dataset_validated.json", 
                       output_path="data/api_dataset_cleaned.json"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []

    print("\n🚀 Starting preprocessing on dataset...\n")
    for entry in tqdm(data, desc="Cleaning entries"):
        # Combine description + endpoints into single text
        endpoints_text = " ".join(entry.get("endpoints", []))
        combined_text = f"{entry['description']} {endpoints_text}"

        # Clean and lemmatize
        cleaned = clean_text(combined_text)
        lemmatized = lemmatize_text(cleaned)

        entry["cleaned_text"] = lemmatized
        cleaned_data.append(entry)

    Path("data").mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4)

    print(f"\n✅ Preprocessing complete. Cleaned dataset saved to: {output_path}")
    print(f"✅ Total entries processed: {len(cleaned_data)}")


if __name__ == "__main__":
    preprocess_dataset()
