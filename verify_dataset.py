# verify_dataset.py
import pandas as pd
import json
from pathlib import Path

# Path to the validated dataset
data_path = Path("data/api_dataset_validated.json")

if not data_path.exists():
    print("❌ File not found: data/api_dataset_validated.json")
    print("Make sure you ran dataset_builder.py first.")
else:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    print("\n=== Dataset Preview (Top 10 Entries) ===\n")
    print(df[["api_name", "version", "popularity_score", "doc_quality", "last_updated"]].head(10))

    print("\n=== Dataset Info ===")
    print(f"Total Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\n✅ File loaded successfully from: {data_path}")
