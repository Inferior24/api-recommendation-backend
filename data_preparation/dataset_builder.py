# dataset_builder.py
from pydantic import BaseModel, HttpUrl, Field, ValidationError
from typing import List, Optional
from datetime import datetime
import json
from pathlib import Path

# -------------------------------
# Step 1: Schema Definition
# -------------------------------
class APIMetadata(BaseModel):
    api_name: str
    description: str
    endpoints: List[str]
    documentation_url: Optional[HttpUrl]
    version: str = Field(default="1.0")
    popularity_score: float = Field(ge=0.0, le=1.0)
    last_updated: datetime
    doc_quality: float = Field(ge=0.0, le=1.0)


# -------------------------------
# Step 2: Validation & Saving
# -------------------------------
def validate_and_save_dataset(input_path="data/api_dataset_raw.json", 
                              output_path="data/api_dataset_validated.json"):
    Path("data").mkdir(exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    validated_entries = []
    for entry in raw_data:
        try:
            # Convert string date → datetime
            if isinstance(entry.get("last_updated"), str):
                entry["last_updated"] = datetime.strptime(entry["last_updated"], "%Y-%m-%d")

            # Validate via schema
            api = APIMetadata(**entry)
            validated_entries.append(api.model_dump())

        except ValidationError as e:
            print(f"❌ Validation failed for {entry.get('api_name', 'UNKNOWN')}")
            print(e)
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(validated_entries, f, indent=4, default=str)

    print(f"✅ Validation complete. Saved cleaned dataset to: {output_path}")
    print(f"✅ Total valid entries: {len(validated_entries)}")


if __name__ == "__main__":
    validate_and_save_dataset()
