# embed_catalog.py
# One-off script to add embeddings to your perfume catalog CSV.

import csv
import json
import os
from openai import OpenAI

# ⚠️ 1) Update this to the name of your "clean" perfume CSV.
# For now, let's assume you export a small test file first:
INPUT_CSV = "perfumes_source.csv"
OUTPUT_CSV = "perfumes_with_embeddings.csv"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def build_embedding_text(row: dict) -> str:
    """
    Combine useful fields into a single description string
    for the embedding model.
    Adjust the keys if your CSV uses slightly different names.
    """
    parts = [
        row.get("name", ""),
        row.get("brand", ""),
        row.get("description", ""),
        row.get("notestop", ""),
        row.get("notesheart", ""),
        row.get("notesbase", ""),
    ]
    # Join with spaces, strip extra whitespace
    return " | ".join(p for p in parts if p).strip()

def main():
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    rows = []

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "embedding" not in (fieldnames or []):
        fieldnames = list(fieldnames) + ["embedding"]

    print(f"📦 Loaded {len(rows)} perfumes from {INPUT_CSV}")

    for i, row in enumerate(rows, start=1):
        text = build_embedding_text(row)
        if not text:
            print(f"⚠️ Row {i} has no text; skipping embedding.")
            row["embedding"] = ""
            continue

        print(f"🧠 Embedding {i}/{len(rows)}: {row.get('name','(no name)')}")
        emb = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        ).data[0].embedding

        # Store as JSON string so Swift can parse it later
        row["embedding"] = json.dumps(emb)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Wrote {len(rows)} perfumes with embeddings to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()



