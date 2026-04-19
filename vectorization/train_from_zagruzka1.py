"""
Build vector DB from Загрузка 1.

For each ТН ВЭД code found in Загрузка 1:
  - collect all product descriptions (name, material, manufacturer, Russian label)
  - aggregate into one rich text
  - embed with text-embedding-3-small
  - store in pgvector

Usage:
    python train_from_zagruzka1.py
"""

import os
import re
import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "../data_files/Загрузка 1.xlsx")
SHEET_NAME = "装车明细"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "customs_vectors"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )


def setup_db(conn):
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tnved_codes (
            id SERIAL PRIMARY KEY,
            code VARCHAR(12) UNIQUE NOT NULL,
            description TEXT NOT NULL,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    print("DB schema ready.")


def clean_code(val) -> str | None:
    if pd.isna(val):
        return None
    code = str(val).split(".")[0].strip()
    if not code.isdigit():
        return None
    return code.zfill(10)


def clean_text(val) -> str:
    if pd.isna(val):
        return ""
    # strip Chinese characters, keep Russian/Latin/digits/punctuation
    text = str(val)
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    return text.strip()


def load_zagruzka1() -> dict[str, list[str]]:
    """
    Returns dict: { tnved_code -> [description1, description2, ...] }
    """
    print(f"Reading {EXCEL_PATH}...")
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=0)
    df.columns = df.columns.str.strip()
    print(f"  {len(df)} rows loaded.")

    code_descriptions: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        code = clean_code(row.get("код"))
        if not code:
            continue

        parts = []
        for col in ["名称(наименование)", "НАИМЕНОВАНИЕ ТОВАРА",
                    "наименование для бирки", "成分（состав)", "изготовитель"]:
            val = clean_text(row.get(col, ""))
            if val:
                parts.append(val)

        description = "; ".join(parts)
        if not description:
            continue

        if code not in code_descriptions:
            code_descriptions[code] = []
        code_descriptions[code].append(description)

    print(f"  Found {len(code_descriptions)} unique ТН ВЭД codes.")
    return code_descriptions


def aggregate(descriptions: list[str]) -> str:
    """Merge all product descriptions for one code into one text."""
    # deduplicate while preserving order
    seen = set()
    unique = []
    for d in descriptions:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return " | ".join(unique[:20])  # cap at 20 examples per code


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]


def main():
    code_descriptions = load_zagruzka1()

    codes = list(code_descriptions.keys())
    aggregated = [aggregate(code_descriptions[c]) for c in codes]

    print(f"\nEmbedding {len(codes)} codes in batches of 500...")
    all_embeddings: list[list[float]] = []

    batch_size = 500
    for i in range(0, len(aggregated), batch_size):
        batch = aggregated[i:i + batch_size]
        embeddings = embed_batch(batch)
        all_embeddings.extend(embeddings)
        print(f"  batch {i // batch_size + 1}/{(len(aggregated) + batch_size - 1) // batch_size} done")

    print("\nConnecting to DB...")
    conn = get_db()
    setup_db(conn)

    cur = conn.cursor()
    data = [
        (codes[i], aggregated[i], all_embeddings[i])
        for i in range(len(codes))
    ]
    execute_batch(cur, """
        INSERT INTO tnved_codes (code, description, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (code) DO UPDATE
            SET description = EXCLUDED.description,
                embedding = EXCLUDED.embedding;
    """, data, page_size=100)
    conn.commit()
    cur.close()
    conn.close()

    print(f"\nDone. {len(codes)} codes stored in pgvector.")


if __name__ == "__main__":
    main()
