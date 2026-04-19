import os
import csv
import psycopg2
from psycopg2.extras import execute_batch
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "customs_vectors"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres")
    )

def embed_codes():
    if not os.path.exists('codes_for_embedding.csv'):
        print("codes_for_embedding.csv not found!")
        return

    codes = []
    texts_to_embed = []
    
    with open('codes_for_embedding.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['code']
            description = row['description']
            text = f"{code} {description}"
            codes.append((code, description))
            texts_to_embed.append(text)

    print(f"Loaded {len(codes)} codes. Starting embeddings in batches...")
    
    batch_size = 500
    all_embeddings = []
    
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        response = client.embeddings.create(
            input=batch_texts,
            model="text-embedding-3-small"
        )
        all_embeddings.extend([res.embedding for res in response.data])
        print(f"Embedded batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size}")

    print("Connecting to PostgreSQL...")
    conn = get_db_connection()
    cur = conn.cursor()
    
    insert_query = """
    INSERT INTO tnved_codes (code, description, embedding)
    VALUES (%s, %s, %s)
    ON CONFLICT (code) DO NOTHING;
    """
    
    data_to_insert = [
        (codes[i][0], codes[i][1], all_embeddings[i]) for i in range(len(codes))
    ]
    
    print("Inserting data into database...")
    execute_batch(cur, insert_query, data_to_insert, page_size=100)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Done! All vectors stored.")

if __name__ == "__main__":
    embed_codes()
