import os
import csv
import psycopg2
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

def test_search():
    if not os.path.exists('products_test_set.csv'):
        print("products_test_set.csv not found!")
        return

    try:
        conn = get_db_connection()
    except psycopg2.Error as e:
        print(f"Could not connect to database: {e}")
        return
        
    top1_hits = 0
    top5_hits = 0
    total = 0
    
    results = []

    with open('products_test_set.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth = row['ground_truth_code']
            product_name = row['product_name']
            manufacturer = row['manufacturer']
            
            search_text = f"{product_name} {manufacturer}".strip()
            
            response = client.embeddings.create(
                input=search_text,
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
            
            cur = conn.cursor()
            cur.execute("""
                SELECT code, description, (embedding <=> %s::vector) AS distance
                FROM tnved_codes
                ORDER BY distance ASC
                LIMIT 5;
            """, (query_embedding,))
            
            top_5_results = cur.fetchall()
            cur.close()
            
            top_5_codes = [r[0] for r in top_5_results]
            
            rank = -1
            if ground_truth in top_5_codes:
                rank = top_5_codes.index(ground_truth) + 1
                top5_hits += 1
                if rank == 1:
                    top1_hits += 1
            
            results.append({
                'product_name': product_name,
                'ground_truth': ground_truth,
                'top_5_codes': ",".join(top_5_codes),
                'rank_of_truth': rank
            })
            total += 1
            
            if total % 10 == 0:
                print(f"Processed {total} test items...")

    conn.close()

    with open('test_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['product_name', 'ground_truth', 'top_5_codes', 'rank_of_truth'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n--- Results ---")
    print(f"Total tested: {total}")
    print(f"Top-1 accuracy: {top1_hits/total*100:.1f}% ({top1_hits}/{total})")
    print(f"Top-5 accuracy: {top5_hits/total*100:.1f}% ({top5_hits}/{total})")

if __name__ == "__main__":
    test_search()
