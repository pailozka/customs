-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the table for HS codes and their embeddings
CREATE TABLE IF NOT EXISTS tnved_codes (
    id SERIAL PRIMARY KEY,
    code VARCHAR(12) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

-- We are intentionally omitting the ivfflat index to use exact cosine similarity search 
-- as it is extremely fast for ~4,000 vectors and avoids potential recall degradation.
