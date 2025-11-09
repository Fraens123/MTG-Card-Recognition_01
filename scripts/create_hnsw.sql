-- Create HNSW index for cosine distance search
-- Adjust vector_cosine_ops to vector_l2_ops if you prefer L2
CREATE INDEX IF NOT EXISTS cards_embedding_hnsw
ON cards
USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 200);
