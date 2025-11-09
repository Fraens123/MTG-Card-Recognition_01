CREATE EXTENSION IF NOT EXISTS vector;

-- cards table stores metadata + embedding
CREATE TABLE IF NOT EXISTS cards (
    id               BIGSERIAL PRIMARY KEY,
    card_uuid        TEXT UNIQUE,
    name             TEXT,
    set_code         TEXT,
    collector_number TEXT,
    image_path       TEXT,
    embedding        vector(512)
);

-- optional: if you use half precision vectors (pgvector >= 0.7)
-- ALTER TABLE cards ALTER COLUMN embedding TYPE halfvec(512);
