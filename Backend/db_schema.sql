-- 1. Enable Vector Extension (for RAG)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create Jobs Table (for tracking analysis tasks)
CREATE TABLE IF NOT EXISTS jobs (
    task_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    filename TEXT,
    message TEXT,
    progress INTEGER DEFAULT 0,
    result_path TEXT,
    error TEXT,
    report_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);

-- 3. Create Document Chunks Table (for RAG/Vector Search)
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    task_id TEXT NOT NULL,
    content TEXT,
    chunk_index INTEGER,
    metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Create Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
ON document_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_document_chunks_task_id ON document_chunks(task_id);
