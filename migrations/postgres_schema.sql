-- PostgreSQL Schema for XYZCare RAG
-- Full-text search with tsvector and automatic updates

-- Enable pg_trgm extension for fuzzy text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Manuals table
CREATE TABLE IF NOT EXISTS manuals (
    manual_id VARCHAR(255) PRIMARY KEY,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    num_pages INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pages table with full-text search
CREATE TABLE IF NOT EXISTS pages (
    id SERIAL PRIMARY KEY,
    manual_id VARCHAR(255) NOT NULL REFERENCES manuals(manual_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    text_content TEXT,
    headings TEXT,
    text_vector tsvector,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(manual_id, page_number)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_pages_manual_id ON pages(manual_id);
CREATE INDEX IF NOT EXISTS idx_pages_text_vector ON pages USING GIN(text_vector);

-- Function to automatically update text_vector on INSERT/UPDATE
CREATE OR REPLACE FUNCTION pages_text_vector_update()
RETURNS TRIGGER AS $$
BEGIN
    NEW.text_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.headings, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.text_content, '')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to call the function before INSERT or UPDATE
DROP TRIGGER IF EXISTS trigger_pages_text_vector_update ON pages;
CREATE TRIGGER trigger_pages_text_vector_update
    BEFORE INSERT OR UPDATE OF text_content, headings ON pages
    FOR EACH ROW
    EXECUTE FUNCTION pages_text_vector_update();
