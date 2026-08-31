CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  id            bigserial PRIMARY KEY,
  chunk_uid     text UNIQUE NOT NULL,
  corpus        text NOT NULL,
  doc_path      text NOT NULL,
  breadcrumb    text NOT NULL,
  heading_level int,
  content       text NOT NULL,
  word_count    int NOT NULL,
  has_code      boolean NOT NULL DEFAULT false,
  identifiers   text[] NOT NULL DEFAULT '{}',
  embedding     vector(768),
  tsv           tsvector GENERATED ALWAYS AS (
                  to_tsvector('english', breadcrumb || ' ' || content)
                ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx    ON chunks USING gin(tsv);
CREATE INDEX IF NOT EXISTS chunks_ident_idx  ON chunks USING gin(identifiers);
CREATE INDEX IF NOT EXISTS chunks_corpus_idx ON chunks (corpus);
