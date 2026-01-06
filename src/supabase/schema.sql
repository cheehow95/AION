-- ============================================================
-- AION Supabase Database Schema
-- ============================================================
-- Run this in Supabase SQL Editor after creating your project

-- ============================================================
-- Profiles Table (extends auth.users)
-- ============================================================
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    display_name TEXT,
    avatar_url TEXT,
    bio TEXT,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Auto-create profile on user signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO profiles (id, display_name)
    VALUES (NEW.id, NEW.raw_user_meta_data->>'display_name');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- ============================================================
-- Knowledge Base Table
-- ============================================================
CREATE TABLE IF NOT EXISTS knowledge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),  -- For semantic search (requires pgvector)
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_knowledge_user ON knowledge(user_id);
CREATE INDEX idx_knowledge_tags ON knowledge USING GIN(tags);

-- ============================================================
-- Memories/Conversations Table
-- ============================================================
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_memories_user_session ON memories(user_id, session_id);
CREATE INDEX idx_memories_created ON memories(created_at DESC);

-- ============================================================
-- User Files Table (tracks storage uploads)
-- ============================================================
CREATE TABLE IF NOT EXISTS user_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    bucket TEXT NOT NULL,
    path TEXT NOT NULL,
    filename TEXT NOT NULL,
    size_bytes BIGINT,
    mime_type TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_files_user ON user_files(user_id);
CREATE UNIQUE INDEX idx_files_path ON user_files(bucket, path);

-- ============================================================
-- Row Level Security (RLS) Policies
-- ============================================================

-- Enable RLS on all tables
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_files ENABLE ROW LEVEL SECURITY;

-- Profiles: Users can only access their own profile
CREATE POLICY "Users can view own profile" 
    ON profiles FOR SELECT 
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" 
    ON profiles FOR UPDATE 
    USING (auth.uid() = id);

-- Knowledge: Users can only access their own knowledge
CREATE POLICY "Users can view own knowledge" 
    ON knowledge FOR SELECT 
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own knowledge" 
    ON knowledge FOR INSERT 
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own knowledge" 
    ON knowledge FOR UPDATE 
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own knowledge" 
    ON knowledge FOR DELETE 
    USING (auth.uid() = user_id);

-- Memories: Users can only access their own memories
CREATE POLICY "Users can view own memories" 
    ON memories FOR SELECT 
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own memories" 
    ON memories FOR INSERT 
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own memories" 
    ON memories FOR DELETE 
    USING (auth.uid() = user_id);

-- User Files: Users can only access their own files
CREATE POLICY "Users can view own files" 
    ON user_files FOR SELECT 
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own files" 
    ON user_files FOR INSERT 
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own files" 
    ON user_files FOR DELETE 
    USING (auth.uid() = user_id);

-- ============================================================
-- Storage Bucket Policies (run in Supabase Storage settings)
-- ============================================================
-- Note: Create these buckets in Supabase Dashboard:
--   1. avatars (public)
--   2. uploads (private)
--   3. exports (private)

-- Example storage policy for 'uploads' bucket:
-- CREATE POLICY "Users can upload to own folder"
-- ON storage.objects FOR INSERT
-- WITH CHECK (bucket_id = 'uploads' AND auth.uid()::text = (storage.foldername(name))[1]);
