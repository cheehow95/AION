"""Tests for AION Streaming Responses"""

import pytest
import asyncio
import sys
sys.path.insert(0, '.')

from src.runtime.streaming import (
    StreamChunk, StreamingResponse, StreamingAgent,
    stream_local_response, stream_cloud_response
)


class TestStreamChunk:
    """Test StreamChunk dataclass."""
    
    def test_chunk_creation(self):
        """Test creating a StreamChunk."""
        chunk = StreamChunk(content="Hello")
        
        assert chunk.content == "Hello"
        assert chunk.done == False
        assert chunk.metadata is None
    
    def test_chunk_done_flag(self):
        """Test chunk with done flag."""
        chunk = StreamChunk(content="", done=True)
        
        assert chunk.done == True
    
    def test_chunk_with_metadata(self):
        """Test chunk with metadata."""
        chunk = StreamChunk(
            content="test",
            metadata={"tokens": 5}
        )
        
        assert chunk.metadata["tokens"] == 5


class TestStreamingResponse:
    """Test StreamingResponse class."""
    
    def test_init(self):
        """Test response initialization."""
        response = StreamingResponse()
        
        assert response._buffer == []
        assert response._done == False
        assert response._callbacks == []
    
    @pytest.mark.asyncio
    async def test_write(self):
        """Test writing to stream."""
        response = StreamingResponse()
        
        await response.write("Hello")
        await response.write(" World")
        
        assert "Hello" in response._buffer
        assert " World" in response._buffer
    
    @pytest.mark.asyncio
    async def test_write_calls_callback(self):
        """Test that write triggers callbacks."""
        response = StreamingResponse()
        chunks_received = []
        
        response.on_chunk(lambda c: chunks_received.append(c.content))
        await response.write("test")
        
        assert "test" in chunks_received
    
    @pytest.mark.asyncio
    async def test_write_calls_async_callback(self):
        """Test async callback support."""
        response = StreamingResponse()
        chunks_received = []
        
        async def async_handler(chunk):
            chunks_received.append(chunk.content)
        
        response.on_chunk(async_handler)
        await response.write("async test")
        
        assert "async test" in chunks_received
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the stream."""
        response = StreamingResponse()
        
        await response.write("data")
        await response.close()
        
        assert response._done == True
        assert response.is_done == True
    
    @pytest.mark.asyncio
    async def test_close_sends_done_chunk(self):
        """Test that close sends done chunk to callbacks."""
        response = StreamingResponse()
        final_chunk = None
        
        def handler(chunk):
            nonlocal final_chunk
            final_chunk = chunk
        
        response.on_chunk(handler)
        await response.close()
        
        assert final_chunk is not None
        assert final_chunk.done == True
    
    def test_get_full_response(self):
        """Test getting full buffered response."""
        response = StreamingResponse()
        response._buffer = ["Hello", " ", "World"]
        
        full = response.get_full_response()
        
        assert full == "Hello World"
    
    def test_is_done_property(self):
        """Test is_done property."""
        response = StreamingResponse()
        
        assert response.is_done == False
        
        response._done = True
        
        assert response.is_done == True
    
    def test_multiple_callbacks(self):
        """Test multiple callback registration."""
        response = StreamingResponse()
        
        response.on_chunk(lambda c: None)
        response.on_chunk(lambda c: None)
        response.on_chunk(lambda c: None)
        
        assert len(response._callbacks) == 3


class TestStreamLocalResponse:
    """Test stream_local_response function."""
    
    @pytest.mark.asyncio
    async def test_stream_local_response(self):
        """Test local streaming."""
        from src.runtime.local_engine import LocalReasoningEngine
        
        engine = LocalReasoningEngine()
        chunks = []
        
        async for chunk in stream_local_response("test prompt", engine):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert any(c.done for c in chunks)  # Should have final done chunk
    
    @pytest.mark.asyncio
    async def test_stream_local_response_content(self):
        """Test that streaming produces content."""
        from src.runtime.local_engine import LocalReasoningEngine
        
        engine = LocalReasoningEngine()
        content = []
        
        async for chunk in stream_local_response("analyze this", engine):
            content.append(chunk.content)
        
        full_response = "".join(content)
        assert len(full_response) > 0


class TestStreamCloudResponse:
    """Test stream_cloud_response function."""
    
    @pytest.mark.asyncio
    async def test_stream_cloud_response(self):
        """Test cloud streaming (simulated)."""
        chunks = []
        
        async for chunk in stream_cloud_response("test", "openai"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        # Last chunk should be done
        assert chunks[-1].done == True
    
    @pytest.mark.asyncio
    async def test_stream_cloud_response_provider(self):
        """Test provider name appears in response."""
        content = []
        
        async for chunk in stream_cloud_response("test", "anthropic"):
            content.append(chunk.content)
        
        full = "".join(content)
        assert "anthropic" in full


class TestStreamingAgent:
    """Test StreamingAgent class."""
    
    def test_init(self):
        """Test agent initialization."""
        agent = StreamingAgent(agent_runner=None)
        
        assert agent.runner is None
    
    @pytest.mark.asyncio
    async def test_run_streaming(self):
        """Test running agent with streaming."""
        agent = StreamingAgent(agent_runner=None)
        chunks = []
        
        async for chunk in agent.run_streaming("TestAgent", "hello"):
            chunks.append(chunk)
        
        assert len(chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
