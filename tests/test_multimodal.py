"""Tests for AION Multimodal Module"""

import pytest
import asyncio
import sys
sys.path.insert(0, '.')

from src.multimodal.vision import (
    VisionInput, VisionAnalysis, VisionProcessor, ImageFormat, DetectedObject
)
from src.multimodal.audio import (
    AudioInput, AudioProcessor, TextToSpeech, AudioFormat, TranscriptionResult, SynthesisRequest
)
from src.multimodal.document import (
    DocumentInput, DocumentProcessor, DocumentFormat, DocumentAnalysis, ExtractedTable
)
from src.multimodal.screen import (
    ScreenCapture, ScreenProcessor, UIElement, ActionPlanner, UIElementType, UIAction
)
from src.multimodal.memory import (
    MultimodalMemory, MultimodalMemoryEntry, MemoryType, MemoryEmbedding
)


class TestVision:
    """Test vision processing."""
    
    def test_image_format_from_extension(self):
        """Test image format detection."""
        assert ImageFormat.from_extension("jpg") == ImageFormat.JPEG
        assert ImageFormat.from_extension("png") == ImageFormat.PNG
        assert ImageFormat.from_extension("webp") == ImageFormat.WEBP
    
    def test_vision_input_creation(self):
        """Test VisionInput creation."""
        input = VisionInput(
            data=b"test",
            format=ImageFormat.PNG,
            width=640,
            height=480
        )
        assert input.width == 640
        assert input.height == 480
        assert input.aspect_ratio == 640 / 480
        assert len(input.id) > 0
    
    def test_vision_input_base64(self):
        """Test base64 encoding."""
        input = VisionInput(data=b"hello")
        assert input.base64 == "aGVsbG8="
    
    @pytest.mark.asyncio
    async def test_vision_processor_analyze(self):
        """Test image analysis."""
        processor = VisionProcessor()
        input = VisionInput(data=b"test", width=100, height=100)
        
        analysis = await processor.analyze(input)
        
        assert analysis.input_id == input.id
        assert isinstance(analysis.description, str)
    
    def test_detected_object(self):
        """Test DetectedObject dataclass."""
        obj = DetectedObject(
            label="cat",
            confidence=0.95,
            bounding_box=(10, 20, 100, 80)
        )
        assert obj.label == "cat"
        assert obj.confidence == 0.95


class TestAudio:
    """Test audio processing."""
    
    def test_audio_format_from_extension(self):
        """Test audio format detection."""
        assert AudioFormat.from_extension("wav") == AudioFormat.WAV
        assert AudioFormat.from_extension("mp3") == AudioFormat.MP3
    
    def test_audio_input_creation(self):
        """Test AudioInput creation."""
        input = AudioInput(
            data=b"audio",
            format=AudioFormat.WAV,
            duration_seconds=5.0,
            sample_rate=16000
        )
        assert input.duration_seconds == 5.0
        assert input.sample_rate == 16000
        assert input.size_bytes == 5
    
    @pytest.mark.asyncio
    async def test_audio_processor_transcribe(self):
        """Test audio transcription."""
        processor = AudioProcessor()
        input = AudioInput(duration_seconds=2.0)
        
        result = await processor.transcribe(input)
        
        assert isinstance(result, TranscriptionResult)
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_text_to_speech(self):
        """Test TTS synthesis."""
        tts = TextToSpeech()
        request = SynthesisRequest(text="Hello world")
        
        output = await tts.synthesize(request)
        
        assert len(output.data) > 0
        assert output.duration_seconds > 0


class TestDocument:
    """Test document processing."""
    
    def test_document_format_from_extension(self):
        """Test document format detection."""
        assert DocumentFormat.from_extension("pdf") == DocumentFormat.PDF
        assert DocumentFormat.from_extension("html") == DocumentFormat.HTML
        assert DocumentFormat.from_extension("md") == DocumentFormat.MD
    
    def test_document_input_creation(self):
        """Test DocumentInput creation."""
        input = DocumentInput(
            data=b"<html></html>",
            format=DocumentFormat.HTML,
            filename="test.html"
        )
        assert input.filename == "test.html"
        assert input.size_bytes == 13
    
    @pytest.mark.asyncio
    async def test_document_extract_text(self):
        """Test text extraction."""
        processor = DocumentProcessor()
        input = DocumentInput(
            data=b"Hello world",
            format=DocumentFormat.TXT
        )
        
        text = await processor.extract_text(input)
        
        assert "Hello world" in text
    
    @pytest.mark.asyncio
    async def test_html_text_extraction(self):
        """Test HTML text extraction."""
        processor = DocumentProcessor()
        input = DocumentInput(
            data=b"<html><body><p>Test content</p></body></html>",
            format=DocumentFormat.HTML
        )
        
        text = await processor.extract_text(input)
        
        assert "Test content" in text
    
    def test_extracted_table_to_markdown(self):
        """Test table to markdown conversion."""
        from src.multimodal.document import TableCell
        
        table = ExtractedTable(
            cells=[
                TableCell(content="A", row=0, column=0),
                TableCell(content="B", row=0, column=1),
                TableCell(content="1", row=1, column=0),
                TableCell(content="2", row=1, column=1),
            ],
            rows=2,
            columns=2
        )
        
        md = table.to_markdown()
        assert "|" in md
        assert "---" in md


class TestScreen:
    """Test screen/UI processing."""
    
    def test_ui_element_creation(self):
        """Test UIElement dataclass."""
        elem = UIElement(
            type=UIElementType.BUTTON,
            text="Click me",
            bounding_box=(100, 200, 80, 30)
        )
        assert elem.type == UIElementType.BUTTON
        assert elem.center == (140, 215)
        assert elem.clickable == True
    
    def test_screen_capture_creation(self):
        """Test ScreenCapture dataclass."""
        capture = ScreenCapture(width=1920, height=1080)
        assert capture.size == (1920, 1080)
    
    @pytest.mark.asyncio
    async def test_screen_processor_capture(self):
        """Test screen capture."""
        processor = ScreenProcessor()
        
        capture = await processor.capture_screen()
        
        assert capture.width > 0
        assert capture.height > 0
    
    @pytest.mark.asyncio
    async def test_screen_processor_detect_elements(self):
        """Test element detection."""
        processor = ScreenProcessor()
        capture = ScreenCapture(width=800, height=600)
        
        elements = await processor.detect_elements(capture)
        
        assert isinstance(elements, list)
    
    @pytest.mark.asyncio
    async def test_action_planner(self):
        """Test action planning."""
        planner = ActionPlanner()
        capture = ScreenCapture(width=800, height=600)
        
        plan = await planner.plan_action("click Submit button", capture)
        
        assert plan.goal == "click Submit button"


class TestMultimodalMemory:
    """Test multimodal memory."""
    
    def test_memory_embedding_similarity(self):
        """Test embedding similarity calculation."""
        e1 = MemoryEmbedding(vector=[1.0, 0.0, 0.0])
        e2 = MemoryEmbedding(vector=[1.0, 0.0, 0.0])
        e3 = MemoryEmbedding(vector=[0.0, 1.0, 0.0])
        
        assert e1.cosine_similarity(e2) == pytest.approx(1.0)
        assert e1.cosine_similarity(e3) == pytest.approx(0.0)
    
    @pytest.mark.asyncio
    async def test_memory_store_and_retrieve(self):
        """Test storing and retrieving memories."""
        memory = MultimodalMemory()
        
        entry = await memory.store(
            "Test memory content",
            memory_type=MemoryType.TEXTUAL,
            tags=["test"]
        )
        
        assert entry.text_content == "Test memory content"
        assert "test" in entry.tags
        
        results = await memory.retrieve("Test memory", limit=1)
        
        assert len(results) > 0
        assert results[0].entry.id == entry.id
    
    @pytest.mark.asyncio
    async def test_memory_association(self):
        """Test memory association."""
        memory = MultimodalMemory()
        
        m1 = await memory.store("First memory")
        m2 = await memory.store("Second memory")
        
        assoc = await memory.create_association(m1.id, m2.id, "related")
        
        assert assoc is not None
        assert assoc.source_id == m1.id
        assert assoc.target_id == m2.id
    
    def test_memory_stats(self):
        """Test memory statistics."""
        memory = MultimodalMemory()
        
        stats = memory.get_stats()
        
        assert stats.total_entries == 0
        assert isinstance(stats.by_type, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
