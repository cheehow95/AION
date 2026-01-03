"""
AION Native Multimodality System - Package Initialization
==========================================================

Unified multimodal processing:
- Phase 3: Vision, Audio, Document, Screen, Memory processing
- Phase 7: Gemini 3 Parity - Modality routing, Video, Audio engine
"""

# Phase 3: Perception modules
from .vision import (
    ImageFormat,
    VisionInput,
    DetectedObject,
    ExtractedText,
    DetectedFace,
    VisionAnalysis,
    VisionProcessor
)

from .audio import (
    AudioFormat,
    VoiceStyle,
    AudioInput,
    TranscriptionWord,
    TranscriptionResult,
    SpeakerSegment,
    AudioAnalysis,
    SynthesisRequest,
    AudioOutput,
    AudioProcessor
)

from .document import (
    DocumentFormat,
    ElementType,
    DocumentInput,
    DocumentElement,
    TableCell,
    ExtractedTable,
    ExtractedImage,
    DocumentSection,
    DocumentAnalysis,
    DocumentProcessor
)

from .screen import (
    UIElementType,
    ActionType,
    UIElement,
    ScreenCapture,
    ScreenAnalysis,
    UIAction,
    ActionPlan,
    ScreenProcessor
)

from .memory import (
    MemoryType,
    RetrievalStrategy,
    MemoryEmbedding,
    MemoryAssociation,
    MultimodalMemoryEntry,
    RetrievalResult,
    MemoryStats,
    MultimodalMemory
)

# Phase 7: Gemini 3 Parity modules
from .modality_router import (
    ModalityType,
    MultimodalInput,
    ModalityRouter
)

from .video_processor import (
    VideoSegment,
    VideoProcessor,
    TemporalEncoder
)

from .audio_engine import (
    AudioEngine,
    SpeechSegment,
    VoiceProfile
)

__all__ = [
    # Phase 3: Vision
    'ImageFormat',
    'VisionInput',
    'DetectedObject',
    'ExtractedText',
    'DetectedFace',
    'VisionAnalysis',
    'VisionProcessor',
    
    # Phase 3: Audio
    'AudioFormat',
    'VoiceStyle',
    'AudioInput',
    'TranscriptionWord',
    'TranscriptionResult',
    'SpeakerSegment',
    'AudioAnalysis',
    'SynthesisRequest',
    'AudioOutput',
    'AudioProcessor',
    
    # Phase 3: Document
    'DocumentFormat',
    'ElementType',
    'DocumentInput',
    'DocumentElement',
    'TableCell',
    'ExtractedTable',
    'ExtractedImage',
    'DocumentSection',
    'DocumentAnalysis',
    'DocumentProcessor',
    
    # Phase 3: Screen
    'UIElementType',
    'ActionType',
    'UIElement',
    'ScreenCapture',
    'ScreenAnalysis',
    'UIAction',
    'ActionPlan',
    'ScreenProcessor',
    
    # Phase 3: Memory
    'MemoryType',
    'RetrievalStrategy',
    'MemoryEmbedding',
    'MemoryAssociation',
    'MultimodalMemoryEntry',
    'RetrievalResult',
    'MemoryStats',
    'MultimodalMemory',
    
    # Phase 7: Router
    'ModalityType',
    'MultimodalInput',
    'ModalityRouter',
    
    # Phase 7: Video
    'VideoSegment',
    'VideoProcessor',
    'TemporalEncoder',
    
    # Phase 7: Audio Engine
    'AudioEngine',
    'SpeechSegment',
    'VoiceProfile',
]
