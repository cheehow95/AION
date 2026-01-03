"""
AION Vision Processing
======================

Vision input processing for image understanding, object detection,
OCR, and scene description.
"""

import base64
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import uuid


# =============================================================================
# IMAGE FORMAT
# =============================================================================

class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    
    @classmethod
    def from_extension(cls, ext: str) -> "ImageFormat":
        """Get format from file extension."""
        ext = ext.lower().lstrip(".")
        mapping = {
            "jpg": cls.JPEG,
            "jpeg": cls.JPEG,
            "png": cls.PNG,
            "webp": cls.WEBP,
            "gif": cls.GIF,
            "bmp": cls.BMP,
            "tiff": cls.TIFF,
            "tif": cls.TIFF,
        }
        return mapping.get(ext, cls.JPEG)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VisionInput:
    """Image data for vision processing."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: bytes = b""
    format: ImageFormat = ImageFormat.JPEG
    width: int = 0
    height: int = 0
    channels: int = 3
    source: str = ""  # file path or URL
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def base64(self) -> str:
        """Get base64-encoded image data."""
        return base64.b64encode(self.data).decode("utf-8")
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get image dimensions."""
        return (self.width, self.height)
    
    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width/height)."""
        if self.height == 0:
            return 0.0
        return self.width / self.height


@dataclass
class DetectedObject:
    """An object detected in an image."""
    label: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedText:
    """Text extracted from an image (OCR)."""
    text: str
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    language: str = "en"


@dataclass
class DetectedFace:
    """A face detected in an image."""
    bounding_box: Tuple[int, int, int, int]
    confidence: float
    landmarks: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)  # age, emotion, etc.


@dataclass
class VisionAnalysis:
    """Complete analysis results for an image."""
    input_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Scene understanding
    description: str = ""
    scene_type: str = ""  # indoor, outdoor, etc.
    
    # Detected elements
    objects: List[DetectedObject] = field(default_factory=list)
    text_regions: List[ExtractedText] = field(default_factory=list)
    faces: List[DetectedFace] = field(default_factory=list)
    
    # Colors and aesthetics
    dominant_colors: List[str] = field(default_factory=list)
    
    # Safety
    is_safe: bool = True
    safety_flags: List[str] = field(default_factory=list)
    
    # Raw response from vision model
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_text(self) -> str:
        """Get all extracted text combined."""
        return " ".join(t.text for t in self.text_regions)
    
    @property
    def object_labels(self) -> List[str]:
        """Get list of detected object labels."""
        return [obj.label for obj in self.objects]


# =============================================================================
# VISION PROCESSOR
# =============================================================================

class VisionProcessor:
    """
    Process vision inputs for image understanding.
    
    Supports loading images from files, base64, URLs, and provides
    analysis capabilities including object detection, OCR, and scene description.
    """
    
    def __init__(self, model: str = "default"):
        self.model = model
        self._cache: Dict[str, VisionAnalysis] = {}
        self._analysis_history: List[VisionAnalysis] = []
    
    def load_from_file(self, file_path: str) -> VisionInput:
        """
        Load an image from a file path.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            VisionInput with loaded image data
        """
        import os
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Get format from extension
        _, ext = os.path.splitext(file_path)
        img_format = ImageFormat.from_extension(ext)
        
        # Read file
        with open(file_path, "rb") as f:
            data = f.read()
        
        # Try to get dimensions
        width, height = self._get_image_dimensions(data, img_format)
        
        return VisionInput(
            data=data,
            format=img_format,
            width=width,
            height=height,
            source=file_path
        )
    
    def load_from_base64(self, b64_data: str, format: ImageFormat = ImageFormat.JPEG) -> VisionInput:
        """
        Load an image from base64-encoded data.
        
        Args:
            b64_data: Base64-encoded image data
            format: Image format
            
        Returns:
            VisionInput with decoded image data
        """
        # Handle data URLs
        if b64_data.startswith("data:"):
            parts = b64_data.split(",", 1)
            if len(parts) == 2:
                header, b64_data = parts
                # Extract format from header
                if "png" in header:
                    format = ImageFormat.PNG
                elif "webp" in header:
                    format = ImageFormat.WEBP
                elif "gif" in header:
                    format = ImageFormat.GIF
        
        data = base64.b64decode(b64_data)
        width, height = self._get_image_dimensions(data, format)
        
        return VisionInput(
            data=data,
            format=format,
            width=width,
            height=height,
            source="base64"
        )
    
    async def load_from_url(self, url: str) -> VisionInput:
        """
        Load an image from a URL.
        
        Args:
            url: URL to the image
            
        Returns:
            VisionInput with downloaded image data
        """
        # Simulated URL loading for now
        # In production, use aiohttp or httpx
        
        # Determine format from URL
        format = ImageFormat.JPEG
        url_lower = url.lower()
        if ".png" in url_lower:
            format = ImageFormat.PNG
        elif ".webp" in url_lower:
            format = ImageFormat.WEBP
        elif ".gif" in url_lower:
            format = ImageFormat.GIF
        
        # Placeholder - actual implementation would fetch from URL
        return VisionInput(
            data=b"",
            format=format,
            source=url,
            metadata={"status": "url_load_simulated"}
        )
    
    def _get_image_dimensions(self, data: bytes, format: ImageFormat) -> Tuple[int, int]:
        """Extract image dimensions from binary data."""
        if len(data) < 24:
            return (0, 0)
        
        try:
            # PNG
            if data[:8] == b'\x89PNG\r\n\x1a\n':
                width = int.from_bytes(data[16:20], 'big')
                height = int.from_bytes(data[20:24], 'big')
                return (width, height)
            
            # JPEG
            if data[:2] == b'\xff\xd8':
                # Search for SOF marker
                i = 2
                while i < len(data) - 9:
                    if data[i] == 0xFF:
                        marker = data[i + 1]
                        if marker in (0xC0, 0xC1, 0xC2):
                            height = int.from_bytes(data[i + 5:i + 7], 'big')
                            width = int.from_bytes(data[i + 7:i + 9], 'big')
                            return (width, height)
                        else:
                            length = int.from_bytes(data[i + 2:i + 4], 'big')
                            i += 2 + length
                    else:
                        i += 1
            
            # GIF
            if data[:6] in (b'GIF87a', b'GIF89a'):
                width = int.from_bytes(data[6:8], 'little')
                height = int.from_bytes(data[8:10], 'little')
                return (width, height)
            
        except Exception:
            pass
        
        return (0, 0)
    
    async def analyze(self, input: VisionInput) -> VisionAnalysis:
        """
        Perform full analysis on an image.
        
        Args:
            input: VisionInput to analyze
            
        Returns:
            VisionAnalysis with detected objects, text, scene description
        """
        # Check cache
        if input.id in self._cache:
            return self._cache[input.id]
        
        # Run analysis components in parallel
        objects_task = asyncio.create_task(self._detect_objects(input))
        text_task = asyncio.create_task(self._extract_text(input))
        faces_task = asyncio.create_task(self._detect_faces(input))
        scene_task = asyncio.create_task(self._describe_scene(input))
        
        objects = await objects_task
        text_regions = await text_task
        faces = await faces_task
        description, scene_type = await scene_task
        
        analysis = VisionAnalysis(
            input_id=input.id,
            description=description,
            scene_type=scene_type,
            objects=objects,
            text_regions=text_regions,
            faces=faces,
            dominant_colors=await self._extract_colors(input)
        )
        
        # Cache and record
        self._cache[input.id] = analysis
        self._analysis_history.append(analysis)
        
        return analysis
    
    async def detect_objects(self, input: VisionInput) -> List[DetectedObject]:
        """Detect objects in an image."""
        return await self._detect_objects(input)
    
    async def extract_text(self, input: VisionInput) -> List[ExtractedText]:
        """Extract text from an image using OCR."""
        return await self._extract_text(input)
    
    async def describe_scene(self, input: VisionInput) -> str:
        """Generate a natural language description of the image."""
        description, _ = await self._describe_scene(input)
        return description
    
    async def compare_images(
        self, 
        image1: VisionInput, 
        image2: VisionInput
    ) -> Dict[str, Any]:
        """
        Compare two images for similarity.
        
        Returns:
            Dict with similarity score and difference analysis
        """
        analysis1 = await self.analyze(image1)
        analysis2 = await self.analyze(image2)
        
        # Compare object sets
        objects1 = set(analysis1.object_labels)
        objects2 = set(analysis2.object_labels)
        object_overlap = len(objects1 & objects2) / max(len(objects1 | objects2), 1)
        
        # Compare scene types
        scene_match = analysis1.scene_type == analysis2.scene_type
        
        # Overall similarity (simplified)
        similarity = (object_overlap + (1.0 if scene_match else 0.0)) / 2
        
        return {
            "similarity_score": similarity,
            "scene_match": scene_match,
            "common_objects": list(objects1 & objects2),
            "unique_to_first": list(objects1 - objects2),
            "unique_to_second": list(objects2 - objects1),
        }
    
    # -------------------------------------------------------------------------
    # Internal analysis methods (simulated - would use actual ML models)
    # -------------------------------------------------------------------------
    
    async def _detect_objects(self, input: VisionInput) -> List[DetectedObject]:
        """Internal object detection."""
        # Simulated object detection
        # In production, integrate with YOLO, DETR, or vision API
        
        await asyncio.sleep(0.01)  # Simulate processing
        
        # Return sample detections based on image presence
        if input.data:
            return [
                DetectedObject(
                    label="object",
                    confidence=0.95,
                    bounding_box=(0, 0, input.width // 2, input.height // 2)
                )
            ]
        return []
    
    async def _extract_text(self, input: VisionInput) -> List[ExtractedText]:
        """Internal OCR."""
        await asyncio.sleep(0.01)
        
        # Simulated OCR
        return []
    
    async def _detect_faces(self, input: VisionInput) -> List[DetectedFace]:
        """Internal face detection."""
        await asyncio.sleep(0.01)
        return []
    
    async def _describe_scene(self, input: VisionInput) -> Tuple[str, str]:
        """Internal scene description."""
        await asyncio.sleep(0.01)
        
        description = f"An image ({input.width}x{input.height})"
        scene_type = "unknown"
        
        return description, scene_type
    
    async def _extract_colors(self, input: VisionInput) -> List[str]:
        """Extract dominant colors."""
        await asyncio.sleep(0.01)
        return ["#FFFFFF", "#000000"]  # Placeholder


# =============================================================================
# DEMO
# =============================================================================

async def demo_vision():
    """Demonstrate vision processing."""
    print("🔍 Vision Processing Demo")
    print("-" * 40)
    
    processor = VisionProcessor()
    
    # Create a test input
    test_input = VisionInput(
        data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,  # Minimal PNG header
        format=ImageFormat.PNG,
        width=640,
        height=480
    )
    
    print(f"Input: {test_input.width}x{test_input.height} {test_input.format.value}")
    
    # Analyze
    analysis = await processor.analyze(test_input)
    
    print(f"Description: {analysis.description}")
    print(f"Scene type: {analysis.scene_type}")
    print(f"Objects detected: {len(analysis.objects)}")
    print(f"Text regions: {len(analysis.text_regions)}")
    print(f"Faces: {len(analysis.faces)}")
    print(f"Colors: {analysis.dominant_colors}")
    
    print("-" * 40)
    print("✅ Vision demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_vision())
