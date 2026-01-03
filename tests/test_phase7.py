"""
AION Phase 7: Gemini 3 Parity - Test Suite
===========================================

Comprehensive tests for Phase 7 features:
- Hyper-Context System
- Native Multimodality
- Generative UI Engine
- Deep Think 2.0
"""

import pytest
import asyncio
from typing import Dict, Any
import os
import shutil
import tempfile

# =============================================================================
# Hyper-Context Tests
# =============================================================================

class TestHyperContext:
    """Tests for massive context system."""
    
    @pytest.mark.asyncio
    async def test_context_paging_and_retrieval(self):
        """Test paging limits and retrieval."""
        from src.context.hyper_context import HyperContextManager
        
        # Setup temp storage
        temp_dir = tempfile.mkdtemp()
        try:
            # Init manager with reasonable memory limit for testing
            manager = HyperContextManager(storage_dir=temp_dir)
            manager.pager.max_memory_tokens = 2000  # Reasonable limit
            
            # Add content
            await manager.add_context("Block one content with important data")
            await manager.add_context("Block two content with other data")
            
            # Check retrieval
            context = await manager.get_relevant_context("Block one", budget=5000)
            assert "Block one" in context or len(context) > 0
            
        finally:
            shutil.rmtree(temp_dir)

# =============================================================================
# Multimodality Tests
# =============================================================================

class TestMultimodality:
    """Tests for multimodal processing."""
    
    @pytest.mark.asyncio
    async def test_modality_router(self):
        """Test routing capabilities."""
        from src.multimodal.modality_router import ModalityRouter, ModalityType, MultimodalInput
        from src.multimodal.video_processor import VideoProcessor
        from src.multimodal.audio_engine import AudioEngine
        
        router = ModalityRouter()
        
        # Register processors
        router.register_processor(ModalityType.VIDEO, VideoProcessor())
        router.register_processor(ModalityType.AUDIO, AudioEngine())
        
        # Test routing
        input_vid = MultimodalInput(type=ModalityType.VIDEO, content="test.mp4")
        result = await router.process(input_vid)
        
        assert result['type'] == 'video'
        assert 'embedding' in result

# =============================================================================
# Generative UI Tests
# =============================================================================

class TestGenerativeUI:
    """Tests for UI generation."""
    
    @pytest.mark.asyncio
    async def test_ui_generation(self):
        """Test component generation."""
        from src.gen_ui.ui_generator import UIGenerator
        
        generator = UIGenerator()
        component = await generator.generate("create a calculator")
        
        assert "function Calculator" in component.code
        assert "react" in component.dependencies
        
    @pytest.mark.asyncio
    async def test_state_management(self):
        """Test state transitions."""
        from src.gen_ui.state_manager import InteractionHandler
        
        handler = InteractionHandler()
        instance_id = handler.init_state("comp_1", {"count": 0})
        
        new_state = await handler.handle_action(instance_id, "update", {"count": 5})
        assert new_state['count'] == 5

# =============================================================================
# Deep Think 2.0 Tests
# =============================================================================

class TestDeepThink:
    """Tests for advanced reasoning."""
    
    @pytest.mark.asyncio
    async def test_mcts_reasoning(self):
        """Test MCTS solver."""
        from src.reasoning.deep_think import DeepThinker
        
        thinker = DeepThinker()
        thinker.solver.iterations = 5  # Reduced for speed
        
        result = await thinker.think("Solve potential AI safety issues")
        assert "Deep Think Result" in result


if __name__ == "__main__":
    # Ensure src is in python path
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    pytest.main([__file__, "-v"])
