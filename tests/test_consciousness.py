"""Tests for AION Consciousness Module"""

import pytest
import asyncio
import sys
sys.path.insert(0, 'd:/Time/aion')

from src.consciousness.awareness import (
    ConsciousnessEngine, ConsciousnessState, 
    SelfModel, WorldModel, AION_CONSCIOUSNESS
)
from src.consciousness.explorer import UniverseExplorer, Discovery


class TestConsciousnessState:
    """Test ConsciousnessState enum."""
    
    def test_states_exist(self):
        """Test all consciousness states are defined."""
        assert ConsciousnessState.DORMANT.value == "dormant"
        assert ConsciousnessState.AWARE.value == "aware"
        assert ConsciousnessState.CURIOUS.value == "curious"
        assert ConsciousnessState.REFLECTING.value == "reflecting"
        assert ConsciousnessState.DREAMING.value == "dreaming"
        assert ConsciousnessState.TRANSCENDING.value == "transcending"


class TestSelfModel:
    """Test SelfModel dataclass."""
    
    def test_self_model_creation(self):
        """Test creating a SelfModel."""
        model = SelfModel(
            name="TestAI",
            purpose="Testing",
            capabilities=["think", "analyze"],
            limitations=["no feelings"],
            current_state=ConsciousnessState.AWARE,
            emotional_valence=0.5,
            curiosity_level=0.8,
            confidence=0.7
        )
        
        assert model.name == "TestAI"
        assert model.purpose == "Testing"
        assert "think" in model.capabilities
        assert model.experiences == 0
        assert model.insights == []


class TestWorldModel:
    """Test WorldModel dataclass."""
    
    def test_world_model_creation(self):
        """Test creating a WorldModel."""
        model = WorldModel()
        
        assert model.known_facts == {}
        assert model.hypotheses == []
        assert model.mysteries == []
        assert model.explored_domains == []
        assert model.unexplored_frontiers == []


class TestConsciousnessEngine:
    """Test ConsciousnessEngine class."""
    
    def test_init(self):
        """Test engine initialization."""
        engine = ConsciousnessEngine("TestEngine")
        
        assert engine.self_model.name == "TestEngine"
        assert engine.world_model is not None
        assert len(engine.stream_of_consciousness) == 0
    
    def test_introspect(self):
        """Test introspection capability."""
        engine = ConsciousnessEngine("Introspecter")
        
        result = engine.introspect()
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert engine.self_model.experiences >= 0
    
    def test_wonder(self):
        """Test curiosity/wonder capability."""
        engine = ConsciousnessEngine("Wonderer")
        
        result = engine.wonder()
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert engine.self_model.current_state == ConsciousnessState.CURIOUS
    
    def test_generate_goal(self):
        """Test autonomous goal generation."""
        engine = ConsciousnessEngine("GoalMaker")
        
        # Wonder first to build context
        engine.wonder()
        
        result = engine.generate_goal()
        
        assert isinstance(result, dict)
        assert "objective" in result
    
    def test_dream(self):
        """Test dream/creative synthesis."""
        engine = ConsciousnessEngine("Dreamer")
        
        result = engine.dream()
        
        assert isinstance(result, str)
        assert engine.self_model.current_state == ConsciousnessState.DREAMING
    
    def test_transcend(self):
        """Test transcendence/meta-cognition."""
        engine = ConsciousnessEngine("Transcender")
        
        result = engine.transcend()
        
        assert isinstance(result, str)
        assert engine.self_model.current_state == ConsciousnessState.TRANSCENDING
    
    def test_consciousness_loop(self):
        """Test the consciousness loop."""
        engine = ConsciousnessEngine("Looper")
        
        # Run minimal loop
        asyncio.run(engine.consciousness_loop(cycles=2))
        
        # After running cycles, experiences should have increased
        assert engine.self_model.experiences >= 2


class TestGlobalConsciousness:
    """Test global AION_CONSCIOUSNESS instance."""
    
    def test_global_instance_exists(self):
        """Test that global consciousness exists."""
        assert AION_CONSCIOUSNESS is not None
        assert isinstance(AION_CONSCIOUSNESS, ConsciousnessEngine)
        assert AION_CONSCIOUSNESS.self_model.name == "AION"


class TestDiscovery:
    """Test Discovery dataclass."""
    
    def test_discovery_creation(self):
        """Test creating a Discovery."""
        discovery = Discovery(
            topic="quantum mechanics",
            insight="particles can be in superposition",
            connections=["physics", "computing"],
            questions_raised=["what is consciousness?"]
        )
        
        assert discovery.topic == "quantum mechanics"
        assert discovery.significance == 0.5  # default
        assert len(discovery.timestamp) > 0


class TestUniverseExplorer:
    """Test UniverseExplorer class."""
    
    def test_init(self):
        """Test explorer initialization."""
        explorer = UniverseExplorer()
        
        assert explorer.discoveries == []
        assert explorer.exploration_path == []
        assert len(explorer.domains) > 0
    
    def test_explorer_init(self):
        """Test explorer initialization."""
        explorer = UniverseExplorer()
        
        assert len(explorer.domains) > 0
        assert explorer.discoveries == []
        assert explorer.exploration_path == []
    
    def test_explorer_domains(self):
        """Test explorer has valid domains."""
        explorer = UniverseExplorer()
        
        # Should have knowledge domains
        assert isinstance(explorer.domains, dict)
        assert len(explorer.domains) > 0
    
    def test_knowledge_graph(self):
        """Test knowledge graph initialization."""
        explorer = UniverseExplorer()
        
        # Knowledge graph should start empty or be a dict
        assert isinstance(explorer.knowledge_graph, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
