"""
AION Phase 6: GPT-5.2 Parity - Test Suite
==========================================

Comprehensive tests for Phase 6 features:
- Extended Context System
- Tiered Agent Variants
- Enhanced Memory System
- Pulse Task Automation
- Canvas Collaboration
- App Integration Directory
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta


# =============================================================================
# Extended Context Tests
# =============================================================================

class TestContextManager:
    """Tests for context manager."""
    
    def test_context_segment_priority(self):
        """Test context segment effective priority."""
        from src.context.context_manager import ContextSegment, ContextPriority
        
        segment = ContextSegment(
            content="Test content",
            priority=ContextPriority.HIGH,
            relevance_score=1.0
        )
        
        assert segment.effective_priority > 0
        assert segment.priority == ContextPriority.HIGH
    
    def test_context_window_tokens(self):
        """Test context window token management."""
        from src.context.context_manager import ContextWindow
        
        window = ContextWindow(max_tokens=256000)
        assert window.max_tokens == 256000
        assert window.available_tokens > 0
    
    def test_context_manager_add(self):
        """Test adding content to context."""
        from src.context.context_manager import ContextManager, ContextPriority
        
        manager = ContextManager()
        segment = manager.add("Test content", ContextPriority.HIGH)
        
        assert segment.id
        assert segment.content == "Test content"
    
    def test_context_budget_allocator(self):
        """Test token budget allocation."""
        from src.context.context_manager import ContextBudgetAllocator
        
        allocator = ContextBudgetAllocator(total_tokens=256000)
        
        assert allocator.get_budget('system') > 0
        assert allocator.get_budget('user') > 0


class TestContextCompression:
    """Tests for context compression."""
    
    @pytest.mark.asyncio
    async def test_semantic_compression(self):
        """Test semantic compression."""
        from src.context.context_compression import SemanticCompressor
        
        compressor = SemanticCompressor()
        text = "This is a very really important document."
        
        result = await compressor.compress(text)
        
        assert result.compressed_tokens <= result.original_tokens
        assert result.compression_ratio <= 1.0
    
    @pytest.mark.asyncio
    async def test_summary_compression(self):
        """Test summary compression."""
        from src.context.context_compression import SummaryCompressor
        
        compressor = SummaryCompressor()
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        
        result = await compressor.compress(text, target_ratio=0.5)
        
        assert result.compressed_tokens <= result.original_tokens


class TestContextChunking:
    """Tests for context chunking."""
    
    def test_semantic_chunking(self):
        """Test semantic chunking."""
        from src.context.context_chunking import SemanticChunker
        
        chunker = SemanticChunker()
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        
        result = chunker.chunk(text, max_chunk_tokens=100)
        
        assert result.chunk_count > 0
        assert len(result.chunks) > 0
    
    def test_overlapping_chunking(self):
        """Test overlapping chunking."""
        from src.context.context_chunking import OverlappingChunker
        
        chunker = OverlappingChunker(overlap_tokens=20)
        text = "A" * 1000  # Long text
        
        result = chunker.chunk(text, max_chunk_tokens=100)
        
        assert result.chunk_count > 0


# =============================================================================
# Tiered Agent Variants Tests
# =============================================================================

class TestInstantAgent:
    """Tests for Instant tier agent."""
    
    @pytest.mark.asyncio
    async def test_instant_respond(self):
        """Test instant response."""
        from src.variants.instant import InstantAgent
        
        agent = InstantAgent()
        response = await agent.respond("What is Python?")
        
        assert response.content
        assert response.latency_ms >= 0
    
    def test_task_classification(self):
        """Test task type classification."""
        from src.variants.instant import InstantAgent, TaskType
        
        agent = InstantAgent()
        
        assert agent.classify_task("Translate to Spanish") == TaskType.TRANSLATION
        assert agent.classify_task("Summarize this") == TaskType.SUMMARIZATION
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test response caching."""
        from src.variants.instant import InstantAgent
        
        agent = InstantAgent()
        
        await agent.respond("Test query")
        response = await agent.respond("Test query")
        
        assert response.cached


class TestThinkingAgent:
    """Tests for Thinking tier agent."""
    
    @pytest.mark.asyncio
    async def test_thinking_process(self):
        """Test thinking process."""
        from src.variants.thinking import ThinkingAgent
        
        agent = ThinkingAgent()
        process = await agent.think("How to design a system?")
        
        assert len(process.thoughts) > 0
        assert process.final_answer
    
    @pytest.mark.asyncio
    async def test_reasoning_chain(self):
        """Test step-by-step reasoning."""
        from src.variants.thinking import ThinkingAgent
        
        agent = ThinkingAgent()
        chain = await agent.reason_step_by_step("Test problem")
        
        assert len(chain.steps) > 0


class TestProAgent:
    """Tests for Pro tier agent."""
    
    @pytest.mark.asyncio
    async def test_deep_analysis(self):
        """Test deep analysis."""
        from src.variants.pro import ProAgent
        
        agent = ProAgent()
        analysis = await agent.analyze("Design a scalable system", depth=2)
        
        assert analysis.domain
        assert analysis.confidence > 0
        assert len(analysis.findings) > 0
    
    def test_domain_detection(self):
        """Test domain detection."""
        from src.variants.pro import ProAgent, DomainExpertise
        
        agent = ProAgent()
        
        assert agent.detect_domain("Fix this bug in code") == DomainExpertise.PROGRAMMING


class TestVariantRouter:
    """Tests for variant router."""
    
    def test_routing_simple(self):
        """Test routing simple queries."""
        from src.variants.router import VariantRouter, TaskComplexity
        
        router = VariantRouter()
        decision = router.route("What is Python?")
        
        assert decision.tier == "instant"
        assert decision.complexity == TaskComplexity.SIMPLE
    
    def test_routing_complex(self):
        """Test routing complex queries."""
        from src.variants.router import VariantRouter
        
        router = VariantRouter()
        decision = router.route("Design and implement a complete microservices architecture")
        
        assert decision.tier in ["thinking", "pro"]


# =============================================================================
# Enhanced Memory Tests
# =============================================================================

class TestPersistentMemory:
    """Tests for persistent memory."""
    
    @pytest.mark.asyncio
    async def test_remember_and_recall(self):
        """Test memory storage and retrieval."""
        from src.memory.persistent_memory import PersistentMemoryManager, MemoryType
        
        manager = PersistentMemoryManager()
        
        await manager.remember("Test fact", MemoryType.FACT, importance=0.8)
        memories = await manager.recall("fact")
        
        assert len(memories) > 0
    
    @pytest.mark.asyncio
    async def test_preferences(self):
        """Test preference management."""
        from src.memory.persistent_memory import PersistentMemoryManager
        
        manager = PersistentMemoryManager()
        
        await manager.set_preference("theme", "dark")
        pref = await manager.get_preference("theme")
        
        assert "dark" in str(pref)


class TestMemoryGraph:
    """Tests for memory graph."""
    
    def test_add_nodes_and_edges(self):
        """Test graph construction."""
        from src.memory.memory_graph import MemoryGraph, MemoryNode, EntityType
        
        graph = MemoryGraph()
        
        node1 = MemoryNode(name="Node1", entity_type=EntityType.CONCEPT)
        node2 = MemoryNode(name="Node2", entity_type=EntityType.CONCEPT)
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge(node1.id, node2.id)
        
        neighbors = graph.get_neighbors(node1.id)
        assert len(neighbors) == 1
    
    def test_traversal(self):
        """Test graph traversal."""
        from src.memory.memory_graph import MemoryGraph, MemoryNode, EntityType
        
        graph = MemoryGraph()
        
        n1 = MemoryNode(name="A", entity_type=EntityType.CONCEPT)
        n2 = MemoryNode(name="B", entity_type=EntityType.CONCEPT)
        
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge(n1.id, n2.id)
        
        result = graph.traverse(n1.id, depth=1)
        assert 'depth_0' in result


class TestPersonalization:
    """Tests for personalization."""
    
    def test_preference_setting(self):
        """Test setting preferences."""
        from src.memory.personalization import PersonalizationEngine
        
        engine = PersonalizationEngine()
        engine.set_preference("response_length", "concise")
        
        assert engine.get_preference("response_length") == "concise"
    
    def test_personalized_prompt(self):
        """Test personalized prompt generation."""
        from src.memory.personalization import PersonalizationEngine
        
        engine = PersonalizationEngine()
        prompt = engine.get_personalized_prompt()
        
        assert len(prompt) > 0


# =============================================================================
# Pulse Automation Tests
# =============================================================================

class TestTaskScheduler:
    """Tests for task scheduler."""
    
    @pytest.mark.asyncio
    async def test_schedule_task(self):
        """Test task scheduling."""
        from src.pulse.scheduler import TaskScheduler
        
        scheduler = TaskScheduler()
        
        task_id = scheduler.schedule_prompt(
            "Test Task",
            "Do something",
            run_at=datetime.now() + timedelta(seconds=10)
        )
        
        assert task_id
        assert task_id in scheduler.tasks
    
    @pytest.mark.asyncio
    async def test_run_due_tasks(self):
        """Test running due tasks."""
        from src.pulse.scheduler import TaskScheduler
        
        scheduler = TaskScheduler()
        scheduler.set_executor(lambda p: f"Executed: {p}")
        
        scheduler.schedule_prompt("Now Task", "Immediate", run_at=datetime.now())
        
        results = await scheduler.run_due_tasks()
        
        assert len(results) > 0


class TestTriggers:
    """Tests for triggers."""
    
    @pytest.mark.asyncio
    async def test_time_trigger(self):
        """Test time trigger."""
        from src.pulse.triggers import TimeTrigger
        import asyncio
        
        # First evaluate initializes _last_interval_check, second triggers if interval passed
        trigger = TimeTrigger(interval_seconds=1)
        await trigger.evaluate()  # Initialize
        await asyncio.sleep(1.1)  # Wait for interval to pass
        result = await trigger.evaluate()
        
        assert result.triggered
    
    @pytest.mark.asyncio
    async def test_event_trigger(self):
        """Test event trigger."""
        from src.pulse.triggers import EventTrigger
        
        trigger = EventTrigger(event_type="test")
        trigger.receive_event({'type': 'test', 'data': 'value'})
        
        result = await trigger.evaluate()
        
        assert result.triggered
    
    @pytest.mark.asyncio
    async def test_condition_trigger(self):
        """Test condition trigger."""
        from src.pulse.triggers import ConditionTrigger
        
        trigger = ConditionTrigger(
            check_value_key="value",
            operator=">",
            threshold=50
        )
        
        result = await trigger.evaluate({'value': 100})
        
        assert result.triggered


# =============================================================================
# Canvas Collaboration Tests
# =============================================================================

class TestRealTimeCollaboration:
    """Tests for real-time collaboration."""
    
    def test_crdt_document_insert(self):
        """Test CRDT document insert."""
        from src.canvas.real_time import CRDTDocument
        
        doc = CRDTDocument()
        doc.insert(0, "Hello", "user1")
        
        assert doc.content == "Hello"
        assert doc.version == 1
    
    def test_collaborative_session(self):
        """Test collaborative session."""
        from src.canvas.real_time import CollaborativeSession, OperationType
        
        session = CollaborativeSession()
        
        alice = session.join("alice", "Alice")
        bob = session.join("bob", "Bob")
        
        session.apply_operation("alice", OperationType.INSERT, 0, "Hello ")
        
        assert "Hello" in session.document.content
        assert len(session.get_active_participants()) == 2


class TestSharing:
    """Tests for sharing."""
    
    def test_permission_management(self):
        """Test permission management."""
        from src.canvas.sharing import ShareManager, Permission
        
        manager = ShareManager()
        
        manager.set_permission("doc1", "user1", Permission.EDIT)
        perm = manager.get_permission("doc1", "user1")
        
        assert perm == Permission.EDIT
    
    def test_access_check(self):
        """Test access checking."""
        from src.canvas.sharing import ShareManager, Permission
        
        manager = ShareManager()
        
        manager.set_permission("doc1", "user1", Permission.EDIT)
        
        assert manager.check_access("doc1", "user1", Permission.VIEW)
        assert manager.check_access("doc1", "user1", Permission.EDIT)
        assert not manager.check_access("doc1", "user1", Permission.ADMIN)
    
    def test_version_history(self):
        """Test version history."""
        from src.canvas.sharing import ShareManager
        
        manager = ShareManager()
        
        manager.save_version("doc1", "Content v1", "user1")
        manager.save_version("doc1", "Content v2", "user1")
        
        versions = manager.get_versions("doc1")
        assert len(versions) == 2


# =============================================================================
# App Integration Tests
# =============================================================================

class TestAppDirectory:
    """Tests for app directory."""
    
    def test_register_and_search(self):
        """Test app registration and search."""
        from src.apps.directory import AppDirectory, App, AppCategory
        
        directory = AppDirectory()
        
        app = App(
            name="Test App",
            description="A test application",
            category=AppCategory.DEVELOPMENT
        )
        directory.register_app(app)
        
        results = directory.search("test")
        assert len(results) > 0
    
    def test_install_and_rate(self):
        """Test app installation and rating."""
        from src.apps.directory import AppDirectory, App
        
        directory = AppDirectory()
        
        app = App(name="My App")
        app_id = directory.register_app(app)
        
        directory.install("user1", app_id)
        directory.rate_app(app_id, "user1", 5, "Great!")
        
        installed = directory.get_installed("user1")
        assert len(installed) == 1
        assert installed[0].average_rating == 5.0


class TestAppConnector:
    """Tests for app connector."""
    
    @pytest.mark.asyncio
    async def test_connect(self):
        """Test app connection."""
        from src.apps.connector import AppConnector
        
        connector = AppConnector()
        session = await connector.connect("test_app", "user1")
        
        assert session.status.value == "connected"
    
    @pytest.mark.asyncio
    async def test_execute_action(self):
        """Test action execution."""
        from src.apps.connector import AppConnector
        
        connector = AppConnector()
        
        async def test_action(session, params):
            return "Success"
        
        connector.register_action("test_app", "test", test_action)
        session = await connector.connect("test_app", "user1")
        
        result = await connector.execute_action(session.id, "test")
        
        assert result['success']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
