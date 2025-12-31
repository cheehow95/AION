"""Tests for AION Advanced Runtime Features"""

import pytest
import asyncio
import sys
sys.path.insert(0, 'd:/Time/aion')

from src.runtime.messaging import (
    AgentMessage, MessageBroker, AgentMailbox, get_broker
)
from src.runtime.router import AdaptiveRouter, HybridRuntime
from src.runtime.reflexion import ReflexionLoop


class TestAgentMessage:
    """Test AgentMessage dataclass."""
    
    def test_message_creation(self):
        """Test creating an AgentMessage."""
        msg = AgentMessage(
            sender="AgentA",
            recipient="AgentB",
            content={"data": "test"},
            message_type="request"
        )
        
        assert msg.sender == "AgentA"
        assert msg.recipient == "AgentB"
        assert msg.content["data"] == "test"


class TestMessageBroker:
    """Test MessageBroker class."""
    
    def test_init(self):
        """Test broker initialization."""
        broker = MessageBroker()
        
        assert broker.agents == {}
        assert broker.subscriptions == {}
    
    def test_register_agent(self):
        """Test agent registration."""
        broker = MessageBroker()
        
        mailbox = broker.register("TestAgent")
        
        assert "TestAgent" in broker.agents
        assert isinstance(mailbox, AgentMailbox)
    
    def test_register_returns_existing(self):
        """Test registering same agent returns existing."""
        broker = MessageBroker()
        
        mailbox1 = broker.register("Agent")
        mailbox2 = broker.register("Agent")
        
        # Should return same mailbox
        assert mailbox1 is mailbox2
    
    def test_unregister(self):
        """Test unregistering an agent."""
        broker = MessageBroker()
        broker.register("TestAgent")
        
        broker.unregister("TestAgent")
        
        assert "TestAgent" not in broker.agents


class TestAgentMailbox:
    """Test AgentMailbox class."""
    
    def test_mailbox_creation(self):
        """Test creating a mailbox."""
        broker = MessageBroker()
        mailbox = AgentMailbox("TestAgent", broker)
        
        assert mailbox.name == "TestAgent"
        assert mailbox.broker is broker


class TestHybridRuntime:
    """Test HybridRuntime class."""
    
    def test_init(self):
        """Test runtime initialization."""
        runtime = HybridRuntime()
        
        assert runtime.router is not None
        assert runtime.local is not None
    
    def test_router_attribute(self):
        """Test router is AdaptiveRouter."""
        runtime = HybridRuntime()
        
        assert isinstance(runtime.router, AdaptiveRouter)


class TestReflexionLoop:
    """Test ReflexionLoop class."""
    
    def test_init(self):
        """Test loop initialization with required callbacks."""
        loop = ReflexionLoop(
            generator=lambda x: f"Generated: {x}",
            evaluator=lambda x: 0.9,
            critique_model=lambda x: "Looks good",
            max_attempts=3
        )
        
        assert loop.max_attempts == 3
        assert loop.traces == []
    
    def test_callbacks_stored(self):
        """Test that callbacks are stored properly."""
        gen = lambda x: x
        eval_ = lambda x: 1.0
        crit = lambda x: "ok"
        
        loop = ReflexionLoop(
            generator=gen,
            evaluator=eval_,
            critique_model=crit
        )
        
        assert loop.generator is gen
        assert loop.evaluator is eval_
        assert loop.critique_model is crit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
