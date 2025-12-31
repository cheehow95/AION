"""
AION Distributed Runtime (Hive Mind)
Enables distributed agent execution using the Actor model.
Agents can run in parallel processes and communicate via messages.
"""

import multiprocessing
import queue
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import asyncio

@dataclass
class AgentMessage:
    sender: str
    recipient: str
    content: Any
    type: str = 'message'
    id: str = ''
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.id: self.id = str(uuid.uuid4())
        if not self.timestamp: self.timestamp = time.time()

class MessageBus:
    """Simple message bus for inter-process communication."""
    def __init__(self):
        self._queue = multiprocessing.Queue()
        self._subscribers = {}  # agent_id -> callback (not picklable, so handled in loop)
    
    def send(self, message: AgentMessage):
        self._queue.put(message)
    
    def receive(self, block=False, timeout=None) -> Optional[AgentMessage]:
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

class AgentNode(multiprocessing.Process):
    """
    Independent process hosting an AION agent.
    Acts as an Actor in the distributed system.
    """
    
    def __init__(self, agent_name: str, bus: MessageBus, source_code: str):
        super().__init__()
        self.agent_name = agent_name
        self.bus = bus
        self.source_code = source_code
        self.running = True
        
    def run(self):
        """Main loop for the agent process."""
        print(f"[{self.agent_name}] Process started (PID: {self.pid})")
        
        # Initialize local engine inside the process
        from src.runtime.local_engine import LocalReasoningEngine
        from src.parser import parse
        
        engine = LocalReasoningEngine()
        
        # Simple simulation of agent lifecycle
        while self.running:
            # Check for messages
            try:
                msg = self.bus.receive(block=False)
                if msg:
                    if msg.recipient == self.agent_name or msg.recipient == 'all':
                        self._handle_message(msg, engine)
                    else:
                        # Re-queue if not for us (simple bus logic)
                        self.bus.send(msg)
            except Exception:
                pass
            
            # Simulate "thinking" cycle
            time.sleep(0.1)
            
    def _handle_message(self, msg: AgentMessage, engine):
        """Process incoming message."""
        print(f"[{self.agent_name}] Received from {msg.sender}: {msg.content}")
        
        if msg.content == "STOP":
            self.running = False
            return
            
        # Reason about the message
        trace = engine.think(f"Incoming message from {msg.sender}: {msg.content}")
        analysis = engine.analyze(msg.content)
        
        # Reply logic
        if analysis['intent'] == 'question':
            reply_content = engine.decide(msg.content)['decision']
            response = AgentMessage(
                sender=self.agent_name,
                recipient=msg.sender,
                content=reply_content,
                type='reply'
            )
            self.bus.send(response)
            print(f"[{self.agent_name}] Replied to {msg.sender}")

class HiveMind:
    """
    Orchestrator for the distributed agent system.
    """
    
    def __init__(self):
        self.bus = MessageBus()
        self.nodes: Dict[str, AgentNode] = {}
        
    def spawn_agent(self, name: str, source_code: str = ""):
        """Spawn a new agent process."""
        if name in self.nodes:
            raise ValueError(f"Agent {name} already exists")
            
        node = AgentNode(name, self.bus, source_code)
        node.start()
        self.nodes[name] = node
        print(f"[HiveMind] Spawned agent '{name}'")
        
    def broadcast(self, content: str, sender: str = "System"):
        """Send a message to all agents."""
        msg = AgentMessage(sender, "all", content)
        self.bus.send(msg)
        
    def terminate(self):
        """Stop all agents."""
        self.broadcast("STOP")
        for name, node in self.nodes.items():
            node.join(timeout=2)
            if node.is_alive():
                node.terminate()
            print(f"[HiveMind] Terminated agent '{name}'")
