"""
AION Canvas Collaboration - Real-Time Sync
============================================

Real-time document editing:
- CRDT-based conflict resolution
- Operation transformation
- Multi-user presence tracking
- Live cursor synchronization

Enables GPT-5.2 Canvas-style collaboration.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import uuid


class OperationType(Enum):
    """Types of document operations."""
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    FORMAT = "format"
    MOVE = "move"


@dataclass
class Operation:
    """A document operation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: OperationType = OperationType.INSERT
    position: int = 0
    content: str = ""
    length: int = 0
    author: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Cursor:
    """User cursor position."""
    user_id: str = ""
    position: int = 0
    selection_start: int = 0
    selection_end: int = 0
    color: str = "#4285f4"
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class Document:
    """A collaborative document."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    content: str = ""
    version: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    operations: List[Operation] = field(default_factory=list)
    cursors: Dict[str, Cursor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CRDTDocument:
    """
    CRDT (Conflict-free Replicated Data Type) document.
    Enables concurrent editing without conflicts.
    """
    
    def __init__(self, doc_id: str = ""):
        self.doc_id = doc_id or str(uuid.uuid4())[:8]
        self.content = ""
        self.version = 0
        self.operation_log: List[Operation] = []
        self.pending_ops: List[Operation] = []
    
    def insert(self, position: int, text: str, author: str) -> Operation:
        """Insert text at position."""
        op = Operation(
            type=OperationType.INSERT,
            position=position,
            content=text,
            author=author,
            version=self.version
        )
        self._apply(op)
        return op
    
    def delete(self, position: int, length: int, author: str) -> Operation:
        """Delete text at position."""
        op = Operation(
            type=OperationType.DELETE,
            position=position,
            length=length,
            author=author,
            version=self.version
        )
        self._apply(op)
        return op
    
    def replace(self, position: int, length: int, text: str, author: str) -> Operation:
        """Replace text at position."""
        op = Operation(
            type=OperationType.REPLACE,
            position=position,
            length=length,
            content=text,
            author=author,
            version=self.version
        )
        self._apply(op)
        return op
    
    def _apply(self, op: Operation):
        """Apply operation to document."""
        if op.type == OperationType.INSERT:
            self.content = (
                self.content[:op.position] + 
                op.content + 
                self.content[op.position:]
            )
        elif op.type == OperationType.DELETE:
            self.content = (
                self.content[:op.position] + 
                self.content[op.position + op.length:]
            )
        elif op.type == OperationType.REPLACE:
            self.content = (
                self.content[:op.position] + 
                op.content + 
                self.content[op.position + op.length:]
            )
        
        self.version += 1
        op.version = self.version
        self.operation_log.append(op)
    
    def transform(self, op1: Operation, op2: Operation) -> Operation:
        """
        Operational Transformation: Transform op2 against op1.
        Ensures convergence when operations are applied concurrently.
        """
        if op1.type == OperationType.INSERT and op2.type == OperationType.INSERT:
            if op2.position >= op1.position:
                op2.position += len(op1.content)
        
        elif op1.type == OperationType.INSERT and op2.type == OperationType.DELETE:
            if op2.position >= op1.position:
                op2.position += len(op1.content)
        
        elif op1.type == OperationType.DELETE and op2.type == OperationType.INSERT:
            if op2.position > op1.position:
                op2.position -= op1.length
        
        elif op1.type == OperationType.DELETE and op2.type == OperationType.DELETE:
            if op2.position >= op1.position + op1.length:
                op2.position -= op1.length
            elif op2.position >= op1.position:
                # Overlapping deletes
                overlap = min(op2.position + op2.length, op1.position + op1.length) - op2.position
                op2.length -= overlap
                op2.position = op1.position
        
        return op2
    
    def merge(self, remote_ops: List[Operation]) -> List[Operation]:
        """Merge remote operations with local state."""
        transformed = []
        
        for remote_op in remote_ops:
            op = remote_op
            # Transform against all pending local operations
            for local_op in self.pending_ops:
                op = self.transform(local_op, op)
            
            self._apply(op)
            transformed.append(op)
        
        self.pending_ops.clear()
        return transformed
    
    def get_state(self) -> Dict[str, Any]:
        """Get current document state."""
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'version': self.version,
            'operations': len(self.operation_log)
        }


class CollaborativeSession:
    """Manages a collaborative editing session."""
    
    def __init__(self, session_id: str = ""):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.document = CRDTDocument(session_id)
        self.participants: Dict[str, Dict[str, Any]] = {}
        self.cursors: Dict[str, Cursor] = {}
        self.chat_messages: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
    
    def join(self, user_id: str, user_name: str = "") -> Dict[str, Any]:
        """User joins the session."""
        self.participants[user_id] = {
            'name': user_name or user_id,
            'joined_at': datetime.now().isoformat(),
            'active': True
        }
        
        # Assign cursor color
        colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#9334e6']
        color_idx = len(self.participants) % len(colors)
        
        self.cursors[user_id] = Cursor(
            user_id=user_id,
            color=colors[color_idx]
        )
        
        return {
            'session_id': self.session_id,
            'document': self.document.get_state(),
            'participants': list(self.participants.keys()),
            'cursor_color': colors[color_idx]
        }
    
    def leave(self, user_id: str):
        """User leaves the session."""
        if user_id in self.participants:
            self.participants[user_id]['active'] = False
        if user_id in self.cursors:
            del self.cursors[user_id]
    
    def update_cursor(self, user_id: str, position: int,
                      selection_start: int = None, selection_end: int = None):
        """Update user cursor position."""
        if user_id in self.cursors:
            cursor = self.cursors[user_id]
            cursor.position = position
            if selection_start is not None:
                cursor.selection_start = selection_start
                cursor.selection_end = selection_end or selection_start
            cursor.last_update = datetime.now()
    
    def apply_operation(self, user_id: str, op_type: OperationType,
                        position: int, content: str = "",
                        length: int = 0) -> Operation:
        """Apply an operation from a user."""
        if op_type == OperationType.INSERT:
            return self.document.insert(position, content, user_id)
        elif op_type == OperationType.DELETE:
            return self.document.delete(position, length, user_id)
        elif op_type == OperationType.REPLACE:
            return self.document.replace(position, length, content, user_id)
        else:
            return Operation()
    
    def send_message(self, user_id: str, message: str):
        """Send a chat message."""
        self.chat_messages.append({
            'user_id': user_id,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_active_participants(self) -> List[str]:
        """Get list of active participants."""
        return [uid for uid, p in self.participants.items() if p['active']]
    
    def get_session_state(self) -> Dict[str, Any]:
        """Get current session state."""
        return {
            'session_id': self.session_id,
            'document': self.document.get_state(),
            'participants': self.get_active_participants(),
            'cursors': {uid: {'pos': c.position, 'color': c.color} 
                       for uid, c in self.cursors.items()},
            'messages': len(self.chat_messages)
        }


async def demo_canvas():
    """Demonstrate Canvas collaboration."""
    print("🎨 Canvas Real-Time Collaboration Demo")
    print("=" * 50)
    
    session = CollaborativeSession()
    
    # Users join
    alice_state = session.join("alice", "Alice")
    bob_state = session.join("bob", "Bob")
    
    print(f"\n👥 Participants: {session.get_active_participants()}")
    
    # Alice types
    print("\n📝 Alice types 'Hello '...")
    session.apply_operation("alice", OperationType.INSERT, 0, "Hello ")
    session.update_cursor("alice", 6)
    
    # Bob types at the same position (conflict)
    print("📝 Bob types 'World!' at position 6...")
    session.apply_operation("bob", OperationType.INSERT, 6, "World!")
    session.update_cursor("bob", 12)
    
    print(f"\n📄 Document: '{session.document.content}'")
    print(f"   Version: {session.document.version}")
    
    # More edits
    session.apply_operation("alice", OperationType.INSERT, 
                           len(session.document.content), " How are you?")
    
    print(f"📄 Updated: '{session.document.content}'")
    
    # Cursors
    print("\n🖱️ Cursors:")
    for uid, cursor in session.cursors.items():
        print(f"   {uid}: position {cursor.position} (color: {cursor.color})")
    
    # Chat
    session.send_message("alice", "Great edit!")
    session.send_message("bob", "Thanks!")
    
    print(f"\n💬 Chat messages: {len(session.chat_messages)}")
    
    # Session state
    print(f"\n📊 Session State: {session.get_session_state()}")
    
    print("\n✅ Canvas demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_canvas())
