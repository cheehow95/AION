"""
AION Swarm Intelligence 2.0 - Self-Organizing Hierarchies
==========================================================

Implements self-organizing hierarchies:
- Dynamic Leadership: Automatic leader election based on capabilities
- Role Assignment: Skill-based role distribution
- Hierarchical Communication: Efficient message routing through hierarchy
- Adaptive Structure: Hierarchy reshaping based on task requirements

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Optional
from datetime import datetime
from enum import Enum
from collections import defaultdict


class AgentRole(Enum):
    """Roles an agent can hold in the hierarchy."""
    LEADER = "leader"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    WORKER = "worker"
    OBSERVER = "observer"


@dataclass
class HierarchyNode:
    """A node in the agent hierarchy."""
    agent_id: str
    role: AgentRole = AgentRole.WORKER
    parent: Optional[str] = None
    children: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    performance_score: float = 0.5
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_leader(self) -> bool:
        return self.role == AgentRole.LEADER
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def add_child(self, child_id: str):
        self.children.add(child_id)
    
    def remove_child(self, child_id: str):
        self.children.discard(child_id)


@dataclass
class HierarchyMessage:
    """Message routed through the hierarchy."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    target: str = ""
    direction: str = "down"  # up, down, lateral
    content: Any = None
    priority: int = 1
    timestamp: datetime = field(default_factory=datetime.now)


class DynamicHierarchy:
    """Self-organizing agent hierarchy with dynamic restructuring."""
    
    def __init__(self, max_children: int = 5, reorg_threshold: float = 0.3):
        self.nodes: Dict[str, HierarchyNode] = {}
        self.root: Optional[str] = None
        self.max_children = max_children
        self.reorg_threshold = reorg_threshold
        self.message_queue: asyncio.Queue = asyncio.Queue()
    
    def add_agent(self, agent_id: str, capabilities: Set[str] = None,
                  performance: float = 0.5) -> HierarchyNode:
        node = HierarchyNode(
            agent_id=agent_id,
            capabilities=capabilities or set(),
            performance_score=performance
        )
        self.nodes[agent_id] = node
        
        if self.root is None:
            self.root = agent_id
            node.role = AgentRole.LEADER
            node.depth = 0
        else:
            self._place_node(node)
        
        return node
    
    def _place_node(self, node: HierarchyNode):
        """Find optimal placement for a new node."""
        # Find node with capacity that has matching capabilities
        best_parent = None
        best_score = -1
        
        for pid, parent in self.nodes.items():
            if pid == node.agent_id:
                continue
            if len(parent.children) >= self.max_children:
                continue
            
            # Score based on capability overlap
            overlap = len(parent.capabilities & node.capabilities)
            score = overlap + (1 - parent.depth / 10)
            
            if score > best_score:
                best_score = score
                best_parent = pid
        
        if best_parent is None:
            best_parent = self.root
        
        self._attach_to_parent(node.agent_id, best_parent)
    
    def _attach_to_parent(self, child_id: str, parent_id: str):
        if child_id not in self.nodes or parent_id not in self.nodes:
            return
        
        child = self.nodes[child_id]
        parent = self.nodes[parent_id]
        
        # Remove from old parent
        if child.parent and child.parent in self.nodes:
            self.nodes[child.parent].remove_child(child_id)
        
        child.parent = parent_id
        child.depth = parent.depth + 1
        parent.add_child(child_id)
        
        # Assign role based on depth
        if child.depth == 1:
            child.role = AgentRole.COORDINATOR
        elif child.depth == 2:
            child.role = AgentRole.SPECIALIST
        else:
            child.role = AgentRole.WORKER
    
    def remove_agent(self, agent_id: str):
        if agent_id not in self.nodes:
            return
        
        node = self.nodes[agent_id]
        
        # Reassign children
        for child_id in list(node.children):
            if node.parent:
                self._attach_to_parent(child_id, node.parent)
            elif self.root and self.root != agent_id:
                self._attach_to_parent(child_id, self.root)
        
        # Remove from parent
        if node.parent and node.parent in self.nodes:
            self.nodes[node.parent].remove_child(agent_id)
        
        # Handle root removal
        if agent_id == self.root:
            self._elect_new_leader()
        
        del self.nodes[agent_id]
    
    def _elect_new_leader(self):
        """Elect a new leader based on performance."""
        if not self.nodes:
            self.root = None
            return
        
        # Find highest performing agent
        best = max(self.nodes.values(), key=lambda n: n.performance_score)
        self.root = best.agent_id
        best.role = AgentRole.LEADER
        best.parent = None
        best.depth = 0
        
        # Reorganize
        self._reorganize()
    
    def update_performance(self, agent_id: str, score: float):
        if agent_id not in self.nodes:
            return
        
        node = self.nodes[agent_id]
        old_score = node.performance_score
        node.performance_score = score
        
        # Check if reorganization needed
        if abs(score - old_score) > self.reorg_threshold:
            self._check_promotion(agent_id)
    
    def _check_promotion(self, agent_id: str):
        """Check if agent should be promoted/demoted."""
        node = self.nodes[agent_id]
        if not node.parent:
            return
        
        parent = self.nodes.get(node.parent)
        if not parent:
            return
        
        # Promote if outperforming parent significantly
        if node.performance_score > parent.performance_score + 0.2:
            self._swap_positions(agent_id, node.parent)
    
    def _swap_positions(self, agent_a: str, agent_b: str):
        """Swap two agents' positions in hierarchy."""
        if agent_a not in self.nodes or agent_b not in self.nodes:
            return
        
        a, b = self.nodes[agent_a], self.nodes[agent_b]
        
        # Swap parents and children
        a.parent, b.parent = b.parent, a.parent
        a.children, b.children = b.children, a.children
        a.depth, b.depth = b.depth, a.depth
        a.role, b.role = b.role, a.role
        
        # Update root if needed
        if self.root == agent_a:
            self.root = agent_b
        elif self.root == agent_b:
            self.root = agent_a
    
    def _reorganize(self):
        """Reorganize entire hierarchy for efficiency."""
        if not self.root:
            return
        
        # Sort all non-root nodes by performance
        others = [(n.agent_id, n.performance_score) 
                  for n in self.nodes.values() if n.agent_id != self.root]
        others.sort(key=lambda x: x[1], reverse=True)
        
        # Reset structure
        for agent_id, _ in others:
            node = self.nodes[agent_id]
            if node.parent and node.parent in self.nodes:
                self.nodes[node.parent].remove_child(agent_id)
            node.parent = None
            node.children.clear()
        
        # Rebuild
        for agent_id, _ in others:
            self._place_node(self.nodes[agent_id])
    
    def get_path_to_root(self, agent_id: str) -> List[str]:
        """Get path from agent to root."""
        path = []
        current = agent_id
        while current:
            path.append(current)
            node = self.nodes.get(current)
            current = node.parent if node else None
        return path
    
    def get_descendants(self, agent_id: str) -> List[str]:
        """Get all descendants of an agent."""
        if agent_id not in self.nodes:
            return []
        
        descendants = []
        queue = [agent_id]
        while queue:
            current = queue.pop(0)
            node = self.nodes.get(current)
            if node:
                for child in node.children:
                    descendants.append(child)
                    queue.append(child)
        return descendants
    
    async def route_message(self, message: HierarchyMessage) -> List[str]:
        """Route message through hierarchy."""
        path = []
        
        if message.direction == "up":
            path = self.get_path_to_root(message.sender)
        elif message.direction == "down":
            path = [message.target] + self.get_descendants(message.target)
        else:  # lateral
            # Go up to common ancestor, then down
            sender_path = set(self.get_path_to_root(message.sender))
            target_path = self.get_path_to_root(message.target)
            for node in target_path:
                path.append(node)
                if node in sender_path:
                    break
        
        return path
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self.nodes:
            return {}
        
        depths = [n.depth for n in self.nodes.values()]
        roles = defaultdict(int)
        for n in self.nodes.values():
            roles[n.role.value] += 1
        
        return {
            'total_agents': len(self.nodes),
            'max_depth': max(depths) if depths else 0,
            'avg_depth': sum(depths) / len(depths) if depths else 0,
            'roles': dict(roles),
            'root': self.root
        }


class RoleAssignment:
    """Skill-based role assignment system."""
    
    def __init__(self):
        self.role_requirements: Dict[AgentRole, Set[str]] = {
            AgentRole.LEADER: {"planning", "coordination", "decision_making"},
            AgentRole.COORDINATOR: {"coordination", "communication"},
            AgentRole.SPECIALIST: {"expertise"},
            AgentRole.WORKER: set(),
            AgentRole.OBSERVER: {"monitoring"},
        }
        self.agent_skills: Dict[str, Set[str]] = {}
    
    def register_agent(self, agent_id: str, skills: Set[str]):
        self.agent_skills[agent_id] = skills
    
    def get_best_role(self, agent_id: str) -> AgentRole:
        skills = self.agent_skills.get(agent_id, set())
        
        best_role = AgentRole.WORKER
        best_match = 0
        
        for role, required in self.role_requirements.items():
            if not required:
                continue
            match = len(skills & required) / len(required)
            if match > best_match:
                best_match = match
                best_role = role
        
        return best_role
    
    def assign_roles(self, agents: List[str]) -> Dict[str, AgentRole]:
        assignments = {}
        for agent in agents:
            assignments[agent] = self.get_best_role(agent)
        return assignments


class HierarchicalRouter:
    """Efficient message routing through agent hierarchy."""
    
    def __init__(self, hierarchy: DynamicHierarchy):
        self.hierarchy = hierarchy
        self.message_log: List[HierarchyMessage] = []
    
    async def broadcast_down(self, sender: str, content: Any) -> int:
        """Broadcast message down the hierarchy."""
        descendants = self.hierarchy.get_descendants(sender)
        count = 0
        for target in descendants:
            msg = HierarchyMessage(sender=sender, target=target, 
                                   direction="down", content=content)
            self.message_log.append(msg)
            count += 1
        return count
    
    async def escalate(self, sender: str, content: Any) -> List[str]:
        """Escalate message up the hierarchy."""
        path = self.hierarchy.get_path_to_root(sender)
        for target in path[1:]:  # Skip sender
            msg = HierarchyMessage(sender=sender, target=target,
                                   direction="up", content=content)
            self.message_log.append(msg)
        return path[1:]
    
    async def send_to_role(self, sender: str, role: AgentRole, content: Any) -> List[str]:
        """Send message to all agents with specific role."""
        targets = [n.agent_id for n in self.hierarchy.nodes.values() if n.role == role]
        for target in targets:
            msg = HierarchyMessage(sender=sender, target=target,
                                   direction="lateral", content=content)
            self.message_log.append(msg)
        return targets


async def demo_hierarchy():
    print("🏛️ Self-Organizing Hierarchy Demo")
    print("=" * 50)
    
    hierarchy = DynamicHierarchy(max_children=3)
    roles = RoleAssignment()
    
    # Add agents
    agents = [
        ("Alpha", {"planning", "coordination", "decision_making"}, 0.9),
        ("Beta", {"coordination", "communication"}, 0.7),
        ("Gamma", {"expertise", "analysis"}, 0.8),
        ("Delta", {"coding", "testing"}, 0.6),
        ("Epsilon", {"monitoring", "reporting"}, 0.5),
    ]
    
    for name, skills, perf in agents:
        hierarchy.add_agent(name, skills, perf)
        roles.register_agent(name, skills)
    
    print("\n📊 Hierarchy Structure:")
    for agent_id, node in hierarchy.nodes.items():
        parent = node.parent or "None"
        children = list(node.children) or []
        print(f"  {agent_id}: role={node.role.value}, parent={parent}, children={children}")
    
    print(f"\n📈 Statistics: {hierarchy.get_statistics()}")
    
    # Test routing
    router = HierarchicalRouter(hierarchy)
    escalated = await router.escalate("Epsilon", "Need help!")
    print(f"\n📬 Escalation path from Epsilon: {escalated}")
    
    print("\n✅ Hierarchy demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_hierarchy())
