"""
AION Self-Evolution v2 - Knowledge Transfer
============================================

Cross-agent knowledge transfer:
- Knowledge Distillation: Transfer learned patterns between agents
- Skill Sharing: Share successful strategies
- Experience Replay: Collective memory building
- Transfer Learning: Domain adaptation

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
from collections import defaultdict


class KnowledgeType(Enum):
    """Types of transferable knowledge."""
    SKILL = "skill"
    FACT = "fact"
    STRATEGY = "strategy"
    EXPERIENCE = "experience"
    MODEL = "model"


@dataclass
class Knowledge:
    """A piece of transferable knowledge."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: KnowledgeType = KnowledgeType.FACT
    domain: str = ""
    content: Any = None
    embedding: Optional[List[float]] = None
    source_agent: str = ""
    quality_score: float = 0.5
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """Edge in knowledge graph."""
    source: str
    target: str
    relation: str
    weight: float = 1.0


class KnowledgeGraph:
    """Graph structure for organizing knowledge."""
    
    def __init__(self):
        self.nodes: Dict[str, Knowledge] = {}
        self.edges: List[KnowledgeEdge] = []
        self.adjacency: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    
    def add_knowledge(self, knowledge: Knowledge):
        """Add knowledge node."""
        self.nodes[knowledge.id] = knowledge
    
    def add_relation(self, source_id: str, target_id: str, 
                     relation: str, weight: float = 1.0):
        """Add relation between knowledge."""
        edge = KnowledgeEdge(source_id, target_id, relation, weight)
        self.edges.append(edge)
        self.adjacency[source_id].append((target_id, relation, weight))
    
    def get_related(self, knowledge_id: str, 
                    relation: str = None) -> List[Knowledge]:
        """Get related knowledge."""
        related = []
        for target_id, rel, _ in self.adjacency.get(knowledge_id, []):
            if relation and rel != relation:
                continue
            if target_id in self.nodes:
                related.append(self.nodes[target_id])
        return related
    
    def find_path(self, start_id: str, end_id: str) -> List[str]:
        """Find path between two knowledge nodes."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return []
        
        visited = {start_id}
        queue = [(start_id, [start_id])]
        
        while queue:
            current, path = queue.pop(0)
            if current == end_id:
                return path
            
            for target_id, _, _ in self.adjacency.get(current, []):
                if target_id not in visited:
                    visited.add(target_id)
                    queue.append((target_id, path + [target_id]))
        
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        by_type = defaultdict(int)
        for k in self.nodes.values():
            by_type[k.type.value] += 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'by_type': dict(by_type)
        }


class KnowledgeDistillation:
    """Distill knowledge from teacher to student agents."""
    
    def __init__(self):
        self.transfers: List[Dict[str, Any]] = []
    
    async def distill(self, teacher_knowledge: List[Knowledge],
                      student_capacity: int = 100,
                      quality_threshold: float = 0.5) -> List[Knowledge]:
        """Distill knowledge for transfer to student."""
        # Filter by quality
        qualified = [k for k in teacher_knowledge if k.quality_score >= quality_threshold]
        
        # Sort by quality and usage
        qualified.sort(key=lambda k: (k.quality_score, k.usage_count), reverse=True)
        
        # Take top N within capacity
        distilled = qualified[:student_capacity]
        
        self.transfers.append({
            'teacher_count': len(teacher_knowledge),
            'distilled_count': len(distilled),
            'timestamp': datetime.now().isoformat()
        })
        
        return distilled
    
    async def compress_knowledge(self, knowledge: Knowledge,
                                 compression_ratio: float = 0.5) -> Knowledge:
        """Compress knowledge for efficient transfer."""
        compressed = Knowledge(
            type=knowledge.type,
            domain=knowledge.domain,
            source_agent=knowledge.source_agent,
            quality_score=knowledge.quality_score * compression_ratio,
            metadata={'compressed_from': knowledge.id}
        )
        
        # Simulate compression
        if isinstance(knowledge.content, dict):
            # Keep only top keys
            items = list(knowledge.content.items())
            keep_count = max(1, int(len(items) * compression_ratio))
            compressed.content = dict(items[:keep_count])
        else:
            compressed.content = knowledge.content
        
        return compressed


@dataclass
class Experience:
    """An experience that can be replayed."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: Dict[str, Any] = field(default_factory=dict)
    action: str = ""
    result: Any = None
    reward: float = 0.0
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class ExperienceReplay:
    """Collective experience replay buffer."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer: List[Experience] = []
        self.capacity = capacity
        self.priorities: Dict[str, float] = {}
    
    def add(self, experience: Experience, priority: float = 1.0):
        """Add experience to buffer."""
        if len(self.buffer) >= self.capacity:
            # Remove lowest priority
            if self.priorities:
                min_id = min(self.priorities, key=self.priorities.get)
                self.buffer = [e for e in self.buffer if e.id != min_id]
                del self.priorities[min_id]
        
        self.buffer.append(experience)
        self.priorities[experience.id] = priority
    
    def sample(self, batch_size: int, prioritized: bool = True) -> List[Experience]:
        """Sample experiences from buffer."""
        if not self.buffer:
            return []
        
        batch_size = min(batch_size, len(self.buffer))
        
        if prioritized:
            # Weighted sampling
            weights = [self.priorities.get(e.id, 1.0) for e in self.buffer]
            total = sum(weights)
            probs = [w / total for w in weights]
            
            indices = []
            for _ in range(batch_size):
                r = __import__('random').random()
                cumsum = 0
                for i, p in enumerate(probs):
                    cumsum += p
                    if r <= cumsum:
                        indices.append(i)
                        break
            
            return [self.buffer[i] for i in indices]
        else:
            return __import__('random').sample(self.buffer, batch_size)
    
    def update_priority(self, experience_id: str, priority: float):
        """Update experience priority."""
        self.priorities[experience_id] = priority
    
    def get_by_agent(self, agent_id: str) -> List[Experience]:
        """Get all experiences from an agent."""
        return [e for e in self.buffer if e.agent_id == agent_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'unique_agents': len(set(e.agent_id for e in self.buffer))
        }


class TransferLearning:
    """Domain adaptation and transfer learning."""
    
    def __init__(self):
        self.domain_mappings: Dict[Tuple[str, str], Callable] = {}
        self.transfer_history: List[Dict[str, Any]] = []
    
    def register_mapping(self, source_domain: str, target_domain: str,
                         adapter: Callable):
        """Register domain adaptation function."""
        self.domain_mappings[(source_domain, target_domain)] = adapter
    
    async def transfer(self, knowledge: Knowledge, 
                       target_domain: str) -> Optional[Knowledge]:
        """Transfer knowledge to a different domain."""
        source_domain = knowledge.domain
        
        if source_domain == target_domain:
            return knowledge
        
        adapter = self.domain_mappings.get((source_domain, target_domain))
        
        if not adapter:
            # Try generic adaptation
            adapted = Knowledge(
                type=knowledge.type,
                domain=target_domain,
                content=knowledge.content,
                source_agent=knowledge.source_agent,
                quality_score=knowledge.quality_score * 0.7,
                metadata={'adapted_from': knowledge.id, 'original_domain': source_domain}
            )
        else:
            adapted_content = adapter(knowledge.content)
            adapted = Knowledge(
                type=knowledge.type,
                domain=target_domain,
                content=adapted_content,
                source_agent=knowledge.source_agent,
                quality_score=knowledge.quality_score * 0.9
            )
        
        self.transfer_history.append({
            'source_domain': source_domain,
            'target_domain': target_domain,
            'knowledge_id': knowledge.id,
            'timestamp': datetime.now().isoformat()
        })
        
        return adapted


async def demo_knowledge_transfer():
    """Demonstrate knowledge transfer."""
    print("🧠 Knowledge Transfer Demo")
    print("=" * 50)
    
    # Create knowledge graph
    graph = KnowledgeGraph()
    
    # Add knowledge
    k1 = Knowledge(type=KnowledgeType.SKILL, domain="coding",
                   content={"pattern": "factory"}, quality_score=0.9, source_agent="expert")
    k2 = Knowledge(type=KnowledgeType.STRATEGY, domain="coding",
                   content={"approach": "test-first"}, quality_score=0.8, source_agent="expert")
    k3 = Knowledge(type=KnowledgeType.FACT, domain="coding",
                   content={"rule": "DRY"}, quality_score=0.95, source_agent="mentor")
    
    graph.add_knowledge(k1)
    graph.add_knowledge(k2)
    graph.add_knowledge(k3)
    graph.add_relation(k1.id, k2.id, "enables")
    graph.add_relation(k2.id, k3.id, "applies")
    
    print(f"\n📊 Knowledge Graph: {graph.get_statistics()}")
    
    # Distillation
    distiller = KnowledgeDistillation()
    distilled = await distiller.distill([k1, k2, k3], student_capacity=2)
    print(f"\n📚 Distilled {len(distilled)} knowledge items for student")
    
    # Experience replay
    replay = ExperienceReplay(capacity=100)
    for i in range(10):
        exp = Experience(
            state={'step': i},
            action=f"action_{i % 3}",
            result="success" if i % 2 == 0 else "failure",
            reward=1.0 if i % 2 == 0 else -0.5,
            agent_id=f"agent_{i % 2}"
        )
        replay.add(exp, priority=exp.reward + 1)
    
    print(f"\n🔄 Experience Buffer: {replay.get_statistics()}")
    
    # Sample experiences
    samples = replay.sample(3, prioritized=True)
    print(f"  Sampled {len(samples)} experiences")
    
    # Transfer learning
    transfer = TransferLearning()
    adapted = await transfer.transfer(k1, "design")
    print(f"\n🔀 Adapted knowledge to domain: {adapted.domain}")
    print(f"  Quality: {k1.quality_score:.2f} → {adapted.quality_score:.2f}")
    
    print("\n✅ Knowledge transfer demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_knowledge_transfer())
