"""
AION Enhanced Memory System - Memory Graph
============================================

Knowledge graph for memory relationships:
- Entity linking and resolution
- Relationship tracking
- Temporal organization
- Graph-based retrieval

Enables rich memory interconnections.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
import uuid


class EntityType(Enum):
    """Types of entities."""
    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    CONCEPT = "concept"
    LOCATION = "location"
    EVENT = "event"
    DOCUMENT = "document"
    CODE = "code"


class RelationType(Enum):
    """Types of relationships."""
    KNOWS = "knows"
    WORKS_ON = "works_on"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    CREATED_BY = "created_by"
    MENTIONED_IN = "mentioned_in"
    FOLLOWS = "follows"
    SIMILAR_TO = "similar_to"


@dataclass
class MemoryNode:
    """A node in the memory graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    entity_type: EntityType = EntityType.CONCEPT
    content: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5
    
    def access(self):
        self.access_count += 1


@dataclass
class MemoryEdge:
    """An edge (relationship) in the memory graph."""
    source_id: str = ""
    target_id: str = ""
    relation_type: RelationType = RelationType.RELATED_TO
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class MemoryGraph:
    """Graph-based memory organization."""
    
    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[MemoryEdge] = []
        self.adjacency: Dict[str, List[str]] = {}  # node_id -> [connected_ids]
    
    def add_node(self, node: MemoryNode) -> str:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self.adjacency[node.id] = []
        return node.id
    
    def add_edge(self, source_id: str, target_id: str,
                 relation: RelationType = RelationType.RELATED_TO,
                 weight: float = 1.0,
                 properties: Dict[str, Any] = None) -> bool:
        """Add an edge between nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation,
            weight=weight,
            properties=properties or {}
        )
        
        self.edges.append(edge)
        
        if target_id not in self.adjacency[source_id]:
            self.adjacency[source_id].append(target_id)
        if source_id not in self.adjacency[target_id]:
            self.adjacency[target_id].append(source_id)
        
        return True
    
    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID."""
        node = self.nodes.get(node_id)
        if node:
            node.access()
        return node
    
    def find_nodes(self, name: str = None,
                   entity_type: EntityType = None,
                   limit: int = 10) -> List[MemoryNode]:
        """Find nodes matching criteria."""
        results = []
        
        for node in self.nodes.values():
            if name and name.lower() not in node.name.lower():
                continue
            if entity_type and node.entity_type != entity_type:
                continue
            results.append(node)
        
        results.sort(key=lambda n: n.importance, reverse=True)
        return results[:limit]
    
    def get_neighbors(self, node_id: str, 
                      relation: RelationType = None) -> List[MemoryNode]:
        """Get neighboring nodes."""
        if node_id not in self.adjacency:
            return []
        
        neighbor_ids = self.adjacency[node_id]
        neighbors = []
        
        for nid in neighbor_ids:
            if relation:
                # Check if the edge has the right relation
                for edge in self.edges:
                    if ((edge.source_id == node_id and edge.target_id == nid) or
                        (edge.target_id == node_id and edge.source_id == nid)):
                        if edge.relation_type == relation:
                            neighbors.append(self.nodes[nid])
                            break
            else:
                neighbors.append(self.nodes[nid])
        
        return neighbors
    
    def traverse(self, start_id: str, depth: int = 2) -> Dict[str, List[MemoryNode]]:
        """Traverse graph from a starting node."""
        result = {f"depth_{d}": [] for d in range(depth + 1)}
        visited = {start_id}
        
        current_level = [start_id]
        result['depth_0'] = [self.nodes[start_id]]
        
        for d in range(1, depth + 1):
            next_level = []
            for node_id in current_level:
                for neighbor_id in self.adjacency.get(node_id, []):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_level.append(neighbor_id)
                        result[f'depth_{d}'].append(self.nodes[neighbor_id])
            current_level = next_level
        
        return result
    
    def shortest_path(self, source_id: str, target_id: str) -> List[str]:
        """Find shortest path between nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return []
        
        from collections import deque
        
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target_id:
                return path
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def get_clusters(self) -> List[Set[str]]:
        """Find connected components (clusters)."""
        visited = set()
        clusters = []
        
        for node_id in self.nodes:
            if node_id not in visited:
                cluster = set()
                stack = [node_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.add(current)
                        stack.extend(self.adjacency.get(current, []))
                
                clusters.append(cluster)
        
        return clusters
    
    def compute_centrality(self) -> Dict[str, float]:
        """Compute node centrality (simplified degree centrality)."""
        centrality = {}
        n = len(self.nodes)
        
        if n <= 1:
            return {nid: 1.0 for nid in self.nodes}
        
        for node_id in self.nodes:
            degree = len(self.adjacency.get(node_id, []))
            centrality[node_id] = degree / (n - 1)
        
        return centrality
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'clusters': len(self.get_clusters()),
            'avg_degree': sum(len(adj) for adj in self.adjacency.values()) / len(self.nodes) if self.nodes else 0
        }


async def demo_memory_graph():
    """Demonstrate memory graph."""
    print("🕸️ Memory Graph Demo")
    print("=" * 50)
    
    graph = MemoryGraph()
    
    # Add nodes
    user = MemoryNode(name="User", entity_type=EntityType.PERSON, importance=1.0)
    aion = MemoryNode(name="AION", entity_type=EntityType.PROJECT, importance=0.9)
    python = MemoryNode(name="Python", entity_type=EntityType.CONCEPT, importance=0.7)
    ai = MemoryNode(name="AI", entity_type=EntityType.CONCEPT, importance=0.8)
    gpt = MemoryNode(name="GPT-5.2", entity_type=EntityType.CONCEPT, importance=0.8)
    
    for node in [user, aion, python, ai, gpt]:
        graph.add_node(node)
    
    # Add edges
    graph.add_edge(user.id, aion.id, RelationType.WORKS_ON, weight=1.0)
    graph.add_edge(aion.id, python.id, RelationType.DEPENDS_ON, weight=0.9)
    graph.add_edge(aion.id, ai.id, RelationType.RELATED_TO, weight=0.8)
    graph.add_edge(ai.id, gpt.id, RelationType.RELATED_TO, weight=0.7)
    
    print(f"\n📊 Graph Stats: {graph.get_stats()}")
    
    # Find nodes
    print("\n🔍 Finding nodes of type PROJECT:")
    projects = graph.find_nodes(entity_type=EntityType.PROJECT)
    for p in projects:
        print(f"   • {p.name} (importance: {p.importance})")
    
    # Get neighbors
    print(f"\n🔗 Neighbors of '{aion.name}':")
    neighbors = graph.get_neighbors(aion.id)
    for n in neighbors:
        print(f"   • {n.name}")
    
    # Traverse
    print(f"\n🚶 Traversing from '{user.name}' (depth=2):")
    traversal = graph.traverse(user.id, depth=2)
    for level, nodes in traversal.items():
        print(f"   {level}: {[n.name for n in nodes]}")
    
    # Shortest path
    path = graph.shortest_path(user.id, gpt.id)
    path_names = [graph.nodes[p].name for p in path]
    print(f"\n📍 Path from User to GPT-5.2: {' → '.join(path_names)}")
    
    # Centrality
    centrality = graph.compute_centrality()
    print("\n📈 Centrality Scores:")
    for node_id, score in sorted(centrality.items(), key=lambda x: x[1], reverse=True):
        print(f"   {graph.nodes[node_id].name}: {score:.2f}")
    
    print("\n✅ Memory graph demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_memory_graph())
