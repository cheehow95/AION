"""
AION Knowledge Graph Engine
===========================

Advanced knowledge representation and reasoning with:
- Entity-Relation-Entity triples
- Semantic reasoning
- Path-based inference
- Graph traversal algorithms
- Knowledge base queries
- Ontology support

Enables structured knowledge for AI reasoning.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from enum import Enum
from collections import defaultdict
import json


# =============================================================================
# KNOWLEDGE TYPES
# =============================================================================

class RelationType(Enum):
    """Standard relation types."""
    IS_A = "is_a"                    # Taxonomy
    PART_OF = "part_of"              # Mereology
    HAS_PROPERTY = "has_property"    # Attributes
    CAUSES = "causes"                # Causation
    BEFORE = "before"                # Temporal
    AFTER = "after"
    LOCATED_IN = "located_in"        # Spatial
    CONNECTED_TO = "connected_to"    # Association
    INSTANCE_OF = "instance_of"      # Class membership
    SUBCLASS_OF = "subclass_of"      # Class hierarchy
    RELATED_TO = "related_to"        # General
    OPPOSITE_OF = "opposite_of"      # Antonyms
    SIMILAR_TO = "similar_to"        # Similarity
    IMPLIES = "implies"              # Logical
    DERIVED_FROM = "derived_from"    # Origin


class EntityType(Enum):
    """Entity type categories."""
    CONCEPT = "concept"
    INSTANCE = "instance"
    CLASS = "class"
    PROPERTY = "property"
    ACTION = "action"
    STATE = "state"
    EVENT = "event"
    AGENT = "agent"
    LOCATION = "location"
    TIME = "time"


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Entity:
    """A knowledge graph entity (node)."""
    id: str
    name: str
    entity_type: EntityType = EntityType.CONCEPT
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # Semantic vector
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False


@dataclass 
class Relation:
    """A knowledge graph relation (edge)."""
    id: str
    source: str  # Entity ID
    target: str  # Entity ID
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Triple:
    """Subject-Predicate-Object triple."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "user"
    
    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


# =============================================================================
# KNOWLEDGE GRAPH
# =============================================================================

class KnowledgeGraph:
    """
    Core knowledge graph structure with reasoning capabilities.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        
        # Indexes for fast lookup
        self._outgoing: Dict[str, Set[str]] = defaultdict(set)  # entity -> relation ids
        self._incoming: Dict[str, Set[str]] = defaultdict(set)
        self._by_type: Dict[RelationType, Set[str]] = defaultdict(set)
        
        self._entity_counter = 0
        self._relation_counter = 0
    
    def _next_entity_id(self) -> str:
        self._entity_counter += 1
        return f"e{self._entity_counter}"
    
    def _next_relation_id(self) -> str:
        self._relation_counter += 1
        return f"r{self._relation_counter}"
    
    # =========================================================================
    # Entity Operations
    # =========================================================================
    
    def add_entity(self, name: str, entity_type: EntityType = EntityType.CONCEPT,
                   properties: Dict = None, entity_id: str = None) -> Entity:
        """Add an entity to the graph."""
        eid = entity_id or self._next_entity_id()
        entity = Entity(
            id=eid,
            name=name,
            entity_type=entity_type,
            properties=properties or {}
        )
        self.entities[eid] = entity
        return entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    def find_entity(self, name: str) -> Optional[Entity]:
        """Find entity by name."""
        for entity in self.entities.values():
            if entity.name.lower() == name.lower():
                return entity
        return None
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove entity and all its relations."""
        if entity_id not in self.entities:
            return False
        
        # Remove all relations
        for rel_id in list(self._outgoing[entity_id]):
            self.remove_relation(rel_id)
        for rel_id in list(self._incoming[entity_id]):
            self.remove_relation(rel_id)
        
        del self.entities[entity_id]
        return True
    
    # =========================================================================
    # Relation Operations
    # =========================================================================
    
    def add_relation(self, source_id: str, target_id: str, 
                     relation_type: RelationType,
                     properties: Dict = None,
                     weight: float = 1.0,
                     confidence: float = 1.0) -> Optional[Relation]:
        """Add a relation between entities."""
        if source_id not in self.entities or target_id not in self.entities:
            return None
        
        rid = self._next_relation_id()
        relation = Relation(
            id=rid,
            source=source_id,
            target=target_id,
            relation_type=relation_type,
            properties=properties or {},
            weight=weight,
            confidence=confidence
        )
        
        self.relations[rid] = relation
        self._outgoing[source_id].add(rid)
        self._incoming[target_id].add(rid)
        self._by_type[relation_type].add(rid)
        
        return relation
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get relation by ID."""
        return self.relations.get(relation_id)
    
    def remove_relation(self, relation_id: str) -> bool:
        """Remove a relation."""
        if relation_id not in self.relations:
            return False
        
        rel = self.relations[relation_id]
        self._outgoing[rel.source].discard(relation_id)
        self._incoming[rel.target].discard(relation_id)
        self._by_type[rel.relation_type].discard(relation_id)
        
        del self.relations[relation_id]
        return True
    
    # =========================================================================
    # Triple Operations (Convenient API)
    # =========================================================================
    
    def add_triple(self, subject: str, predicate: str, obj: str,
                   confidence: float = 1.0) -> Tuple[Entity, Relation, Entity]:
        """Add a triple (creates entities if needed)."""
        # Get or create subject
        subj_entity = self.find_entity(subject)
        if not subj_entity:
            subj_entity = self.add_entity(subject)
        
        # Get or create object
        obj_entity = self.find_entity(obj)
        if not obj_entity:
            obj_entity = self.add_entity(obj)
        
        # Determine relation type
        rel_type = self._parse_relation_type(predicate)
        
        # Create relation
        relation = self.add_relation(
            subj_entity.id, obj_entity.id, rel_type,
            properties={'predicate': predicate},
            confidence=confidence
        )
        
        return (subj_entity, relation, obj_entity)
    
    def _parse_relation_type(self, predicate: str) -> RelationType:
        """Parse predicate string to relation type."""
        pred_lower = predicate.lower().replace(' ', '_').replace('-', '_')
        
        type_map = {
            'is_a': RelationType.IS_A,
            'is': RelationType.IS_A,
            'part_of': RelationType.PART_OF,
            'has': RelationType.HAS_PROPERTY,
            'has_property': RelationType.HAS_PROPERTY,
            'causes': RelationType.CAUSES,
            'before': RelationType.BEFORE,
            'after': RelationType.AFTER,
            'located_in': RelationType.LOCATED_IN,
            'in': RelationType.LOCATED_IN,
            'connected_to': RelationType.CONNECTED_TO,
            'instance_of': RelationType.INSTANCE_OF,
            'subclass_of': RelationType.SUBCLASS_OF,
            'related_to': RelationType.RELATED_TO,
            'opposite_of': RelationType.OPPOSITE_OF,
            'similar_to': RelationType.SIMILAR_TO,
            'implies': RelationType.IMPLIES,
            'derived_from': RelationType.DERIVED_FROM,
        }
        
        return type_map.get(pred_lower, RelationType.RELATED_TO)
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def get_neighbors(self, entity_id: str, 
                      direction: str = 'both') -> List[Tuple[Entity, Relation]]:
        """Get neighboring entities."""
        neighbors = []
        
        if direction in ('out', 'both'):
            for rel_id in self._outgoing.get(entity_id, set()):
                rel = self.relations[rel_id]
                target = self.entities.get(rel.target)
                if target:
                    neighbors.append((target, rel))
        
        if direction in ('in', 'both'):
            for rel_id in self._incoming.get(entity_id, set()):
                rel = self.relations[rel_id]
                source = self.entities.get(rel.source)
                if source:
                    neighbors.append((source, rel))
        
        return neighbors
    
    def query(self, subject: str = None, predicate: str = None, 
              obj: str = None) -> List[Triple]:
        """Query triples with pattern matching (None = wildcard)."""
        results = []
        
        for rel in self.relations.values():
            source = self.entities.get(rel.source)
            target = self.entities.get(rel.target)
            
            if not source or not target:
                continue
            
            # Check matches
            if subject and source.name.lower() != subject.lower():
                continue
            if obj and target.name.lower() != obj.lower():
                continue
            if predicate:
                pred_type = self._parse_relation_type(predicate)
                if rel.relation_type != pred_type:
                    continue
            
            results.append(Triple(
                subject=source.name,
                predicate=rel.relation_type.value,
                object=target.name,
                confidence=rel.confidence
            ))
        
        return results
    
    def get_by_relation(self, relation_type: RelationType) -> List[Relation]:
        """Get all relations of a type."""
        return [self.relations[rid] for rid in self._by_type.get(relation_type, set())]
    
    # =========================================================================
    # Reasoning Operations
    # =========================================================================
    
    def find_path(self, source_id: str, target_id: str, 
                  max_depth: int = 5) -> Optional[List[str]]:
        """Find shortest path between entities (BFS)."""
        if source_id not in self.entities or target_id not in self.entities:
            return None
        
        if source_id == target_id:
            return [source_id]
        
        visited = {source_id}
        queue = [(source_id, [source_id])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for neighbor, _ in self.get_neighbors(current, 'out'):
                if neighbor.id == target_id:
                    return path + [target_id]
                
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [neighbor.id]))
        
        return None
    
    def infer_transitive(self, relation_type: RelationType) -> List[Triple]:
        """Infer new triples via transitivity (e.g., is_a, part_of)."""
        inferred = []
        relations = self.get_by_relation(relation_type)
        
        # Build adjacency
        adj: Dict[str, Set[str]] = defaultdict(set)
        for rel in relations:
            adj[rel.source].add(rel.target)
        
        # Transitive closure (Floyd-Warshall simplified)
        for start in adj:
            visited = set()
            stack = list(adj[start])
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                
                # Check if this is a new inference
                if current not in adj[start]:
                    source = self.entities.get(start)
                    target = self.entities.get(current)
                    if source and target:
                        inferred.append(Triple(
                            subject=source.name,
                            predicate=relation_type.value,
                            object=target.name,
                            confidence=0.8  # Lower confidence for inferred
                        ))
                
                stack.extend(adj.get(current, set()))
        
        return inferred
    
    def get_subgraph(self, entity_id: str, depth: int = 2) -> 'KnowledgeGraph':
        """Extract subgraph around an entity."""
        subgraph = KnowledgeGraph(f"{self.name}_sub_{entity_id}")
        
        visited = set()
        queue = [(entity_id, 0)]
        
        while queue:
            current, d = queue.pop(0)
            if current in visited or d > depth:
                continue
            visited.add(current)
            
            entity = self.entities.get(current)
            if entity:
                subgraph.entities[current] = entity
                
                for neighbor, rel in self.get_neighbors(current):
                    if rel.id not in subgraph.relations:
                        subgraph.relations[rel.id] = rel
                        subgraph._outgoing[rel.source].add(rel.id)
                        subgraph._incoming[rel.target].add(rel.id)
                    
                    if neighbor.id not in visited:
                        queue.append((neighbor.id, d + 1))
        
        return subgraph
    
    def get_ancestors(self, entity_id: str, 
                      relation_type: RelationType = RelationType.IS_A) -> List[Entity]:
        """Get all ancestors via transitive relation."""
        ancestors = []
        visited = set()
        queue = [entity_id]
        
        while queue:
            current = queue.pop(0)
            for rel_id in self._outgoing.get(current, set()):
                rel = self.relations[rel_id]
                if rel.relation_type == relation_type:
                    if rel.target not in visited:
                        visited.add(rel.target)
                        entity = self.entities.get(rel.target)
                        if entity:
                            ancestors.append(entity)
                            queue.append(rel.target)
        
        return ancestors
    
    def get_descendants(self, entity_id: str,
                       relation_type: RelationType = RelationType.IS_A) -> List[Entity]:
        """Get all descendants via transitive relation."""
        descendants = []
        visited = set()
        queue = [entity_id]
        
        while queue:
            current = queue.pop(0)
            for rel_id in self._incoming.get(current, set()):
                rel = self.relations[rel_id]
                if rel.relation_type == relation_type:
                    if rel.source not in visited:
                        visited.add(rel.source)
                        entity = self.entities.get(rel.source)
                        if entity:
                            descendants.append(entity)
                            queue.append(rel.source)
        
        return descendants
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def stats(self) -> Dict:
        """Get graph statistics."""
        return {
            'entities': len(self.entities),
            'relations': len(self.relations),
            'relation_types': {rt.value: len(self._by_type.get(rt, set())) 
                              for rt in RelationType},
            'entity_types': self._count_entity_types(),
            'avg_degree': self._avg_degree()
        }
    
    def _count_entity_types(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for e in self.entities.values():
            counts[e.entity_type.value] += 1
        return dict(counts)
    
    def _avg_degree(self) -> float:
        if not self.entities:
            return 0.0
        total = sum(len(self._outgoing.get(e, set())) + len(self._incoming.get(e, set()))
                   for e in self.entities)
        return total / len(self.entities)
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'entities': [
                {
                    'id': e.id,
                    'name': e.name,
                    'type': e.entity_type.value,
                    'properties': e.properties
                } for e in self.entities.values()
            ],
            'relations': [
                {
                    'id': r.id,
                    'source': r.source,
                    'target': r.target,
                    'type': r.relation_type.value,
                    'weight': r.weight,
                    'confidence': r.confidence,
                    'properties': r.properties
                } for r in self.relations.values()
            ]
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeGraph':
        """Deserialize from dictionary."""
        kg = cls(data.get('name', 'default'))
        
        for e in data.get('entities', []):
            entity = Entity(
                id=e['id'],
                name=e['name'],
                entity_type=EntityType(e.get('type', 'concept')),
                properties=e.get('properties', {})
            )
            kg.entities[entity.id] = entity
        
        for r in data.get('relations', []):
            relation = Relation(
                id=r['id'],
                source=r['source'],
                target=r['target'],
                relation_type=RelationType(r.get('type', 'related_to')),
                weight=r.get('weight', 1.0),
                confidence=r.get('confidence', 1.0),
                properties=r.get('properties', {})
            )
            kg.relations[relation.id] = relation
            kg._outgoing[relation.source].add(relation.id)
            kg._incoming[relation.target].add(relation.id)
            kg._by_type[relation.relation_type].add(relation.id)
        
        return kg


# =============================================================================
# KNOWLEDGE GRAPH ENGINE
# =============================================================================

class KnowledgeGraphEngine:
    """
    AION Knowledge Graph Engine.
    
    High-level interface for knowledge representation and reasoning.
    """
    
    def __init__(self):
        self.graphs: Dict[str, KnowledgeGraph] = {}
        self.default_graph = KnowledgeGraph("default")
        self.graphs["default"] = self.default_graph
    
    def create_graph(self, name: str) -> KnowledgeGraph:
        """Create a new knowledge graph."""
        kg = KnowledgeGraph(name)
        self.graphs[name] = kg
        return kg
    
    def get_graph(self, name: str = "default") -> Optional[KnowledgeGraph]:
        """Get a knowledge graph by name."""
        return self.graphs.get(name)
    
    def add(self, subject: str, predicate: str, obj: str, 
            graph: str = "default") -> Dict:
        """Add a triple to the graph."""
        kg = self.graphs.get(graph, self.default_graph)
        s, r, o = kg.add_triple(subject, predicate, obj)
        
        return {
            'subject': {'id': s.id, 'name': s.name},
            'predicate': r.relation_type.value if r else predicate,
            'object': {'id': o.id, 'name': o.name},
            'success': r is not None
        }
    
    def query(self, subject: str = None, predicate: str = None, 
              obj: str = None, graph: str = "default") -> List[Dict]:
        """Query triples."""
        kg = self.graphs.get(graph, self.default_graph)
        triples = kg.query(subject, predicate, obj)
        
        return [
            {
                'subject': t.subject,
                'predicate': t.predicate,
                'object': t.object,
                'confidence': t.confidence
            } for t in triples
        ]
    
    def find_path(self, source: str, target: str, 
                  graph: str = "default") -> Optional[List[str]]:
        """Find path between entities."""
        kg = self.graphs.get(graph, self.default_graph)
        
        source_entity = kg.find_entity(source)
        target_entity = kg.find_entity(target)
        
        if not source_entity or not target_entity:
            return None
        
        path_ids = kg.find_path(source_entity.id, target_entity.id)
        
        if path_ids:
            return [kg.entities[eid].name for eid in path_ids]
        return None
    
    def infer(self, relation: str = "is_a", graph: str = "default") -> List[Dict]:
        """Infer new knowledge via transitivity."""
        kg = self.graphs.get(graph, self.default_graph)
        rel_type = kg._parse_relation_type(relation)
        inferred = kg.infer_transitive(rel_type)
        
        return [
            {
                'subject': t.subject,
                'predicate': t.predicate,
                'object': t.object,
                'confidence': t.confidence,
                'inferred': True
            } for t in inferred
        ]
    
    def stats(self, graph: str = "default") -> Dict:
        """Get graph statistics."""
        kg = self.graphs.get(graph, self.default_graph)
        return kg.stats()
    
    def build_taxonomy(self, root: str, children: List[str], 
                       graph: str = "default") -> Dict:
        """Build a taxonomy tree."""
        for child in children:
            self.add(child, "is_a", root, graph)
        
        return {
            'root': root,
            'children': children,
            'relation': 'is_a'
        }
    
    def build_from_triples(self, triples: List[Tuple[str, str, str]], 
                           graph: str = "default") -> Dict:
        """Build graph from list of triples."""
        for s, p, o in triples:
            self.add(s, p, o, graph)
        
        return self.stats(graph)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Knowledge Graph Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🧠 AION KNOWLEDGE GRAPH ENGINE 🧠                                ║
║                                                                           ║
║     Entity-Relation Triples, Reasoning, Inference, Queries              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = KnowledgeGraphEngine()
    
    # Build taxonomy
    print("📚 Building Animal Taxonomy:")
    print("-" * 50)
    engine.add("Dog", "is_a", "Mammal")
    engine.add("Cat", "is_a", "Mammal")
    engine.add("Mammal", "is_a", "Animal")
    engine.add("Bird", "is_a", "Animal")
    engine.add("Sparrow", "is_a", "Bird")
    
    # Add properties
    engine.add("Dog", "has", "fur")
    engine.add("Dog", "has", "four legs")
    engine.add("Sparrow", "has", "wings")
    engine.add("Sparrow", "has", "feathers")
    
    stats = engine.stats()
    print(f"   Entities: {stats['entities']}")
    print(f"   Relations: {stats['relations']}")
    
    # Query
    print("\n🔍 Query: What is a Mammal?")
    print("-" * 50)
    results = engine.query(predicate="is_a", obj="Mammal")
    for r in results:
        print(f"   {r['subject']} is_a {r['object']}")
    
    # Query properties
    print("\n🔍 Query: What does Dog have?")
    print("-" * 50)
    results = engine.query(subject="Dog", predicate="has")
    for r in results:
        print(f"   Dog has {r['object']}")
    
    # Find path
    print("\n🛤️ Path: Dog → Animal")
    print("-" * 50)
    path = engine.find_path("Dog", "Animal")
    if path:
        print(f"   {' → '.join(path)}")
    
    # Inference
    print("\n💡 Inferred Knowledge (transitive is_a):")
    print("-" * 50)
    inferred = engine.infer("is_a")
    for i in inferred[:5]:
        print(f"   {i['subject']} is_a {i['object']} (inferred)")


if __name__ == "__main__":
    demo()
