"""
AION State Graph
================

Knowledge graph for tracking world state.
Represents entities, relationships, and observations.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json


class EntityType(Enum):
    """Types of entities in the world model."""
    OBJECT = "object"
    AGENT = "agent"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    STATE = "state"


class RelationType(Enum):
    """Types of relationships."""
    IS_A = "is_a"
    HAS = "has"
    LOCATED_AT = "located_at"
    CAUSES = "causes"
    PRECEDES = "precedes"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    AFFECTS = "affects"
    DEPENDS_ON = "depends_on"


@dataclass
class Entity:
    """An entity in the world model."""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    source: str = "observed"  # observed, inferred, hypothetical
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Entity) and self.id == other.id
    
    def update(self, properties: Dict[str, Any]):
        """Update entity properties."""
        self.properties.update(properties)
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type.value,
            "properties": self.properties,
            "confidence": self.confidence,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=EntityType(data["type"]),
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "observed")
        )


@dataclass
class Relation:
    """A relationship between entities."""
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    confidence: float = 1.0
    
    def __hash__(self):
        return hash(self.id)
    
    def is_valid(self, at_time: datetime = None) -> bool:
        """Check if relation is valid at a given time."""
        at_time = at_time or datetime.now()
        
        if self.valid_from and at_time < self.valid_from:
            return False
        if self.valid_until and at_time > self.valid_until:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.relation_type.value,
            "properties": self.properties,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relation':
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["type"]),
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class Observation:
    """A recorded observation or event."""
    id: str
    timestamp: datetime
    observer: str
    content: Dict[str, Any]
    entities_affected: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "observer": self.observer,
            "content": self.content,
            "entities_affected": self.entities_affected
        }


class StateGraph:
    """
    Knowledge graph for world state representation.
    
    Tracks entities, relationships, and changes over time.
    Supports:
    - Entity CRUD operations
    - Relationship management
    - Temporal queries
    - Subgraph extraction
    """
    
    def __init__(self, name: str = "world"):
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.observations: List[Observation] = []
        self.created_at = datetime.now()
        
        # Indices for fast lookup
        self._entity_by_name: Dict[str, str] = {}  # name -> id
        self._entity_by_type: Dict[EntityType, Set[str]] = {}
        self._relations_by_source: Dict[str, Set[str]] = {}
        self._relations_by_target: Dict[str, Set[str]] = {}
    
    # ============ Entity Operations ============
    
    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        properties: Dict[str, Any] = None,
        confidence: float = 1.0,
        source: str = "observed"
    ) -> Entity:
        """Add a new entity to the graph."""
        entity_id = str(uuid.uuid4())[:8]
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            confidence=confidence,
            source=source
        )
        
        self.entities[entity_id] = entity
        
        # Update indices
        self._entity_by_name[name] = entity_id
        if entity_type not in self._entity_by_type:
            self._entity_by_type[entity_type] = set()
        self._entity_by_type[entity_type].add(entity_id)
        
        return entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        entity_id = self._entity_by_name.get(name)
        if entity_id:
            return self.entities.get(entity_id)
        return None
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a type."""
        entity_ids = self._entity_by_type.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def update_entity(self, entity_id: str, properties: Dict[str, Any]):
        """Update entity properties."""
        if entity_id in self.entities:
            self.entities[entity_id].update(properties)
    
    def remove_entity(self, entity_id: str):
        """Remove an entity and its relations."""
        if entity_id not in self.entities:
            return
        
        entity = self.entities[entity_id]
        
        # Remove from indices
        if entity.name in self._entity_by_name:
            del self._entity_by_name[entity.name]
        if entity.entity_type in self._entity_by_type:
            self._entity_by_type[entity.entity_type].discard(entity_id)
        
        # Remove related relations
        for rel_id in list(self._relations_by_source.get(entity_id, set())):
            self.remove_relation(rel_id)
        for rel_id in list(self._relations_by_target.get(entity_id, set())):
            self.remove_relation(rel_id)
        
        del self.entities[entity_id]
    
    # ============ Relation Operations ============
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: Dict[str, Any] = None,
        confidence: float = 1.0
    ) -> Optional[Relation]:
        """Add a relationship between entities."""
        if source_id not in self.entities or target_id not in self.entities:
            return None
        
        relation_id = str(uuid.uuid4())[:8]
        relation = Relation(
            id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            confidence=confidence
        )
        
        self.relations[relation_id] = relation
        
        # Update indices
        if source_id not in self._relations_by_source:
            self._relations_by_source[source_id] = set()
        self._relations_by_source[source_id].add(relation_id)
        
        if target_id not in self._relations_by_target:
            self._relations_by_target[target_id] = set()
        self._relations_by_target[target_id].add(relation_id)
        
        return relation
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get a relation by ID."""
        return self.relations.get(relation_id)
    
    def get_relations_from(self, entity_id: str) -> List[Relation]:
        """Get all relations from an entity."""
        rel_ids = self._relations_by_source.get(entity_id, set())
        return [self.relations[rid] for rid in rel_ids if rid in self.relations]
    
    def get_relations_to(self, entity_id: str) -> List[Relation]:
        """Get all relations to an entity."""
        rel_ids = self._relations_by_target.get(entity_id, set())
        return [self.relations[rid] for rid in rel_ids if rid in self.relations]
    
    def get_neighbors(self, entity_id: str) -> List[Entity]:
        """Get all entities connected to an entity."""
        neighbors = set()
        
        for rel in self.get_relations_from(entity_id):
            neighbors.add(rel.target_id)
        for rel in self.get_relations_to(entity_id):
            neighbors.add(rel.source_id)
        
        return [self.entities[eid] for eid in neighbors if eid in self.entities]
    
    def remove_relation(self, relation_id: str):
        """Remove a relation."""
        if relation_id not in self.relations:
            return
        
        relation = self.relations[relation_id]
        
        # Update indices
        if relation.source_id in self._relations_by_source:
            self._relations_by_source[relation.source_id].discard(relation_id)
        if relation.target_id in self._relations_by_target:
            self._relations_by_target[relation.target_id].discard(relation_id)
        
        del self.relations[relation_id]
    
    # ============ Queries ============
    
    def find_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 5
    ) -> Optional[List[Tuple[Entity, Relation]]]:
        """Find a path between two entities."""
        if from_id not in self.entities or to_id not in self.entities:
            return None
        
        if from_id == to_id:
            return [(self.entities[from_id], None)]
        
        # BFS
        visited = {from_id}
        queue = [(from_id, [])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            for rel in self.get_relations_from(current_id):
                next_id = rel.target_id
                if next_id in visited:
                    continue
                
                new_path = path + [(self.entities[current_id], rel)]
                
                if next_id == to_id:
                    new_path.append((self.entities[to_id], None))
                    return new_path
                
                visited.add(next_id)
                queue.append((next_id, new_path))
        
        return None
    
    def query(
        self,
        entity_type: EntityType = None,
        properties: Dict[str, Any] = None,
        min_confidence: float = 0.0
    ) -> List[Entity]:
        """Query entities with filters."""
        results = []
        
        candidates = self.entities.values()
        if entity_type:
            candidates = self.get_entities_by_type(entity_type)
        
        for entity in candidates:
            if entity.confidence < min_confidence:
                continue
            
            if properties:
                match = all(
                    entity.properties.get(k) == v
                    for k, v in properties.items()
                )
                if not match:
                    continue
            
            results.append(entity)
        
        return results
    
    # ============ Observations ============
    
    def record_observation(
        self,
        observer: str,
        content: Dict[str, Any],
        entities_affected: List[str] = None
    ) -> Observation:
        """Record an observation."""
        obs = Observation(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            observer=observer,
            content=content,
            entities_affected=entities_affected or []
        )
        self.observations.append(obs)
        return obs
    
    def get_observations(
        self,
        entity_id: str = None,
        since: datetime = None,
        limit: int = 100
    ) -> List[Observation]:
        """Get observations, optionally filtered."""
        results = self.observations
        
        if entity_id:
            results = [o for o in results if entity_id in o.entities_affected]
        
        if since:
            results = [o for o in results if o.timestamp >= since]
        
        return results[-limit:]
    
    # ============ Serialization ============
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "name": self.name,
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "relations": {rid: r.to_dict() for rid, r in self.relations.items()},
            "observations": [o.to_dict() for o in self.observations[-100:]]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateGraph':
        """Deserialize graph from dictionary."""
        graph = cls(name=data.get("name", "world"))
        
        # Load entities first
        for eid, edata in data.get("entities", {}).items():
            entity = Entity.from_dict(edata)
            graph.entities[entity.id] = entity
            graph._entity_by_name[entity.name] = entity.id
            if entity.entity_type not in graph._entity_by_type:
                graph._entity_by_type[entity.entity_type] = set()
            graph._entity_by_type[entity.entity_type].add(entity.id)
        
        # Load relations
        for rid, rdata in data.get("relations", {}).items():
            relation = Relation.from_dict(rdata)
            graph.relations[relation.id] = relation
            
            if relation.source_id not in graph._relations_by_source:
                graph._relations_by_source[relation.source_id] = set()
            graph._relations_by_source[relation.source_id].add(relation.id)
            
            if relation.target_id not in graph._relations_by_target:
                graph._relations_by_target[relation.target_id] = set()
            graph._relations_by_target[relation.target_id].add(relation.id)
        
        return graph
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "observation_count": len(self.observations),
            "entity_types": {
                t.value: len(ids)
                for t, ids in self._entity_by_type.items()
            }
        }
