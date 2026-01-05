"""
AION Knowledge Ingester
=======================

Ingest extracted knowledge into AION's memory systems:
- Entity extraction and linking
- Relation extraction
- Knowledge graph integration
- Memory consolidation
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from datetime import datetime
from collections import defaultdict


@dataclass
class Entity:
    """An extracted entity."""
    id: str
    name: str
    entity_type: str  # PERSON, ORG, LOCATION, CONCEPT, EVENT
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    source_urls: List[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    mention_count: int = 1


@dataclass
class Relation:
    """A relation between entities."""
    id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float
    source_url: str
    timestamp: datetime


@dataclass
class Fact:
    """A verified fact for knowledge base."""
    id: str
    content: str
    subject_entity: Optional[str]
    predicate: Optional[str]
    object_entity: Optional[str]
    source_urls: List[str]
    confidence: float
    verification_status: str
    timestamp: datetime
    expiry: Optional[datetime] = None  # For temporal facts


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge for memory storage."""
    id: str
    content: str
    embedding: Optional[List[float]]
    source_type: str  # news, forum, article, media
    source_url: str
    entities: List[str]
    topics: List[str]
    importance: float
    timestamp: datetime


class KnowledgeIngester:
    """
    Ingest and organize knowledge from various sources.
    """
    
    def __init__(self):
        # Knowledge stores
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.facts: Dict[str, Fact] = {}
        self.chunks: List[KnowledgeChunk] = []
        
        # Indexes
        self.entity_by_name: Dict[str, str] = {}  # name -> entity_id
        self.facts_by_entity: Dict[str, List[str]] = defaultdict(list)
        self.chunks_by_topic: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'entities_created': 0,
            'relations_created': 0,
            'facts_stored': 0,
            'chunks_stored': 0
        }
    
    def ingest_content(self, content: Dict, source_type: str) -> Dict:
        """
        Ingest content and extract knowledge.
        
        Args:
            content: Dictionary with 'text', 'url', 'title', 'metadata'
            source_type: Type of source (news, forum, article, media)
        
        Returns:
            Summary of ingestion results
        """
        
        text = content.get('text', '')
        url = content.get('url', '')
        title = content.get('title', '')
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Extract relations
        relations = self._extract_relations(text, entities)
        
        # Extract facts
        facts = self._extract_facts(text, entities)
        
        # Create knowledge chunk
        chunk = self._create_chunk(content, source_type, entities)
        
        # Store everything
        stored_entities = self._store_entities(entities, url)
        stored_relations = self._store_relations(relations, url)
        stored_facts = self._store_facts(facts, url)
        self._store_chunk(chunk)
        
        return {
            'entities_found': len(entities),
            'relations_found': len(relations),
            'facts_extracted': len(facts),
            'entities_stored': stored_entities,
            'relations_stored': stored_relations,
            'facts_stored': stored_facts
        }
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text."""
        import re
        
        entities = []
        
        # Pattern-based NER
        # Capitalized phrases (potential names, organizations, places)
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        for match in re.finditer(name_pattern, text):
            name = match.group(1)
            if len(name) > 3:
                entities.append({
                    'name': name,
                    'type': self._classify_entity(name),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for e in entities:
            if e['name'] not in seen:
                seen.add(e['name'])
                unique_entities.append(e)
        
        return unique_entities[:50]  # Limit per document
    
    def _classify_entity(self, name: str) -> str:
        """Classify entity type."""
        # Simple heuristics
        name_lower = name.lower()
        
        # Organization patterns
        org_patterns = ['inc', 'corp', 'ltd', 'company', 'university', 'institute']
        if any(p in name_lower for p in org_patterns):
            return 'ORGANIZATION'
        
        # Location patterns
        loc_patterns = ['city', 'country', 'state', 'river', 'mountain']
        if any(p in name_lower for p in loc_patterns):
            return 'LOCATION'
        
        # Default to PERSON for capitalized names
        words = name.split()
        if len(words) == 2 and all(w.istitle() for w in words):
            return 'PERSON'
        
        return 'CONCEPT'
    
    def _extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relations between entities."""
        import re
        
        relations = []
        
        # Relation patterns
        patterns = [
            (r'(\w+) is (?:a|the) (\w+)', 'is_a'),
            (r'(\w+) works (?:for|at) (\w+)', 'works_for'),
            (r'(\w+) founded (\w+)', 'founded'),
            (r'(\w+) is located in (\w+)', 'located_in'),
            (r'(\w+) acquired (\w+)', 'acquired'),
            (r'(\w+) said (\w+)', 'stated'),
        ]
        
        text_lower = text.lower()
        
        for pattern, predicate in patterns:
            for match in re.finditer(pattern, text_lower):
                relations.append({
                    'subject': match.group(1),
                    'predicate': predicate,
                    'object': match.group(2)
                })
        
        return relations[:20]
    
    def _extract_facts(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract factual statements."""
        import re
        
        facts = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        
        for sent in sentences:
            sent = sent.strip()
            
            # Skip too short or too long
            if len(sent) < 20 or len(sent) > 500:
                continue
            
            # Look for factual patterns
            if any(p in sent.lower() for p in ['is', 'was', 'are', 'were', 'has', 'have']):
                # Check if contains entities
                entity_names = [e['name'].lower() for e in entities]
                if any(name in sent.lower() for name in entity_names):
                    facts.append({
                        'content': sent,
                        'entities': [e['name'] for e in entities if e['name'].lower() in sent.lower()]
                    })
        
        return facts[:30]
    
    def _create_chunk(self, content: Dict, source_type: str, 
                     entities: List[Dict]) -> KnowledgeChunk:
        """Create knowledge chunk for storage."""
        
        text = content.get('text', '')[:2000]  # Limit chunk size
        url = content.get('url', '')
        
        # Extract topics (simple keyword extraction)
        topics = self._extract_topics(text)
        
        # Calculate importance
        importance = self._calculate_importance(content, entities)
        
        return KnowledgeChunk(
            id=hashlib.md5(url.encode()).hexdigest()[:16],
            content=text,
            embedding=None,  # Would compute with embedding model
            source_type=source_type,
            source_url=url,
            entities=[e['name'] for e in entities],
            topics=topics,
            importance=importance,
            timestamp=datetime.now()
        )
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords."""
        import re
        
        # Common stopwords to filter
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                    'as', 'into', 'through', 'that', 'this', 'these', 'those',
                    'and', 'but', 'or', 'nor', 'so', 'yet'}
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_counts = defaultdict(int)
        
        for word in words:
            if word not in stopwords:
                word_counts[word] += 1
        
        # Return top topics
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:10]]
    
    def _calculate_importance(self, content: Dict, entities: List[Dict]) -> float:
        """Calculate importance score for content."""
        
        score = 0.5  # Base score
        
        # More entities = more important
        score += min(0.2, len(entities) * 0.02)
        
        # Longer content = more substantial
        text_len = len(content.get('text', ''))
        score += min(0.1, text_len / 10000)
        
        # Metadata presence
        if content.get('author'):
            score += 0.05
        if content.get('publish_date'):
            score += 0.05
        
        return min(1.0, score)
    
    def _store_entities(self, entities: List[Dict], source_url: str) -> int:
        """Store or update entities."""
        stored = 0
        
        for e in entities:
            name = e['name']
            
            # Check if entity exists
            if name in self.entity_by_name:
                # Update existing
                entity_id = self.entity_by_name[name]
                self.entities[entity_id].mention_count += 1
                if source_url not in self.entities[entity_id].source_urls:
                    self.entities[entity_id].source_urls.append(source_url)
            else:
                # Create new
                entity_id = hashlib.md5(name.encode()).hexdigest()[:12]
                entity = Entity(
                    id=entity_id,
                    name=name,
                    entity_type=e.get('type', 'CONCEPT'),
                    source_urls=[source_url]
                )
                self.entities[entity_id] = entity
                self.entity_by_name[name] = entity_id
                self.stats['entities_created'] += 1
                stored += 1
        
        return stored
    
    def _store_relations(self, relations: List[Dict], source_url: str) -> int:
        """Store relations."""
        stored = 0
        
        for r in relations:
            rel = Relation(
                id=hashlib.md5(f"{r['subject']}{r['predicate']}{r['object']}".encode()).hexdigest()[:12],
                subject_id=self.entity_by_name.get(r['subject'], ''),
                predicate=r['predicate'],
                object_id=self.entity_by_name.get(r['object'], ''),
                confidence=0.7,
                source_url=source_url,
                timestamp=datetime.now()
            )
            self.relations.append(rel)
            self.stats['relations_created'] += 1
            stored += 1
        
        return stored
    
    def _store_facts(self, facts: List[Dict], source_url: str) -> int:
        """Store extracted facts."""
        stored = 0
        
        for f in facts:
            fact_id = hashlib.md5(f['content'][:100].encode()).hexdigest()[:12]
            
            if fact_id not in self.facts:
                fact = Fact(
                    id=fact_id,
                    content=f['content'],
                    subject_entity=f['entities'][0] if f['entities'] else None,
                    predicate=None,
                    object_entity=None,
                    source_urls=[source_url],
                    confidence=0.6,
                    verification_status='unverified',
                    timestamp=datetime.now()
                )
                self.facts[fact_id] = fact
                
                # Index by entity
                for entity in f.get('entities', []):
                    self.facts_by_entity[entity].append(fact_id)
                
                self.stats['facts_stored'] += 1
                stored += 1
        
        return stored
    
    def _store_chunk(self, chunk: KnowledgeChunk):
        """Store knowledge chunk."""
        self.chunks.append(chunk)
        
        # Index by topic
        for topic in chunk.topics:
            self.chunks_by_topic[topic].append(chunk.id)
        
        self.stats['chunks_stored'] += 1
    
    def query_by_entity(self, entity_name: str) -> Dict:
        """Query knowledge about an entity."""
        
        if entity_name not in self.entity_by_name:
            return {'found': False}
        
        entity_id = self.entity_by_name[entity_name]
        entity = self.entities[entity_id]
        
        # Get related facts
        fact_ids = self.facts_by_entity.get(entity_name, [])
        related_facts = [self.facts[fid] for fid in fact_ids if fid in self.facts]
        
        # Get related relations
        related_relations = [
            r for r in self.relations 
            if r.subject_id == entity_id or r.object_id == entity_id
        ]
        
        return {
            'found': True,
            'entity': entity,
            'facts': related_facts[:10],
            'relations': related_relations[:10]
        }
    
    def query_by_topic(self, topic: str, limit: int = 10) -> List[KnowledgeChunk]:
        """Query knowledge chunks by topic."""
        chunk_ids = self.chunks_by_topic.get(topic, [])
        return [c for c in self.chunks if c.id in chunk_ids][:limit]
    
    def get_stats(self) -> Dict:
        """Get ingestion statistics."""
        return {
            **self.stats,
            'total_entities': len(self.entities),
            'total_relations': len(self.relations),
            'total_facts': len(self.facts),
            'total_chunks': len(self.chunks)
        }


def demo():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🧠 AION KNOWLEDGE INGESTER 🧠                                    ║
║                                                                           ║
║     Entity Extraction, Relations, Knowledge Graph                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    ingester = KnowledgeIngester()
    
    # Test with sample content
    sample_content = {
        'text': '''
        Google announced today that CEO Sundar Pichai will lead the new AI initiative.
        The company, headquartered in Mountain View, California, is investing heavily
        in artificial intelligence research. Microsoft and Amazon are also competing
        in this space. Pichai said the future of technology is AI-powered.
        ''',
        'url': 'https://example.com/tech-news',
        'title': 'Google AI Announcement'
    }
    
    result = ingester.ingest_content(sample_content, 'news')
    
    print("✓ Sample content ingested:")
    print(f"   • Entities found: {result['entities_found']}")
    print(f"   • Relations found: {result['relations_found']}")
    print(f"   • Facts extracted: {result['facts_extracted']}")
    
    stats = ingester.get_stats()
    print(f"\n✓ Knowledge base stats:")
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Knowledge Ingester ready to build AION's brain! 🧠💡")


if __name__ == "__main__":
    demo()
