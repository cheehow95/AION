"""
Unified Web Search
==================
Combines multiple search providers for comprehensive results.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from .duckduckgo import DuckDuckGo, SearchResult
from .wikipedia import Wikipedia


@dataclass
class KnowledgeResult:
    """A knowledge item from any source."""
    title: str
    content: str
    source: str
    url: str = None
    type: str = "text"  # text, answer, definition, article
    confidence: float = 1.0
    metadata: Dict = None


class WebSearch:
    """
    Unified web search interface.
    
    Combines DuckDuckGo and Wikipedia for comprehensive knowledge retrieval.
    No API keys required - completely free.
    """
    
    def __init__(self):
        self.ddg = DuckDuckGo()
        self.wiki = Wikipedia()
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            num_results: Max results per source
            
        Returns:
            List of search results from all sources
        """
        results = []
        
        # DuckDuckGo web search
        ddg_results = self.ddg.search(query, num_results)
        for r in ddg_results:
            results.append({
                'title': r.title,
                'snippet': r.snippet,
                'url': r.url,
                'source': 'duckduckgo'
            })
        
        # Wikipedia search
        wiki_results = self.wiki.search(query, min(3, num_results))
        for r in wiki_results:
            results.append({
                'title': r['title'],
                'snippet': r['snippet'],
                'url': r['url'],
                'source': 'wikipedia'
            })
        
        return results
    
    def answer(self, query: str) -> Optional[KnowledgeResult]:
        """
        Get a direct answer to a question.
        
        Tries instant answers first, then Wikipedia summaries.
        
        Args:
            query: Question or query
            
        Returns:
            KnowledgeResult with answer or None
        """
        # Try DuckDuckGo instant answer first
        instant = self.ddg.instant_answer(query)
        
        if instant:
            if instant['type'] == 'answer':
                return KnowledgeResult(
                    title=query,
                    content=instant['text'],
                    source='duckduckgo',
                    type='answer',
                    confidence=0.95
                )
            elif instant['type'] == 'abstract':
                return KnowledgeResult(
                    title=instant.get('title', query),
                    content=instant['text'],
                    source=instant.get('source', 'duckduckgo'),
                    url=instant.get('url'),
                    type='article',
                    confidence=0.9
                )
            elif instant['type'] == 'definition':
                return KnowledgeResult(
                    title=query,
                    content=instant['text'],
                    source=instant.get('source', 'duckduckgo'),
                    url=instant.get('url'),
                    type='definition',
                    confidence=0.85
                )
        
        # Try Wikipedia summary
        # Extract main topic from query
        topic = self._extract_topic(query)
        wiki_summary = self.wiki.summary(topic)
        
        if wiki_summary and wiki_summary.get('extract'):
            return KnowledgeResult(
                title=wiki_summary.get('title', topic),
                content=wiki_summary['extract'],
                source='wikipedia',
                url=wiki_summary.get('url'),
                type='article',
                confidence=0.8,
                metadata={'image': wiki_summary.get('image')}
            )
        
        return None
    
    def lookup(self, topic: str) -> Optional[Dict]:
        """
        Look up a specific topic/entity.
        
        Args:
            topic: Topic to look up (person, place, concept)
            
        Returns:
            Dict with comprehensive info or None
        """
        # Get Wikipedia summary
        summary = self.wiki.summary(topic)
        
        if not summary:
            return None
        
        # Get DuckDuckGo instant answer for additional context
        instant = self.ddg.instant_answer(topic)
        
        result = {
            'title': summary.get('title'),
            'summary': summary.get('extract'),
            'description': summary.get('description'),
            'url': summary.get('url'),
            'image': summary.get('image'),
            'source': 'wikipedia'
        }
        
        # Add related topics from DuckDuckGo
        if instant and instant.get('type') == 'related':
            result['related'] = instant.get('topics', [])
        
        return result
    
    def quick_facts(self, query: str) -> Dict[str, Any]:
        """
        Get quick facts about a query.
        
        Args:
            query: Query string
            
        Returns:
            Dict with facts from various sources
        """
        facts = {
            'query': query,
            'instant_answer': None,
            'wikipedia': None,
            'web_results': []
        }
        
        # Instant answer
        instant = self.ddg.instant_answer(query)
        if instant:
            facts['instant_answer'] = instant
        
        # Wikipedia
        topic = self._extract_topic(query)
        wiki = self.wiki.summary(topic)
        if wiki:
            facts['wikipedia'] = {
                'title': wiki.get('title'),
                'extract': wiki.get('extract'),
                'url': wiki.get('url')
            }
        
        # Top web results
        web = self.ddg.search(query, 3)
        facts['web_results'] = [asdict(r) for r in web]
        
        return facts
    
    def _extract_topic(self, query: str) -> str:
        """Extract main topic from a question."""
        # Remove common question words
        question_words = [
            'what is', 'who is', 'where is', 'when was', 'how does',
            'why is', 'what are', 'who are', 'tell me about', 'define',
            'explain', 'describe', 'the', 'a', 'an'
        ]
        
        topic = query.lower()
        for word in question_words:
            topic = topic.replace(word, '')
        
        # Clean up
        topic = ' '.join(topic.split())
        topic = topic.strip('?.,!')
        
        return topic or query
