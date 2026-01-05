"""
AION Article Processor
======================

Process articles and documents for knowledge extraction:
- Wikipedia integration
- Academic paper parsing
- Blog content processing
- Readability optimization
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class ProcessedArticle:
    """A processed article."""
    url: str
    title: str
    content: str
    summary: str
    sections: List[Dict]
    key_facts: List[str]
    entities: List[Dict]
    categories: List[str]
    references: List[str]
    word_count: int
    quality_score: float


class ArticleProcessor:
    """Process articles for knowledge extraction."""
    
    def process(self, url: str, html: str, metadata: Dict) -> ProcessedArticle:
        """Process an article."""
        # Extract sections
        sections = self._extract_sections(html)
        
        # Extract key facts
        facts = self._extract_key_facts(html)
        
        # Get quality score
        quality = self._calculate_quality(metadata, sections, facts)
        
        return ProcessedArticle(
            url=url,
            title=metadata.get('title', 'Untitled'),
            content=metadata.get('text', ''),
            summary=metadata.get('summary', ''),
            sections=sections,
            key_facts=facts,
            entities=[],
            categories=[],
            references=self._extract_references(html),
            word_count=len(metadata.get('text', '').split()),
            quality_score=quality
        )
    
    def _extract_sections(self, html: str) -> List[Dict]:
        """Extract document sections."""
        sections = []
        headings = re.findall(r'<h([1-6])[^>]*>([^<]+)</h\1>', html, re.IGNORECASE)
        for level, title in headings:
            sections.append({'level': int(level), 'title': title.strip()})
        return sections
    
    def _extract_key_facts(self, html: str) -> List[str]:
        """Extract key factual statements."""
        facts = []
        text = re.sub(r'<[^>]+>', ' ', html)
        sentences = re.split(r'[.!?]', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20 and any(w in sent.lower() for w in ['is', 'was', 'are', 'were']):
                facts.append(sent[:200])
        return facts[:20]
    
    def _extract_references(self, html: str) -> List[str]:
        """Extract reference links."""
        refs = re.findall(r'href=["\']([^"\']+)["\']', html)
        return [r for r in refs if r.startswith('http')][:30]
    
    def _calculate_quality(self, metadata: Dict, sections: List, facts: List) -> float:
        """Calculate article quality score."""
        score = 0.5
        if len(sections) >= 3: score += 0.1
        if len(facts) >= 10: score += 0.1
        if metadata.get('author'): score += 0.1
        if metadata.get('publish_date'): score += 0.1
        return min(1.0, score)


def demo():
    print("📄 AION Article Processor ready!")


if __name__ == "__main__":
    demo()
