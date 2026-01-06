"""
DuckDuckGo Search (Free, No API Key)
====================================
Web search via DuckDuckGo HTML scraping and Instant Answer API.
"""

import re
import urllib.parse
from typing import List, Dict, Optional
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str


class DuckDuckGo:
    """
    DuckDuckGo search client.
    
    Uses HTML scraping for web results and the instant answer API.
    No API key required.
    """
    
    SEARCH_URL = "https://html.duckduckgo.com/html/"
    INSTANT_URL = "https://api.duckduckgo.com/"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        Search DuckDuckGo for web results.
        
        Args:
            query: Search query
            num_results: Maximum results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            response = requests.post(
                self.SEARCH_URL,
                data={'q': query, 'b': ''},
                headers=self.HEADERS,
                timeout=10
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            results = []
            
            for result in soup.select('.result'):
                title_elem = result.select_one('.result__title a')
                snippet_elem = result.select_one('.result__snippet')
                
                if not title_elem:
                    continue
                
                # Extract URL (DuckDuckGo wraps URLs)
                href = title_elem.get('href', '')
                url = self._extract_url(href)
                
                if not url:
                    continue
                
                results.append(SearchResult(
                    title=title_elem.get_text(strip=True),
                    url=url,
                    snippet=snippet_elem.get_text(strip=True) if snippet_elem else ''
                ))
                
                if len(results) >= num_results:
                    break
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def instant_answer(self, query: str) -> Optional[Dict]:
        """
        Get instant answer from DuckDuckGo API.
        
        Args:
            query: Search query
            
        Returns:
            Dict with answer info or None
        """
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            response = requests.get(
                self.INSTANT_URL,
                params=params,
                headers=self.HEADERS,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for abstract (main answer)
            if data.get('Abstract'):
                return {
                    'type': 'abstract',
                    'title': data.get('Heading', ''),
                    'text': data.get('Abstract'),
                    'source': data.get('AbstractSource'),
                    'url': data.get('AbstractURL'),
                    'image': data.get('Image')
                }
            
            # Check for answer (calculations, conversions)
            if data.get('Answer'):
                return {
                    'type': 'answer',
                    'text': data.get('Answer'),
                    'answer_type': data.get('AnswerType')
                }
            
            # Check for definition
            if data.get('Definition'):
                return {
                    'type': 'definition',
                    'text': data.get('Definition'),
                    'source': data.get('DefinitionSource'),
                    'url': data.get('DefinitionURL')
                }
            
            # Check for related topics
            if data.get('RelatedTopics'):
                topics = []
                for topic in data['RelatedTopics'][:5]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        topics.append({
                            'text': topic.get('Text'),
                            'url': topic.get('FirstURL')
                        })
                
                if topics:
                    return {
                        'type': 'related',
                        'topics': topics
                    }
            
            return None
            
        except Exception as e:
            print(f"DuckDuckGo instant answer error: {e}")
            return None
    
    def _extract_url(self, href: str) -> Optional[str]:
        """Extract actual URL from DuckDuckGo redirect link."""
        if not href:
            return None
        
        # DuckDuckGo wraps URLs like //duckduckgo.com/l/?uddg=https%3A%2F%2F...
        if 'uddg=' in href:
            match = re.search(r'uddg=([^&]+)', href)
            if match:
                return urllib.parse.unquote(match.group(1))
        
        # Direct URL
        if href.startswith('http'):
            return href
        
        return None
