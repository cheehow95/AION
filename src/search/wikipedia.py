"""
Wikipedia API Client (Free, No API Key)
=======================================
Access Wikipedia articles and summaries.
"""

from typing import Optional, Dict, List
import requests


class Wikipedia:
    """
    Wikipedia API client.
    
    Free access to Wikipedia content with no API key required.
    """
    
    API_URL = "https://en.wikipedia.org/api/rest_v1"
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"
    
    HEADERS = {
        'User-Agent': 'AION/1.0 (AI Assistant; contact@example.com)',
        'Accept': 'application/json'
    }
    
    def summary(self, title: str) -> Optional[Dict]:
        """
        Get summary for a Wikipedia article.
        
        Args:
            title: Article title (e.g., "Quantum computing")
            
        Returns:
            Dict with title, extract, image, url or None
        """
        try:
            # Clean title for URL
            title_clean = title.replace(' ', '_')
            
            response = requests.get(
                f"{self.API_URL}/page/summary/{title_clean}",
                headers=self.HEADERS,
                timeout=10
            )
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            data = response.json()
            
            return {
                'title': data.get('title'),
                'extract': data.get('extract'),
                'description': data.get('description'),
                'url': data.get('content_urls', {}).get('desktop', {}).get('page'),
                'image': data.get('thumbnail', {}).get('source'),
                'type': data.get('type')
            }
            
        except Exception as e:
            print(f"Wikipedia summary error: {e}")
            return None
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search Wikipedia for articles.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of {title, snippet, pageid}
        """
        try:
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'srlimit': limit,
                'format': 'json',
                'utf8': 1
            }
            
            response = requests.get(
                self.SEARCH_URL,
                params=params,
                headers=self.HEADERS,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('query', {}).get('search', []):
                # Clean HTML from snippet
                snippet = item.get('snippet', '')
                snippet = snippet.replace('<span class="searchmatch">', '')
                snippet = snippet.replace('</span>', '')
                
                results.append({
                    'title': item.get('title'),
                    'snippet': snippet,
                    'pageid': item.get('pageid'),
                    'url': f"https://en.wikipedia.org/wiki/{item.get('title', '').replace(' ', '_')}"
                })
            
            return results
            
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []
    
    def full_article(self, title: str) -> Optional[str]:
        """
        Get full article text.
        
        Args:
            title: Article title
            
        Returns:
            Plain text content or None
        """
        try:
            params = {
                'action': 'query',
                'titles': title,
                'prop': 'extracts',
                'explaintext': True,
                'format': 'json'
            }
            
            response = requests.get(
                self.SEARCH_URL,
                params=params,
                headers=self.HEADERS,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page in pages.items():
                if page_id != '-1':
                    return page.get('extract')
            
            return None
            
        except Exception as e:
            print(f"Wikipedia article error: {e}")
            return None
    
    def random(self) -> Optional[Dict]:
        """
        Get a random Wikipedia article summary.
        
        Returns:
            Article summary dict or None
        """
        try:
            response = requests.get(
                f"{self.API_URL}/page/random/summary",
                headers=self.HEADERS,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                'title': data.get('title'),
                'extract': data.get('extract'),
                'url': data.get('content_urls', {}).get('desktop', {}).get('page')
            }
            
        except Exception as e:
            print(f"Wikipedia random error: {e}")
            return None
