"""
AION News Aggregator
====================

Aggregate news from multiple sources:
- RSS feed monitoring
- Breaking news detection
- Topic clustering
- Source credibility scoring
"""

import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import xml.etree.ElementTree as ET


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class NewsCategory(Enum):
    """News categories."""
    WORLD = "world"
    POLITICS = "politics"
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    HEALTH = "health"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    OPINION = "opinion"
    LOCAL = "local"
    OTHER = "other"


class BreakingLevel(Enum):
    """Breaking news importance level."""
    NORMAL = 0
    DEVELOPING = 1
    BREAKING = 2
    URGENT = 3


@dataclass
class NewsSource:
    """A news source (outlet)."""
    name: str
    url: str
    rss_feeds: List[str]
    credibility_score: float  # 0-1
    bias_rating: str  # left, center-left, center, center-right, right
    categories: List[NewsCategory]
    language: str = "en"
    country: str = "US"


@dataclass
class NewsArticle:
    """A news article."""
    id: str
    title: str
    url: str
    summary: str
    content: Optional[str]
    source: str
    author: Optional[str]
    publish_date: datetime
    fetch_date: datetime
    category: NewsCategory
    keywords: List[str]
    entities: List[Dict]  # People, organizations, locations
    sentiment: float  # -1 to 1
    breaking_level: BreakingLevel
    image_url: Optional[str]
    related_articles: List[str] = field(default_factory=list)
    
    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.publish_date).total_seconds() / 3600
    
    @property
    def is_fresh(self) -> bool:
        return self.age_hours < 24


@dataclass
class NewsTopic:
    """A clustered news topic (story)."""
    id: str
    title: str
    summary: str
    articles: List[NewsArticle]
    main_entities: List[str]
    first_seen: datetime
    last_updated: datetime
    trending_score: float
    
    @property
    def article_count(self) -> int:
        return len(self.articles)
    
    @property
    def sources(self) -> List[str]:
        return list(set(a.source for a in self.articles))


# =============================================================================
# RSS PARSER
# =============================================================================

class RSSParser:
    """Parse RSS and Atom feeds."""
    
    # Common RSS date formats
    DATE_FORMATS = [
        '%a, %d %b %Y %H:%M:%S %z',
        '%a, %d %b %Y %H:%M:%S GMT',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S',
    ]
    
    def parse(self, xml_content: str, source_name: str) -> List[Dict]:
        """Parse RSS/Atom feed content."""
        articles = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Detect feed type
            if root.tag == 'rss' or root.find('channel') is not None:
                articles = self._parse_rss(root, source_name)
            elif 'feed' in root.tag.lower() or root.tag.endswith('}feed'):
                articles = self._parse_atom(root, source_name)
            
        except ET.ParseError:
            pass
        
        return articles
    
    def _parse_rss(self, root: ET.Element, source: str) -> List[Dict]:
        """Parse RSS 2.0 feed."""
        articles = []
        channel = root.find('channel')
        
        if channel is None:
            return articles
        
        for item in channel.findall('item'):
            article = {
                'title': self._get_text(item, 'title'),
                'url': self._get_text(item, 'link'),
                'summary': self._get_text(item, 'description') or '',
                'author': self._get_text(item, 'author') or self._get_text(item, 'dc:creator'),
                'publish_date': self._parse_date(self._get_text(item, 'pubDate')),
                'source': source,
                'categories': [c.text for c in item.findall('category') if c.text],
            }
            
            # Get enclosure (media)
            enclosure = item.find('enclosure')
            if enclosure is not None:
                article['image_url'] = enclosure.get('url')
            
            if article['title'] and article['url']:
                articles.append(article)
        
        return articles
    
    def _parse_atom(self, root: ET.Element, source: str) -> List[Dict]:
        """Parse Atom feed."""
        articles = []
        
        # Handle namespaces
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('.//entry') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
        
        for entry in entries:
            # Get link
            link_elem = entry.find('link') or entry.find('{http://www.w3.org/2005/Atom}link')
            url = link_elem.get('href') if link_elem is not None else None
            
            article = {
                'title': self._get_text(entry, 'title') or self._get_text(entry, '{http://www.w3.org/2005/Atom}title'),
                'url': url,
                'summary': self._get_text(entry, 'summary') or self._get_text(entry, 'content'),
                'author': self._get_text(entry, 'author/name'),
                'publish_date': self._parse_date(self._get_text(entry, 'published') or self._get_text(entry, 'updated')),
                'source': source,
            }
            
            if article['title'] and article['url']:
                articles.append(article)
        
        return articles
    
    def _get_text(self, elem: ET.Element, path: str) -> Optional[str]:
        """Get text content of element."""
        child = elem.find(path)
        if child is not None and child.text:
            return child.text.strip()
        return None
    
    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """Parse date string."""
        if not date_str:
            return datetime.now()
        
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return datetime.now()


# =============================================================================
# BREAKING NEWS DETECTOR
# =============================================================================

class BreakingNewsDetector:
    """Detect breaking and developing news."""
    
    BREAKING_KEYWORDS = {
        'breaking', 'urgent', 'alert', 'just in', 'developing',
        'breaking news', 'live updates', 'happening now'
    }
    
    IMPORTANCE_KEYWORDS = {
        'death', 'killed', 'attack', 'explosion', 'earthquake',
        'crash', 'shooting', 'emergency', 'disaster', 'crisis',
        'war', 'invasion', 'election', 'resignation'
    }
    
    def detect(self, article: Dict, recent_articles: List[Dict] = None) -> BreakingLevel:
        """Detect if article is breaking news."""
        title_lower = article.get('title', '').lower()
        summary_lower = article.get('summary', '').lower()
        combined = f"{title_lower} {summary_lower}"
        
        # Check for explicit breaking indicators
        for keyword in self.BREAKING_KEYWORDS:
            if keyword in combined:
                return BreakingLevel.BREAKING
        
        # Check for important event keywords
        importance_count = sum(1 for kw in self.IMPORTANCE_KEYWORDS if kw in combined)
        if importance_count >= 2:
            return BreakingLevel.DEVELOPING
        
        # Check how many sources are covering same story
        if recent_articles:
            similar_count = self._count_similar(article, recent_articles)
            if similar_count >= 5:
                return BreakingLevel.DEVELOPING
        
        return BreakingLevel.NORMAL
    
    def _count_similar(self, article: Dict, recent: List[Dict]) -> int:
        """Count similar recent articles (same story)."""
        title_words = set(article.get('title', '').lower().split())
        count = 0
        
        for other in recent:
            other_words = set(other.get('title', '').lower().split())
            overlap = len(title_words & other_words)
            if overlap >= 3:
                count += 1
        
        return count


# =============================================================================
# TOPIC CLUSTERER
# =============================================================================

class TopicClusterer:
    """Cluster related news articles into topics."""
    
    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold
        self.topics: Dict[str, NewsTopic] = {}
    
    def add_article(self, article: NewsArticle) -> Optional[str]:
        """Add article to appropriate topic or create new one."""
        
        # Find matching topic
        best_topic = None
        best_score = 0
        
        for topic_id, topic in self.topics.items():
            score = self._calculate_similarity(article, topic)
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_topic = topic_id
        
        if best_topic:
            # Add to existing topic
            self.topics[best_topic].articles.append(article)
            self.topics[best_topic].last_updated = datetime.now()
            self._update_trending_score(self.topics[best_topic])
            return best_topic
        else:
            # Create new topic
            topic_id = hashlib.md5(article.title.encode()).hexdigest()[:12]
            self.topics[topic_id] = NewsTopic(
                id=topic_id,
                title=article.title,
                summary=article.summary,
                articles=[article],
                main_entities=[e['name'] for e in article.entities[:5]],
                first_seen=article.publish_date,
                last_updated=datetime.now(),
                trending_score=1.0
            )
            return topic_id
    
    def _calculate_similarity(self, article: NewsArticle, topic: NewsTopic) -> float:
        """Calculate similarity between article and topic."""
        # Simple word overlap similarity
        article_words = set(article.title.lower().split())
        article_words.update(article.summary.lower().split())
        
        topic_words = set(topic.title.lower().split())
        topic_words.update(topic.summary.lower().split())
        
        overlap = len(article_words & topic_words)
        total = len(article_words | topic_words)
        
        if total == 0:
            return 0
        
        return overlap / total
    
    def _update_trending_score(self, topic: NewsTopic):
        """Update topic trending score."""
        # Score based on article count and recency
        article_score = min(10, len(topic.articles)) / 10
        
        # Freshness decay
        hours_since_update = (datetime.now() - topic.last_updated).total_seconds() / 3600
        freshness = max(0, 1 - (hours_since_update / 48))
        
        # Source diversity
        source_count = len(set(a.source for a in topic.articles))
        diversity = min(5, source_count) / 5
        
        topic.trending_score = (article_score * 0.4 + freshness * 0.4 + diversity * 0.2)
    
    def get_trending(self, limit: int = 10) -> List[NewsTopic]:
        """Get top trending topics."""
        sorted_topics = sorted(
            self.topics.values(),
            key=lambda t: t.trending_score,
            reverse=True
        )
        return sorted_topics[:limit]


# =============================================================================
# NEWS AGGREGATOR
# =============================================================================

class NewsAggregator:
    """
    Main news aggregation engine.
    Collects, processes, and organizes news from multiple sources.
    """
    
    # Default news sources (no API key needed - RSS feeds)
    DEFAULT_SOURCES = [
        NewsSource(
            name="BBC News",
            url="https://www.bbc.com/news",
            rss_feeds=["https://feeds.bbci.co.uk/news/rss.xml"],
            credibility_score=0.95,
            bias_rating="center",
            categories=[NewsCategory.WORLD, NewsCategory.POLITICS],
            country="UK"
        ),
        NewsSource(
            name="Reuters",
            url="https://www.reuters.com",
            rss_feeds=["https://www.reutersagency.com/feed/"],
            credibility_score=0.95,
            bias_rating="center",
            categories=[NewsCategory.WORLD, NewsCategory.BUSINESS]
        ),
        NewsSource(
            name="NPR",
            url="https://www.npr.org",
            rss_feeds=["https://feeds.npr.org/1001/rss.xml"],
            credibility_score=0.90,
            bias_rating="center-left",
            categories=[NewsCategory.WORLD, NewsCategory.POLITICS]
        ),
        NewsSource(
            name="TechCrunch",
            url="https://techcrunch.com",
            rss_feeds=["https://techcrunch.com/feed/"],
            credibility_score=0.85,
            bias_rating="center",
            categories=[NewsCategory.TECHNOLOGY]
        ),
        NewsSource(
            name="Ars Technica",
            url="https://arstechnica.com",
            rss_feeds=["https://feeds.arstechnica.com/arstechnica/features"],
            credibility_score=0.90,
            bias_rating="center",
            categories=[NewsCategory.TECHNOLOGY, NewsCategory.SCIENCE]
        ),
        NewsSource(
            name="Science Daily",
            url="https://www.sciencedaily.com",
            rss_feeds=["https://www.sciencedaily.com/rss/all.xml"],
            credibility_score=0.90,
            bias_rating="center",
            categories=[NewsCategory.SCIENCE]
        ),
    ]
    
    def __init__(self):
        self.sources: List[NewsSource] = self.DEFAULT_SOURCES.copy()
        self.rss_parser = RSSParser()
        self.breaking_detector = BreakingNewsDetector()
        self.topic_clusterer = TopicClusterer()
        self.articles: List[NewsArticle] = []
        self.seen_urls: Set[str] = set()
    
    def add_source(self, source: NewsSource):
        """Add a news source."""
        self.sources.append(source)
    
    async def fetch_all(self, session) -> List[NewsArticle]:
        """Fetch news from all sources."""
        new_articles = []
        
        for source in self.sources:
            for feed_url in source.rss_feeds:
                try:
                    articles = await self._fetch_feed(session, feed_url, source)
                    new_articles.extend(articles)
                except Exception as e:
                    print(f"Error fetching {feed_url}: {e}")
        
        return new_articles
    
    async def _fetch_feed(self, session, feed_url: str, source: NewsSource) -> List[NewsArticle]:
        """Fetch and parse a single RSS feed."""
        articles = []
        
        try:
            async with session.get(feed_url, timeout=30) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    raw_articles = self.rss_parser.parse(xml_content, source.name)
                    
                    for raw in raw_articles:
                        if raw['url'] in self.seen_urls:
                            continue
                        
                        self.seen_urls.add(raw['url'])
                        
                        # Create article
                        article = self._create_article(raw, source)
                        
                        # Detect breaking level
                        article.breaking_level = self.breaking_detector.detect(raw, self.articles[-100:])
                        
                        # Add to topic cluster
                        self.topic_clusterer.add_article(article)
                        
                        articles.append(article)
                        self.articles.append(article)
        
        except Exception:
            pass
        
        return articles
    
    def _create_article(self, raw: Dict, source: NewsSource) -> NewsArticle:
        """Create NewsArticle from raw data."""
        return NewsArticle(
            id=hashlib.md5(raw['url'].encode()).hexdigest()[:16],
            title=raw.get('title', 'Untitled'),
            url=raw['url'],
            summary=raw.get('summary', '')[:500],
            content=None,
            source=source.name,
            author=raw.get('author'),
            publish_date=raw.get('publish_date', datetime.now()),
            fetch_date=datetime.now(),
            category=self._detect_category(raw, source),
            keywords=self._extract_keywords(raw),
            entities=self._extract_entities(raw),
            sentiment=0.0,
            breaking_level=BreakingLevel.NORMAL,
            image_url=raw.get('image_url')
        )
    
    def _detect_category(self, raw: Dict, source: NewsSource) -> NewsCategory:
        """Detect article category."""
        # Use source default if only one category
        if len(source.categories) == 1:
            return source.categories[0]
        
        # Check feed categories
        categories = raw.get('categories', [])
        text = f"{raw.get('title', '')} {' '.join(categories)}".lower()
        
        category_keywords = {
            NewsCategory.TECHNOLOGY: ['tech', 'software', 'ai', 'computer', 'startup', 'app'],
            NewsCategory.SCIENCE: ['science', 'research', 'study', 'discovery', 'space'],
            NewsCategory.POLITICS: ['politics', 'election', 'congress', 'president', 'vote'],
            NewsCategory.BUSINESS: ['business', 'market', 'stock', 'economy', 'company'],
            NewsCategory.HEALTH: ['health', 'medical', 'disease', 'hospital', 'drug'],
            NewsCategory.SPORTS: ['sports', 'game', 'team', 'player', 'championship'],
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in text for kw in keywords):
                return category
        
        return NewsCategory.OTHER
    
    def _extract_keywords(self, raw: Dict) -> List[str]:
        """Extract keywords from article."""
        text = f"{raw.get('title', '')} {raw.get('summary', '')}"
        
        # Simple keyword extraction (stop word removal)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                    'as', 'into', 'through', 'during', 'before', 'after', 'above',
                    'below', 'between', 'under', 'again', 'further', 'then', 'once',
                    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                    'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
                    'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
                    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                    'such', 'no', 'any', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]
        
        # Return most common
        from collections import Counter
        return [w for w, _ in Counter(keywords).most_common(10)]
    
    def _extract_entities(self, raw: Dict) -> List[Dict]:
        """Extract named entities from article."""
        # Simple pattern-based NER
        text = f"{raw.get('title', '')} {raw.get('summary', '')}"
        entities = []
        
        # Find capitalized phrases (potential entities)
        pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b'
        matches = re.findall(pattern, text)
        
        for match in set(matches):
            if len(match) > 2 and match not in ['The', 'This', 'That']:
                entities.append({
                    'name': match,
                    'type': 'UNKNOWN'
                })
        
        return entities[:10]
    
    def get_headlines(self, limit: int = 10, 
                     category: Optional[NewsCategory] = None) -> List[NewsArticle]:
        """Get top headlines."""
        articles = self.articles
        
        if category:
            articles = [a for a in articles if a.category == category]
        
        # Sort by freshness and breaking level
        articles = sorted(
            articles,
            key=lambda a: (a.breaking_level.value, -a.age_hours),
            reverse=True
        )
        
        return articles[:limit]
    
    def get_trending_topics(self, limit: int = 10) -> List[NewsTopic]:
        """Get trending topics."""
        return self.topic_clusterer.get_trending(limit)
    
    def search(self, query: str, limit: int = 20) -> List[NewsArticle]:
        """Search articles by query."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for article in self.articles:
            text = f"{article.title} {article.summary}".lower()
            if all(word in text for word in query_words):
                results.append(article)
        
        return results[:limit]
    
    def get_stats(self) -> Dict:
        """Get aggregator statistics."""
        return {
            'total_articles': len(self.articles),
            'sources': len(self.sources),
            'topics': len(self.topic_clusterer.topics),
            'categories': len(set(a.category for a in self.articles)),
            'breaking_news': sum(1 for a in self.articles if a.breaking_level.value >= 2)
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the news aggregator."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          📰 AION NEWS AGGREGATOR 📰                                       ║
║                                                                           ║
║     Multi-Source News Collection & Topic Clustering                      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    aggregator = NewsAggregator()
    
    print(f"✓ Initialized with {len(aggregator.sources)} news sources:")
    for source in aggregator.sources:
        print(f"   • {source.name} ({source.credibility_score:.0%} credibility)")
    
    # Test RSS parser
    sample_rss = '''<?xml version="1.0"?>
    <rss version="2.0">
    <channel>
        <title>Test Feed</title>
        <item>
            <title>Breaking: Major Event Happens</title>
            <link>https://example.com/news/1</link>
            <description>This is a breaking news story.</description>
            <pubDate>Mon, 01 Jan 2026 12:00:00 GMT</pubDate>
        </item>
    </channel>
    </rss>'''
    
    parser = RSSParser()
    articles = parser.parse(sample_rss, "Test Source")
    print(f"\n✓ RSS Parser: Found {len(articles)} articles")
    
    # Test breaking detection
    detector = BreakingNewsDetector()
    level = detector.detect({'title': 'Breaking: Major earthquake strikes', 'summary': ''})
    print(f"✓ Breaking Detector: {level.name}")
    
    # Test topic clusterer  
    print(f"\n✓ Topic Clusterer ready")
    print(f"   • Similarity threshold: {aggregator.topic_clusterer.similarity_threshold}")
    
    print("\n" + "=" * 60)
    print("News Aggregator ready to collect world knowledge! 📰🌍")


if __name__ == "__main__":
    demo()
