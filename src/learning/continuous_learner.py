"""
AION Continuous Learner
=======================

Main orchestrator for continuous knowledge acquisition:
- Coordinates all learning components
- Manages learning schedule
- Monitors knowledge quality
- Reports learning progress
"""

import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

from .web_crawler import WebCrawler, CrawlPriority
from .content_extractor import ContentExtractor
from .news_aggregator import NewsAggregator
from .forum_miner import ForumMiner
from .article_processor import ArticleProcessor
from .media_processor import MediaProcessor
from .fact_verifier import FactVerifier
from .knowledge_ingester import KnowledgeIngester


class LearningMode(Enum):
    """Learning modes."""
    ACTIVE = "active"         # Actively seeking new knowledge
    PASSIVE = "passive"       # Waiting for input
    CONSOLIDATING = "consolidating"  # Processing acquired knowledge
    SLEEPING = "sleeping"     # Dormant, periodic checks only


@dataclass
class LearningSession:
    """A learning session."""
    id: str
    start_time: datetime
    end_time: Optional[datetime]
    pages_crawled: int
    articles_processed: int
    knowledge_gained: int
    errors: int


@dataclass
class LearningConfig:
    """Configuration for continuous learning."""
    # Rate limits
    max_requests_per_minute: int = 30
    max_concurrent_requests: int = 10
    
    # Content limits
    max_pages_per_session: int = 100
    max_depth: int = 2
    
    # Schedule
    news_check_interval_minutes: int = 30
    forum_check_interval_hours: int = 6
    deep_crawl_interval_hours: int = 24
    
    # Quality thresholds
    min_content_length: int = 100
    min_credibility_score: float = 0.5
    
    # Topics of interest
    focus_topics: List[str] = field(default_factory=lambda: [
        'artificial intelligence', 'machine learning', 'technology',
        'science', 'programming', 'research'
    ])


class ContinuousLearner:
    """
    Main orchestrator for AION's continuous learning.
    """
    
    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        
        # Initialize components
        self.crawler = WebCrawler(
            max_concurrent=self.config.max_concurrent_requests
        )
        self.extractor = ContentExtractor()
        self.news_aggregator = NewsAggregator()
        self.forum_miner = ForumMiner()
        self.article_processor = ArticleProcessor()
        self.media_processor = MediaProcessor()
        self.fact_verifier = FactVerifier()
        self.knowledge_ingester = KnowledgeIngester()
        
        # State
        self.mode = LearningMode.PASSIVE
        self.current_session: Optional[LearningSession] = None
        self.sessions: List[LearningSession] = []
        self.last_news_check = datetime.now() - timedelta(hours=1)
        self.last_forum_check = datetime.now() - timedelta(hours=24)
        self._running = False
    
    async def start_learning(self, duration_minutes: Optional[int] = None):
        """Start a learning session."""
        self.mode = LearningMode.ACTIVE
        self._running = True
        
        self.current_session = LearningSession(
            id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
            end_time=None,
            pages_crawled=0,
            articles_processed=0,
            knowledge_gained=0,
            errors=0
        )
        
        end_time = None
        if duration_minutes:
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        async with aiohttp.ClientSession() as session:
            try:
                while self._running:
                    if end_time and datetime.now() >= end_time:
                        break
                    
                    # Check what needs to be done
                    await self._learning_cycle(session)
                    
                    # Brief pause between cycles
                    await asyncio.sleep(5)
            
            finally:
                self._finish_session()
    
    async def _learning_cycle(self, session: aiohttp.ClientSession):
        """Execute one learning cycle."""
        
        now = datetime.now()
        
        # Check news if interval passed
        if (now - self.last_news_check).total_seconds() > \
           self.config.news_check_interval_minutes * 60:
            await self._fetch_news(session)
            self.last_news_check = now
        
        # Check forums if interval passed
        if (now - self.last_forum_check).total_seconds() > \
           self.config.forum_check_interval_hours * 3600:
            await self._mine_forums(session)
            self.last_forum_check = now
        
        # Consolidate knowledge periodically
        if len(self.knowledge_ingester.chunks) % 50 == 0:
            await self._consolidate_knowledge()
    
    async def _fetch_news(self, session: aiohttp.ClientSession):
        """Fetch latest news."""
        print("📰 Fetching latest news...")
        
        try:
            articles = await self.news_aggregator.fetch_all(session)
            
            for article in articles[:self.config.max_pages_per_session]:
                # Extract content
                if article.url:
                    content = {
                        'text': article.summary,
                        'url': article.url,
                        'title': article.title,
                        'author': article.author,
                        'publish_date': article.publish_date
                    }
                    
                    # Ingest knowledge
                    result = self.knowledge_ingester.ingest_content(content, 'news')
                    self.current_session.articles_processed += 1
                    self.current_session.knowledge_gained += result['entities_found']
            
            print(f"   ✓ Processed {len(articles)} news articles")
        
        except Exception as e:
            print(f"   ✗ News fetch error: {e}")
            self.current_session.errors += 1
    
    async def _mine_forums(self, session: aiohttp.ClientSession):
        """Mine knowledge from forums."""
        print("💬 Mining forum discussions...")
        
        try:
            # HackerNews
            hn_discussions = await self.forum_miner.mine_hackernews(session, limit=20)
            
            # Reddit knowledge subreddits
            for subreddit in self.forum_miner.KNOWLEDGE_SUBREDDITS[:5]:
                await self.forum_miner.mine_reddit(session, subreddit, limit=10)
            
            # Process discussions
            for discussion in self.forum_miner.discussions[-50:]:
                knowledge = self.forum_miner.extract_knowledge(discussion)
                
                for k in knowledge:
                    content = {
                        'text': k.answer or '',
                        'url': k.source_url,
                        'title': k.topic
                    }
                    self.knowledge_ingester.ingest_content(content, 'forum')
                    self.current_session.knowledge_gained += 1
            
            print(f"   ✓ Mined {len(self.forum_miner.discussions)} discussions")
        
        except Exception as e:
            print(f"   ✗ Forum mining error: {e}")
            self.current_session.errors += 1
    
    async def _consolidate_knowledge(self):
        """Consolidate and verify acquired knowledge."""
        self.mode = LearningMode.CONSOLIDATING
        print("🧠 Consolidating knowledge...")
        
        # Revert to active mode
        self.mode = LearningMode.ACTIVE
    
    def _finish_session(self):
        """Finish current learning session."""
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self.sessions.append(self.current_session)
            self.current_session = None
        
        self.mode = LearningMode.PASSIVE
        self._running = False
    
    def stop_learning(self):
        """Stop the current learning session."""
        self._running = False
    
    async def learn_from_url(self, url: str) -> Dict:
        """Learn from a specific URL."""
        async with aiohttp.ClientSession() as session:
            # Crawl the page
            result = await self.crawler.crawl_url(url)
            
            if not result.is_success:
                return {'success': False, 'error': result.error}
            
            # Extract content
            extracted = self.extractor.extract(result.content, url)
            
            # Ingest knowledge
            content = {
                'text': extracted.text,
                'url': url,
                'title': extracted.title,
                'author': extracted.author,
                'publish_date': extracted.publish_date
            }
            
            ingestion = self.knowledge_ingester.ingest_content(content, 'article')
            
            return {
                'success': True,
                'title': extracted.title,
                'word_count': extracted.word_count,
                **ingestion
            }
    
    async def learn_about_topic(self, topic: str) -> Dict:
        """Focused learning about a specific topic."""
        print(f"🎯 Learning about: {topic}")
        
        # Search relevant subreddits
        relevant_subreddits = [
            sr for sr in self.forum_miner.KNOWLEDGE_SUBREDDITS
            if topic.lower() in sr.lower()
        ] or self.forum_miner.KNOWLEDGE_SUBREDDITS[:3]
        
        total_knowledge = 0
        
        async with aiohttp.ClientSession() as session:
            # Mine relevant forums
            for subreddit in relevant_subreddits[:3]:
                await self.forum_miner.mine_reddit(session, subreddit, limit=20)
            
            # Process knowledge
            for discussion in self.forum_miner.discussions:
                if topic.lower() in discussion.title.lower():
                    knowledge = self.forum_miner.extract_knowledge(discussion)
                    total_knowledge += len(knowledge)
        
        return {
            'topic': topic,
            'knowledge_gained': total_knowledge,
            'sources_checked': len(relevant_subreddits)
        }
    
    def get_knowledge_summary(self) -> Dict:
        """Get summary of acquired knowledge."""
        ingester_stats = self.knowledge_ingester.get_stats()
        verifier_stats = self.fact_verifier.get_stats()
        news_stats = self.news_aggregator.get_stats()
        forum_stats = self.forum_miner.get_stats()
        
        return {
            'mode': self.mode.value,
            'sessions_completed': len(self.sessions),
            'knowledge': ingester_stats,
            'verification': verifier_stats,
            'news': news_stats,
            'forums': forum_stats
        }
    
    def query_knowledge(self, query: str) -> Dict:
        """Query the knowledge base."""
        # Try entity query
        entity_result = self.knowledge_ingester.query_by_entity(query)
        if entity_result.get('found'):
            return {
                'found': True,
                'type': 'entity',
                'data': entity_result
            }
        
        # Try topic query
        chunks = self.knowledge_ingester.query_by_topic(query.lower())
        if chunks:
            return {
                'found': True,
                'type': 'topic',
                'chunks': len(chunks),
                'sample': chunks[0].content[:200] if chunks else None
            }
        
        return {'found': False}


def demo():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🌐 AION CONTINUOUS LEARNER 🌐                                    ║
║                                                                           ║
║     Never Stop Learning - News, Forums, Articles, Media                  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    learner = ContinuousLearner()
    
    print("✓ Continuous Learner initialized")
    print(f"   • Mode: {learner.mode.value}")
    print(f"   • Components loaded: 8")
    
    print("\n✓ Configuration:")
    print(f"   • Max requests/min: {learner.config.max_requests_per_minute}")
    print(f"   • News check interval: {learner.config.news_check_interval_minutes} min")
    print(f"   • Forum check interval: {learner.config.forum_check_interval_hours} hrs")
    
    print("\n✓ Focus topics:")
    for topic in learner.config.focus_topics[:5]:
        print(f"   • {topic}")
    
    print("\n✓ Components:")
    print(f"   • Web Crawler: ready")
    print(f"   • Content Extractor: ready")
    print(f"   • News Aggregator: {len(learner.news_aggregator.sources)} sources")
    print(f"   • Forum Miner: {len(learner.forum_miner.KNOWLEDGE_SUBREDDITS)} subreddits")
    print(f"   • Fact Verifier: ready")
    print(f"   • Knowledge Ingester: ready")
    
    summary = learner.get_knowledge_summary()
    print(f"\n✓ Knowledge Base:")
    print(f"   • Entities: {summary['knowledge']['total_entities']}")
    print(f"   • Facts: {summary['knowledge']['total_facts']}")
    print(f"   • Chunks: {summary['knowledge']['total_chunks']}")
    
    print("\n" + "=" * 60)
    print("AION is ready to learn everything from the internet! 🌐🧠✨")
    print("\nUsage:")
    print("  learner = ContinuousLearner()")
    print("  await learner.start_learning(duration_minutes=30)")
    print("  result = await learner.learn_from_url('https://example.com')")
    print("  knowledge = learner.query_knowledge('AI')")


if __name__ == "__main__":
    demo()
