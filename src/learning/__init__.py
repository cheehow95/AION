"""
AION Learning Module
====================

Internet knowledge acquisition and learning system.
"""

from .web_crawler import WebCrawler, CrawlResult
from .content_extractor import ContentExtractor, ExtractedContent
from .news_aggregator import NewsAggregator, NewsArticle
from .forum_miner import ForumMiner, Discussion
from .article_processor import ArticleProcessor
from .media_processor import MediaProcessor
from .fact_verifier import FactVerifier, VerificationResult
from .knowledge_ingester import KnowledgeIngester
from .continuous_learner import ContinuousLearner

__all__ = [
    'WebCrawler',
    'CrawlResult', 
    'ContentExtractor',
    'ExtractedContent',
    'NewsAggregator',
    'NewsArticle',
    'ForumMiner',
    'Discussion',
    'ArticleProcessor',
    'MediaProcessor',
    'FactVerifier',
    'VerificationResult',
    'KnowledgeIngester',
    'ContinuousLearner',
]
