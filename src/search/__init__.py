"""
AION Web Search Module
======================
Free web search using DuckDuckGo and Wikipedia (no API keys required).
"""

from .duckduckgo import DuckDuckGo
from .wikipedia import Wikipedia
from .search import WebSearch

__all__ = [
    'DuckDuckGo',
    'Wikipedia', 
    'WebSearch'
]
