"""
AION Web Crawler Engine
=======================

Asynchronous web crawler with rate limiting, robots.txt compliance,
and intelligent scheduling for knowledge acquisition.
"""

import asyncio
import aiohttp
import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Callable, Any
from urllib.parse import urlparse, urljoin
from collections import defaultdict
from enum import Enum
import json


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class CrawlPriority(Enum):
    """Priority levels for crawl tasks."""
    CRITICAL = 0    # Breaking news, urgent updates
    HIGH = 1        # Important sources
    NORMAL = 2      # Regular crawling
    LOW = 3         # Background exploration
    DISCOVERY = 4   # New source discovery


@dataclass
class CrawlResult:
    """Result of a crawl operation."""
    url: str
    status_code: int
    content: str
    content_type: str
    headers: Dict[str, str]
    timestamp: float
    response_time: float
    redirected_url: Optional[str] = None
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300
    
    @property
    def is_html(self) -> bool:
        return 'text/html' in self.content_type.lower()
    
    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class CrawlTask:
    """A task to be crawled."""
    url: str
    priority: CrawlPriority = CrawlPriority.NORMAL
    depth: int = 0
    parent_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc


@dataclass
class RobotsRules:
    """Parsed robots.txt rules."""
    allowed_paths: List[str] = field(default_factory=list)
    disallowed_paths: List[str] = field(default_factory=list)
    crawl_delay: float = 1.0
    sitemaps: List[str] = field(default_factory=list)


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Per-domain rate limiting."""
    
    def __init__(self, default_delay: float = 1.0):
        self.default_delay = default_delay
        self.domain_delays: Dict[str, float] = {}
        self.last_requests: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    def set_delay(self, domain: str, delay: float):
        """Set delay for a specific domain."""
        self.domain_delays[domain] = delay
    
    async def wait(self, domain: str):
        """Wait if necessary before making request to domain."""
        async with self._lock:
            delay = self.domain_delays.get(domain, self.default_delay)
            last_request = self.last_requests.get(domain, 0)
            
            elapsed = time.time() - last_request
            if elapsed < delay:
                await asyncio.sleep(delay - elapsed)
            
            self.last_requests[domain] = time.time()


# =============================================================================
# ROBOTS.TXT PARSER
# =============================================================================

class RobotsParser:
    """Parse and check robots.txt."""
    
    def __init__(self, user_agent: str = "AION-Bot"):
        self.user_agent = user_agent
        self.rules_cache: Dict[str, RobotsRules] = {}
    
    async def fetch_robots(self, session: aiohttp.ClientSession, 
                           domain: str) -> RobotsRules:
        """Fetch and parse robots.txt for a domain."""
        if domain in self.rules_cache:
            return self.rules_cache[domain]
        
        robots_url = f"https://{domain}/robots.txt"
        rules = RobotsRules()
        
        try:
            async with session.get(robots_url, timeout=10) as response:
                if response.status == 200:
                    text = await response.text()
                    rules = self._parse_robots(text)
        except Exception:
            # If can't fetch robots.txt, use permissive defaults
            pass
        
        self.rules_cache[domain] = rules
        return rules
    
    def _parse_robots(self, content: str) -> RobotsRules:
        """Parse robots.txt content."""
        rules = RobotsRules()
        current_agent = None
        applies_to_us = False
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key == 'user-agent':
                current_agent = value
                applies_to_us = value == '*' or self.user_agent.lower() in value.lower()
            elif applies_to_us:
                if key == 'disallow':
                    rules.disallowed_paths.append(value)
                elif key == 'allow':
                    rules.allowed_paths.append(value)
                elif key == 'crawl-delay':
                    try:
                        rules.crawl_delay = float(value)
                    except ValueError:
                        pass
                elif key == 'sitemap':
                    rules.sitemaps.append(value)
        
        return rules
    
    def is_allowed(self, rules: RobotsRules, path: str) -> bool:
        """Check if a path is allowed according to rules."""
        # Check allowed first (more specific)
        for allowed in rules.allowed_paths:
            if path.startswith(allowed):
                return True
        
        # Check disallowed
        for disallowed in rules.disallowed_paths:
            if path.startswith(disallowed):
                return False
        
        return True


# =============================================================================
# URL FRONTIER
# =============================================================================

class URLFrontier:
    """
    Manages URLs to be crawled with priority queuing.
    """
    
    def __init__(self, max_per_domain: int = 1000):
        self.queues: Dict[CrawlPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in CrawlPriority
        }
        self.seen_urls: Set[str] = set()
        self.domain_counts: Dict[str, int] = defaultdict(int)
        self.max_per_domain = max_per_domain
        self._lock = asyncio.Lock()
    
    async def add(self, task: CrawlTask) -> bool:
        """Add a URL to the frontier."""
        async with self._lock:
            # Normalize URL
            url = self._normalize_url(task.url)
            
            # Skip if already seen
            if url in self.seen_urls:
                return False
            
            # Check domain limit
            domain = task.domain
            if self.domain_counts[domain] >= self.max_per_domain:
                return False
            
            self.seen_urls.add(url)
            self.domain_counts[domain] += 1
            
            task.url = url
            await self.queues[task.priority].put(task)
            return True
    
    async def get(self) -> Optional[CrawlTask]:
        """Get next URL to crawl (priority order)."""
        for priority in CrawlPriority:
            queue = self.queues[priority]
            if not queue.empty():
                return await queue.get()
        return None
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        # Remove fragments, normalize path
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    @property
    def size(self) -> int:
        """Total URLs in frontier."""
        return sum(q.qsize() for q in self.queues.values())
    
    @property
    def is_empty(self) -> bool:
        return self.size == 0


# =============================================================================
# LINK EXTRACTOR
# =============================================================================

class LinkExtractor:
    """Extract links from HTML content."""
    
    LINK_PATTERN = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
    
    def extract(self, html: str, base_url: str) -> List[str]:
        """Extract all links from HTML."""
        links = []
        
        for match in self.LINK_PATTERN.finditer(html):
            href = match.group(1)
            
            # Skip non-http links
            if href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                continue
            
            # Convert relative to absolute
            absolute_url = urljoin(base_url, href)
            
            # Only keep http(s) links
            if absolute_url.startswith(('http://', 'https://')):
                links.append(absolute_url)
        
        return links
    
    def filter_by_domain(self, links: List[str], 
                         allowed_domains: Optional[List[str]] = None) -> List[str]:
        """Filter links to only allowed domains."""
        if not allowed_domains:
            return links
        
        filtered = []
        for link in links:
            domain = urlparse(link).netloc
            if any(d in domain for d in allowed_domains):
                filtered.append(link)
        
        return filtered


# =============================================================================
# WEB CRAWLER
# =============================================================================

class WebCrawler:
    """
    Asynchronous web crawler with rate limiting and robots.txt compliance.
    """
    
    def __init__(self, 
                 user_agent: str = "AION-Bot/1.0",
                 max_concurrent: int = 10,
                 default_delay: float = 1.0,
                 timeout: int = 30,
                 respect_robots: bool = True):
        
        self.user_agent = user_agent
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.respect_robots = respect_robots
        
        self.rate_limiter = RateLimiter(default_delay)
        self.robots_parser = RobotsParser(user_agent)
        self.frontier = URLFrontier()
        self.link_extractor = LinkExtractor()
        
        self.stats = {
            'pages_crawled': 0,
            'pages_failed': 0,
            'bytes_downloaded': 0,
            'domains_crawled': set()
        }
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._callbacks: List[Callable[[CrawlResult], None]] = []
    
    def on_result(self, callback: Callable[[CrawlResult], None]):
        """Register callback for crawl results."""
        self._callbacks.append(callback)
    
    async def start(self):
        """Start the crawler."""
        headers = {'User-Agent': self.user_agent}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        self._running = True
    
    async def stop(self):
        """Stop the crawler."""
        self._running = False
        if self._session:
            await self._session.close()
    
    async def add_url(self, url: str, priority: CrawlPriority = CrawlPriority.NORMAL):
        """Add a URL to be crawled."""
        task = CrawlTask(url=url, priority=priority)
        await self.frontier.add(task)
    
    async def add_urls(self, urls: List[str], priority: CrawlPriority = CrawlPriority.NORMAL):
        """Add multiple URLs to be crawled."""
        for url in urls:
            await self.add_url(url, priority)
    
    async def crawl_url(self, url: str) -> CrawlResult:
        """Crawl a single URL."""
        domain = urlparse(url).netloc
        path = urlparse(url).path or '/'
        
        # Check robots.txt
        if self.respect_robots and self._session:
            rules = await self.robots_parser.fetch_robots(self._session, domain)
            if not self.robots_parser.is_allowed(rules, path):
                return CrawlResult(
                    url=url,
                    status_code=403,
                    content="",
                    content_type="",
                    headers={},
                    timestamp=time.time(),
                    response_time=0,
                    error="Blocked by robots.txt"
                )
            
            # Update rate limiter with crawl-delay
            if rules.crawl_delay > 0:
                self.rate_limiter.set_delay(domain, rules.crawl_delay)
        
        # Wait for rate limit
        await self.rate_limiter.wait(domain)
        
        # Make request
        start_time = time.time()
        
        try:
            async with self._session.get(url) as response:
                content = await response.text()
                response_time = time.time() - start_time
                
                result = CrawlResult(
                    url=url,
                    status_code=response.status,
                    content=content,
                    content_type=response.headers.get('Content-Type', ''),
                    headers=dict(response.headers),
                    timestamp=time.time(),
                    response_time=response_time,
                    redirected_url=str(response.url) if str(response.url) != url else None
                )
                
                self._update_stats(result)
                return result
                
        except asyncio.TimeoutError:
            return CrawlResult(
                url=url,
                status_code=408,
                content="",
                content_type="",
                headers={},
                timestamp=time.time(),
                response_time=time.time() - start_time,
                error="Timeout"
            )
        except Exception as e:
            return CrawlResult(
                url=url,
                status_code=0,
                content="",
                content_type="",
                headers={},
                timestamp=time.time(),
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    async def run(self, max_pages: Optional[int] = None):
        """Run the crawler until frontier is empty or max_pages reached."""
        if not self._session:
            await self.start()
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []
        pages_crawled = 0
        
        while self._running:
            if max_pages and pages_crawled >= max_pages:
                break
            
            if self.frontier.is_empty:
                if tasks:
                    await asyncio.gather(*tasks)
                break
            
            task = await self.frontier.get()
            if task:
                async def crawl_task(t: CrawlTask):
                    async with semaphore:
                        result = await self.crawl_url(t.url)
                        
                        # Call callbacks
                        for callback in self._callbacks:
                            try:
                                callback(result)
                            except Exception:
                                pass
                        
                        # Extract and add new links if HTML
                        if result.is_success and result.is_html and t.depth < 3:
                            links = self.link_extractor.extract(result.content, result.url)
                            for link in links[:20]:  # Limit links per page
                                await self.frontier.add(CrawlTask(
                                    url=link,
                                    priority=CrawlPriority.LOW,
                                    depth=t.depth + 1,
                                    parent_url=result.url
                                ))
                
                tasks.append(asyncio.create_task(crawl_task(task)))
                pages_crawled += 1
        
        await self.stop()
    
    def _update_stats(self, result: CrawlResult):
        """Update crawler statistics."""
        if result.is_success:
            self.stats['pages_crawled'] += 1
            self.stats['bytes_downloaded'] += len(result.content)
            self.stats['domains_crawled'].add(urlparse(result.url).netloc)
        else:
            self.stats['pages_failed'] += 1
    
    def get_stats(self) -> Dict:
        """Get crawler statistics."""
        return {
            **self.stats,
            'domains_crawled': len(self.stats['domains_crawled']),
            'frontier_size': self.frontier.size
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the web crawler."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🌐 AION WEB CRAWLER 🌐                                           ║
║                                                                           ║
║     Async, Rate-Limited, Robots.txt Compliant                            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    crawler = WebCrawler()
    print(f"✓ WebCrawler initialized")
    print(f"  - Max concurrent: {crawler.max_concurrent}")
    print(f"  - User agent: {crawler.user_agent}")
    print(f"  - Respects robots.txt: {crawler.respect_robots}")
    
    # Test link extractor
    html = '''
    <html>
    <a href="/page1">Page 1</a>
    <a href="https://example.com/page2">Page 2</a>
    <a href="javascript:void(0)">Skip</a>
    </html>
    '''
    
    links = crawler.link_extractor.extract(html, "https://base.com")
    print(f"\n✓ Link Extractor found {len(links)} valid links")
    
    # Test robots parser
    robots_txt = """
    User-agent: *
    Disallow: /admin/
    Disallow: /private/
    Allow: /public/
    Crawl-delay: 2
    Sitemap: https://example.com/sitemap.xml
    """
    
    rules = crawler.robots_parser._parse_robots(robots_txt)
    print(f"\n✓ Robots.txt Parser")
    print(f"  - Crawl delay: {rules.crawl_delay}s")
    print(f"  - Disallowed paths: {len(rules.disallowed_paths)}")
    print(f"  - Sitemaps found: {len(rules.sitemaps)}")
    
    # Test frontier
    print("\n✓ URL Frontier ready")
    print(f"  - Priority queues: {len(CrawlPriority)}")
    
    print("\n" + "=" * 60)
    print("Web Crawler ready for internet knowledge acquisition! 🚀")


if __name__ == "__main__":
    demo()
