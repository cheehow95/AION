"""
AION Content Extractor
======================

Extract clean content from web pages:
- HTML to text conversion
- Main content detection
- Metadata extraction
- Media link extraction
"""

import re
import html
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse, urljoin


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractedContent:
    """Extracted content from a web page."""
    url: str
    title: str
    text: str
    summary: str
    author: Optional[str]
    publish_date: Optional[datetime]
    language: str
    word_count: int
    reading_time_minutes: int
    images: List[Dict]
    videos: List[Dict]
    links: List[str]
    metadata: Dict
    source_domain: str
    
    @property
    def is_article(self) -> bool:
        return self.word_count > 100 and len(self.title) > 10


@dataclass
class MediaItem:
    """Extracted media item."""
    url: str
    type: str  # image, video, audio
    alt_text: Optional[str]
    caption: Optional[str]
    width: Optional[int]
    height: Optional[int]


# =============================================================================
# HTML CLEANER
# =============================================================================

class HTMLCleaner:
    """Clean and normalize HTML content."""
    
    # Tags to completely remove (including content)
    REMOVE_TAGS = {'script', 'style', 'noscript', 'iframe', 'svg', 'path'}
    
    # Tags to unwrap (keep content, remove tag)
    UNWRAP_TAGS = {'span', 'font', 'b', 'i', 'u', 'strong', 'em', 'a', 'div'}
    
    # Block-level tags (add newline after)
    BLOCK_TAGS = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'br', 
                  'tr', 'td', 'th', 'article', 'section', 'header', 'footer'}
    
    def __init__(self):
        # Regex patterns
        self.tag_pattern = re.compile(r'<[^>]+>', re.DOTALL)
        self.comment_pattern = re.compile(r'<!--.*?-->', re.DOTALL)
        self.whitespace_pattern = re.compile(r'\s+')
        self.remove_pattern = re.compile(
            r'<(' + '|'.join(self.REMOVE_TAGS) + r')[^>]*>.*?</\1>',
            re.DOTALL | re.IGNORECASE
        )
    
    def clean(self, html_content: str) -> str:
        """Clean HTML to plain text."""
        if not html_content:
            return ""
        
        # Remove comments
        text = self.comment_pattern.sub('', html_content)
        
        # Remove script, style, etc.
        text = self.remove_pattern.sub('', text)
        
        # Add newlines for block elements
        for tag in self.BLOCK_TAGS:
            text = re.sub(rf'</?{tag}[^>]*>', '\n', text, flags=re.IGNORECASE)
        
        # Remove all remaining tags
        text = self.tag_pattern.sub('', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Clean up multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()


# =============================================================================
# CONTENT DETECTOR
# =============================================================================

class ContentDetector:
    """
    Detect main content area in HTML.
    Uses heuristics to find the main article content.
    """
    
    # Content container selectors (in priority order)
    MAIN_SELECTORS = [
        'article', 'main', '[role="main"]',
        '.article', '.post', '.content', '.entry',
        '#article', '#content', '#main', '#post'
    ]
    
    # Elements likely to be noise
    NOISE_SELECTORS = [
        'nav', 'header', 'footer', 'aside', '.sidebar',
        '.comments', '.advertisement', '.ad', '.social',
        '.related', '.recommended', '.navigation'
    ]
    
    def __init__(self):
        self.cleaner = HTMLCleaner()
    
    def extract_main_content(self, html_content: str) -> Tuple[str, str]:
        """
        Extract main content from HTML.
        Returns (main_html, cleaned_text).
        """
        # Simple heuristic: find largest text block
        # In production, would use proper DOM parsing
        
        # Remove noise sections first
        clean_html = html_content
        for noise in self.NOISE_SELECTORS:
            pattern = rf'<{noise}[^>]*>.*?</{noise}>'
            clean_html = re.sub(pattern, '', clean_html, flags=re.DOTALL | re.IGNORECASE)
        
        # Try to find article content
        for selector in self.MAIN_SELECTORS:
            if selector.startswith('.') or selector.startswith('#'):
                # Class or ID selector
                attr = 'class' if selector.startswith('.') else 'id'
                name = selector[1:]
                pattern = rf'<\w+[^>]*{attr}=["\'][^"\']*{name}[^"\']*["\'][^>]*>(.*?)</\w+>'
            elif selector.startswith('['):
                # Attribute selector
                pattern = rf'<\w+[^>]*{selector[1:-1]}[^>]*>(.*?)</\w+>'
            else:
                # Tag selector
                pattern = rf'<{selector}[^>]*>(.*?)</{selector}>'
            
            match = re.search(pattern, clean_html, re.DOTALL | re.IGNORECASE)
            if match:
                main_html = match.group(1)
                return main_html, self.cleaner.clean(main_html)
        
        # Fallback: use body content
        body_match = re.search(r'<body[^>]*>(.*?)</body>', clean_html, re.DOTALL | re.IGNORECASE)
        if body_match:
            return body_match.group(1), self.cleaner.clean(body_match.group(1))
        
        return html_content, self.cleaner.clean(html_content)


# =============================================================================
# METADATA EXTRACTOR
# =============================================================================

class MetadataExtractor:
    """Extract metadata from HTML."""
    
    def extract(self, html_content: str, url: str) -> Dict:
        """Extract all metadata from HTML."""
        metadata = {
            'title': self._extract_title(html_content),
            'description': self._extract_meta('description', html_content),
            'keywords': self._extract_meta('keywords', html_content),
            'author': self._extract_author(html_content),
            'publish_date': self._extract_date(html_content),
            'og_title': self._extract_og('title', html_content),
            'og_description': self._extract_og('description', html_content),
            'og_image': self._extract_og('image', html_content),
            'og_type': self._extract_og('type', html_content),
            'canonical_url': self._extract_canonical(html_content),
            'language': self._extract_language(html_content),
        }
        
        return {k: v for k, v in metadata.items() if v}
    
    def _extract_title(self, html_content: str) -> Optional[str]:
        """Extract page title."""
        # Try <title> tag
        match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1).strip())
        
        # Try h1
        match = re.search(r'<h1[^>]*>([^<]+)</h1>', html_content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1).strip())
        
        return None
    
    def _extract_meta(self, name: str, html_content: str) -> Optional[str]:
        """Extract meta tag content."""
        pattern = rf'<meta[^>]*name=["\']?{name}["\']?[^>]*content=["\']([^"\']+)["\'][^>]*>'
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1).strip())
        
        # Try reverse order (content before name)
        pattern = rf'<meta[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']?{name}["\']?[^>]*>'
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1).strip())
        
        return None
    
    def _extract_og(self, property_name: str, html_content: str) -> Optional[str]:
        """Extract Open Graph meta tag."""
        pattern = rf'<meta[^>]*property=["\']og:{property_name}["\'][^>]*content=["\']([^"\']+)["\'][^>]*>'
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1).strip())
        return None
    
    def _extract_author(self, html_content: str) -> Optional[str]:
        """Extract author information."""
        # Try meta author
        author = self._extract_meta('author', html_content)
        if author:
            return author
        
        # Try schema.org author
        pattern = r'"author"\s*:\s*(?:\{[^}]*"name"\s*:\s*"([^"]+)"|\s*"([^"]+)")'
        match = re.search(pattern, html_content)
        if match:
            return match.group(1) or match.group(2)
        
        return None
    
    def _extract_date(self, html_content: str) -> Optional[str]:
        """Extract publish date."""
        # Try common date meta tags
        for name in ['article:published_time', 'datePublished', 'date', 'pubdate']:
            pattern = rf'<meta[^>]*(?:property|name)=["\']?{name}["\']?[^>]*content=["\']([^"\']+)["\'][^>]*>'
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Try time element
        pattern = r'<time[^>]*datetime=["\']([^"\']+)["\'][^>]*>'
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _extract_canonical(self, html_content: str) -> Optional[str]:
        """Extract canonical URL."""
        pattern = r'<link[^>]*rel=["\']canonical["\'][^>]*href=["\']([^"\']+)["\'][^>]*>'
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_language(self, html_content: str) -> str:
        """Extract page language."""
        pattern = r'<html[^>]*lang=["\']([^"\']+)["\'][^>]*>'
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            return match.group(1).lower()[:2]
        return 'en'


# =============================================================================
# MEDIA EXTRACTOR
# =============================================================================

class MediaExtractor:
    """Extract media (images, videos) from HTML."""
    
    def extract_images(self, html_content: str, base_url: str) -> List[Dict]:
        """Extract all images from HTML."""
        images = []
        
        # Match img tags
        pattern = r'<img[^>]+>'
        for match in re.finditer(pattern, html_content, re.IGNORECASE):
            img_tag = match.group()
            
            # Extract src
            src_match = re.search(r'src=["\']([^"\']+)["\']', img_tag)
            if not src_match:
                continue
            
            src = src_match.group(1)
            if not src.startswith(('http://', 'https://')):
                src = urljoin(base_url, src)
            
            # Skip data URIs and small icons
            if src.startswith('data:') or 'icon' in src.lower():
                continue
            
            # Extract alt text
            alt_match = re.search(r'alt=["\']([^"\']*)["\']', img_tag)
            alt = alt_match.group(1) if alt_match else None
            
            # Extract dimensions
            width_match = re.search(r'width=["\']?(\d+)', img_tag)
            height_match = re.search(r'height=["\']?(\d+)', img_tag)
            
            images.append({
                'url': src,
                'alt': alt,
                'width': int(width_match.group(1)) if width_match else None,
                'height': int(height_match.group(1)) if height_match else None
            })
        
        return images
    
    def extract_videos(self, html_content: str, base_url: str) -> List[Dict]:
        """Extract video links from HTML."""
        videos = []
        
        # Match video tags
        pattern = r'<video[^>]*>.*?</video>'
        for match in re.finditer(pattern, html_content, re.DOTALL | re.IGNORECASE):
            video_tag = match.group()
            
            # Look for source tags
            src_pattern = r'<source[^>]*src=["\']([^"\']+)["\'][^>]*>'
            src_match = re.search(src_pattern, video_tag)
            if src_match:
                src = src_match.group(1)
                if not src.startswith(('http://', 'https://')):
                    src = urljoin(base_url, src)
                videos.append({'url': src, 'type': 'video'})
        
        # Match YouTube embeds
        yt_pattern = r'(?:youtube\.com/embed/|youtu\.be/)([a-zA-Z0-9_-]+)'
        for match in re.finditer(yt_pattern, html_content):
            video_id = match.group(1)
            videos.append({
                'url': f'https://www.youtube.com/watch?v={video_id}',
                'type': 'youtube',
                'video_id': video_id
            })
        
        return videos


# =============================================================================
# CONTENT EXTRACTOR
# =============================================================================

class ContentExtractor:
    """
    Main content extraction engine.
    Extracts clean text, metadata, and media from web pages.
    """
    
    def __init__(self):
        self.cleaner = HTMLCleaner()
        self.content_detector = ContentDetector()
        self.metadata_extractor = MetadataExtractor()
        self.media_extractor = MediaExtractor()
    
    def extract(self, html_content: str, url: str) -> ExtractedContent:
        """Extract all content from HTML page."""
        # Extract metadata first
        metadata = self.metadata_extractor.extract(html_content, url)
        
        # Extract main content
        main_html, text = self.content_detector.extract_main_content(html_content)
        
        # Extract media
        images = self.media_extractor.extract_images(main_html, url)
        videos = self.media_extractor.extract_videos(main_html, url)
        
        # Calculate reading time (avg 200 words per minute)
        word_count = len(text.split())
        reading_time = max(1, word_count // 200)
        
        # Extract links
        link_pattern = r'href=["\']([^"\']+)["\']'
        links = re.findall(link_pattern, main_html)
        
        # Create summary (first paragraph or truncated text)
        summary = self._create_summary(text)
        
        return ExtractedContent(
            url=url,
            title=metadata.get('og_title') or metadata.get('title', 'Untitled'),
            text=text,
            summary=summary,
            author=metadata.get('author'),
            publish_date=self._parse_date(metadata.get('publish_date')),
            language=metadata.get('language', 'en'),
            word_count=word_count,
            reading_time_minutes=reading_time,
            images=images,
            videos=videos,
            links=links[:50],  # Limit links
            metadata=metadata,
            source_domain=urlparse(url).netloc
        )
    
    def _create_summary(self, text: str, max_length: int = 300) -> str:
        """Create summary from text."""
        # Use first paragraph or truncate
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:
                if len(para) <= max_length:
                    return para
                return para[:max_length-3] + '...'
        
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + '...'
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        
        # Common date formats
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d',
            '%B %d, %Y',
            '%d %B %Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str[:19], fmt)
            except ValueError:
                continue
        
        return None


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the content extractor."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          📄 AION CONTENT EXTRACTOR 📄                                     ║
║                                                                           ║
║     Clean Text, Metadata, Media from HTML                                ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Sample HTML
    sample_html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Sample Article - News Site</title>
        <meta name="author" content="John Doe">
        <meta name="description" content="A sample article for testing">
        <meta property="og:title" content="Sample Article">
        <meta property="og:image" content="https://example.com/image.jpg">
    </head>
    <body>
        <nav>Navigation here</nav>
        <article>
            <h1>Sample Article Title</h1>
            <p>This is the first paragraph of the article. It contains 
            important information that should be extracted.</p>
            <p>This is the second paragraph with more content.</p>
            <img src="/images/photo.jpg" alt="A sample photo">
        </article>
        <aside>Sidebar content</aside>
        <footer>Footer content</footer>
    </body>
    </html>
    '''
    
    extractor = ContentExtractor()
    result = extractor.extract(sample_html, 'https://example.com/article')
    
    print(f"✓ Title: {result.title}")
    print(f"✓ Author: {result.author}")
    print(f"✓ Language: {result.language}")
    print(f"✓ Word count: {result.word_count}")
    print(f"✓ Reading time: {result.reading_time_minutes} min")
    print(f"✓ Images found: {len(result.images)}")
    print(f"✓ Summary: {result.summary[:100]}...")
    
    print("\n" + "=" * 60)
    print("Content Extractor ready! 📄")


if __name__ == "__main__":
    demo()
