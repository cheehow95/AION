"""
AION Forum Miner
================

Extract knowledge from forums and discussion platforms:
- Reddit (no API key - scraping)
- HackerNews
- StackOverflow/StackExchange
- General forum detection
"""

import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime
from enum import Enum
from collections import defaultdict


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class DiscussionType(Enum):
    """Type of discussion."""
    QUESTION = "question"
    DISCUSSION = "discussion"
    ANNOUNCEMENT = "announcement"
    NEWS = "news"
    HELP_REQUEST = "help_request"
    OPINION = "opinion"
    TUTORIAL = "tutorial"


class Platform(Enum):
    """Discussion platform."""
    REDDIT = "reddit"
    HACKERNEWS = "hackernews"
    STACKOVERFLOW = "stackoverflow"
    GENERIC_FORUM = "generic"


@dataclass
class ForumUser:
    """A forum user/contributor."""
    username: str
    platform: Platform
    karma_score: int = 0
    expertise_areas: List[str] = field(default_factory=list)
    contribution_count: int = 0


@dataclass
class Comment:
    """A comment/reply in a discussion."""
    id: str
    author: str
    content: str
    timestamp: datetime
    score: int
    parent_id: Optional[str]
    is_accepted: bool = False  # For StackOverflow
    depth: int = 0
    
    @property
    def is_quality(self) -> bool:
        return self.score >= 5 or self.is_accepted


@dataclass
class Discussion:
    """A forum discussion/thread."""
    id: str
    title: str
    url: str
    author: str
    content: str
    platform: Platform
    subreddit_or_category: str
    timestamp: datetime
    score: int
    comment_count: int
    comments: List[Comment]
    discussion_type: DiscussionType
    tags: List[str]
    is_answered: bool = False
    
    @property
    def quality_score(self) -> float:
        """Calculate quality score for knowledge extraction."""
        base_score = min(100, self.score) / 100
        comment_bonus = min(50, self.comment_count) / 100
        answer_bonus = 0.2 if self.is_answered else 0
        return base_score + comment_bonus + answer_bonus
    
    @property
    def best_comments(self) -> List[Comment]:
        """Get top quality comments."""
        return sorted(
            [c for c in self.comments if c.is_quality],
            key=lambda c: c.score,
            reverse=True
        )[:10]


@dataclass
class ExtractedKnowledge:
    """Knowledge extracted from discussions."""
    topic: str
    question: Optional[str]
    answer: Optional[str]
    alternative_answers: List[str]
    source_url: str
    platform: Platform
    confidence: float
    upvotes: int
    timestamp: datetime


# =============================================================================
# REDDIT PARSER (No API - HTML Parsing)
# =============================================================================

class RedditParser:
    """Parse Reddit content from HTML (old.reddit.com for easier parsing)."""
    
    def parse_subreddit(self, html: str, subreddit: str) -> List[Dict]:
        """Parse subreddit listing page."""
        posts = []
        
        # Match Reddit post entries
        pattern = r'<div[^>]*class="[^"]*thing[^"]*"[^>]*data-fullname="([^"]+)"[^>]*>'
        pattern += r'.*?<a[^>]*class="[^"]*title[^"]*"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
        pattern += r'.*?data-score="(\d+)"'
        
        for match in re.finditer(pattern, html, re.DOTALL):
            posts.append({
                'id': match.group(1),
                'url': match.group(2),
                'title': match.group(3),
                'score': int(match.group(4)),
                'subreddit': subreddit
            })
        
        # Simpler fallback pattern
        if not posts:
            title_pattern = r'<a[^>]*href="(/r/[^"]+/comments/[^"]+)"[^>]*>([^<]+)</a>'
            for match in re.finditer(title_pattern, html):
                posts.append({
                    'url': 'https://old.reddit.com' + match.group(1),
                    'title': match.group(2),
                    'score': 0,
                    'subreddit': subreddit
                })
        
        return posts
    
    def parse_comments(self, html: str) -> List[Comment]:
        """Parse comments from a Reddit post page."""
        comments = []
        
        # Match comment structure
        comment_pattern = r'<div[^>]*class="[^"]*comment[^"]*"[^>]*data-fullname="([^"]+)"'
        comment_pattern += r'.*?<a[^>]*class="[^"]*author[^"]*"[^>]*>([^<]+)</a>'
        comment_pattern += r'.*?<div[^>]*class="[^"]*md[^"]*"[^>]*>(.+?)</div>'
        
        for match in re.finditer(comment_pattern, html, re.DOTALL):
            comments.append(Comment(
                id=match.group(1),
                author=match.group(2),
                content=self._clean_html(match.group(3)),
                timestamp=datetime.now(),  # Would parse from page
                score=0,
                parent_id=None
            ))
        
        return comments
    
    def _clean_html(self, html: str) -> str:
        """Clean HTML to plain text."""
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# =============================================================================
# HACKERNEWS PARSER
# =============================================================================

class HackerNewsParser:
    """Parse HackerNews content."""
    
    BASE_URL = "https://news.ycombinator.com"
    
    def parse_frontpage(self, html: str) -> List[Dict]:
        """Parse HN front page."""
        items = []
        
        # Match story rows
        pattern = r'<tr class="athing" id="(\d+)">'
        pattern += r'.*?<a href="([^"]+)"[^>]*class="titleline"[^>]*><span>([^<]+)</span></a>'
        
        for match in re.finditer(pattern, html, re.DOTALL):
            items.append({
                'id': match.group(1),
                'url': match.group(2),
                'title': match.group(3),
                'platform': Platform.HACKERNEWS
            })
        
        return items
    
    def parse_item(self, html: str, item_id: str) -> Optional[Discussion]:
        """Parse individual HN item page."""
        # Extract title and URL
        title_match = re.search(r'<span class="titleline"><a href="[^"]*">([^<]+)</a>', html)
        if not title_match:
            return None
        
        title = title_match.group(1)
        
        # Extract score
        score_match = re.search(r'(\d+) points', html)
        score = int(score_match.group(1)) if score_match else 0
        
        # Extract comments
        comments = self._parse_hn_comments(html)
        
        return Discussion(
            id=item_id,
            title=title,
            url=f"{self.BASE_URL}/item?id={item_id}",
            author="",
            content="",
            platform=Platform.HACKERNEWS,
            subreddit_or_category="HackerNews",
            timestamp=datetime.now(),
            score=score,
            comment_count=len(comments),
            comments=comments,
            discussion_type=DiscussionType.NEWS,
            tags=[]
        )
    
    def _parse_hn_comments(self, html: str) -> List[Comment]:
        """Parse HN comments."""
        comments = []
        
        # Match comment structure
        pattern = r'<tr class="comtr" id="(\d+)">'
        pattern += r'.*?<a href="user\?id=([^"]+)"[^>]*>.*?</a>'
        pattern += r'.*?<span class="commtext[^"]*">(.+?)</span>'
        
        for match in re.finditer(pattern, html, re.DOTALL):
            comments.append(Comment(
                id=match.group(1),
                author=match.group(2),
                content=re.sub(r'<[^>]+>', ' ', match.group(3)).strip(),
                timestamp=datetime.now(),
                score=0,
                parent_id=None
            ))
        
        return comments


# =============================================================================
# STACKOVERFLOW PARSER
# =============================================================================

class StackOverflowParser:
    """Parse StackOverflow content."""
    
    def parse_question_list(self, html: str) -> List[Dict]:
        """Parse question listing page."""
        questions = []
        
        pattern = r'<div[^>]*class="[^"]*question-summary[^"]*"[^>]*id="question-summary-(\d+)"'
        pattern += r'.*?<a href="([^"]+)" class="[^"]*question-hyperlink[^"]*">([^<]+)</a>'
        
        for match in re.finditer(pattern, html, re.DOTALL):
            questions.append({
                'id': match.group(1),
                'url': 'https://stackoverflow.com' + match.group(2),
                'title': match.group(3)
            })
        
        return questions
    
    def parse_question_page(self, html: str, question_id: str) -> Optional[Discussion]:
        """Parse a question page with answers."""
        # Extract title
        title_match = re.search(r'<h1[^>]*itemprop="name"[^>]*>.*?<a[^>]*>([^<]+)</a>', html, re.DOTALL)
        if not title_match:
            return None
        
        title = title_match.group(1).strip()
        
        # Extract question body
        body_match = re.search(r'<div[^>]*class="[^"]*s-prose[^"]*"[^>]*itemprop="text"[^>]*>(.+?)</div>', html, re.DOTALL)
        body = re.sub(r'<[^>]+>', ' ', body_match.group(1)).strip() if body_match else ""
        
        # Check if answered
        is_answered = 'accepted-answer' in html
        
        # Extract answers as comments
        answers = self._parse_answers(html)
        
        # Extract tags
        tags = re.findall(r'<a[^>]*class="[^"]*post-tag[^"]*"[^>]*>([^<]+)</a>', html)
        
        return Discussion(
            id=question_id,
            title=title,
            url=f'https://stackoverflow.com/questions/{question_id}',
            author="",
            content=body,
            platform=Platform.STACKOVERFLOW,
            subreddit_or_category="StackOverflow",
            timestamp=datetime.now(),
            score=0,
            comment_count=len(answers),
            comments=answers,
            discussion_type=DiscussionType.QUESTION,
            tags=tags,
            is_answered=is_answered
        )
    
    def _parse_answers(self, html: str) -> List[Comment]:
        """Parse answers as Comment objects."""
        answers = []
        
        # Match answer blocks
        pattern = r'<div[^>]*class="[^"]*answer[^"]*"[^>]*data-answerid="(\d+)"'
        pattern += r'.*?<div[^>]*class="[^"]*s-prose[^"]*"[^>]*>(.+?)</div>'
        
        for i, match in enumerate(re.finditer(pattern, html, re.DOTALL)):
            is_accepted = f'accepted-answer' in html[max(0, match.start()-100):match.start()]
            
            answers.append(Comment(
                id=match.group(1),
                author="",
                content=re.sub(r'<[^>]+>', ' ', match.group(2)).strip()[:2000],
                timestamp=datetime.now(),
                score=0,
                parent_id=None,
                is_accepted=is_accepted
            ))
        
        return answers


# =============================================================================
# FORUM MINER
# =============================================================================

class ForumMiner:
    """
    Main forum mining engine.
    Extracts knowledge from various discussion platforms.
    """
    
    # Valuable subreddits for knowledge
    KNOWLEDGE_SUBREDDITS = [
        'science', 'askscience', 'todayilearned', 'explainlikeimfive',
        'programming', 'learnprogramming', 'MachineLearning', 'artificial',
        'technology', 'futurology', 'space', 'physics', 'chemistry',
        'biology', 'medicine', 'economics', 'philosophy', 'history'
    ]
    
    def __init__(self):
        self.reddit_parser = RedditParser()
        self.hn_parser = HackerNewsParser()
        self.so_parser = StackOverflowParser()
        self.discussions: List[Discussion] = []
        self.extracted_knowledge: List[ExtractedKnowledge] = []
    
    async def mine_reddit(self, session, subreddit: str, 
                         limit: int = 25) -> List[Discussion]:
        """Mine knowledge from a subreddit."""
        discussions = []
        url = f"https://old.reddit.com/r/{subreddit}/top/?t=week"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    posts = self.reddit_parser.parse_subreddit(html, subreddit)
                    
                    for post in posts[:limit]:
                        discussion = Discussion(
                            id=post.get('id', hashlib.md5(post['url'].encode()).hexdigest()[:12]),
                            title=post['title'],
                            url=post['url'],
                            author="",
                            content="",
                            platform=Platform.REDDIT,
                            subreddit_or_category=subreddit,
                            timestamp=datetime.now(),
                            score=post.get('score', 0),
                            comment_count=0,
                            comments=[],
                            discussion_type=self._classify_discussion(post['title']),
                            tags=[subreddit]
                        )
                        discussions.append(discussion)
        except Exception:
            pass
        
        self.discussions.extend(discussions)
        return discussions
    
    async def mine_hackernews(self, session, limit: int = 30) -> List[Discussion]:
        """Mine knowledge from HackerNews front page."""
        discussions = []
        url = "https://news.ycombinator.com/"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    items = self.hn_parser.parse_frontpage(html)
                    
                    for item in items[:limit]:
                        discussion = Discussion(
                            id=item.get('id', ''),
                            title=item['title'],
                            url=item['url'],
                            author="",
                            content="",
                            platform=Platform.HACKERNEWS,
                            subreddit_or_category="HackerNews",
                            timestamp=datetime.now(),
                            score=0,
                            comment_count=0,
                            comments=[],
                            discussion_type=DiscussionType.NEWS,
                            tags=['tech']
                        )
                        discussions.append(discussion)
        except Exception:
            pass
        
        self.discussions.extend(discussions)
        return discussions
    
    async def mine_stackoverflow(self, session, tag: str = "python",
                                 limit: int = 20) -> List[Discussion]:
        """Mine knowledge from StackOverflow."""
        discussions = []
        url = f"https://stackoverflow.com/questions/tagged/{tag}?tab=votes"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    questions = self.so_parser.parse_question_list(html)
                    
                    for q in questions[:limit]:
                        discussion = Discussion(
                            id=q['id'],
                            title=q['title'],
                            url=q['url'],
                            author="",
                            content="",
                            platform=Platform.STACKOVERFLOW,
                            subreddit_or_category=tag,
                            timestamp=datetime.now(),
                            score=0,
                            comment_count=0,
                            comments=[],
                            discussion_type=DiscussionType.QUESTION,
                            tags=[tag]
                        )
                        discussions.append(discussion)
        except Exception:
            pass
        
        self.discussions.extend(discussions)
        return discussions
    
    def extract_knowledge(self, discussion: Discussion) -> List[ExtractedKnowledge]:
        """Extract actionable knowledge from a discussion."""
        knowledge = []
        
        if discussion.discussion_type == DiscussionType.QUESTION:
            # Extract Q&A knowledge
            if discussion.best_comments:
                best = discussion.best_comments[0]
                knowledge.append(ExtractedKnowledge(
                    topic=discussion.title,
                    question=discussion.title,
                    answer=best.content[:1000],
                    alternative_answers=[c.content[:500] for c in discussion.best_comments[1:3]],
                    source_url=discussion.url,
                    platform=discussion.platform,
                    confidence=0.8 if best.is_accepted else 0.6,
                    upvotes=discussion.score,
                    timestamp=discussion.timestamp
                ))
        
        else:
            # Extract general knowledge
            key_points = self._extract_key_points(discussion)
            if key_points:
                knowledge.append(ExtractedKnowledge(
                    topic=discussion.title,
                    question=None,
                    answer=key_points,
                    alternative_answers=[],
                    source_url=discussion.url,
                    platform=discussion.platform,
                    confidence=min(0.9, 0.5 + discussion.quality_score * 0.4),
                    upvotes=discussion.score,
                    timestamp=discussion.timestamp
                ))
        
        self.extracted_knowledge.extend(knowledge)
        return knowledge
    
    def _classify_discussion(self, title: str) -> DiscussionType:
        """Classify discussion type from title."""
        title_lower = title.lower()
        
        if any(w in title_lower for w in ['how to', 'how do', 'why', 'what is', '?']):
            return DiscussionType.QUESTION
        elif any(w in title_lower for w in ['eli5', 'explain', 'help']):
            return DiscussionType.HELP_REQUEST
        elif any(w in title_lower for w in ['tutorial', 'guide', 'learn']):
            return DiscussionType.TUTORIAL
        elif any(w in title_lower for w in ['opinion', 'think', 'feel']):
            return DiscussionType.OPINION
        elif any(w in title_lower for w in ['til', 'today i learned', 'discovered']):
            return DiscussionType.DISCUSSION
        elif any(w in title_lower for w in ['new', 'just released', 'announced']):
            return DiscussionType.NEWS
        
        return DiscussionType.DISCUSSION
    
    def _extract_key_points(self, discussion: Discussion) -> str:
        """Extract key points from discussion."""
        points = []
        
        # From main content
        if discussion.content:
            points.append(discussion.content[:500])
        
        # From top comments
        for comment in discussion.best_comments[:3]:
            points.append(comment.content[:300])
        
        return ' '.join(points)[:1500]
    
    def get_stats(self) -> Dict:
        """Get mining statistics."""
        platform_counts = defaultdict(int)
        for d in self.discussions:
            platform_counts[d.platform.value] += 1
        
        return {
            'total_discussions': len(self.discussions),
            'knowledge_extracted': len(self.extracted_knowledge),
            'platforms': dict(platform_counts),
            'answered_questions': sum(1 for d in self.discussions if d.is_answered)
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the forum miner."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          💬 AION FORUM MINER 💬                                           ║
║                                                                           ║
║     Extract Knowledge from Discussions                                   ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    miner = ForumMiner()
    
    print(f"✓ Forum Miner initialized")
    print(f"   • Reddit parser ready")
    print(f"   • HackerNews parser ready")
    print(f"   • StackOverflow parser ready")
    
    print(f"\n✓ Knowledge subreddits configured:")
    for sr in miner.KNOWLEDGE_SUBREDDITS[:5]:
        print(f"   • r/{sr}")
    print(f"   ... and {len(miner.KNOWLEDGE_SUBREDDITS) - 5} more")
    
    # Test discussion classification
    test_titles = [
        "How do I implement a neural network in Python?",
        "TIL that the Great Barrier Reef can be seen from space",
        "Opinion: AI will transform healthcare by 2030",
        "Tutorial: Complete guide to Docker containers"
    ]
    
    print(f"\n✓ Discussion Classification:")
    for title in test_titles:
        dtype = miner._classify_discussion(title)
        print(f"   • '{title[:40]}...' → {dtype.value}")
    
    print("\n" + "=" * 60)
    print("Forum Miner ready to extract community knowledge! 💬🧠")


if __name__ == "__main__":
    demo()
