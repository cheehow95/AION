"""
AION App Integration Directory - App Catalog
=============================================

Third-party app catalog:
- App discovery and search
- Categories and ratings
- Featured and trending apps
- App installation management

Matches GPT-5.2 App Directory feature.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum


class AppCategory(Enum):
    """App categories."""
    PRODUCTIVITY = "productivity"
    DEVELOPMENT = "development"
    DATA = "data"
    COMMUNICATION = "communication"
    DESIGN = "design"
    WRITING = "writing"
    RESEARCH = "research"
    FINANCE = "finance"
    MARKETING = "marketing"
    OTHER = "other"


class AppStatus(Enum):
    """App status."""
    PUBLISHED = "published"
    BETA = "beta"
    DEPRECATED = "deprecated"
    UNPUBLISHED = "unpublished"


@dataclass
class AppRating:
    """App rating and review."""
    user_id: str = ""
    rating: float = 0.0  # 1-5 stars
    review: str = ""
    helpful_votes: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class App:
    """A third-party app."""
    id: str = ""
    name: str = ""
    description: str = ""
    short_description: str = ""
    category: AppCategory = AppCategory.OTHER
    author: str = ""
    website: str = ""
    icon_url: str = ""
    status: AppStatus = AppStatus.PUBLISHED
    version: str = "1.0.0"
    installs: int = 0
    ratings: List[AppRating] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    price: float = 0.0  # 0 = free
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def average_rating(self) -> float:
        if not self.ratings:
            return 0.0
        return sum(r.rating for r in self.ratings) / len(self.ratings)
    
    @property
    def is_free(self) -> bool:
        return self.price == 0


class AppDirectory:
    """App catalog and discovery."""
    
    def __init__(self):
        self.apps: Dict[str, App] = {}
        self.installed: Dict[str, Set[str]] = {}  # user_id -> {app_ids}
        self.featured: List[str] = []
    
    def register_app(self, app: App) -> str:
        """Register an app in the directory."""
        if not app.id:
            app.id = f"app_{len(self.apps) + 1}"
        self.apps[app.id] = app
        return app.id
    
    def get_app(self, app_id: str) -> Optional[App]:
        """Get app by ID."""
        return self.apps.get(app_id)
    
    def search(self, query: str = "", 
               category: AppCategory = None,
               free_only: bool = False,
               min_rating: float = 0,
               limit: int = 20) -> List[App]:
        """Search for apps."""
        results = []
        query_lower = query.lower()
        
        for app in self.apps.values():
            # Filter by status
            if app.status != AppStatus.PUBLISHED:
                continue
            
            # Filter by category
            if category and app.category != category:
                continue
            
            # Filter by price
            if free_only and not app.is_free:
                continue
            
            # Filter by rating
            if app.average_rating < min_rating:
                continue
            
            # Search query
            if query:
                searchable = (app.name + app.description + 
                             ' '.join(app.tags)).lower()
                if query_lower not in searchable:
                    continue
            
            results.append(app)
        
        # Sort by relevance (installs + rating)
        results.sort(
            key=lambda a: a.installs * 0.3 + a.average_rating * 0.7,
            reverse=True
        )
        
        return results[:limit]
    
    def get_featured(self, limit: int = 10) -> List[App]:
        """Get featured apps."""
        featured_apps = []
        for app_id in self.featured[:limit]:
            app = self.apps.get(app_id)
            if app:
                featured_apps.append(app)
        return featured_apps
    
    def get_trending(self, limit: int = 10) -> List[App]:
        """Get trending apps by recent installs."""
        published = [a for a in self.apps.values() 
                    if a.status == AppStatus.PUBLISHED]
        published.sort(key=lambda a: a.installs, reverse=True)
        return published[:limit]
    
    def get_by_category(self, category: AppCategory,
                        limit: int = 20) -> List[App]:
        """Get apps by category."""
        return self.search(category=category, limit=limit)
    
    def install(self, user_id: str, app_id: str) -> bool:
        """Install an app for a user."""
        app = self.apps.get(app_id)
        if not app:
            return False
        
        if user_id not in self.installed:
            self.installed[user_id] = set()
        
        if app_id not in self.installed[user_id]:
            self.installed[user_id].add(app_id)
            app.installs += 1
            return True
        
        return False
    
    def uninstall(self, user_id: str, app_id: str) -> bool:
        """Uninstall an app for a user."""
        if user_id in self.installed and app_id in self.installed[user_id]:
            self.installed[user_id].remove(app_id)
            return True
        return False
    
    def get_installed(self, user_id: str) -> List[App]:
        """Get user's installed apps."""
        app_ids = self.installed.get(user_id, set())
        return [self.apps[aid] for aid in app_ids if aid in self.apps]
    
    def rate_app(self, app_id: str, user_id: str, 
                 rating: float, review: str = "") -> bool:
        """Rate an app."""
        app = self.apps.get(app_id)
        if not app:
            return False
        
        # Remove existing rating
        app.ratings = [r for r in app.ratings if r.user_id != user_id]
        
        # Add new rating
        app.ratings.append(AppRating(
            user_id=user_id,
            rating=max(1, min(5, rating)),
            review=review
        ))
        
        return True
    
    def set_featured(self, app_ids: List[str]):
        """Set featured apps."""
        self.featured = [aid for aid in app_ids if aid in self.apps]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get directory statistics."""
        by_category = {}
        for app in self.apps.values():
            cat = app.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
        
        total_installs = sum(a.installs for a in self.apps.values())
        
        return {
            'total_apps': len(self.apps),
            'published': sum(1 for a in self.apps.values() if a.status == AppStatus.PUBLISHED),
            'by_category': by_category,
            'total_installs': total_installs,
            'featured_count': len(self.featured)
        }


async def demo_directory():
    """Demonstrate app directory."""
    print("📱 App Directory Demo")
    print("=" * 50)
    
    directory = AppDirectory()
    
    # Register apps
    apps = [
        App(
            name="GitHub Integration",
            description="Connect your GitHub repositories for code review and PR management",
            short_description="GitHub code integration",
            category=AppCategory.DEVELOPMENT,
            author="AION Team",
            capabilities=["read_code", "create_pr", "review"],
            tags={"github", "code", "development"}
        ),
        App(
            name="Slack Connector",
            description="Send messages and read channels from Slack workspaces",
            short_description="Slack messaging",
            category=AppCategory.COMMUNICATION,
            author="AION Team",
            capabilities=["send_message", "read_channel"],
            tags={"slack", "messaging", "team"}
        ),
        App(
            name="Data Analyzer Pro",
            description="Advanced data analysis and visualization tool",
            short_description="Data analytics",
            category=AppCategory.DATA,
            author="DataViz Inc",
            capabilities=["analyze_csv", "create_chart"],
            price=9.99,
            tags={"data", "analytics", "charts"}
        ),
        App(
            name="Writing Assistant",
            description="AI-powered writing enhancement and grammar checking",
            short_description="Writing helper",
            category=AppCategory.WRITING,
            author="WriteWell",
            capabilities=["check_grammar", "suggest_edits"],
            tags={"writing", "grammar", "editing"}
        ),
    ]
    
    for app in apps:
        directory.register_app(app)
    
    print(f"\n📊 Stats: {directory.get_stats()}")
    
    # Search
    print("\n🔍 Search 'code':")
    results = directory.search("code")
    for app in results:
        print(f"   • {app.name} ({app.category.value})")
    
    # Install
    directory.install("user1", "app_1")
    directory.install("user1", "app_2")
    directory.install("user2", "app_1")
    
    print(f"\n📦 User1 installed:")
    for app in directory.get_installed("user1"):
        print(f"   • {app.name}")
    
    # Rate
    directory.rate_app("app_1", "user1", 5, "Great app!")
    directory.rate_app("app_1", "user2", 4, "Good but could be better")
    
    app = directory.get_app("app_1")
    print(f"\n⭐ {app.name} rating: {app.average_rating:.1f}/5 ({len(app.ratings)} reviews)")
    
    # Trending
    print("\n🔥 Trending Apps:")
    for app in directory.get_trending(3):
        print(f"   • {app.name} ({app.installs} installs)")
    
    # By category
    print("\n💻 Development Apps:")
    for app in directory.get_by_category(AppCategory.DEVELOPMENT):
        print(f"   • {app.name}")
    
    print("\n✅ Directory demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_directory())
