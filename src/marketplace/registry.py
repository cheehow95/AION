"""
AION Agent Marketplace - Package Registry
==========================================

Public/private registries:
- Package Storage: Blob storage backend
- Metadata Index: Searchable package catalog
- Access Control: Private registry authentication
- Mirroring: Registry synchronization

Auto-generated for Phase 5: Scale
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum


class RegistryType(Enum):
    """Types of registries."""
    PUBLIC = "public"
    PRIVATE = "private"
    MIRROR = "mirror"


@dataclass
class PackageEntry:
    """An entry in the package registry."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    downloads: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    size_bytes: int = 0
    keywords: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    storage_url: str = ""
    
    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass
class RegistryUser:
    """A registry user."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    api_key: str = ""
    scopes: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)


class Registry:
    """Package registry."""
    
    def __init__(self, name: str, registry_type: RegistryType = RegistryType.PUBLIC):
        self.name = name
        self.registry_type = registry_type
        self.packages: Dict[str, Dict[str, PackageEntry]] = {}  # name -> version -> entry
        self.users: Dict[str, RegistryUser] = {}
        self.access_tokens: Dict[str, str] = {}  # token -> user_id
    
    def register_user(self, username: str, email: str) -> RegistryUser:
        """Register a new user."""
        user = RegistryUser(
            username=username,
            email=email,
            api_key=str(uuid.uuid4()),
            scopes={"read", "publish"}
        )
        self.users[user.id] = user
        self.access_tokens[user.api_key] = user.id
        return user
    
    def authenticate(self, api_key: str) -> Optional[RegistryUser]:
        """Authenticate with API key."""
        user_id = self.access_tokens.get(api_key)
        return self.users.get(user_id) if user_id else None
    
    def publish(self, entry: PackageEntry, user_id: str = None) -> bool:
        """Publish a package."""
        if self.registry_type == RegistryType.PRIVATE and not user_id:
            return False
        
        if entry.name not in self.packages:
            self.packages[entry.name] = {}
        
        # Check if version already exists
        if entry.version in self.packages[entry.name]:
            return False  # Cannot overwrite
        
        entry.updated_at = datetime.now()
        self.packages[entry.name][entry.version] = entry
        return True
    
    def get_package(self, name: str, version: str = None) -> Optional[PackageEntry]:
        """Get a package entry."""
        if name not in self.packages:
            return None
        
        versions = self.packages[name]
        
        if version:
            return versions.get(version)
        else:
            # Return latest
            if versions:
                latest = max(versions.keys())
                return versions[latest]
        
        return None
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a package."""
        if name not in self.packages:
            return []
        return sorted(self.packages[name].keys(), reverse=True)
    
    def search(self, query: str, limit: int = 20) -> List[PackageEntry]:
        """Search packages by name or keywords."""
        results = []
        query_lower = query.lower()
        
        for name, versions in self.packages.items():
            if not versions:
                continue
            
            latest = versions[max(versions.keys())]
            
            # Match name or keywords
            if (query_lower in name.lower() or
                query_lower in latest.description.lower() or
                any(query_lower in kw.lower() for kw in latest.keywords)):
                results.append(latest)
        
        # Sort by downloads
        results.sort(key=lambda p: p.downloads, reverse=True)
        return results[:limit]
    
    def record_download(self, name: str, version: str):
        """Record a package download."""
        entry = self.get_package(name, version)
        if entry:
            entry.downloads += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_packages = len(self.packages)
        total_versions = sum(len(v) for v in self.packages.values())
        total_downloads = sum(
            sum(e.downloads for e in versions.values())
            for versions in self.packages.values()
        )
        
        return {
            'name': self.name,
            'type': self.registry_type.value,
            'total_packages': total_packages,
            'total_versions': total_versions,
            'total_downloads': total_downloads,
            'registered_users': len(self.users)
        }


class RegistryClient:
    """Client for interacting with a registry."""
    
    def __init__(self, registry_url: str = None, api_key: str = None):
        self.registry_url = registry_url
        self.api_key = api_key
        self._local_registry: Optional[Registry] = None
    
    def connect_local(self, registry: Registry):
        """Connect to a local registry (for testing)."""
        self._local_registry = registry
    
    async def publish(self, entry: PackageEntry) -> bool:
        """Publish a package."""
        if self._local_registry:
            return self._local_registry.publish(entry)
        # In real implementation, would make HTTP request
        return False
    
    async def install(self, name: str, version: str = None) -> Optional[PackageEntry]:
        """Install a package."""
        if self._local_registry:
            entry = self._local_registry.get_package(name, version)
            if entry:
                self._local_registry.record_download(name, entry.version)
            return entry
        return None
    
    async def search(self, query: str) -> List[PackageEntry]:
        """Search for packages."""
        if self._local_registry:
            return self._local_registry.search(query)
        return []
    
    async def list_versions(self, name: str) -> List[str]:
        """List package versions."""
        if self._local_registry:
            return self._local_registry.list_versions(name)
        return []


class RegistryMirror:
    """Mirrors packages between registries."""
    
    def __init__(self, source: Registry, target: Registry):
        self.source = source
        self.target = target
        self.sync_history: List[Dict[str, Any]] = []
    
    async def sync_package(self, name: str, version: str = None) -> bool:
        """Sync a specific package."""
        if version:
            entry = self.source.get_package(name, version)
            if entry:
                return self.target.publish(entry)
        else:
            # Sync all versions
            versions = self.source.list_versions(name)
            success = True
            for v in versions:
                entry = self.source.get_package(name, v)
                if entry:
                    if not self.target.publish(entry):
                        success = False
            return success
        return False
    
    async def sync_all(self) -> Dict[str, Any]:
        """Sync all packages from source to target."""
        synced = 0
        failed = 0
        
        for name in self.source.packages:
            if await self.sync_package(name):
                synced += 1
            else:
                failed += 1
        
        result = {
            'synced': synced,
            'failed': failed,
            'timestamp': datetime.now().isoformat()
        }
        self.sync_history.append(result)
        
        return result


async def demo_registry():
    """Demonstrate registry system."""
    print("📚 Package Registry Demo")
    print("=" * 50)
    
    # Create registries
    public = Registry("aion-registry", RegistryType.PUBLIC)
    private = Registry("company-registry", RegistryType.PRIVATE)
    
    # Register user
    user = private.register_user("developer", "dev@example.com")
    print(f"\n👤 Registered user: {user.username}")
    
    # Publish packages
    packages = [
        PackageEntry(name="reasoning-agent", version="1.0.0",
                    description="Advanced reasoning agent", author="AION",
                    keywords=["reasoning", "agent", "ai"]),
        PackageEntry(name="reasoning-agent", version="1.1.0",
                    description="Advanced reasoning agent with memory", author="AION",
                    keywords=["reasoning", "agent", "ai", "memory"]),
        PackageEntry(name="coding-assistant", version="2.0.0",
                    description="AI coding assistant", author="AION",
                    keywords=["coding", "assistant", "developer"]),
    ]
    
    for pkg in packages:
        public.publish(pkg)
    
    print(f"\n📦 Published {len(packages)} packages")
    
    # Search
    results = public.search("reasoning")
    print(f"\n🔍 Search 'reasoning': {len(results)} results")
    for r in results:
        print(f"  - {r.name}@{r.version}: {r.description}")
    
    # Client operations
    client = RegistryClient()
    client.connect_local(public)
    
    installed = await client.install("reasoning-agent", "1.1.0")
    print(f"\n📥 Installed: {installed.id if installed else 'None'}")
    
    versions = await client.list_versions("reasoning-agent")
    print(f"  Available versions: {versions}")
    
    # Mirror
    mirror = RegistryMirror(public, private)
    sync_result = await mirror.sync_all()
    print(f"\n🔄 Mirror sync: {sync_result}")
    
    print(f"\n📊 Public registry stats: {public.get_statistics()}")
    print("\n✅ Registry demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_registry())
