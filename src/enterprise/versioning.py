"""
AION Prompt Versioning
======================

Version control for prompts with rollback capability,
diff comparison, and history tracking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import hashlib
import json


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PromptVersion:
    """A versioned prompt."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: int = 1
    content: str = ""
    
    # Metadata
    author: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    
    # Changes
    changelog: str = ""
    parent_version: Optional[int] = None
    
    # Hash for integrity
    content_hash: str = ""
    
    # Tags and labels
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    is_draft: bool = False
    
    # Performance tracking
    usage_count: int = 0
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute content hash."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "content": self.content,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "changelog": self.changelog,
            "parent_version": self.parent_version,
            "content_hash": self.content_hash,
            "tags": self.tags,
            "is_active": self.is_active,
            "is_draft": self.is_draft,
        }


@dataclass
class PromptDiff:
    """Difference between two prompt versions."""
    name: str
    from_version: int
    to_version: int
    
    # Diff content
    additions: List[str] = field(default_factory=list)
    deletions: List[str] = field(default_factory=list)
    
    # Stats
    lines_added: int = 0
    lines_removed: int = 0
    
    @property
    def has_changes(self) -> bool:
        return self.lines_added > 0 or self.lines_removed > 0


# =============================================================================
# PROMPT REGISTRY
# =============================================================================

class PromptRegistry:
    """
    Registry for managing versioned prompts.
    
    Provides version control, rollback, and comparison
    capabilities for prompt management.
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path
        self._prompts: Dict[str, Dict[int, PromptVersion]] = {}  # name -> version -> prompt
        self._active_versions: Dict[str, int] = {}  # name -> active version
    
    def register(
        self,
        name: str,
        content: str,
        author: str = "system",
        changelog: str = "",
        tags: List[str] = None,
        is_draft: bool = False
    ) -> PromptVersion:
        """
        Register a new prompt or create a new version.
        
        Args:
            name: Prompt name (identifier)
            content: Prompt content
            author: Author of this version
            changelog: Description of changes
            tags: Tags for categorization
            is_draft: Whether this is a draft
            
        Returns:
            Created PromptVersion
        """
        if name not in self._prompts:
            self._prompts[name] = {}
            version = 1
            parent = None
        else:
            version = max(self._prompts[name].keys()) + 1
            parent = version - 1
        
        prompt = PromptVersion(
            name=name,
            version=version,
            content=content,
            author=author,
            changelog=changelog,
            parent_version=parent,
            tags=tags or [],
            is_draft=is_draft
        )
        
        self._prompts[name][version] = prompt
        
        # Set as active if not a draft
        if not is_draft:
            self._active_versions[name] = version
        
        return prompt
    
    def get(self, name: str) -> Optional[PromptVersion]:
        """Get the active version of a prompt."""
        if name not in self._active_versions:
            return None
        
        version = self._active_versions[name]
        return self._prompts[name].get(version)
    
    def get_version(self, name: str, version: int) -> Optional[PromptVersion]:
        """Get a specific version of a prompt."""
        if name not in self._prompts:
            return None
        return self._prompts[name].get(version)
    
    def list_prompts(self) -> List[str]:
        """List all prompt names."""
        return list(self._prompts.keys())
    
    def list_versions(self, name: str) -> List[PromptVersion]:
        """List all versions of a prompt."""
        if name not in self._prompts:
            return []
        
        versions = sorted(self._prompts[name].values(), key=lambda p: p.version)
        return versions
    
    def rollback(self, name: str, version: int) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            name: Prompt name
            version: Version to rollback to
            
        Returns:
            True if rollback successful
        """
        if name not in self._prompts:
            return False
        
        if version not in self._prompts[name]:
            return False
        
        self._active_versions[name] = version
        return True
    
    def diff(self, name: str, from_version: int, to_version: int) -> Optional[PromptDiff]:
        """
        Compare two versions of a prompt.
        
        Args:
            name: Prompt name
            from_version: Source version
            to_version: Target version
            
        Returns:
            PromptDiff with changes
        """
        from_prompt = self.get_version(name, from_version)
        to_prompt = self.get_version(name, to_version)
        
        if not from_prompt or not to_prompt:
            return None
        
        from_lines = from_prompt.content.splitlines()
        to_lines = to_prompt.content.splitlines()
        
        # Simple line-based diff
        from_set = set(from_lines)
        to_set = set(to_lines)
        
        additions = [line for line in to_lines if line not in from_set]
        deletions = [line for line in from_lines if line not in to_set]
        
        return PromptDiff(
            name=name,
            from_version=from_version,
            to_version=to_version,
            additions=additions,
            deletions=deletions,
            lines_added=len(additions),
            lines_removed=len(deletions)
        )
    
    def delete(self, name: str, version: int = None) -> bool:
        """
        Delete a prompt or specific version.
        
        Args:
            name: Prompt name
            version: Specific version to delete (None = all)
            
        Returns:
            True if deleted
        """
        if name not in self._prompts:
            return False
        
        if version is None:
            # Delete all versions
            del self._prompts[name]
            if name in self._active_versions:
                del self._active_versions[name]
        else:
            # Delete specific version
            if version in self._prompts[name]:
                del self._prompts[name][version]
                
                # Update active if needed
                if self._active_versions.get(name) == version:
                    remaining = sorted(self._prompts[name].keys())
                    if remaining:
                        self._active_versions[name] = remaining[-1]
                    else:
                        del self._active_versions[name]
        
        return True
    
    def export(self, name: str = None) -> Dict[str, Any]:
        """
        Export prompts to dictionary.
        
        Args:
            name: Specific prompt to export (None = all)
            
        Returns:
            Export dictionary
        """
        if name:
            if name not in self._prompts:
                return {}
            return {
                "name": name,
                "versions": [p.to_dict() for p in self._prompts[name].values()],
                "active_version": self._active_versions.get(name)
            }
        
        return {
            "prompts": {
                n: [p.to_dict() for p in versions.values()]
                for n, versions in self._prompts.items()
            },
            "active_versions": self._active_versions.copy()
        }
    
    def import_prompts(self, data: Dict[str, Any]) -> int:
        """
        Import prompts from dictionary.
        
        Args:
            data: Export dictionary
            
        Returns:
            Number of prompts imported
        """
        count = 0
        
        if "prompts" in data:
            for name, versions in data["prompts"].items():
                for v_data in versions:
                    self._prompts.setdefault(name, {})[v_data["version"]] = PromptVersion(
                        id=v_data.get("id", str(uuid.uuid4())),
                        name=v_data["name"],
                        version=v_data["version"],
                        content=v_data["content"],
                        author=v_data.get("author", "system"),
                        changelog=v_data.get("changelog", ""),
                        tags=v_data.get("tags", [])
                    )
                    count += 1
            
            self._active_versions.update(data.get("active_versions", {}))
        
        return count
    
    def record_usage(
        self, 
        name: str, 
        latency_ms: float, 
        success: bool = True
    ) -> None:
        """Record usage metrics for a prompt."""
        prompt = self.get(name)
        if prompt:
            prompt.usage_count += 1
            
            # Update rolling average latency
            n = prompt.usage_count
            prompt.avg_latency_ms = (
                (prompt.avg_latency_ms * (n - 1) + latency_ms) / n
            )
            
            # Update success rate
            prompt.success_rate = (
                (prompt.success_rate * (n - 1) + (1.0 if success else 0.0)) / n
            )


# =============================================================================
# DEMO
# =============================================================================

def demo_versioning():
    """Demonstrate prompt versioning."""
    print("📝 Prompt Versioning Demo")
    print("-" * 40)
    
    registry = PromptRegistry()
    
    # Create initial prompt
    v1 = registry.register(
        name="summarizer",
        content="Summarize the following text:\n{text}",
        author="developer",
        changelog="Initial version"
    )
    print(f"Created v{v1.version}: {v1.content_hash}")
    
    # Create new version
    v2 = registry.register(
        name="summarizer",
        content="Summarize the following text concisely in 2-3 sentences:\n{text}",
        author="developer",
        changelog="Added length constraint"
    )
    print(f"Created v{v2.version}: {v2.content_hash}")
    
    # Get active version
    active = registry.get("summarizer")
    print(f"Active version: v{active.version}")
    
    # Compare versions
    diff = registry.diff("summarizer", 1, 2)
    print(f"Diff v1->v2: +{diff.lines_added}/-{diff.lines_removed}")
    
    # Rollback
    registry.rollback("summarizer", 1)
    active = registry.get("summarizer")
    print(f"After rollback: v{active.version}")
    
    # List versions
    versions = registry.list_versions("summarizer")
    print(f"All versions: {[v.version for v in versions]}")
    
    # Export
    exported = registry.export("summarizer")
    print(f"Exported {len(exported['versions'])} versions")
    
    print("-" * 40)
    print("✅ Versioning demo complete!")


if __name__ == "__main__":
    demo_versioning()
