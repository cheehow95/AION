"""
AION Agent Marketplace - Version Management
============================================

Semantic version management:
- Version Parsing: SemVer compliance
- Dependency Resolution: Constraint satisfaction
- Upgrade Paths: Safe version migration
- Changelog Generation: Automatic release notes

Auto-generated for Phase 5: Scale
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum


@dataclass
class SemanticVersion:
    """Semantic version representation."""
    major: int = 0
    minor: int = 1
    patch: int = 0
    prerelease: str = ""
    build: str = ""
    
    @classmethod
    def parse(cls, version_str: str) -> 'SemanticVersion':
        """Parse a version string."""
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$'
        match = re.match(pattern, version_str)
        if not match:
            raise ValueError(f"Invalid version: {version_str}")
        
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4) or "",
            build=match.group(5) or ""
        )
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    def __lt__(self, other: 'SemanticVersion') -> bool:
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        # Prerelease versions are less than release
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        return self.prerelease < other.prerelease
    
    def __le__(self, other: 'SemanticVersion') -> bool:
        return self == other or self < other
    
    def __gt__(self, other: 'SemanticVersion') -> bool:
        return other < self
    
    def __ge__(self, other: 'SemanticVersion') -> bool:
        return self == other or self > other
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (self.major == other.major and 
                self.minor == other.minor and 
                self.patch == other.patch and
                self.prerelease == other.prerelease)
    
    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))
    
    def bump_major(self) -> 'SemanticVersion':
        return SemanticVersion(self.major + 1, 0, 0)
    
    def bump_minor(self) -> 'SemanticVersion':
        return SemanticVersion(self.major, self.minor + 1, 0)
    
    def bump_patch(self) -> 'SemanticVersion':
        return SemanticVersion(self.major, self.minor, self.patch + 1)


class ConstraintOperator(Enum):
    """Version constraint operators."""
    EQ = "="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    CARET = "^"  # Compatible with
    TILDE = "~"  # Approximately


@dataclass
class VersionConstraint:
    """A version constraint."""
    operator: ConstraintOperator
    version: SemanticVersion
    
    @classmethod
    def parse(cls, constraint_str: str) -> 'VersionConstraint':
        """Parse a constraint string."""
        patterns = [
            (r'^>=(.+)$', ConstraintOperator.GE),
            (r'^<=(.+)$', ConstraintOperator.LE),
            (r'^>(.+)$', ConstraintOperator.GT),
            (r'^<(.+)$', ConstraintOperator.LT),
            (r'^!=(.+)$', ConstraintOperator.NE),
            (r'^\^(.+)$', ConstraintOperator.CARET),
            (r'^~(.+)$', ConstraintOperator.TILDE),
            (r'^=(.+)$', ConstraintOperator.EQ),
            (r'^(.+)$', ConstraintOperator.EQ),
        ]
        
        for pattern, op in patterns:
            match = re.match(pattern, constraint_str.strip())
            if match:
                version = SemanticVersion.parse(match.group(1))
                return cls(operator=op, version=version)
        
        raise ValueError(f"Invalid constraint: {constraint_str}")
    
    def satisfies(self, version: SemanticVersion) -> bool:
        """Check if a version satisfies this constraint."""
        if self.operator == ConstraintOperator.EQ:
            return version == self.version
        elif self.operator == ConstraintOperator.NE:
            return version != self.version
        elif self.operator == ConstraintOperator.GT:
            return version > self.version
        elif self.operator == ConstraintOperator.GE:
            return version >= self.version
        elif self.operator == ConstraintOperator.LT:
            return version < self.version
        elif self.operator == ConstraintOperator.LE:
            return version <= self.version
        elif self.operator == ConstraintOperator.CARET:
            # ^1.2.3 means >=1.2.3 <2.0.0
            if version < self.version:
                return False
            return version.major == self.version.major
        elif self.operator == ConstraintOperator.TILDE:
            # ~1.2.3 means >=1.2.3 <1.3.0
            if version < self.version:
                return False
            return (version.major == self.version.major and 
                    version.minor == self.version.minor)
        return False


@dataclass
class Dependency:
    """A package dependency."""
    name: str
    constraints: List[VersionConstraint] = field(default_factory=list)
    optional: bool = False
    
    def satisfies(self, version: SemanticVersion) -> bool:
        """Check if version satisfies all constraints."""
        return all(c.satisfies(version) for c in self.constraints)


class DependencyResolver:
    """Resolves package dependencies."""
    
    def __init__(self):
        self.available: Dict[str, List[SemanticVersion]] = {}  # package -> versions
        self.dependencies: Dict[str, Dict[str, List[Dependency]]] = {}  # pkg -> version -> deps
    
    def register_package(self, name: str, version: SemanticVersion,
                         dependencies: List[Dependency] = None):
        """Register an available package version."""
        if name not in self.available:
            self.available[name] = []
        self.available[name].append(version)
        self.available[name].sort(reverse=True)
        
        if dependencies:
            if name not in self.dependencies:
                self.dependencies[name] = {}
            self.dependencies[name][str(version)] = dependencies
    
    def resolve(self, requirements: List[Dependency]) -> Optional[Dict[str, SemanticVersion]]:
        """Resolve dependencies to specific versions."""
        resolved: Dict[str, SemanticVersion] = {}
        
        def backtrack(remaining: List[Dependency]) -> bool:
            if not remaining:
                return True
            
            dep = remaining[0]
            rest = remaining[1:]
            
            # Get candidate versions
            candidates = self.available.get(dep.name, [])
            candidates = [v for v in candidates if dep.satisfies(v)]
            
            for version in candidates:
                # Check if already resolved with different version
                if dep.name in resolved:
                    if resolved[dep.name] == version:
                        if backtrack(rest):
                            return True
                    continue
                
                resolved[dep.name] = version
                
                # Get transitive dependencies
                trans_deps = self.dependencies.get(dep.name, {}).get(str(version), [])
                
                if backtrack(rest + trans_deps):
                    return True
                
                del resolved[dep.name]
            
            return dep.optional  # OK to fail if optional
        
        if backtrack(requirements):
            return resolved
        return None
    
    def check_conflicts(self, resolved: Dict[str, SemanticVersion]) -> List[str]:
        """Check for conflicts in resolved dependencies."""
        conflicts = []
        
        for pkg, version in resolved.items():
            deps = self.dependencies.get(pkg, {}).get(str(version), [])
            for dep in deps:
                if dep.name in resolved:
                    if not dep.satisfies(resolved[dep.name]):
                        conflicts.append(
                            f"{pkg}@{version} requires {dep.name} {dep.constraints} "
                            f"but got {resolved[dep.name]}"
                        )
        
        return conflicts


@dataclass
class ChangelogEntry:
    """A changelog entry."""
    version: SemanticVersion
    date: datetime = field(default_factory=datetime.now)
    added: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    deprecated: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    fixed: List[str] = field(default_factory=list)
    security: List[str] = field(default_factory=list)


class ChangelogGenerator:
    """Generates changelogs from entries."""
    
    def __init__(self):
        self.entries: List[ChangelogEntry] = []
    
    def add_entry(self, entry: ChangelogEntry):
        """Add a changelog entry."""
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e.version, reverse=True)
    
    def generate_markdown(self) -> str:
        """Generate changelog in Markdown format."""
        lines = ["# Changelog\n"]
        
        for entry in self.entries:
            lines.append(f"\n## [{entry.version}] - {entry.date.strftime('%Y-%m-%d')}\n")
            
            sections = [
                ("Added", entry.added),
                ("Changed", entry.changed),
                ("Deprecated", entry.deprecated),
                ("Removed", entry.removed),
                ("Fixed", entry.fixed),
                ("Security", entry.security),
            ]
            
            for title, items in sections:
                if items:
                    lines.append(f"\n### {title}\n")
                    for item in items:
                        lines.append(f"- {item}\n")
        
        return "".join(lines)
    
    def infer_version_bump(self, entry: ChangelogEntry, 
                           current: SemanticVersion) -> SemanticVersion:
        """Infer next version based on changes."""
        if entry.removed or entry.security:
            return current.bump_major()
        elif entry.added or entry.changed:
            return current.bump_minor()
        else:
            return current.bump_patch()


async def demo_versioning():
    """Demonstrate versioning system."""
    print("🏷️ Version Management Demo")
    print("=" * 50)
    
    # Parse versions
    v1 = SemanticVersion.parse("1.0.0")
    v2 = SemanticVersion.parse("1.2.3")
    v3 = SemanticVersion.parse("2.0.0-beta.1")
    
    print(f"\n📋 Versions: {v1}, {v2}, {v3}")
    print(f"  {v1} < {v2}: {v1 < v2}")
    print(f"  {v2} < {v3}: {v2 < v3}")
    
    # Constraints
    c1 = VersionConstraint.parse(">=1.0.0")
    c2 = VersionConstraint.parse("^1.0.0")
    
    print(f"\n📐 Constraints:")
    print(f"  '{c1.operator.value}{c1.version}' satisfies {v2}: {c1.satisfies(v2)}")
    print(f"  '{c2.operator.value}{c2.version}' satisfies {v3}: {c2.satisfies(v3)}")
    
    # Dependency resolution
    resolver = DependencyResolver()
    resolver.register_package("core", SemanticVersion.parse("1.0.0"))
    resolver.register_package("core", SemanticVersion.parse("2.0.0"))
    resolver.register_package("utils", SemanticVersion.parse("1.0.0"),
                             [Dependency("core", [VersionConstraint.parse(">=1.0.0")])])
    
    requirements = [
        Dependency("utils", [VersionConstraint.parse(">=1.0.0")]),
        Dependency("core", [VersionConstraint.parse("^2.0.0")])
    ]
    
    resolved = resolver.resolve(requirements)
    print(f"\n🔧 Resolved: {resolved}")
    
    # Changelog
    changelog = ChangelogGenerator()
    changelog.add_entry(ChangelogEntry(
        version=SemanticVersion.parse("1.1.0"),
        added=["New reasoning engine", "Memory optimization"],
        fixed=["Token counting bug"]
    ))
    
    print(f"\n📝 Changelog preview:")
    print(changelog.generate_markdown()[:200] + "...")
    
    print("\n✅ Versioning demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_versioning())
