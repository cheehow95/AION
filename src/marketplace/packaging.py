"""
AION Agent Marketplace - Packaging Format
==========================================

Agent packaging format (.aion-pkg):
- Package Manifest: Metadata and dependencies
- Asset Bundling: Code, models, and configurations
- Signature Verification: Cryptographic signing
- Compression: Efficient package distribution

Auto-generated for Phase 5: Scale
"""

import asyncio
import uuid
import json
import hashlib
import gzip
import tarfile
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path
from enum import Enum


class PackageType(Enum):
    """Types of AION packages."""
    AGENT = "agent"
    TOOL = "tool"
    TEMPLATE = "template"
    EXTENSION = "extension"
    MODEL = "model"


@dataclass
class PackageManifest:
    """Manifest for an AION package."""
    name: str = ""
    version: str = "0.1.0"
    description: str = ""
    package_type: PackageType = PackageType.AGENT
    author: str = ""
    license: str = "MIT"
    homepage: str = ""
    repository: str = ""
    
    # Dependencies
    dependencies: Dict[str, str] = field(default_factory=dict)  # name -> version constraint
    python_requires: str = ">=3.9"
    
    # Contents
    entry_point: str = ""
    files: List[str] = field(default_factory=list)
    
    # Metadata
    keywords: List[str] = field(default_factory=list)
    classifiers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'type': self.package_type.value,
            'author': self.author,
            'license': self.license,
            'homepage': self.homepage,
            'repository': self.repository,
            'dependencies': self.dependencies,
            'python_requires': self.python_requires,
            'entry_point': self.entry_point,
            'files': self.files,
            'keywords': self.keywords,
            'classifiers': self.classifiers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PackageManifest':
        return cls(
            name=data.get('name', ''),
            version=data.get('version', '0.1.0'),
            description=data.get('description', ''),
            package_type=PackageType(data.get('type', 'agent')),
            author=data.get('author', ''),
            license=data.get('license', 'MIT'),
            dependencies=data.get('dependencies', {}),
            entry_point=data.get('entry_point', ''),
            files=data.get('files', [])
        )


@dataclass
class PackageSignature:
    """Cryptographic signature for package."""
    algorithm: str = "sha256"
    signature: str = ""
    signer: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def verify(self, content: bytes, public_key: str = None) -> bool:
        """Verify signature (simplified)."""
        computed = hashlib.sha256(content).hexdigest()
        return computed == self.signature


@dataclass
class AionPackage:
    """An AION package (.aion-pkg)."""
    manifest: PackageManifest = field(default_factory=PackageManifest)
    files: Dict[str, bytes] = field(default_factory=dict)  # path -> content
    signature: Optional[PackageSignature] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def id(self) -> str:
        return f"{self.manifest.name}@{self.manifest.version}"
    
    def compute_checksum(self) -> str:
        """Compute package checksum."""
        content = json.dumps(self.manifest.to_dict(), sort_keys=True)
        for path, data in sorted(self.files.items()):
            content += path + hashlib.sha256(data).hexdigest()
        return hashlib.sha256(content.encode()).hexdigest()


class PackageBuilder:
    """Builds AION packages."""
    
    def __init__(self):
        self.manifest: Optional[PackageManifest] = None
        self.files: Dict[str, bytes] = {}
    
    def set_manifest(self, manifest: PackageManifest) -> 'PackageBuilder':
        """Set package manifest."""
        self.manifest = manifest
        return self
    
    def add_file(self, path: str, content: bytes) -> 'PackageBuilder':
        """Add a file to the package."""
        self.files[path] = content
        return self
    
    def add_file_from_path(self, source: Path, dest: str = None) -> 'PackageBuilder':
        """Add a file from filesystem."""
        dest = dest or source.name
        self.files[dest] = source.read_bytes()
        return self
    
    def add_directory(self, source: Path, prefix: str = "") -> 'PackageBuilder':
        """Add a directory recursively."""
        for path in source.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(source)
                dest = f"{prefix}/{rel_path}" if prefix else str(rel_path)
                self.files[dest] = path.read_bytes()
        return self
    
    def build(self) -> AionPackage:
        """Build the package."""
        if not self.manifest:
            raise ValueError("Manifest not set")
        
        # Update manifest with file list
        self.manifest.files = list(self.files.keys())
        
        package = AionPackage(
            manifest=self.manifest,
            files=self.files
        )
        
        return package
    
    def sign(self, package: AionPackage, signer: str = "builder") -> AionPackage:
        """Sign a package."""
        checksum = package.compute_checksum()
        package.signature = PackageSignature(
            signature=checksum,
            signer=signer
        )
        return package
    
    def export(self, package: AionPackage, output_path: Path) -> Path:
        """Export package to .aion-pkg file."""
        # Create tar archive
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
            # Add manifest
            manifest_data = json.dumps(package.manifest.to_dict(), indent=2).encode()
            manifest_info = tarfile.TarInfo(name='manifest.json')
            manifest_info.size = len(manifest_data)
            tar.addfile(manifest_info, io.BytesIO(manifest_data))
            
            # Add files
            for path, content in package.files.items():
                info = tarfile.TarInfo(name=f'files/{path}')
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            
            # Add signature
            if package.signature:
                sig_data = json.dumps({
                    'algorithm': package.signature.algorithm,
                    'signature': package.signature.signature,
                    'signer': package.signature.signer,
                    'timestamp': package.signature.timestamp.isoformat()
                }).encode()
                sig_info = tarfile.TarInfo(name='signature.json')
                sig_info.size = len(sig_data)
                tar.addfile(sig_info, io.BytesIO(sig_data))
        
        # Write to file
        output_path = output_path.with_suffix('.aion-pkg')
        output_path.write_bytes(buffer.getvalue())
        
        return output_path


class PackageVerifier:
    """Verifies AION packages."""
    
    def __init__(self):
        self.trusted_signers: Set[str] = set()
    
    def add_trusted_signer(self, signer: str):
        """Add a trusted signer."""
        self.trusted_signers.add(signer)
    
    def load_package(self, path: Path) -> AionPackage:
        """Load package from file."""
        with tarfile.open(path, 'r:gz') as tar:
            # Read manifest
            manifest_file = tar.extractfile('manifest.json')
            manifest_data = json.loads(manifest_file.read().decode())
            manifest = PackageManifest.from_dict(manifest_data)
            
            # Read files
            files = {}
            for member in tar.getmembers():
                if member.name.startswith('files/'):
                    file_path = member.name[6:]  # Remove 'files/' prefix
                    content = tar.extractfile(member).read()
                    files[file_path] = content
            
            # Read signature
            signature = None
            try:
                sig_file = tar.extractfile('signature.json')
                sig_data = json.loads(sig_file.read().decode())
                signature = PackageSignature(
                    algorithm=sig_data['algorithm'],
                    signature=sig_data['signature'],
                    signer=sig_data['signer']
                )
            except (KeyError, TypeError):
                pass
            
            return AionPackage(manifest=manifest, files=files, signature=signature)
    
    def verify(self, package: AionPackage) -> Dict[str, Any]:
        """Verify a package."""
        results = {
            'valid': True,
            'signature_valid': False,
            'trusted_signer': False,
            'issues': []
        }
        
        # Check manifest
        if not package.manifest.name:
            results['issues'].append("Missing package name")
            results['valid'] = False
        
        if not package.manifest.version:
            results['issues'].append("Missing version")
            results['valid'] = False
        
        # Check signature
        if package.signature:
            expected = package.compute_checksum()
            results['signature_valid'] = package.signature.signature == expected
            results['trusted_signer'] = package.signature.signer in self.trusted_signers
            
            if not results['signature_valid']:
                results['issues'].append("Invalid signature")
                results['valid'] = False
        else:
            results['issues'].append("Package not signed")
        
        return results


async def demo_packaging():
    """Demonstrate packaging system."""
    print("📦 AION Packaging Demo")
    print("=" * 50)
    
    # Create manifest
    manifest = PackageManifest(
        name="reasoning-agent",
        version="1.0.0",
        description="Advanced reasoning agent for complex problem solving",
        package_type=PackageType.AGENT,
        author="AION Team",
        dependencies={"aion-core": ">=2.0.0"},
        entry_point="agent.py"
    )
    
    # Build package
    builder = PackageBuilder()
    package = (builder
        .set_manifest(manifest)
        .add_file("agent.py", b"# Reasoning agent implementation\n")
        .add_file("config.yaml", b"model: gpt-4\ntemperature: 0.7\n")
        .add_file("prompts/system.txt", b"You are a reasoning assistant.\n")
        .build())
    
    # Sign package
    package = builder.sign(package, signer="official")
    
    print(f"\n📋 Package: {package.id}")
    print(f"  Files: {len(package.files)}")
    print(f"  Checksum: {package.compute_checksum()[:16]}...")
    print(f"  Signed by: {package.signature.signer}")
    
    # Verify
    verifier = PackageVerifier()
    verifier.add_trusted_signer("official")
    
    results = verifier.verify(package)
    print(f"\n✅ Verification:")
    print(f"  Valid: {results['valid']}")
    print(f"  Signature valid: {results['signature_valid']}")
    print(f"  Trusted signer: {results['trusted_signer']}")
    
    print("\n✅ Packaging demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_packaging())
