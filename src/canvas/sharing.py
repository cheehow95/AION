"""
AION Canvas Collaboration - Sharing
====================================

Collaborative sharing capabilities:
- Permission management
- Invitation system
- Access control
- Version history

Enables secure document sharing.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib


class Permission(Enum):
    """Permission levels."""
    VIEW = "view"
    COMMENT = "comment"
    EDIT = "edit"
    ADMIN = "admin"


@dataclass
class ShareSettings:
    """Sharing settings for a document."""
    document_id: str = ""
    is_public: bool = False
    allow_copy: bool = True
    allow_print: bool = True
    require_login: bool = True
    expires_at: Optional[datetime] = None
    password_hash: Optional[str] = None


@dataclass
class CollaborationInvite:
    """An invitation to collaborate."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    document_id: str = ""
    inviter_id: str = ""
    invitee_email: str = ""
    permission: Permission = Permission.VIEW
    message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))
    accepted: bool = False
    token: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


@dataclass
class DocumentVersion:
    """A version snapshot of a document."""
    version: int = 0
    content: str = ""
    author: str = ""
    title: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    message: str = ""


class ShareManager:
    """Manages document sharing and permissions."""
    
    def __init__(self):
        self.settings: Dict[str, ShareSettings] = {}
        self.permissions: Dict[str, Dict[str, Permission]] = {}  # doc_id -> {user_id: perm}
        self.invites: Dict[str, CollaborationInvite] = {}
        self.versions: Dict[str, List[DocumentVersion]] = {}  # doc_id -> versions
    
    def create_share_link(self, document_id: str, 
                          permission: Permission = Permission.VIEW,
                          password: str = None,
                          expires_days: int = None) -> str:
        """Create a shareable link."""
        settings = self.settings.get(document_id, ShareSettings(document_id=document_id))
        
        if password:
            settings.password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if expires_days:
            settings.expires_at = datetime.now() + timedelta(days=expires_days)
        
        self.settings[document_id] = settings
        
        # Generate link token
        token = hashlib.sha256(f"{document_id}:{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        return f"https://aion.io/share/{document_id}?token={token}"
    
    def set_permission(self, document_id: str, user_id: str, permission: Permission):
        """Set user permission for a document."""
        if document_id not in self.permissions:
            self.permissions[document_id] = {}
        self.permissions[document_id][user_id] = permission
    
    def get_permission(self, document_id: str, user_id: str) -> Optional[Permission]:
        """Get user permission for a document."""
        return self.permissions.get(document_id, {}).get(user_id)
    
    def check_access(self, document_id: str, user_id: str,
                     required: Permission) -> bool:
        """Check if user has required permission."""
        user_perm = self.get_permission(document_id, user_id)
        if not user_perm:
            return False
        
        permission_hierarchy = {
            Permission.VIEW: 1,
            Permission.COMMENT: 2,
            Permission.EDIT: 3,
            Permission.ADMIN: 4
        }
        
        return permission_hierarchy.get(user_perm, 0) >= permission_hierarchy.get(required, 0)
    
    def revoke_access(self, document_id: str, user_id: str) -> bool:
        """Revoke user access to a document."""
        if document_id in self.permissions and user_id in self.permissions[document_id]:
            del self.permissions[document_id][user_id]
            return True
        return False
    
    def send_invite(self, document_id: str, inviter_id: str,
                    invitee_email: str, permission: Permission = Permission.VIEW,
                    message: str = "") -> CollaborationInvite:
        """Send a collaboration invite."""
        invite = CollaborationInvite(
            document_id=document_id,
            inviter_id=inviter_id,
            invitee_email=invitee_email,
            permission=permission,
            message=message
        )
        
        self.invites[invite.token] = invite
        return invite
    
    def accept_invite(self, token: str, user_id: str) -> bool:
        """Accept a collaboration invite."""
        invite = self.invites.get(token)
        if not invite or invite.is_expired or invite.accepted:
            return False
        
        invite.accepted = True
        self.set_permission(invite.document_id, user_id, invite.permission)
        return True
    
    def save_version(self, document_id: str, content: str, author: str,
                     title: str = "", message: str = "") -> DocumentVersion:
        """Save a document version."""
        if document_id not in self.versions:
            self.versions[document_id] = []
        
        versions = self.versions[document_id]
        version_num = len(versions) + 1
        
        version = DocumentVersion(
            version=version_num,
            content=content,
            author=author,
            title=title or f"Version {version_num}",
            message=message
        )
        
        versions.append(version)
        
        # Keep only last 50 versions
        if len(versions) > 50:
            self.versions[document_id] = versions[-50:]
        
        return version
    
    def get_versions(self, document_id: str, limit: int = 10) -> List[DocumentVersion]:
        """Get document version history."""
        versions = self.versions.get(document_id, [])
        return versions[-limit:][::-1]  # Most recent first
    
    def restore_version(self, document_id: str, version: int) -> Optional[str]:
        """Restore a document to a specific version."""
        versions = self.versions.get(document_id, [])
        for v in versions:
            if v.version == version:
                return v.content
        return None
    
    def get_collaborators(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all collaborators for a document."""
        perms = self.permissions.get(document_id, {})
        return [
            {'user_id': uid, 'permission': p.value}
            for uid, p in perms.items()
        ]
    
    def get_stats(self, document_id: str) -> Dict[str, Any]:
        """Get sharing statistics for a document."""
        return {
            'collaborators': len(self.permissions.get(document_id, {})),
            'versions': len(self.versions.get(document_id, [])),
            'pending_invites': sum(1 for i in self.invites.values() 
                                  if i.document_id == document_id and not i.accepted),
            'settings': self.settings.get(document_id) is not None
        }


async def demo_sharing():
    """Demonstrate sharing system."""
    print("🔗 Canvas Sharing Demo")
    print("=" * 50)
    
    manager = ShareManager()
    doc_id = "doc_123"
    
    # Create share link
    link = manager.create_share_link(doc_id, password="secret123", expires_days=7)
    print(f"\n🔗 Share Link: {link}")
    
    # Set permissions
    manager.set_permission(doc_id, "owner", Permission.ADMIN)
    manager.set_permission(doc_id, "editor1", Permission.EDIT)
    manager.set_permission(doc_id, "viewer1", Permission.VIEW)
    
    print(f"\n👥 Collaborators:")
    for collab in manager.get_collaborators(doc_id):
        print(f"   • {collab['user_id']}: {collab['permission']}")
    
    # Check access
    print(f"\n🔒 Access Check:")
    print(f"   owner can EDIT: {manager.check_access(doc_id, 'owner', Permission.EDIT)}")
    print(f"   viewer1 can EDIT: {manager.check_access(doc_id, 'viewer1', Permission.EDIT)}")
    
    # Send invite
    invite = manager.send_invite(doc_id, "owner", "newuser@example.com",
                                 Permission.EDIT, "Join our project!")
    print(f"\n📧 Invite sent: {invite.invitee_email} (token: {invite.token[:8]}...)")
    
    # Accept invite
    success = manager.accept_invite(invite.token, "newuser")
    print(f"   Accepted: {success}")
    print(f"   New permission: {manager.get_permission(doc_id, 'newuser')}")
    
    # Version history
    manager.save_version(doc_id, "Initial content", "owner", message="First draft")
    manager.save_version(doc_id, "Updated content", "editor1", message="Added section 2")
    manager.save_version(doc_id, "Final content", "owner", message="Ready for review")
    
    print(f"\n📚 Version History:")
    for v in manager.get_versions(doc_id):
        print(f"   v{v.version}: {v.message} by {v.author}")
    
    print(f"\n📊 Stats: {manager.get_stats(doc_id)}")
    print("\n✅ Sharing demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_sharing())
