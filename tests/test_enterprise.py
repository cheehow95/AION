"""Tests for AION Enterprise Features Module"""

import pytest
import sys
sys.path.insert(0, '.')

from src.enterprise.versioning import (
    PromptVersion, PromptRegistry, PromptDiff
)
from src.enterprise.audit import (
    AuditEvent, AuditLogger, AuditLevel, ActionType, AuditQuery
)
from src.enterprise.pii import (
    PIIDetector, PIIMasker, PIIType, PIIMatch, PIIConfig, MaskingStrategy
)
from src.enterprise.quotas import (
    QuotaManager, QuotaPolicy, QuotaType, ResetPeriod, EnforcementAction
)


class TestVersioning:
    """Test prompt versioning."""
    
    def test_prompt_version_creation(self):
        """Test PromptVersion creation."""
        version = PromptVersion(
            name="test_prompt",
            version=1,
            content="Hello {name}",
            author="developer"
        )
        assert version.name == "test_prompt"
        assert version.version == 1
        assert len(version.content_hash) > 0
    
    def test_registry_register(self):
        """Test registering prompts."""
        registry = PromptRegistry()
        
        v1 = registry.register(
            name="test",
            content="Version 1",
            author="dev"
        )
        
        assert v1.version == 1
        assert v1.name == "test"
    
    def test_registry_versioning(self):
        """Test automatic versioning."""
        registry = PromptRegistry()
        
        v1 = registry.register("test", "Content 1")
        v2 = registry.register("test", "Content 2")
        
        assert v1.version == 1
        assert v2.version == 2
        assert v2.parent_version == 1
    
    def test_registry_get_active(self):
        """Test getting active version."""
        registry = PromptRegistry()
        
        registry.register("test", "v1")
        registry.register("test", "v2")
        
        active = registry.get("test")
        assert active.version == 2
    
    def test_registry_rollback(self):
        """Test rollback."""
        registry = PromptRegistry()
        
        registry.register("test", "v1")
        registry.register("test", "v2")
        
        assert registry.rollback("test", 1) == True
        
        active = registry.get("test")
        assert active.version == 1
    
    def test_registry_diff(self):
        """Test version diff."""
        registry = PromptRegistry()
        
        registry.register("test", "Line 1\nLine 2")
        registry.register("test", "Line 1\nLine 3")
        
        diff = registry.diff("test", 1, 2)
        
        assert diff is not None
        assert diff.lines_added > 0 or diff.lines_removed > 0
    
    def test_registry_export_import(self):
        """Test export/import."""
        registry = PromptRegistry()
        registry.register("test", "Content")
        
        exported = registry.export("test")
        
        new_registry = PromptRegistry()
        count = new_registry.import_prompts({"prompts": {"test": exported["versions"]}})
        
        assert count == 1


class TestAudit:
    """Test audit logging."""
    
    def test_audit_event_creation(self):
        """Test AuditEvent creation."""
        event = AuditEvent(
            level=AuditLevel.INFO,
            actor="user@example.com",
            action=ActionType.READ,
            resource_type="document"
        )
        assert event.level == AuditLevel.INFO
        assert len(event.event_hash) > 0
    
    def test_logger_log(self):
        """Test basic logging."""
        logger = AuditLogger()
        
        event = logger.log(
            action=ActionType.CREATE,
            resource_type="user",
            actor="admin"
        )
        
        assert event.action == ActionType.CREATE
        assert len(logger._events) == 1
    
    def test_logger_log_security(self):
        """Test security logging."""
        logger = AuditLogger()
        
        event = logger.log_security(
            action=ActionType.AUTHENTICATE,
            resource_type="session",
            actor="user"
        )
        
        assert event.level == AuditLevel.SECURITY
    
    def test_logger_query(self):
        """Test query filtering."""
        logger = AuditLogger()
        
        logger.log(action=ActionType.READ, resource_type="doc", level=AuditLevel.INFO)
        logger.log_security(action=ActionType.AUTHENTICATE, resource_type="session")
        
        query = AuditQuery(levels=[AuditLevel.SECURITY])
        results = logger.query(query)
        
        assert len(results) == 1
        assert results[0].level == AuditLevel.SECURITY
    
    def test_logger_integrity(self):
        """Test integrity verification."""
        logger = AuditLogger()
        
        logger.log(action=ActionType.READ, resource_type="a")
        logger.log(action=ActionType.CREATE, resource_type="b")
        
        result = logger.verify_integrity()
        
        assert result["valid"] == True
        assert result["events_checked"] == 2
    
    def test_logger_export(self):
        """Test export."""
        logger = AuditLogger()
        logger.log(action=ActionType.READ, resource_type="test")
        
        json_export = logger.export("json")
        csv_export = logger.export("csv")
        
        assert len(json_export) > 0
        assert "timestamp" in csv_export


class TestPII:
    """Test PII detection and masking."""
    
    def test_detector_email(self):
        """Test email detection."""
        detector = PIIDetector()
        
        matches = detector.detect("Contact me at john@example.com")
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL
        assert "john@example.com" in matches[0].value
    
    def test_detector_phone(self):
        """Test phone detection."""
        detector = PIIDetector()
        
        matches = detector.detect("Call 555-123-4567")
        
        assert len(matches) >= 1
        assert any(m.pii_type == PIIType.PHONE for m in matches)
    
    def test_detector_ssn(self):
        """Test SSN detection."""
        detector = PIIDetector()
        
        # Use a realistic SSN (not 123456789 which is flagged as invalid)
        matches = detector.detect("SSN: 456-78-9012")
        
        assert len(matches) >= 1
        assert any(m.pii_type == PIIType.SSN for m in matches)
    
    def test_detector_credit_card(self):
        """Test credit card detection."""
        detector = PIIDetector()
        
        matches = detector.detect("Card: 4111-1111-1111-1111")
        
        assert len(matches) >= 1
        assert any(m.pii_type == PIIType.CREDIT_CARD for m in matches)
    
    def test_detector_report(self):
        """Test full report generation."""
        detector = PIIDetector()
        
        report = detector.detect_all("Email: test@test.com, Phone: 555-555-5555")
        
        assert report.total_matches >= 2
        assert report.has_pii == True
        assert report.risk_score > 0
    
    def test_masker_mask(self):
        """Test PII masking."""
        masker = PIIMasker()
        
        masked, matches = masker.mask("Email: test@test.com")
        
        assert "test@test.com" not in masked
        assert len(matches) > 0
    
    def test_masker_redact(self):
        """Test redaction."""
        masker = PIIMasker()
        
        result = masker.redact("Call 555-123-4567")
        
        assert "555-123-4567" not in result
        assert "REDACTED" in result
    
    def test_masker_tokenize(self):
        """Test tokenization."""
        masker = PIIMasker()
        
        tokenized, token_map = masker.tokenize("Email: a@b.com")
        
        assert "a@b.com" not in tokenized
        assert len(token_map) > 0
        
        # Test detokenize
        restored = masker.detokenize(tokenized, token_map)
        assert "a@b.com" in restored


class TestQuotas:
    """Test quota management."""
    
    def test_create_policy(self):
        """Test policy creation."""
        manager = QuotaManager()
        
        policy = manager.create_policy(
            name="test_tokens",
            quota_type=QuotaType.TOKENS,
            limit=1000
        )
        
        assert policy.name == "test_tokens"
        assert policy.limit == 1000
        assert policy.enabled == True
    
    def test_check_quota(self):
        """Test quota checking."""
        manager = QuotaManager()
        policy = manager.create_policy("test", QuotaType.REQUESTS, limit=10)
        
        result = manager.check_quota(policy.id, "user_1", amount=5)
        
        assert result.allowed == True
        assert result.remaining == 10
    
    def test_consume_quota(self):
        """Test quota consumption."""
        manager = QuotaManager()
        policy = manager.create_policy("test", QuotaType.TOKENS, limit=100)
        
        result = manager.consume(policy.id, "user_1", 30)
        
        assert result.allowed == True
        assert result.current_usage == 30
        assert result.remaining == 70
    
    def test_quota_exceeded(self):
        """Test quota exceed behavior."""
        manager = QuotaManager()
        policy = manager.create_policy(
            "test",
            QuotaType.REQUESTS,
            limit=10,
            enforcement=EnforcementAction.BLOCK
        )
        
        # Consume all
        manager.consume(policy.id, "user_1", 10)
        
        # Try to consume more
        result = manager.check_quota(policy.id, "user_1", 1)
        
        assert result.allowed == False
    
    def test_quota_alerts(self):
        """Test quota alerts."""
        manager = QuotaManager()
        policy = manager.create_policy(
            "test",
            QuotaType.TOKENS,
            limit=100,
            soft_limit=80
        )
        
        # Consume past soft limit
        manager.consume(policy.id, "user_1", 85)
        
        alerts = manager.get_alerts(policy_id=policy.id)
        
        assert len(alerts) > 0
        assert any(a.alert_type == "approaching" for a in alerts)
    
    def test_reset_usage(self):
        """Test usage reset."""
        manager = QuotaManager()
        policy = manager.create_policy("test", QuotaType.REQUESTS, limit=100)
        
        manager.consume(policy.id, "user_1", 50)
        manager.reset_usage(policy.id, "user_1")
        
        usage = manager.get_usage(policy.id, "user_1")
        assert usage.current == 0
    
    def test_get_summary(self):
        """Test summary generation."""
        manager = QuotaManager()
        policy = manager.create_policy("test", QuotaType.TOKENS, limit=100)
        manager.consume(policy.id, "user_1", 25)
        
        summary = manager.get_summary("user_1")
        
        assert summary["total_policies"] == 1
        assert policy.id in summary["usage"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
