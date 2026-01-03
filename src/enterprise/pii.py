"""
AION PII Detection and Masking
==============================

Personally Identifiable Information (PII) detection
and masking for data privacy compliance.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Pattern, Callable, Tuple
from enum import Enum
import hashlib


# =============================================================================
# PII TYPES
# =============================================================================

class PIIType(Enum):
    """Types of PII that can be detected."""
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID = "national_id"
    MEDICAL_ID = "medical_id"
    USERNAME = "username"
    PASSWORD = "password"
    API_KEY = "api_key"
    LOCATION = "location"
    BIOMETRIC = "biometric"
    GENETIC = "genetic"
    CUSTOM = "custom"


class MaskingStrategy(Enum):
    """Strategies for masking PII."""
    REDACT = "redact"          # Replace with [REDACTED]
    MASK = "mask"              # Replace with ***
    PARTIAL = "partial"        # Show partial (e.g., ****1234)
    HASH = "hash"              # Replace with hash
    TOKENIZE = "tokenize"      # Replace with token
    PSEUDONYMIZE = "pseudonymize"  # Replace with fake data


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PIIMatch:
    """A detected PII match."""
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float = 1.0
    context: str = ""
    
    @property
    def length(self) -> int:
        return self.end - self.start
    
    def masked_value(self, strategy: MaskingStrategy = MaskingStrategy.REDACT) -> str:
        """Get masked version of the value."""
        if strategy == MaskingStrategy.REDACT:
            return f"[{self.pii_type.value.upper()}_REDACTED]"
        elif strategy == MaskingStrategy.MASK:
            return "*" * len(self.value)
        elif strategy == MaskingStrategy.PARTIAL:
            if len(self.value) > 4:
                return "*" * (len(self.value) - 4) + self.value[-4:]
            return "*" * len(self.value)
        elif strategy == MaskingStrategy.HASH:
            return hashlib.sha256(self.value.encode()).hexdigest()[:16]
        else:
            return f"[{self.pii_type.value.upper()}]"


@dataclass
class PIIConfig:
    """Configuration for PII detection."""
    # Types to detect
    enabled_types: List[PIIType] = field(default_factory=lambda: list(PIIType))
    
    # Masking settings
    default_strategy: MaskingStrategy = MaskingStrategy.REDACT
    type_strategies: Dict[PIIType, MaskingStrategy] = field(default_factory=dict)
    
    # Detection settings
    min_confidence: float = 0.5
    detect_context: bool = True
    context_window: int = 50
    
    # Custom patterns
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    
    def get_strategy(self, pii_type: PIIType) -> MaskingStrategy:
        """Get masking strategy for a PII type."""
        return self.type_strategies.get(pii_type, self.default_strategy)


@dataclass
class PIIReport:
    """Report of PII detection results."""
    text_length: int = 0
    total_matches: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    risk_score: float = 0.0  # 0-1
    matches: List[PIIMatch] = field(default_factory=list)
    
    @property
    def has_pii(self) -> bool:
        return self.total_matches > 0


# =============================================================================
# PII DETECTOR
# =============================================================================

class PIIDetector:
    """
    Detect PII in text using regex patterns and heuristics.
    """
    
    def __init__(self, config: PIIConfig = None):
        self.config = config or PIIConfig()
        self._patterns = self._build_patterns()
    
    def _build_patterns(self) -> Dict[PIIType, List[Pattern]]:
        """Build regex patterns for each PII type."""
        patterns = {
            PIIType.EMAIL: [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE)
            ],
            PIIType.PHONE: [
                # US formats
                re.compile(r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
                # International
                re.compile(r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'),
            ],
            PIIType.SSN: [
                re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
            ],
            PIIType.CREDIT_CARD: [
                # Major card formats
                re.compile(r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'),
                re.compile(r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})\d{12}\b'),
            ],
            PIIType.IP_ADDRESS: [
                # IPv4
                re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
                # IPv6 (simplified)
                re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
            ],
            PIIType.DATE_OF_BIRTH: [
                re.compile(r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b'),
                re.compile(r'\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b'),
            ],
            PIIType.API_KEY: [
                # Generic API key patterns
                re.compile(r'\b(?:api[_-]?key|apikey|api_secret|access[_-]?token)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', re.IGNORECASE),
                re.compile(r'\b[a-zA-Z0-9]{32,64}\b'),  # Generic long alphanumeric
            ],
            PIIType.PASSWORD: [
                re.compile(r'(?:password|passwd|pwd)\s*[:=]\s*["\']?([^\s"\']+)["\']?', re.IGNORECASE),
            ],
        }
        
        # Add custom patterns
        for name, pattern in self.config.custom_patterns.items():
            if PIIType.CUSTOM not in patterns:
                patterns[PIIType.CUSTOM] = []
            patterns[PIIType.CUSTOM].append(re.compile(pattern))
        
        return patterns
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect all PII in text.
        
        Args:
            text: Text to scan
            
        Returns:
            List of detected PII matches
        """
        matches = []
        
        for pii_type, patterns in self._patterns.items():
            if pii_type not in self.config.enabled_types:
                continue
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    confidence = self._calculate_confidence(match, pii_type)
                    
                    if confidence < self.config.min_confidence:
                        continue
                    
                    context = ""
                    if self.config.detect_context:
                        start = max(0, match.start() - self.config.context_window)
                        end = min(len(text), match.end() + self.config.context_window)
                        context = text[start:end]
                    
                    pii_match = PIIMatch(
                        pii_type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        context=context
                    )
                    matches.append(pii_match)
        
        # Remove duplicates and overlapping matches
        matches = self._deduplicate(matches)
        
        return matches
    
    def detect_all(self, text: str) -> PIIReport:
        """
        Detect PII and generate a detailed report.
        
        Args:
            text: Text to scan
            
        Returns:
            PIIReport with all findings
        """
        matches = self.detect(text)
        
        by_type = {}
        for match in matches:
            key = match.pii_type.value
            by_type[key] = by_type.get(key, 0) + 1
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(matches, len(text))
        
        return PIIReport(
            text_length=len(text),
            total_matches=len(matches),
            by_type=by_type,
            risk_score=risk_score,
            matches=matches
        )
    
    def _calculate_confidence(self, match: re.Match, pii_type: PIIType) -> float:
        """Calculate confidence score for a match."""
        # Base confidence from pattern match
        confidence = 0.8
        
        # Adjust based on type-specific validation
        value = match.group()
        
        if pii_type == PIIType.EMAIL:
            if "." in value and "@" in value:
                confidence = 0.95
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Luhn check
            if self._luhn_check(re.sub(r'[\s.-]', '', value)):
                confidence = 0.95
            else:
                confidence = 0.5
        
        elif pii_type == PIIType.SSN:
            # Check for common invalid patterns
            digits = re.sub(r'\D', '', value)
            if digits in ['000000000', '111111111', '123456789']:
                confidence = 0.3
        
        return confidence
    
    def _luhn_check(self, number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        try:
            digits = [int(d) for d in number]
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            
            total = sum(odd_digits)
            for d in even_digits:
                d *= 2
                total += d if d < 10 else d - 9
            
            return total % 10 == 0
        except (ValueError, IndexError):
            return False
    
    def _deduplicate(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove duplicate and overlapping matches."""
        if not matches:
            return []
        
        # Sort by position and confidence
        sorted_matches = sorted(matches, key=lambda m: (m.start, -m.confidence))
        
        result = []
        last_end = -1
        
        for match in sorted_matches:
            if match.start >= last_end:
                result.append(match)
                last_end = match.end
        
        return result
    
    def _calculate_risk_score(self, matches: List[PIIMatch], text_length: int) -> float:
        """Calculate overall risk score."""
        if not matches:
            return 0.0
        
        # Weight by PII type sensitivity
        weights = {
            PIIType.SSN: 1.0,
            PIIType.CREDIT_CARD: 1.0,
            PIIType.BANK_ACCOUNT: 1.0,
            PIIType.PASSWORD: 1.0,
            PIIType.API_KEY: 0.9,
            PIIType.PASSPORT: 0.9,
            PIIType.MEDICAL_ID: 0.9,
            PIIType.EMAIL: 0.5,
            PIIType.PHONE: 0.5,
            PIIType.IP_ADDRESS: 0.4,
            PIIType.NAME: 0.3,
        }
        
        total_weight = sum(
            weights.get(m.pii_type, 0.5) * m.confidence
            for m in matches
        )
        
        # Normalize
        max_possible = len(matches) * 1.0
        risk = total_weight / max_possible if max_possible > 0 else 0
        
        return min(1.0, risk)


# =============================================================================
# PII MASKER
# =============================================================================

class PIIMasker:
    """
    Mask detected PII in text.
    """
    
    def __init__(self, config: PIIConfig = None):
        self.config = config or PIIConfig()
        self.detector = PIIDetector(config)
        self._tokens: Dict[str, str] = {}  # For tokenization
    
    def mask(self, text: str) -> Tuple[str, List[PIIMatch]]:
        """
        Mask all PII in text.
        
        Args:
            text: Text to mask
            
        Returns:
            Tuple of (masked_text, matches)
        """
        matches = self.detector.detect(text)
        
        if not matches:
            return text, []
        
        # Sort by position (reverse) to replace from end
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)
        
        result = text
        for match in sorted_matches:
            strategy = self.config.get_strategy(match.pii_type)
            masked = match.masked_value(strategy)
            result = result[:match.start] + masked + result[match.end:]
        
        return result, matches
    
    def redact(self, text: str) -> str:
        """Redact all PII (shorthand for mask with REDACT strategy)."""
        config = PIIConfig(default_strategy=MaskingStrategy.REDACT)
        self.config = config
        masked, _ = self.mask(text)
        return masked
    
    def hash_pii(self, text: str) -> str:
        """Hash all PII for pseudonymization."""
        config = PIIConfig(default_strategy=MaskingStrategy.HASH)
        self.config = config
        masked, _ = self.mask(text)
        return masked
    
    def tokenize(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replace PII with tokens (reversible).
        
        Args:
            text: Text to tokenize
            
        Returns:
            Tuple of (tokenized_text, token_map)
        """
        matches = self.detector.detect(text)
        
        if not matches:
            return text, {}
        
        token_map = {}
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)
        
        result = text
        for i, match in enumerate(sorted_matches):
            token = f"<PII_TOKEN_{i}>"
            token_map[token] = match.value
            result = result[:match.start] + token + result[match.end:]
        
        self._tokens.update(token_map)
        return result, token_map
    
    def detokenize(self, text: str, token_map: Dict[str, str] = None) -> str:
        """Reverse tokenization."""
        tokens = token_map or self._tokens
        result = text
        for token, value in tokens.items():
            result = result.replace(token, value)
        return result


# =============================================================================
# DEMO
# =============================================================================

def demo_pii():
    """Demonstrate PII detection and masking."""
    print("🔒 PII Detection Demo")
    print("-" * 40)
    
    detector = PIIDetector()
    masker = PIIMasker()
    
    # Test text with various PII
    text = """
    Contact John Doe at john.doe@example.com or call 555-123-4567.
    His SSN is 123-45-6789 and credit card is 4111-1111-1111-1111.
    IP address: 192.168.1.100
    API_KEY=sk_test_4eC39HqLyjWDarjtT1zdp7dc
    """
    
    # Detect
    report = detector.detect_all(text)
    print(f"Text length: {report.text_length}")
    print(f"PII found: {report.total_matches}")
    print(f"By type: {report.by_type}")
    print(f"Risk score: {report.risk_score:.2f}")
    
    print("\nMatches:")
    for match in report.matches:
        print(f"  - {match.pii_type.value}: {match.value[:20]}... (conf: {match.confidence:.2f})")
    
    # Mask
    masked, _ = masker.mask(text)
    print(f"\nMasked text:\n{masked}")
    
    # Tokenize
    tokenized, tokens = masker.tokenize(text)
    print(f"\nTokens created: {len(tokens)}")
    
    print("-" * 40)
    print("✅ PII demo complete!")


if __name__ == "__main__":
    demo_pii()
