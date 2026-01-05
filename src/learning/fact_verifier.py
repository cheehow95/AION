"""
AION Fact Verifier
==================

Verify factual claims from multiple sources:
- Cross-reference checking
- Source credibility assessment
- Temporal validation
- Conflict detection
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
from collections import defaultdict


class VerificationStatus(Enum):
    """Status of fact verification."""
    VERIFIED = "verified"        # Multiple credible sources agree
    LIKELY_TRUE = "likely_true"  # Good evidence, not fully verified
    UNCERTAIN = "uncertain"      # Mixed evidence
    DISPUTED = "disputed"        # Sources disagree
    LIKELY_FALSE = "likely_false"  # Evidence against
    UNVERIFIED = "unverified"    # No verification possible


class SourceCredibility(Enum):
    """Source credibility levels."""
    VERY_HIGH = 5    # Scientific journals, official sources
    HIGH = 4         # Major news outlets, established sources
    MEDIUM = 3       # General news, blogs with track record
    LOW = 2          # User-generated content, forums
    VERY_LOW = 1     # Unknown sources, unverified


@dataclass
class Claim:
    """A factual claim to verify."""
    id: str
    content: str
    source_url: str
    source_name: str
    timestamp: datetime
    entity_subject: Optional[str] = None
    entity_predicate: Optional[str] = None
    entity_object: Optional[str] = None


@dataclass
class Evidence:
    """Evidence for or against a claim."""
    source_url: str
    source_name: str
    credibility: SourceCredibility
    supports_claim: bool
    relevance_score: float
    excerpt: str
    timestamp: datetime


@dataclass
class VerificationResult:
    """Result of fact verification."""
    claim: Claim
    status: VerificationStatus
    confidence: float
    supporting_evidence: List[Evidence]
    contradicting_evidence: List[Evidence]
    summary: str
    verified_at: datetime
    
    @property
    def evidence_count(self) -> int:
        return len(self.supporting_evidence) + len(self.contradicting_evidence)


class SourceCredibilityDB:
    """Database of source credibility scores."""
    
    # Pre-rated sources
    KNOWN_SOURCES = {
        # Very High
        'nature.com': SourceCredibility.VERY_HIGH,
        'science.org': SourceCredibility.VERY_HIGH,
        'arxiv.org': SourceCredibility.VERY_HIGH,
        'gov': SourceCredibility.VERY_HIGH,
        'edu': SourceCredibility.VERY_HIGH,
        
        # High
        'bbc.com': SourceCredibility.HIGH,
        'reuters.com': SourceCredibility.HIGH,
        'apnews.com': SourceCredibility.HIGH,
        'npr.org': SourceCredibility.HIGH,
        'nytimes.com': SourceCredibility.HIGH,
        'theguardian.com': SourceCredibility.HIGH,
        'wikipedia.org': SourceCredibility.HIGH,
        
        # Medium
        'techcrunch.com': SourceCredibility.MEDIUM,
        'arstechnica.com': SourceCredibility.HIGH,
        'wired.com': SourceCredibility.MEDIUM,
        'theverge.com': SourceCredibility.MEDIUM,
        
        # Low (user content)
        'reddit.com': SourceCredibility.LOW,
        'twitter.com': SourceCredibility.LOW,
        'facebook.com': SourceCredibility.LOW,
    }
    
    def get_credibility(self, domain: str) -> SourceCredibility:
        """Get credibility score for a domain."""
        # Check exact match
        if domain in self.KNOWN_SOURCES:
            return self.KNOWN_SOURCES[domain]
        
        # Check TLD patterns
        for pattern, cred in self.KNOWN_SOURCES.items():
            if domain.endswith('.' + pattern) or domain.endswith(pattern):
                return cred
        
        # Default to medium
        return SourceCredibility.MEDIUM


class FactVerifier:
    """
    Verify factual claims through cross-referencing.
    """
    
    def __init__(self):
        self.credibility_db = SourceCredibilityDB()
        self.verification_cache: Dict[str, VerificationResult] = {}
        self.claim_history: List[Claim] = []
    
    def verify_claim(self, claim: Claim, 
                     related_content: List[Dict]) -> VerificationResult:
        """Verify a claim against related content."""
        
        # Gather evidence
        supporting = []
        contradicting = []
        
        for content in related_content:
            evidence = self._analyze_evidence(claim, content)
            if evidence:
                if evidence.supports_claim:
                    supporting.append(evidence)
                else:
                    contradicting.append(evidence)
        
        # Determine status
        status, confidence = self._determine_status(supporting, contradicting)
        
        # Generate summary
        summary = self._generate_summary(claim, status, supporting, contradicting)
        
        result = VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            summary=summary,
            verified_at=datetime.now()
        )
        
        # Cache result
        self.verification_cache[claim.id] = result
        self.claim_history.append(claim)
        
        return result
    
    def _analyze_evidence(self, claim: Claim, content: Dict) -> Optional[Evidence]:
        """Analyze content as evidence for/against claim."""
        
        text = content.get('text', '')
        if not text:
            return None
        
        # Check if content is relevant to claim
        relevance = self._calculate_relevance(claim.content, text)
        if relevance < 0.3:
            return None
        
        # Determine if supports or contradicts
        supports = self._check_support(claim.content, text)
        
        # Get source credibility
        source_url = content.get('url', '')
        domain = self._extract_domain(source_url)
        credibility = self.credibility_db.get_credibility(domain)
        
        return Evidence(
            source_url=source_url,
            source_name=content.get('source', domain),
            credibility=credibility,
            supports_claim=supports,
            relevance_score=relevance,
            excerpt=text[:300],
            timestamp=content.get('timestamp', datetime.now())
        )
    
    def _calculate_relevance(self, claim: str, text: str) -> float:
        """Calculate relevance of text to claim."""
        # Word overlap similarity
        claim_words = set(claim.lower().split())
        text_words = set(text.lower().split())
        
        overlap = len(claim_words & text_words)
        if len(claim_words) == 0:
            return 0
        
        return overlap / len(claim_words)
    
    def _check_support(self, claim: str, text: str) -> bool:
        """Check if text supports or contradicts claim."""
        # Simplified check - look for negation patterns
        negation_patterns = [
            'not true', 'false', 'incorrect', 'wrong', 'debunked',
            'misleading', 'inaccurate', 'no evidence', 'disputed'
        ]
        
        text_lower = text.lower()
        for pattern in negation_patterns:
            if pattern in text_lower:
                return False
        
        return True
    
    def _determine_status(self, supporting: List[Evidence], 
                          contradicting: List[Evidence]) -> Tuple[VerificationStatus, float]:
        """Determine verification status from evidence."""
        
        # Calculate weighted scores
        def evidence_score(evidence_list: List[Evidence]) -> float:
            if not evidence_list:
                return 0
            
            total = sum(e.credibility.value * e.relevance_score for e in evidence_list)
            return total
        
        support_score = evidence_score(supporting)
        contradict_score = evidence_score(contradicting)
        total_score = support_score + contradict_score
        
        if total_score == 0:
            return VerificationStatus.UNVERIFIED, 0.0
        
        support_ratio = support_score / total_score
        
        # High credibility sources
        high_cred_support = sum(1 for e in supporting if e.credibility.value >= 4)
        high_cred_contradict = sum(1 for e in contradicting if e.credibility.value >= 4)
        
        if support_ratio >= 0.8 and high_cred_support >= 2:
            return VerificationStatus.VERIFIED, 0.9
        elif support_ratio >= 0.7:
            return VerificationStatus.LIKELY_TRUE, 0.7
        elif support_ratio <= 0.3 and high_cred_contradict >= 2:
            return VerificationStatus.LIKELY_FALSE, 0.3
        elif 0.4 <= support_ratio <= 0.6 and len(contradicting) > 0:
            return VerificationStatus.DISPUTED, 0.5
        else:
            return VerificationStatus.UNCERTAIN, 0.5
    
    def _generate_summary(self, claim: Claim, status: VerificationStatus,
                         supporting: List, contradicting: List) -> str:
        """Generate verification summary."""
        
        templates = {
            VerificationStatus.VERIFIED: 
                "Verified by {support_count} sources including {top_sources}.",
            VerificationStatus.LIKELY_TRUE: 
                "Supported by {support_count} sources, but not fully verified.",
            VerificationStatus.UNCERTAIN: 
                "Evidence is mixed. {support_count} supporting, {contradict_count} contradicting.",
            VerificationStatus.DISPUTED: 
                "Actively disputed. Sources disagree on this claim.",
            VerificationStatus.LIKELY_FALSE: 
                "Multiple credible sources contradict this claim.",
            VerificationStatus.UNVERIFIED: 
                "Unable to verify. No relevant sources found."
        }
        
        top_sources = ', '.join([e.source_name for e in supporting[:3]]) or "N/A"
        
        return templates[status].format(
            support_count=len(supporting),
            contradict_count=len(contradicting),
            top_sources=top_sources
        )
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        import re
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1) if match else url
    
    def check_consistency(self, new_fact: Dict, 
                         existing_facts: List[Dict]) -> List[Dict]:
        """Check if new fact is consistent with existing knowledge."""
        conflicts = []
        
        new_subject = new_fact.get('subject', '')
        new_predicate = new_fact.get('predicate', '')
        new_object = new_fact.get('object', '')
        
        for existing in existing_facts:
            # Check for direct conflicts
            if (existing.get('subject') == new_subject and 
                existing.get('predicate') == new_predicate and
                existing.get('object') != new_object):
                
                conflicts.append({
                    'new_fact': new_fact,
                    'existing_fact': existing,
                    'conflict_type': 'value_mismatch'
                })
        
        return conflicts
    
    def get_stats(self) -> Dict:
        """Get verification statistics."""
        status_counts = defaultdict(int)
        for result in self.verification_cache.values():
            status_counts[result.status.value] += 1
        
        return {
            'total_verified': len(self.verification_cache),
            'claims_processed': len(self.claim_history),
            'status_breakdown': dict(status_counts)
        }


def demo():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          ✅ AION FACT VERIFIER ✅                                         ║
║                                                                           ║
║     Multi-Source Fact Checking & Verification                            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    verifier = FactVerifier()
    
    print("✓ Source credibility database loaded:")
    for source, cred in list(verifier.credibility_db.KNOWN_SOURCES.items())[:5]:
        print(f"   • {source}: {cred.name}")
    
    print("\n✓ Verification statuses:")
    for status in VerificationStatus:
        print(f"   • {status.value}")
    
    print("\n" + "=" * 60)
    print("Fact Verifier ready to separate truth from fiction! ✅🔍")


if __name__ == "__main__":
    demo()
