"""
AION Industry Templates - Legal
================================

Legal-specific agent templates:
- Contract Analysis: Clause extraction and risk identification
- Case Research: Legal precedent discovery
- Compliance Advisory: Regulatory guidance
- Document Review: Legal document processing

Auto-generated for Phase 5: Scale
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import re


class ClauseType(Enum):
    """Types of contract clauses."""
    TERMINATION = "termination"
    INDEMNIFICATION = "indemnification"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    CONFIDENTIALITY = "confidentiality"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    GOVERNING_LAW = "governing_law"
    DISPUTE_RESOLUTION = "dispute_resolution"
    FORCE_MAJEURE = "force_majeure"
    WARRANTY = "warranty"
    PAYMENT = "payment"


class RiskLevel(Enum):
    """Contract risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ContractClause:
    """A contract clause."""
    type: ClauseType
    text: str
    location: str  # Section reference
    risk_level: RiskLevel = RiskLevel.LOW
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class Contract:
    """A legal contract."""
    id: str = ""
    title: str = ""
    parties: List[str] = field(default_factory=list)
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    full_text: str = ""
    clauses: List[ContractClause] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContractAnalyzer:
    """Contract analysis system."""
    
    CLAUSE_PATTERNS = {
        ClauseType.TERMINATION: [
            r'terminat(e|ion)', r'cancel(l)?ation', r'end of (agreement|contract)'
        ],
        ClauseType.INDEMNIFICATION: [
            r'indemnif(y|ication)', r'hold harmless', r'defend and indemnify'
        ],
        ClauseType.LIMITATION_OF_LIABILITY: [
            r'limitation of liability', r'limit(ed)? liability', r'cap on damages'
        ],
        ClauseType.CONFIDENTIALITY: [
            r'confidential(ity)?', r'non-disclosure', r'proprietary information'
        ],
        ClauseType.GOVERNING_LAW: [
            r'governing law', r'jurisdiction', r'laws of (the )?state'
        ],
        ClauseType.FORCE_MAJEURE: [
            r'force majeure', r'act of god', r'beyond (reasonable )?control'
        ],
    }
    
    RISK_INDICATORS = {
        'unlimited liability': RiskLevel.CRITICAL,
        'sole remedy': RiskLevel.HIGH,
        'waive': RiskLevel.HIGH,
        'perpetual': RiskLevel.MEDIUM,
        'automatic renewal': RiskLevel.MEDIUM,
        'unilateral': RiskLevel.HIGH,
    }
    
    def __init__(self):
        self.analyzed_contracts: List[str] = []
    
    def extract_clauses(self, contract: Contract) -> List[ContractClause]:
        """Extract clauses from contract text."""
        clauses = []
        text_lower = contract.full_text.lower()
        
        for clause_type, patterns in self.CLAUSE_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    # Extract surrounding context
                    start = max(0, match.start() - 200)
                    end = min(len(contract.full_text), match.end() + 500)
                    excerpt = contract.full_text[start:end]
                    
                    clause = ContractClause(
                        type=clause_type,
                        text=excerpt,
                        location=f"Position {match.start()}"
                    )
                    
                    # Assess risk
                    clause.risk_level = self._assess_clause_risk(excerpt)
                    clause.issues = self._identify_issues(excerpt, clause_type)
                    clause.recommendations = self._generate_recommendations(clause)
                    
                    clauses.append(clause)
                    break  # Only first match per type
        
        return clauses
    
    def _assess_clause_risk(self, text: str) -> RiskLevel:
        """Assess risk level of clause text."""
        text_lower = text.lower()
        
        for indicator, risk in self.RISK_INDICATORS.items():
            if indicator in text_lower:
                return risk
        
        return RiskLevel.LOW
    
    def _identify_issues(self, text: str, clause_type: ClauseType) -> List[str]:
        """Identify potential issues in clause."""
        issues = []
        text_lower = text.lower()
        
        if clause_type == ClauseType.LIMITATION_OF_LIABILITY:
            if 'unlimited' in text_lower:
                issues.append("Unlimited liability exposure")
            if 'consequential' in text_lower and 'not' not in text_lower:
                issues.append("May include consequential damages")
        
        if clause_type == ClauseType.TERMINATION:
            if 'without cause' in text_lower and 'notice' not in text_lower:
                issues.append("Termination without notice period")
        
        if clause_type == ClauseType.INDEMNIFICATION:
            if 'gross negligence' not in text_lower and 'willful' not in text_lower:
                issues.append("Broad indemnification scope")
        
        return issues
    
    def _generate_recommendations(self, clause: ContractClause) -> List[str]:
        """Generate recommendations for clause."""
        recommendations = []
        
        if clause.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("Request legal review before signing")
        
        if clause.type == ClauseType.LIMITATION_OF_LIABILITY:
            recommendations.append("Consider negotiating a liability cap")
        
        if clause.type == ClauseType.INDEMNIFICATION:
            recommendations.append("Ensure mutual indemnification")
        
        return recommendations
    
    def analyze_contract(self, contract: Contract) -> Dict[str, Any]:
        """Perform full contract analysis."""
        contract.clauses = self.extract_clauses(contract)
        self.analyzed_contracts.append(contract.id)
        
        risk_summary = {}
        for clause in contract.clauses:
            risk_summary[clause.type.value] = clause.risk_level.value
        
        overall_risk = RiskLevel.LOW
        for clause in contract.clauses:
            if clause.risk_level == RiskLevel.CRITICAL:
                overall_risk = RiskLevel.CRITICAL
                break
            elif clause.risk_level == RiskLevel.HIGH and overall_risk != RiskLevel.CRITICAL:
                overall_risk = RiskLevel.HIGH
        
        return {
            'contract_id': contract.id,
            'parties': contract.parties,
            'clauses_found': len(contract.clauses),
            'risk_summary': risk_summary,
            'overall_risk': overall_risk.value,
            'issues': [issue for clause in contract.clauses for issue in clause.issues],
            'recommendations': list(set(rec for clause in contract.clauses for rec in clause.recommendations))
        }


@dataclass
class CasePrecedent:
    """A legal case precedent."""
    citation: str = ""
    court: str = ""
    year: int = 0
    summary: str = ""
    jurisdiction: str = ""
    relevance_score: float = 0.0
    key_holdings: List[str] = field(default_factory=list)


class CaseResearcher:
    """Legal case research system."""
    
    def __init__(self):
        self.case_database: List[CasePrecedent] = []
    
    def add_case(self, case: CasePrecedent):
        """Add case to database."""
        self.case_database.append(case)
    
    def search(self, query: str, jurisdiction: str = None,
               max_results: int = 10) -> List[CasePrecedent]:
        """Search for relevant cases."""
        results = []
        query_terms = set(query.lower().split())
        
        for case in self.case_database:
            # Simple relevance scoring
            case_text = (case.summary + ' ' + ' '.join(case.key_holdings)).lower()
            case_terms = set(case_text.split())
            
            overlap = len(query_terms & case_terms)
            if overlap == 0:
                continue
            
            score = overlap / len(query_terms)
            
            # Jurisdiction boost
            if jurisdiction and case.jurisdiction.lower() == jurisdiction.lower():
                score *= 1.5
            
            case.relevance_score = score
            results.append(case)
        
        results.sort(key=lambda c: c.relevance_score, reverse=True)
        return results[:max_results]
    
    def get_recent_cases(self, topic: str, since_year: int) -> List[CasePrecedent]:
        """Get recent cases on a topic."""
        cases = self.search(topic)
        return [c for c in cases if c.year >= since_year]


class ComplianceAdvisor:
    """Regulatory compliance advisory system."""
    
    def __init__(self):
        self.regulations: Dict[str, Dict[str, Any]] = {}
        self.deadlines: List[Dict[str, Any]] = []
    
    def add_regulation(self, reg_id: str, name: str, 
                       requirements: List[str], jurisdictions: List[str]):
        """Add a regulation."""
        self.regulations[reg_id] = {
            'name': name,
            'requirements': requirements,
            'jurisdictions': jurisdictions
        }
    
    def check_compliance(self, business_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance based on business profile."""
        applicable_regs = []
        requirements = []
        
        for reg_id, reg in self.regulations.items():
            jurisdictions = business_profile.get('jurisdictions', [])
            if any(j in reg['jurisdictions'] for j in jurisdictions):
                applicable_regs.append(reg['name'])
                requirements.extend(reg['requirements'])
        
        return {
            'applicable_regulations': applicable_regs,
            'requirements': list(set(requirements)),
            'compliance_score': 0.0,  # Would be calculated based on actual compliance
        }


class LegalAgent:
    """Legal-specialized AION agent."""
    
    def __init__(self, agent_id: str = "legal-agent"):
        self.agent_id = agent_id
        self.contract_analyzer = ContractAnalyzer()
        self.case_researcher = CaseResearcher()
        self.compliance_advisor = ComplianceAdvisor()
    
    async def analyze_contract(self, contract: Contract) -> Dict[str, Any]:
        """Analyze a contract."""
        result = self.contract_analyzer.analyze_contract(contract)
        result['disclaimer'] = (
            "This analysis is for informational purposes only and does not "
            "constitute legal advice. Please consult a qualified attorney."
        )
        return result
    
    async def research_cases(self, query: str, 
                            jurisdiction: str = None) -> List[CasePrecedent]:
        """Research relevant cases."""
        return self.case_researcher.search(query, jurisdiction)
    
    async def check_compliance(self, 
                              business_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Check regulatory compliance."""
        return self.compliance_advisor.check_compliance(business_profile)
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            'agent_id': self.agent_id,
            'contracts_analyzed': len(self.contract_analyzer.analyzed_contracts),
            'cases_in_database': len(self.case_researcher.case_database),
            'regulations_tracked': len(self.compliance_advisor.regulations)
        }


async def demo_legal():
    """Demonstrate legal template."""
    print("⚖️ Legal Template Demo")
    print("=" * 50)
    
    agent = LegalAgent()
    
    # Sample contract
    contract = Contract(
        id="C001",
        title="Software License Agreement",
        parties=["ACME Corp", "Client Inc"],
        full_text="""
        This Software License Agreement is entered into...
        
        5. TERMINATION
        Either party may terminate this Agreement without cause upon 30 days
        written notice to the other party.
        
        6. LIMITATION OF LIABILITY
        IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
        SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES. The total liability
        shall not exceed the fees paid in the preceding 12 months.
        
        7. INDEMNIFICATION
        Client shall indemnify and hold harmless Licensor from any claims
        arising from Client's use of the Software.
        
        8. CONFIDENTIALITY
        Each party agrees to keep confidential all proprietary information
        received from the other party.
        
        9. GOVERNING LAW
        This Agreement shall be governed by the laws of the State of Delaware.
        """
    )
    
    # Analyze contract
    print("\n📄 Analyzing contract...")
    analysis = await agent.analyze_contract(contract)
    
    print(f"\n📋 Contract Analysis:")
    print(f"  Contract: {contract.title}")
    print(f"  Parties: {', '.join(contract.parties)}")
    print(f"  Clauses Found: {analysis['clauses_found']}")
    print(f"  Overall Risk: {analysis['overall_risk'].upper()}")
    
    print("\n⚠️ Issues Identified:")
    for issue in analysis['issues'][:5]:
        print(f"  - {issue}")
    
    print("\n💡 Recommendations:")
    for rec in analysis['recommendations'][:3]:
        print(f"  - {rec}")
    
    # Add sample cases
    agent.case_researcher.add_case(CasePrecedent(
        citation="ABC v. XYZ Corp, 123 F.3d 456 (2023)",
        court="Federal Circuit",
        year=2023,
        summary="Software licensing dispute regarding indemnification clauses",
        jurisdiction="Federal",
        key_holdings=["Broad indemnification clauses enforceable", "Limitation of liability upheld"]
    ))
    
    # Research cases
    print("\n🔍 Case Research:")
    cases = await agent.research_cases("software license indemnification")
    for case in cases[:3]:
        print(f"  {case.citation}: {case.summary[:50]}...")
    
    print(f"\n📊 Status: {agent.get_status()}")
    print("\n✅ Legal template demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_legal())
