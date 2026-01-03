"""
AION Industry Templates - Package Initialization
=================================================

Pre-configured agents for specific industries.
"""

from .healthcare import (
    HealthcareAgent,
    HIPAACompliance,
    ClinicalDecisionSupport,
    MedicalKnowledgeBase
)

from .finance import (
    FinanceAgent,
    RiskAnalyzer,
    ComplianceChecker,
    MarketDataProcessor
)

from .legal import (
    LegalAgent,
    ContractAnalyzer,
    CaseResearcher,
    ComplianceAdvisor
)

from .engineering import (
    EngineeringAgent,
    CodeReviewer,
    ArchitectureAnalyzer,
    TechnicalDocumentor
)

from .science import (
    ScienceAgent,
    ResearchAssistant,
    DataAnalyzer,
    LiteratureReviewer
)

__all__ = [
    # Healthcare
    'HealthcareAgent',
    'HIPAACompliance',
    'ClinicalDecisionSupport',
    'MedicalKnowledgeBase',
    # Finance
    'FinanceAgent',
    'RiskAnalyzer',
    'ComplianceChecker',
    'MarketDataProcessor',
    # Legal
    'LegalAgent',
    'ContractAnalyzer',
    'CaseResearcher',
    'ComplianceAdvisor',
    # Engineering
    'EngineeringAgent',
    'CodeReviewer',
    'ArchitectureAnalyzer',
    'TechnicalDocumentor',
    # Science
    'ScienceAgent',
    'ResearchAssistant',
    'DataAnalyzer',
    'LiteratureReviewer',
]
