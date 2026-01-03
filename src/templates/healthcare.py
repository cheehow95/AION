"""
AION Industry Templates - Healthcare
=====================================

Healthcare-specific agent templates:
- HIPAA Compliance: Patient data protection
- Clinical Decision Support: Evidence-based recommendations
- Medical Knowledge Base: Domain-specific knowledge
- Patient Communication: Empathetic interaction

Auto-generated for Phase 5: Scale
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum


class DataSensitivity(Enum):
    """Data sensitivity levels."""
    PHI = "phi"  # Protected Health Information
    PII = "pii"  # Personally Identifiable Information
    CLINICAL = "clinical"
    ADMINISTRATIVE = "administrative"
    PUBLIC = "public"


@dataclass
class PatientContext:
    """Patient interaction context."""
    patient_id: str = ""
    age_group: str = ""
    conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    consent_given: bool = False
    data_sensitivity: DataSensitivity = DataSensitivity.PHI


class HIPAACompliance:
    """HIPAA compliance utilities."""
    
    PHI_IDENTIFIERS = {
        "names", "geographic_data", "dates", "phone_numbers", "fax_numbers",
        "email_addresses", "ssn", "medical_record_numbers", "health_plan_numbers",
        "account_numbers", "certificate_numbers", "vehicle_identifiers",
        "device_identifiers", "urls", "ip_addresses", "biometric_identifiers",
        "photos", "unique_identifiers"
    }
    
    def __init__(self):
        self.access_log: List[Dict[str, Any]] = []
        self.audit_trail: List[Dict[str, Any]] = []
    
    def check_phi_access(self, user_id: str, patient_id: str,
                         purpose: str, has_consent: bool) -> bool:
        """Check if PHI access is permitted."""
        # Check minimum necessary standard
        permitted = has_consent and purpose in [
            "treatment", "payment", "healthcare_operations",
            "public_health", "research_with_irb"
        ]
        
        self.access_log.append({
            'user_id': user_id,
            'patient_id': patient_id,
            'purpose': purpose,
            'permitted': permitted,
            'timestamp': datetime.now().isoformat()
        })
        
        return permitted
    
    def redact_phi(self, text: str, identifiers_to_redact: Set[str] = None) -> str:
        """Redact PHI from text (simplified implementation)."""
        # In production, would use NER/regex patterns
        redacted = text
        # Placeholder redaction
        import re
        
        # Redact SSN patterns
        redacted = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED-SSN]', redacted)
        # Redact phone patterns
        redacted = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED-PHONE]', redacted)
        # Redact email patterns
        redacted = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[REDACTED-EMAIL]', redacted)
        
        return redacted
    
    def log_audit(self, action: str, user_id: str, details: Dict[str, Any]):
        """Log action for audit trail."""
        self.audit_trail.append({
            'action': action,
            'user_id': user_id,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })


@dataclass
class ClinicalEvidence:
    """Clinical evidence for decision support."""
    source: str = ""
    level: str = ""  # A, B, C, D (evidence levels)
    recommendation: str = ""
    contraindications: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


class ClinicalDecisionSupport:
    """Clinical decision support system."""
    
    def __init__(self):
        self.guidelines: Dict[str, List[ClinicalEvidence]] = {}
        self.drug_interactions: Dict[str, Set[str]] = {}
    
    def add_guideline(self, condition: str, evidence: ClinicalEvidence):
        """Add a clinical guideline."""
        if condition not in self.guidelines:
            self.guidelines[condition] = []
        self.guidelines[condition].append(evidence)
    
    def add_drug_interaction(self, drug1: str, drug2: str):
        """Record a drug interaction."""
        d1, d2 = drug1.lower(), drug2.lower()
        if d1 not in self.drug_interactions:
            self.drug_interactions[d1] = set()
        if d2 not in self.drug_interactions:
            self.drug_interactions[d2] = set()
        self.drug_interactions[d1].add(d2)
        self.drug_interactions[d2].add(d1)
    
    def check_interactions(self, medications: List[str]) -> List[tuple]:
        """Check for drug interactions."""
        interactions = []
        meds = [m.lower() for m in medications]
        
        for i, med1 in enumerate(meds):
            for med2 in meds[i+1:]:
                if med2 in self.drug_interactions.get(med1, set()):
                    interactions.append((med1, med2))
        
        return interactions
    
    def get_recommendations(self, context: PatientContext) -> List[Dict[str, Any]]:
        """Get clinical recommendations for patient context."""
        recommendations = []
        
        # Check drug interactions
        interactions = self.check_interactions(context.medications)
        if interactions:
            recommendations.append({
                'type': 'warning',
                'category': 'drug_interaction',
                'message': f"Potential interactions: {interactions}",
                'priority': 'high'
            })
        
        # Check allergy conflicts
        for med in context.medications:
            if med.lower() in [a.lower() for a in context.allergies]:
                recommendations.append({
                    'type': 'alert',
                    'category': 'allergy',
                    'message': f"Patient allergic to prescribed medication: {med}",
                    'priority': 'critical'
                })
        
        # Get condition-specific guidelines
        for condition in context.conditions:
            if condition in self.guidelines:
                for evidence in self.guidelines[condition]:
                    recommendations.append({
                        'type': 'guideline',
                        'category': condition,
                        'message': evidence.recommendation,
                        'evidence_level': evidence.level,
                        'source': evidence.source,
                        'priority': 'medium'
                    })
        
        return recommendations


class MedicalKnowledgeBase:
    """Medical domain knowledge base."""
    
    def __init__(self):
        self.conditions: Dict[str, Dict[str, Any]] = {}
        self.medications: Dict[str, Dict[str, Any]] = {}
        self.procedures: Dict[str, Dict[str, Any]] = {}
    
    def add_condition(self, name: str, icd_code: str, 
                      symptoms: List[str], treatments: List[str]):
        """Add a medical condition."""
        self.conditions[name.lower()] = {
            'name': name,
            'icd_code': icd_code,
            'symptoms': symptoms,
            'treatments': treatments
        }
    
    def add_medication(self, name: str, drug_class: str,
                       indications: List[str], side_effects: List[str]):
        """Add a medication."""
        self.medications[name.lower()] = {
            'name': name,
            'drug_class': drug_class,
            'indications': indications,
            'side_effects': side_effects
        }
    
    def lookup_condition(self, query: str) -> Optional[Dict[str, Any]]:
        """Look up a medical condition."""
        query_lower = query.lower()
        return self.conditions.get(query_lower)
    
    def lookup_medication(self, query: str) -> Optional[Dict[str, Any]]:
        """Look up a medication."""
        query_lower = query.lower()
        return self.medications.get(query_lower)
    
    def differential_diagnosis(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """Get differential diagnosis based on symptoms."""
        matches = []
        
        for condition in self.conditions.values():
            condition_symptoms = set(s.lower() for s in condition['symptoms'])
            patient_symptoms = set(s.lower() for s in symptoms)
            
            overlap = len(condition_symptoms & patient_symptoms)
            if overlap > 0:
                score = overlap / len(condition_symptoms)
                matches.append({
                    'condition': condition['name'],
                    'icd_code': condition['icd_code'],
                    'match_score': score,
                    'matched_symptoms': list(condition_symptoms & patient_symptoms)
                })
        
        return sorted(matches, key=lambda x: x['match_score'], reverse=True)


class HealthcareAgent:
    """Healthcare-specialized AION agent."""
    
    def __init__(self, agent_id: str = "healthcare-agent"):
        self.agent_id = agent_id
        self.hipaa = HIPAACompliance()
        self.cds = ClinicalDecisionSupport()
        self.knowledge = MedicalKnowledgeBase()
        self.active_contexts: Dict[str, PatientContext] = {}
    
    def set_patient_context(self, patient_id: str, context: PatientContext):
        """Set patient context for interaction."""
        self.active_contexts[patient_id] = context
    
    async def process_query(self, query: str, patient_id: str = None,
                            user_id: str = "system") -> Dict[str, Any]:
        """Process a healthcare query."""
        response = {'query': query, 'patient_id': patient_id}
        
        # Check HIPAA compliance if patient-specific
        if patient_id:
            context = self.active_contexts.get(patient_id)
            if context:
                permitted = self.hipaa.check_phi_access(
                    user_id, patient_id, "treatment", context.consent_given
                )
                if not permitted:
                    return {
                        'error': 'Access denied',
                        'reason': 'HIPAA compliance check failed'
                    }
                
                # Get clinical recommendations
                response['recommendations'] = self.cds.get_recommendations(context)
        
        # Redact any PHI in response
        response['query'] = self.hipaa.redact_phi(query)
        
        # Add disclaimer
        response['disclaimer'] = (
            "This information is for educational purposes only and should not "
            "replace professional medical advice. Always consult a qualified "
            "healthcare provider."
        )
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            'agent_id': self.agent_id,
            'active_patients': len(self.active_contexts),
            'access_log_entries': len(self.hipaa.access_log),
            'knowledge_conditions': len(self.knowledge.conditions),
            'knowledge_medications': len(self.knowledge.medications)
        }


async def demo_healthcare():
    """Demonstrate healthcare template."""
    print("🏥 Healthcare Template Demo")
    print("=" * 50)
    
    agent = HealthcareAgent()
    
    # Setup knowledge base
    agent.knowledge.add_condition(
        "Diabetes Type 2", "E11",
        symptoms=["fatigue", "increased thirst", "frequent urination", "blurred vision"],
        treatments=["metformin", "lifestyle changes", "insulin"]
    )
    
    agent.knowledge.add_medication(
        "Metformin", "Biguanide",
        indications=["Type 2 diabetes"],
        side_effects=["nausea", "diarrhea", "vitamin B12 deficiency"]
    )
    
    # Setup drug interaction
    agent.cds.add_drug_interaction("metformin", "contrast_dye")
    
    # Create patient context
    context = PatientContext(
        patient_id="P001",
        age_group="adult",
        conditions=["Diabetes Type 2"],
        medications=["metformin", "lisinopril"],
        allergies=["penicillin"],
        consent_given=True
    )
    agent.set_patient_context("P001", context)
    
    print("\n👤 Patient context set")
    
    # Process query
    response = await agent.process_query(
        "What are the treatment options for my condition?",
        patient_id="P001",
        user_id="physician_1"
    )
    
    print(f"\n📋 Query Response:")
    if 'recommendations' in response:
        for rec in response['recommendations']:
            print(f"  [{rec['priority']}] {rec['type']}: {rec['message']}")
    
    # Differential diagnosis
    symptoms = ["fatigue", "increased thirst", "blurred vision"]
    diagnosis = agent.knowledge.differential_diagnosis(symptoms)
    
    print(f"\n🔍 Differential Diagnosis:")
    for d in diagnosis[:3]:
        print(f"  {d['condition']} ({d['icd_code']}): {d['match_score']:.0%} match")
    
    # HIPAA redaction
    phi_text = "Patient John Smith, SSN 123-45-6789, phone 555-123-4567"
    redacted = agent.hipaa.redact_phi(phi_text)
    print(f"\n🔒 PHI Redaction:")
    print(f"  Original: {phi_text}")
    print(f"  Redacted: {redacted}")
    
    print(f"\n📊 Status: {agent.get_status()}")
    print("\n✅ Healthcare template demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_healthcare())
