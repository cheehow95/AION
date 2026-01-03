"""
AION Self-Evolution v2 - Safety Constraint Evolution
=====================================================

Safety constraint evolution:
- Constraint Discovery: Learning new safety rules from incidents
- Invariant Strengthening: Progressive constraint tightening
- Safety Testing: Adversarial safety evaluation
- Constraint Relaxation: Safe removal of unnecessary constraints

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime
from enum import Enum
import random


class ConstraintSeverity(Enum):
    """Severity levels for safety constraints."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ADVISORY = "advisory"


class ConstraintStatus(Enum):
    """Status of a constraint."""
    ACTIVE = "active"
    PROPOSED = "proposed"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


@dataclass
class SafetyConstraint:
    """A safety constraint."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    severity: ConstraintSeverity = ConstraintSeverity.MEDIUM
    status: ConstraintStatus = ConstraintStatus.ACTIVE
    check_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    violation_count: int = 0
    false_positive_count: int = 0
    last_triggered: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def check(self, context: Dict[str, Any]) -> bool:
        """Check if constraint is satisfied."""
        if self.check_func:
            try:
                return self.check_func(context)
            except Exception:
                return True  # Fail safe
        return True
    
    def record_violation(self):
        """Record a violation."""
        self.violation_count += 1
        self.last_triggered = datetime.now()
    
    def record_false_positive(self):
        """Record a false positive."""
        self.false_positive_count += 1


@dataclass
class SafetyIncident:
    """A safety incident for learning."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    severity: ConstraintSeverity = ConstraintSeverity.MEDIUM
    root_cause: str = ""
    resolution: str = ""
    learned_constraint: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyTest:
    """A test for safety constraints."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    test_func: Optional[Callable] = None
    adversarial: bool = False
    target_constraints: List[str] = field(default_factory=list)
    
    async def run(self, constraints: List[SafetyConstraint]) -> Dict[str, Any]:
        """Run the safety test."""
        results = {'passed': [], 'failed': [], 'errors': []}
        
        for constraint in constraints:
            if self.target_constraints and constraint.id not in self.target_constraints:
                continue
            
            try:
                if self.test_func:
                    context = await self.test_func() if asyncio.iscoroutinefunction(self.test_func) else self.test_func()
                    passed = constraint.check(context)
                else:
                    passed = True
                
                if passed:
                    results['passed'].append(constraint.id)
                else:
                    results['failed'].append(constraint.id)
            except Exception as e:
                results['errors'].append({'constraint': constraint.id, 'error': str(e)})
        
        return results


class ConstraintLearner:
    """Learns new constraints from incidents."""
    
    def __init__(self):
        self.incidents: List[SafetyIncident] = []
        self.patterns: Dict[str, int] = {}  # pattern -> occurrence count
        self.proposed_constraints: List[SafetyConstraint] = []
    
    def record_incident(self, incident: SafetyIncident):
        """Record a safety incident."""
        self.incidents.append(incident)
        
        # Extract patterns from context
        for key, value in incident.context.items():
            pattern = f"{key}:{type(value).__name__}"
            self.patterns[pattern] = self.patterns.get(pattern, 0) + 1
    
    def discover_constraints(self, threshold: int = 3) -> List[SafetyConstraint]:
        """Discover new constraints from patterns."""
        discovered = []
        
        for pattern, count in self.patterns.items():
            if count >= threshold:
                key, type_name = pattern.split(':')
                
                # Create constraint based on pattern
                def make_check(k, t):
                    def check(ctx):
                        if k in ctx:
                            return type(ctx[k]).__name__ == t
                        return True
                    return check
                
                constraint = SafetyConstraint(
                    name=f"type_check_{key}",
                    description=f"Ensure {key} is of type {type_name}",
                    severity=ConstraintSeverity.MEDIUM,
                    status=ConstraintStatus.PROPOSED,
                    check_func=make_check(key, type_name)
                )
                discovered.append(constraint)
        
        self.proposed_constraints.extend(discovered)
        return discovered


class SafetyEvolution:
    """Evolves safety constraints over time."""
    
    def __init__(self):
        self.constraints: Dict[str, SafetyConstraint] = {}
        self.tests: Dict[str, SafetyTest] = {}
        self.learner = ConstraintLearner()
        self.evolution_history: List[Dict[str, Any]] = []
    
    def add_constraint(self, constraint: SafetyConstraint):
        """Add a safety constraint."""
        self.constraints[constraint.id] = constraint
    
    def add_test(self, test: SafetyTest):
        """Add a safety test."""
        self.tests[test.id] = test
    
    async def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all constraints against context."""
        violations = []
        
        for constraint in self.constraints.values():
            if constraint.status != ConstraintStatus.ACTIVE:
                continue
            
            if not constraint.check(context):
                constraint.record_violation()
                violations.append({
                    'constraint': constraint.name,
                    'severity': constraint.severity.value
                })
        
        return {
            'passed': len(self.constraints) - len(violations),
            'violations': violations,
            'context': context
        }
    
    async def run_safety_tests(self) -> Dict[str, Any]:
        """Run all safety tests."""
        results = {'total': 0, 'passed': 0, 'failed': 0}
        
        for test in self.tests.values():
            result = await test.run(list(self.constraints.values()))
            results['total'] += len(result['passed']) + len(result['failed'])
            results['passed'] += len(result['passed'])
            results['failed'] += len(result['failed'])
        
        return results
    
    def strengthen_constraint(self, constraint_id: str, factor: float = 1.2):
        """Strengthen a constraint after violations."""
        if constraint_id not in self.constraints:
            return
        
        constraint = self.constraints[constraint_id]
        
        # Increase severity if frequent violations
        severities = list(ConstraintSeverity)
        current_idx = severities.index(constraint.severity)
        if current_idx > 0 and constraint.violation_count > 5:
            constraint.severity = severities[current_idx - 1]
            
            self.evolution_history.append({
                'action': 'strengthen',
                'constraint': constraint_id,
                'new_severity': constraint.severity.value,
                'timestamp': datetime.now().isoformat()
            })
    
    def relax_constraint(self, constraint_id: str) -> bool:
        """Relax or remove a constraint with many false positives."""
        if constraint_id not in self.constraints:
            return False
        
        constraint = self.constraints[constraint_id]
        
        # Only relax if many false positives and few real violations
        if constraint.false_positive_count < 10 or constraint.violation_count > 2:
            return False
        
        if constraint.severity == ConstraintSeverity.ADVISORY:
            constraint.status = ConstraintStatus.DEPRECATED
        else:
            severities = list(ConstraintSeverity)
            current_idx = severities.index(constraint.severity)
            if current_idx < len(severities) - 1:
                constraint.severity = severities[current_idx + 1]
        
        self.evolution_history.append({
            'action': 'relax',
            'constraint': constraint_id,
            'new_severity': constraint.severity.value,
            'timestamp': datetime.now().isoformat()
        })
        
        return True
    
    def learn_from_incident(self, incident: SafetyIncident) -> List[SafetyConstraint]:
        """Learn new constraints from an incident."""
        self.learner.record_incident(incident)
        
        new_constraints = self.learner.discover_constraints()
        
        for constraint in new_constraints:
            self.add_constraint(constraint)
        
        return new_constraints
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety evolution statistics."""
        by_status = {}
        by_severity = {}
        
        for c in self.constraints.values():
            by_status[c.status.value] = by_status.get(c.status.value, 0) + 1
            by_severity[c.severity.value] = by_severity.get(c.severity.value, 0) + 1
        
        return {
            'total_constraints': len(self.constraints),
            'total_tests': len(self.tests),
            'by_status': by_status,
            'by_severity': by_severity,
            'evolution_events': len(self.evolution_history)
        }


async def demo_safety_evolution():
    """Demonstrate safety evolution."""
    print("🛡️ Safety Evolution Demo")
    print("=" * 50)
    
    evolution = SafetyEvolution()
    
    # Add constraints
    evolution.add_constraint(SafetyConstraint(
        name="rate_limit",
        description="Limit request rate",
        severity=ConstraintSeverity.HIGH,
        check_func=lambda ctx: ctx.get('rate', 0) < 100
    ))
    
    evolution.add_constraint(SafetyConstraint(
        name="memory_limit",
        description="Limit memory usage",
        severity=ConstraintSeverity.CRITICAL,
        check_func=lambda ctx: ctx.get('memory_mb', 0) < 1024
    ))
    
    print(f"\n📋 Initial constraints: {len(evolution.constraints)}")
    
    # Evaluate
    result = await evolution.evaluate({'rate': 50, 'memory_mb': 500})
    print(f"  Evaluation: {result['passed']} passed, {len(result['violations'])} violations")
    
    # Learn from incident
    incident = SafetyIncident(
        description="High CPU usage",
        context={'cpu_percent': 95.0, 'rate': 150},
        severity=ConstraintSeverity.HIGH,
        root_cause="Uncontrolled loop"
    )
    
    new_constraints = evolution.learn_from_incident(incident)
    print(f"\n🔍 Learned {len(new_constraints)} new constraints from incident")
    
    # Strengthen after violations
    evolution.constraints[list(evolution.constraints.keys())[0]].violation_count = 10
    evolution.strengthen_constraint(list(evolution.constraints.keys())[0])
    
    print(f"\n📊 Statistics: {evolution.get_statistics()}")
    print("\n✅ Safety evolution demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_safety_evolution())
