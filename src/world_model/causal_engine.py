"""
AION Causal Engine
==================

Causal reasoning for understanding cause-and-effect relationships.
Enables agents to:
- Define causal rules
- Infer effects from actions
- Answer counterfactual questions
- Predict consequences
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re


class CausalStrength(Enum):
    """Strength of causal relationships."""
    WEAK = 0.25
    MODERATE = 0.5
    STRONG = 0.75
    DETERMINISTIC = 1.0


@dataclass
class CausalRule:
    """
    A causal rule: If condition, then effect.
    
    Example:
        CausalRule(
            name="fire_causes_heat",
            condition={"entity_type": "fire", "state": "burning"},
            effect={"property": "temperature", "change": "+100"},
            strength=CausalStrength.STRONG
        )
    """
    name: str
    condition: Dict[str, Any]
    effect: Dict[str, Any]
    strength: CausalStrength = CausalStrength.MODERATE
    delay: float = 0.0  # Time delay in seconds
    reversible: bool = True
    description: str = ""
    
    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if condition matches context."""
        for key, value in self.condition.items():
            if key not in context:
                return False
            
            ctx_value = context[key]
            
            # Handle pattern matching
            if isinstance(value, str) and value.startswith("pattern:"):
                pattern = value[8:]
                if not re.match(pattern, str(ctx_value)):
                    return False
            # Handle range matching
            elif isinstance(value, dict) and "min" in value and "max" in value:
                if not (value["min"] <= ctx_value <= value["max"]):
                    return False
            # Handle list (any match)
            elif isinstance(value, list):
                if ctx_value not in value:
                    return False
            # Exact match
            elif ctx_value != value:
                return False
        
        return True
    
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the effect to a context."""
        result = context.copy()
        
        for key, value in self.effect.items():
            if isinstance(value, str) and value.startswith("+"):
                # Relative change
                delta = float(value[1:])
                result[key] = result.get(key, 0) + delta
            elif isinstance(value, str) and value.startswith("-"):
                delta = float(value[1:])
                result[key] = result.get(key, 0) - delta
            elif isinstance(value, str) and value.startswith("*"):
                factor = float(value[1:])
                result[key] = result.get(key, 1) * factor
            else:
                result[key] = value
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "condition": self.condition,
            "effect": self.effect,
            "strength": self.strength.value,
            "delay": self.delay,
            "reversible": self.reversible,
            "description": self.description
        }


@dataclass
class CausalQuery:
    """A causal query (what would happen if...)."""
    query_type: str  # "predict", "explain", "counterfactual"
    target: Dict[str, Any]
    intervention: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalResult:
    """Result of a causal query."""
    effects: List[Tuple[CausalRule, Dict[str, Any]]]
    probability: float
    confidence: float
    explanation: str
    chain_of_causation: List[str] = field(default_factory=list)


class CausalEngine:
    """
    Engine for causal reasoning.
    
    Supports:
    - Rule-based causal inference
    - Causal chain discovery
    - Counterfactual reasoning
    - Effect prediction
    """
    
    def __init__(self):
        self.rules: Dict[str, CausalRule] = {}
        self.rule_chains: Dict[str, List[str]] = {}  # effect -> [causing rules]
        self._create_default_rules()
    
    def _create_default_rules(self):
        """Create common sense causal rules."""
        defaults = [
            CausalRule(
                name="action_causes_state_change",
                condition={"type": "action"},
                effect={"state_changed": True},
                strength=CausalStrength.MODERATE
            ),
            CausalRule(
                name="error_causes_failure",
                condition={"error": True},
                effect={"success": False},
                strength=CausalStrength.STRONG
            ),
            CausalRule(
                name="timeout_causes_failure",
                condition={"timeout": True},
                effect={"success": False, "error": "timeout"},
                strength=CausalStrength.DETERMINISTIC
            ),
        ]
        
        for rule in defaults:
            self.add_rule(rule)
    
    def add_rule(self, rule: CausalRule):
        """Add a causal rule."""
        self.rules[rule.name] = rule
        
        # Update chain index
        for effect_key in rule.effect:
            if effect_key not in self.rule_chains:
                self.rule_chains[effect_key] = []
            self.rule_chains[effect_key].append(rule.name)
    
    def remove_rule(self, rule_name: str):
        """Remove a causal rule."""
        if rule_name in self.rules:
            rule = self.rules[rule_name]
            for effect_key in rule.effect:
                if effect_key in self.rule_chains:
                    self.rule_chains[effect_key] = [
                        r for r in self.rule_chains[effect_key]
                        if r != rule_name
                    ]
            del self.rules[rule_name]
    
    def infer(self, context: Dict[str, Any]) -> CausalResult:
        """
        Infer effects from current context.
        
        Args:
            context: Current state/situation
        
        Returns:
            CausalResult with predicted effects
        """
        matching_rules = []
        total_probability = 0.0
        chain = []
        
        for rule in self.rules.values():
            if rule.matches(context):
                effect = rule.apply(context)
                matching_rules.append((rule, effect))
                total_probability = max(total_probability, rule.strength.value)
                chain.append(f"{rule.name}: {rule.condition} → {rule.effect}")
        
        explanation = self._generate_explanation(matching_rules)
        
        return CausalResult(
            effects=matching_rules,
            probability=total_probability,
            confidence=min(1.0, len(matching_rules) * 0.2 + 0.3),
            explanation=explanation,
            chain_of_causation=chain
        )
    
    def predict(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any] = None,
        depth: int = 3
    ) -> CausalResult:
        """
        Predict outcomes of an action.
        
        Args:
            action: The action to predict outcomes for
            context: Current context
            depth: How many levels of causation to explore
        
        Returns:
            CausalResult with predicted outcomes
        """
        context = context or {}
        merged = {**context, **action, "type": "action"}
        
        all_effects = []
        chain = []
        current_state = merged.copy()
        
        for level in range(depth):
            result = self.infer(current_state)
            
            if not result.effects:
                break
            
            for rule, effect in result.effects:
                all_effects.append((rule, effect))
                chain.extend(result.chain_of_causation)
                current_state.update(effect)
        
        return CausalResult(
            effects=all_effects,
            probability=self._calculate_chain_probability(all_effects),
            confidence=min(1.0, 0.9 - depth * 0.1),
            explanation=self._generate_explanation(all_effects),
            chain_of_causation=chain
        )
    
    def counterfactual(
        self,
        original_context: Dict[str, Any],
        original_outcome: Dict[str, Any],
        intervention: Dict[str, Any]
    ) -> CausalResult:
        """
        Answer counterfactual: What if X had been different?
        
        Args:
            original_context: What actually happened
            original_outcome: What was the actual outcome
            intervention: What we're changing
        
        Returns:
            CausalResult showing alternative outcome
        """
        # Create counterfactual context
        altered_context = {**original_context, **intervention}
        
        # Predict from altered context
        result = self.predict(altered_context)
        
        # Compare to original outcome
        differences = []
        for rule, effect in result.effects:
            for key, value in effect.items():
                if key in original_outcome and original_outcome[key] != value:
                    differences.append(f"{key}: {original_outcome[key]} → {value}")
        
        result.explanation = (
            f"With intervention {intervention}, the outcome would differ: " +
            ", ".join(differences) if differences else "No significant difference"
        )
        
        return result
    
    def explain(
        self,
        effect: str,
        context: Dict[str, Any]
    ) -> List[CausalRule]:
        """
        Explain what could cause an effect.
        
        Args:
            effect: The effect to explain
            context: Current context
        
        Returns:
            List of rules that could cause the effect
        """
        potential_causes = []
        
        rule_names = self.rule_chains.get(effect, [])
        for name in rule_names:
            rule = self.rules.get(name)
            if rule:
                potential_causes.append(rule)
        
        return potential_causes
    
    def find_intervention(
        self,
        current_context: Dict[str, Any],
        desired_effect: Dict[str, Any],
        max_changes: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find what intervention would achieve desired effect.
        
        Args:
            current_context: Current state
            desired_effect: What we want to achieve
            max_changes: Maximum number of changes to consider
        
        Returns:
            List of possible interventions
        """
        interventions = []
        
        for effect_key, target_value in desired_effect.items():
            causing_rules = self.rule_chains.get(effect_key, [])
            
            for rule_name in causing_rules:
                rule = self.rules.get(rule_name)
                if not rule:
                    continue
                
                # Check if rule produces desired effect
                test_effect = rule.effect.get(effect_key)
                if test_effect == target_value:
                    # Build intervention from conditions
                    intervention = {}
                    changes = 0
                    for cond_key, cond_value in rule.condition.items():
                        if current_context.get(cond_key) != cond_value:
                            intervention[cond_key] = cond_value
                            changes += 1
                            if changes > max_changes:
                                break
                    
                    if intervention and changes <= max_changes:
                        interventions.append(intervention)
        
        return interventions
    
    def _calculate_chain_probability(
        self,
        effects: List[Tuple[CausalRule, Dict[str, Any]]]
    ) -> float:
        """Calculate probability of a causal chain."""
        if not effects:
            return 0.0
        
        probability = 1.0
        for rule, _ in effects:
            probability *= rule.strength.value
        
        return probability
    
    def _generate_explanation(
        self,
        effects: List[Tuple[CausalRule, Dict[str, Any]]]
    ) -> str:
        """Generate a natural language explanation."""
        if not effects:
            return "No causal effects identified."
        
        explanations = []
        for rule, effect in effects:
            if rule.description:
                explanations.append(rule.description)
            else:
                explanations.append(f"{rule.name}: leads to {effect}")
        
        return " → ".join(explanations)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_rules": len(self.rules),
            "indexed_effects": len(self.rule_chains),
            "rules_by_strength": {
                s.name: sum(1 for r in self.rules.values() if r.strength == s)
                for s in CausalStrength
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state."""
        return {
            "rules": {name: rule.to_dict() for name, rule in self.rules.items()}
        }
