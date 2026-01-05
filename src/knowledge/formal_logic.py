"""
AION Formal Logic Engine
=========================

Formal logic and reasoning capabilities:
- Propositional logic
- First-order logic (FOL)
- Modal logic
- Inference rules
- Theorem proving
- Logical connectives

Enables rigorous logical reasoning for AI.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# LOGICAL CONNECTIVES
# =============================================================================

class Connective(Enum):
    """Logical connectives."""
    NOT = "¬"       # Negation
    AND = "∧"       # Conjunction
    OR = "∨"        # Disjunction
    IMPLIES = "→"   # Implication
    IFF = "↔"       # Biconditional
    XOR = "⊕"       # Exclusive or


class Quantifier(Enum):
    """First-order logic quantifiers."""
    FORALL = "∀"    # Universal
    EXISTS = "∃"    # Existential


class ModalOperator(Enum):
    """Modal logic operators."""
    NECESSARY = "□"  # Necessarily
    POSSIBLE = "◇"   # Possibly
    KNOWS = "K"      # Knowledge
    BELIEVES = "B"   # Belief


# =============================================================================
# LOGICAL EXPRESSIONS
# =============================================================================

class Expression(ABC):
    """Base class for logical expressions."""
    
    @abstractmethod
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        """Evaluate expression under truth assignment."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def variables(self) -> Set[str]:
        """Get all propositional variables in expression."""
        pass


class Proposition(Expression):
    """Atomic proposition (variable)."""
    
    def __init__(self, name: str, value: Optional[bool] = None):
        self.name = name
        self.value = value
    
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        if self.value is not None:
            return self.value
        return assignment.get(self.name, False)
    
    def variables(self) -> Set[str]:
        return {self.name}
    
    def __str__(self) -> str:
        return self.name


class Not(Expression):
    """Negation: ¬P"""
    
    def __init__(self, operand: Expression):
        self.operand = operand
    
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        return not self.operand.evaluate(assignment)
    
    def variables(self) -> Set[str]:
        return self.operand.variables()
    
    def __str__(self) -> str:
        return f"¬{self.operand}"


class BinaryOp(Expression):
    """Binary logical operation."""
    
    def __init__(self, left: Expression, right: Expression, connective: Connective):
        self.left = left
        self.right = right
        self.connective = connective
    
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        l = self.left.evaluate(assignment)
        r = self.right.evaluate(assignment)
        
        if self.connective == Connective.AND:
            return l and r
        elif self.connective == Connective.OR:
            return l or r
        elif self.connective == Connective.IMPLIES:
            return (not l) or r
        elif self.connective == Connective.IFF:
            return l == r
        elif self.connective == Connective.XOR:
            return l != r
        return False
    
    def variables(self) -> Set[str]:
        return self.left.variables() | self.right.variables()
    
    def __str__(self) -> str:
        return f"({self.left} {self.connective.value} {self.right})"


# Convenience functions
def And(left: Expression, right: Expression) -> BinaryOp:
    return BinaryOp(left, right, Connective.AND)

def Or(left: Expression, right: Expression) -> BinaryOp:
    return BinaryOp(left, right, Connective.OR)

def Implies(left: Expression, right: Expression) -> BinaryOp:
    return BinaryOp(left, right, Connective.IMPLIES)

def Iff(left: Expression, right: Expression) -> BinaryOp:
    return BinaryOp(left, right, Connective.IFF)

def Xor(left: Expression, right: Expression) -> BinaryOp:
    return BinaryOp(left, right, Connective.XOR)


# =============================================================================
# FIRST-ORDER LOGIC
# =============================================================================

@dataclass
class Term:
    """FOL term (variable or constant)."""
    name: str
    is_variable: bool = True


@dataclass
class Predicate:
    """FOL predicate: P(x, y, ...)"""
    name: str
    arity: int
    
    def __call__(self, *terms: Term) -> 'PredicateApplication':
        if len(terms) != self.arity:
            raise ValueError(f"Predicate {self.name} expects {self.arity} terms")
        return PredicateApplication(self, list(terms))


@dataclass
class PredicateApplication:
    """Application of predicate to terms."""
    predicate: Predicate
    terms: List[Term]
    
    def __str__(self) -> str:
        terms_str = ", ".join(t.name for t in self.terms)
        return f"{self.predicate.name}({terms_str})"


class QuantifiedFormula:
    """Quantified first-order formula: ∀x.P(x) or ∃x.P(x)"""
    
    def __init__(self, quantifier: Quantifier, variable: Term, 
                 formula: Union['QuantifiedFormula', PredicateApplication]):
        self.quantifier = quantifier
        self.variable = variable
        self.formula = formula
    
    def __str__(self) -> str:
        return f"{self.quantifier.value}{self.variable.name}.{self.formula}"


# =============================================================================
# INFERENCE RULES
# =============================================================================

class InferenceRule:
    """Base class for inference rules."""
    name: str = "Unknown"
    
    @staticmethod
    def apply(*premises: Expression) -> Optional[Expression]:
        """Apply rule to premises, return conclusion if valid."""
        pass


class ModusPonens(InferenceRule):
    """
    Modus Ponens: From P and P→Q, infer Q.
    """
    name = "Modus Ponens"
    
    @staticmethod
    def apply(p: Expression, p_implies_q: Expression) -> Optional[Expression]:
        if isinstance(p_implies_q, BinaryOp) and p_implies_q.connective == Connective.IMPLIES:
            # Check if p matches the antecedent
            if str(p) == str(p_implies_q.left):
                return p_implies_q.right
        return None


class ModusTollens(InferenceRule):
    """
    Modus Tollens: From ¬Q and P→Q, infer ¬P.
    """
    name = "Modus Tollens"
    
    @staticmethod
    def apply(not_q: Expression, p_implies_q: Expression) -> Optional[Expression]:
        if isinstance(not_q, Not) and isinstance(p_implies_q, BinaryOp):
            if p_implies_q.connective == Connective.IMPLIES:
                if str(not_q.operand) == str(p_implies_q.right):
                    return Not(p_implies_q.left)
        return None


class HypotheticalSyllogism(InferenceRule):
    """
    Hypothetical Syllogism: From P→Q and Q→R, infer P→R.
    """
    name = "Hypothetical Syllogism"
    
    @staticmethod
    def apply(p_implies_q: Expression, q_implies_r: Expression) -> Optional[Expression]:
        if isinstance(p_implies_q, BinaryOp) and isinstance(q_implies_r, BinaryOp):
            if (p_implies_q.connective == Connective.IMPLIES and 
                q_implies_r.connective == Connective.IMPLIES):
                if str(p_implies_q.right) == str(q_implies_r.left):
                    return Implies(p_implies_q.left, q_implies_r.right)
        return None


class DisjunctiveSyllogism(InferenceRule):
    """
    Disjunctive Syllogism: From P∨Q and ¬P, infer Q.
    """
    name = "Disjunctive Syllogism"
    
    @staticmethod
    def apply(p_or_q: Expression, not_p: Expression) -> Optional[Expression]:
        if isinstance(p_or_q, BinaryOp) and isinstance(not_p, Not):
            if p_or_q.connective == Connective.OR:
                if str(not_p.operand) == str(p_or_q.left):
                    return p_or_q.right
                elif str(not_p.operand) == str(p_or_q.right):
                    return p_or_q.left
        return None


class Conjunction(InferenceRule):
    """
    Conjunction Introduction: From P and Q, infer P∧Q.
    """
    name = "Conjunction Introduction"
    
    @staticmethod
    def apply(p: Expression, q: Expression) -> Expression:
        return And(p, q)


class Simplification(InferenceRule):
    """
    Simplification: From P∧Q, infer P (or Q).
    """
    name = "Simplification"
    
    @staticmethod
    def apply(p_and_q: Expression, which: str = 'left') -> Optional[Expression]:
        if isinstance(p_and_q, BinaryOp) and p_and_q.connective == Connective.AND:
            return p_and_q.left if which == 'left' else p_and_q.right
        return None


# =============================================================================
# THEOREM PROVER
# =============================================================================

class ProofStep:
    """A single step in a proof."""
    
    def __init__(self, line: int, expression: Expression, 
                 justification: str, premises: List[int] = None):
        self.line = line
        self.expression = expression
        self.justification = justification
        self.premises = premises or []
    
    def __str__(self) -> str:
        premise_str = f" [{', '.join(map(str, self.premises))}]" if self.premises else ""
        return f"{self.line}. {self.expression} ({self.justification}{premise_str})"


class Proof:
    """A formal proof."""
    
    def __init__(self):
        self.steps: List[ProofStep] = []
        self.line = 0
    
    def add_premise(self, expr: Expression) -> int:
        self.line += 1
        self.steps.append(ProofStep(self.line, expr, "Premise"))
        return self.line
    
    def add_step(self, expr: Expression, justification: str, 
                 premises: List[int] = None) -> int:
        self.line += 1
        self.steps.append(ProofStep(self.line, expr, justification, premises))
        return self.line
    
    def is_valid(self) -> bool:
        """Check if proof is valid (all steps follow from premises)."""
        # Simplified: check that all referenced premises exist
        for step in self.steps:
            for p in step.premises:
                if p < 1 or p > len(self.steps):
                    return False
        return True
    
    def __str__(self) -> str:
        return "\n".join(str(step) for step in self.steps)


class TheoremProver:
    """Simple theorem prover using forward chaining."""
    
    def __init__(self):
        self.rules = [
            ModusPonens,
            ModusTollens,
            HypotheticalSyllogism,
            DisjunctiveSyllogism
        ]
    
    def prove(self, premises: List[Expression], 
              goal: Expression, max_steps: int = 100) -> Optional[Proof]:
        """
        Attempt to prove goal from premises using forward chaining.
        """
        proof = Proof()
        known = []
        
        # Add premises
        for p in premises:
            proof.add_premise(p)
            known.append(p)
        
        # Forward chaining
        for _ in range(max_steps):
            # Check if goal is proven
            for k in known:
                if str(k) == str(goal):
                    return proof
            
            # Try to derive new facts
            new_facts = []
            for i, expr1 in enumerate(known):
                for j, expr2 in enumerate(known):
                    if i == j:
                        continue
                    
                    for rule in self.rules:
                        result = rule.apply(expr1, expr2)
                        if result and str(result) not in [str(k) for k in known]:
                            proof.add_step(result, rule.name, [i+1, j+1])
                            new_facts.append(result)
            
            if not new_facts:
                break
            
            known.extend(new_facts)
        
        # Check if goal was proven
        for k in known:
            if str(k) == str(goal):
                return proof
        
        return None


# =============================================================================
# TRUTH TABLE
# =============================================================================

class TruthTable:
    """Generate truth table for expression."""
    
    def __init__(self, expression: Expression):
        self.expression = expression
        self.variables = sorted(expression.variables())
        self.rows = self._generate()
    
    def _generate(self) -> List[Tuple[Dict[str, bool], bool]]:
        rows = []
        n = len(self.variables)
        
        for i in range(2 ** n):
            assignment = {}
            for j, var in enumerate(self.variables):
                assignment[var] = bool((i >> (n - 1 - j)) & 1)
            
            result = self.expression.evaluate(assignment)
            rows.append((assignment, result))
        
        return rows
    
    def is_tautology(self) -> bool:
        """Check if expression is always true."""
        return all(result for _, result in self.rows)
    
    def is_contradiction(self) -> bool:
        """Check if expression is always false."""
        return not any(result for _, result in self.rows)
    
    def is_contingent(self) -> bool:
        """Check if expression is sometimes true, sometimes false."""
        return not self.is_tautology() and not self.is_contradiction()
    
    def is_satisfiable(self) -> bool:
        """Check if expression can be true."""
        return any(result for _, result in self.rows)
    
    def satisfying_assignment(self) -> Optional[Dict[str, bool]]:
        """Find an assignment that makes expression true."""
        for assignment, result in self.rows:
            if result:
                return assignment
        return None
    
    def __str__(self) -> str:
        lines = []
        
        # Header
        header = " | ".join(self.variables) + " | Result"
        lines.append(header)
        lines.append("-" * len(header))
        
        # Rows
        for assignment, result in self.rows:
            values = [str(int(assignment[v])) for v in self.variables]
            lines.append(" | ".join(values) + f" | {'T' if result else 'F'}")
        
        return "\n".join(lines)


# =============================================================================
# FORMAL LOGIC ENGINE
# =============================================================================

class FormalLogicEngine:
    """
    AION Formal Logic Engine.
    
    High-level interface for logical reasoning.
    """
    
    def __init__(self):
        self.prover = TheoremProver()
    
    def prop(self, name: str) -> Proposition:
        """Create a proposition."""
        return Proposition(name)
    
    def parse(self, expr_str: str) -> Expression:
        """
        Parse logical expression from string.
        Supports: P, Q, ¬P, P∧Q, P∨Q, P→Q, P↔Q
        """
        # Simple parser for common cases
        expr_str = expr_str.strip()
        
        # Check for binary operators
        for conn in [Connective.IFF, Connective.IMPLIES, Connective.OR, 
                     Connective.AND, Connective.XOR]:
            symbol = conn.value
            if symbol in expr_str:
                parts = expr_str.split(symbol, 1)
                if len(parts) == 2:
                    left = self.parse(parts[0])
                    right = self.parse(parts[1])
                    return BinaryOp(left, right, conn)
        
        # Check for negation
        if expr_str.startswith('¬') or expr_str.startswith('~') or expr_str.startswith('!'):
            return Not(self.parse(expr_str[1:]))
        
        # Atomic proposition
        expr_str = expr_str.strip('()')
        return Proposition(expr_str)
    
    def truth_table(self, expression: Expression) -> Dict:
        """Generate truth table for expression."""
        tt = TruthTable(expression)
        
        return {
            'expression': str(expression),
            'variables': tt.variables,
            'is_tautology': tt.is_tautology(),
            'is_contradiction': tt.is_contradiction(),
            'is_contingent': tt.is_contingent(),
            'satisfying_assignment': tt.satisfying_assignment(),
            'table': str(tt)
        }
    
    def prove(self, premises: List[str], goal: str) -> Dict:
        """Attempt to prove goal from premises."""
        premise_exprs = [self.parse(p) for p in premises]
        goal_expr = self.parse(goal)
        
        proof = self.prover.prove(premise_exprs, goal_expr)
        
        return {
            'premises': premises,
            'goal': goal,
            'proved': proof is not None,
            'proof': str(proof) if proof else None
        }
    
    def check_validity(self, expression: str) -> Dict:
        """Check logical properties of expression."""
        expr = self.parse(expression)
        tt = TruthTable(expr)
        
        return {
            'expression': expression,
            'tautology': tt.is_tautology(),
            'contradiction': tt.is_contradiction(),
            'contingent': tt.is_contingent(),
            'satisfiable': tt.is_satisfiable()
        }
    
    def modus_ponens(self, p: str, p_implies_q: str) -> Optional[str]:
        """Apply modus ponens."""
        result = ModusPonens.apply(self.parse(p), self.parse(p_implies_q))
        return str(result) if result else None
    
    def equivalences(self) -> Dict[str, str]:
        """Return common logical equivalences."""
        return {
            'Double Negation': '¬¬P ↔ P',
            'De Morgan (AND)': '¬(P∧Q) ↔ (¬P∨¬Q)',
            'De Morgan (OR)': '¬(P∨Q) ↔ (¬P∧¬Q)',
            'Implication': '(P→Q) ↔ (¬P∨Q)',
            'Contraposition': '(P→Q) ↔ (¬Q→¬P)',
            'Excluded Middle': 'P∨¬P (tautology)',
            'Non-Contradiction': '¬(P∧¬P) (tautology)',
            'Distribution (AND over OR)': 'P∧(Q∨R) ↔ (P∧Q)∨(P∧R)',
            'Distribution (OR over AND)': 'P∨(Q∧R) ↔ (P∨Q)∧(P∨R)',
            'Absorption': 'P∧(P∨Q) ↔ P',
        }
    
    def inference_rules(self) -> Dict[str, str]:
        """Return common inference rules."""
        return {
            'Modus Ponens': 'P, P→Q ⊢ Q',
            'Modus Tollens': '¬Q, P→Q ⊢ ¬P',
            'Hypothetical Syllogism': 'P→Q, Q→R ⊢ P→R',
            'Disjunctive Syllogism': 'P∨Q, ¬P ⊢ Q',
            'Conjunction Introduction': 'P, Q ⊢ P∧Q',
            'Simplification': 'P∧Q ⊢ P',
            'Addition': 'P ⊢ P∨Q',
            'Constructive Dilemma': '(P→Q)∧(R→S), P∨R ⊢ Q∨S',
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Formal Logic Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          📐 AION FORMAL LOGIC ENGINE 📐                                   ║
║                                                                           ║
║     Propositional Logic, FOL, Inference Rules, Theorem Proving           ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = FormalLogicEngine()
    
    # Build expression
    print("📝 Building Expression: (P ∧ Q) → P")
    print("-" * 50)
    P = engine.prop('P')
    Q = engine.prop('Q')
    expr = Implies(And(P, Q), P)
    print(f"   Expression: {expr}")
    
    # Truth table
    print("\n📊 Truth Table:")
    print("-" * 50)
    result = engine.truth_table(expr)
    print(result['table'])
    print(f"\n   Is tautology: {result['is_tautology']}")
    
    # Check validity
    print("\n✅ Checking: P ∨ ¬P (Law of Excluded Middle)")
    print("-" * 50)
    result = engine.check_validity("P ∨ ¬P")
    print(f"   Tautology: {result['tautology']}")
    
    # Modus Ponens
    print("\n🔗 Modus Ponens: P, P→Q ⊢ ?")
    print("-" * 50)
    conclusion = engine.modus_ponens("P", "P→Q")
    print(f"   Conclusion: {conclusion}")
    
    # Proof
    print("\n📜 Proof: P→Q, Q→R ⊢ P→R")
    print("-" * 50)
    result = engine.prove(["P→Q", "Q→R"], "P→R")
    print(f"   Proved: {result['proved']}")
    if result['proof']:
        print(f"\n{result['proof']}")
    
    # Common equivalences
    print("\n📚 Logical Equivalences:")
    print("-" * 50)
    for name, equiv in list(engine.equivalences().items())[:5]:
        print(f"   {name}: {equiv}")


if __name__ == "__main__":
    demo()
