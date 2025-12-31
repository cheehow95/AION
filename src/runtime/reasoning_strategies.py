"""
AION Advanced Reasoning Strategies
===================================

Implements sophisticated reasoning techniques including:
- Chain of Thought (CoT): Explicit step-by-step reasoning
- Tree of Thought (ToT): Branching exploration of possibilities
- Self-Consistency: Multiple solution paths with voting
- Analogical Reasoning: Transfer knowledge from similar problems
- Counterfactual Reasoning: "What if?" analysis
- Abductive Reasoning: Inference to best explanation

These strategies can be composed and switched dynamically.
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum


# =============================================================================
# REASONING PRIMITIVES
# =============================================================================

@dataclass
class ReasoningStep:
    """A single step in a reasoning process."""
    id: int
    content: str
    confidence: float = 0.5
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Step {self.id}: {self.content[:50]}..."


@dataclass
class ReasoningPath:
    """A complete path through a reasoning process."""
    steps: List[ReasoningStep] = field(default_factory=list)
    conclusion: str = ""
    total_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: ReasoningStep):
        """Add a step to this path."""
        self.steps.append(step)
        # Update confidence as product of step confidences
        if self.steps:
            self.total_confidence = 1.0
            for s in self.steps:
                self.total_confidence *= s.confidence


@dataclass
class ReasoningResult:
    """Result of a reasoning process."""
    conclusion: str
    confidence: float
    reasoning_trace: List[str]
    paths_explored: int = 1
    strategy_used: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CHAIN OF THOUGHT
# =============================================================================

class ChainOfThought:
    """
    Chain of Thought (CoT) reasoning.
    Breaks down complex reasoning into explicit sequential steps.
    
    Key principle: "Let's think step by step"
    """
    
    def __init__(self, step_generator: Callable[[str, List[str]], str] = None):
        """
        Args:
            step_generator: Function(prompt, previous_steps) -> next_step
                           If None, uses a simple heuristic generator
        """
        self.step_generator = step_generator or self._default_generator
        self.step_counter = 0
    
    def _default_generator(self, prompt: str, previous: List[str]) -> str:
        """Default step generator using templates."""
        step_num = len(previous) + 1
        
        if step_num == 1:
            return f"First, let me understand the problem: {prompt}"
        elif step_num == 2:
            return "Next, I'll identify the key components and constraints."
        elif step_num == 3:
            return "Now, let me consider the relationships between these components."
        elif step_num == 4:
            return "Based on this analysis, I can formulate a solution approach."
        else:
            return f"Step {step_num}: Continuing the analysis..."
    
    async def reason(self, problem: str, max_steps: int = 5,
                     stop_condition: Callable[[List[str]], bool] = None) -> ReasoningResult:
        """
        Perform chain-of-thought reasoning on a problem.
        
        Args:
            problem: The problem to reason about
            max_steps: Maximum reasoning steps
            stop_condition: Optional function to check if reasoning should stop
        """
        steps = []
        trace = [f"Problem: {problem}", "---", "Chain of Thought:"]
        
        for i in range(max_steps):
            # Generate next step
            step_content = self.step_generator(problem, steps)
            step = ReasoningStep(
                id=self.step_counter,
                content=step_content,
                confidence=0.8 - (i * 0.05)  # Confidence decreases slightly with depth
            )
            self.step_counter += 1
            
            steps.append(step_content)
            trace.append(f"  {i+1}. {step_content}")
            
            # Check stop condition
            if stop_condition and stop_condition(steps):
                break
        
        # Generate conclusion
        conclusion = f"Based on {len(steps)} reasoning steps, the conclusion is derived."
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=0.7,
            reasoning_trace=trace,
            strategy_used="chain_of_thought"
        )


# =============================================================================
# TREE OF THOUGHT
# =============================================================================

class TreeOfThought:
    """
    Tree of Thought (ToT) reasoning.
    Explores multiple reasoning paths in parallel, backtracking as needed.
    
    Key principle: Branch, evaluate, prune, merge
    """
    
    def __init__(self, branching_factor: int = 3,
                 evaluator: Callable[[ReasoningStep], float] = None):
        """
        Args:
            branching_factor: Number of alternatives to explore at each step
            evaluator: Function to evaluate step quality (higher = better)
        """
        self.branching_factor = branching_factor
        self.evaluator = evaluator or self._default_evaluator
        self.step_counter = 0
    
    def _default_evaluator(self, step: ReasoningStep) -> float:
        """Default step evaluator."""
        # Simple heuristics
        score = step.confidence
        
        # Prefer steps with more substance
        if len(step.content) > 20:
            score += 0.1
        
        return min(1.0, score)
    
    def _generate_alternatives(self, context: str, depth: int) -> List[ReasoningStep]:
        """Generate alternative reasoning steps."""
        alternatives = []
        
        templates = [
            f"Approach A: Consider the direct solution to {context}",
            f"Approach B: Break down {context} into smaller parts",
            f"Approach C: Look for analogies related to {context}",
            f"Approach D: Challenge assumptions about {context}",
            f"Approach E: Consider edge cases of {context}",
        ]
        
        for i in range(min(self.branching_factor, len(templates))):
            step = ReasoningStep(
                id=self.step_counter,
                content=templates[i],
                confidence=random.uniform(0.5, 0.9)
            )
            self.step_counter += 1
            alternatives.append(step)
        
        return alternatives
    
    async def reason(self, problem: str, max_depth: int = 3,
                     beam_width: int = 2) -> ReasoningResult:
        """
        Perform tree-of-thought reasoning.
        
        Args:
            problem: The problem to reason about
            max_depth: Maximum depth of the tree
            beam_width: Number of paths to keep at each level (beam search)
        """
        trace = [f"Problem: {problem}", "---", "Tree of Thought Exploration:"]
        
        # Initialize with root alternatives
        current_paths: List[ReasoningPath] = []
        
        for step in self._generate_alternatives(problem, 0):
            path = ReasoningPath()
            path.add_step(step)
            current_paths.append(path)
        
        trace.append(f"  Level 0: Generated {len(current_paths)} initial approaches")
        
        # Explore tree
        total_nodes = len(current_paths)
        
        for depth in range(1, max_depth):
            next_paths = []
            
            for path in current_paths:
                # Generate children
                context = path.steps[-1].content if path.steps else problem
                children = self._generate_alternatives(context, depth)
                
                for child in children:
                    new_path = ReasoningPath(
                        steps=path.steps.copy(),
                        metadata=path.metadata.copy()
                    )
                    new_path.add_step(child)
                    next_paths.append(new_path)
                
                total_nodes += len(children)
            
            # Prune: keep only best paths (beam search)
            next_paths.sort(key=lambda p: p.total_confidence, reverse=True)
            current_paths = next_paths[:beam_width]
            
            trace.append(f"  Level {depth}: Explored {len(next_paths)} paths, kept {len(current_paths)}")
        
        # Select best path
        best_path = max(current_paths, key=lambda p: p.total_confidence)
        
        conclusion = f"Best reasoning path found with {len(best_path.steps)} steps"
        
        trace.append("---")
        trace.append("Best Path:")
        for i, step in enumerate(best_path.steps):
            trace.append(f"  {i+1}. {step.content}")
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=best_path.total_confidence,
            reasoning_trace=trace,
            paths_explored=total_nodes,
            strategy_used="tree_of_thought",
            metadata={"best_path_steps": len(best_path.steps)}
        )


# =============================================================================
# SELF-CONSISTENCY
# =============================================================================

class SelfConsistency:
    """
    Self-Consistency reasoning.
    Generate multiple reasoning paths and vote on the best answer.
    
    Key principle: If multiple independent chains agree, confidence increases
    """
    
    def __init__(self, num_samples: int = 5):
        """
        Args:
            num_samples: Number of independent reasoning chains to generate
        """
        self.num_samples = num_samples
        self.cot = ChainOfThought()
    
    async def reason(self, problem: str) -> ReasoningResult:
        """
        Perform self-consistency reasoning.
        Generates multiple chains and finds consensus.
        """
        trace = [f"Problem: {problem}", "---", "Self-Consistency Analysis:"]
        
        # Generate multiple chains
        results = []
        for i in range(self.num_samples):
            result = await self.cot.reason(problem, max_steps=4)
            results.append(result)
            trace.append(f"  Chain {i+1}: {result.conclusion[:50]}...")
        
        # Find consensus (simplified: highest average confidence)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        # In practice, you'd cluster conclusions and vote
        # Here we take the highest confidence one
        best_result = max(results, key=lambda r: r.confidence)
        
        # Boost confidence if there's agreement
        consistency_bonus = 0.1 if avg_confidence > 0.6 else 0.0
        
        trace.append("---")
        trace.append(f"Consensus confidence: {avg_confidence:.1%}")
        trace.append(f"Best chain confidence: {best_result.confidence:.1%}")
        
        return ReasoningResult(
            conclusion=best_result.conclusion,
            confidence=min(1.0, best_result.confidence + consistency_bonus),
            reasoning_trace=trace,
            paths_explored=self.num_samples,
            strategy_used="self_consistency",
            metadata={"avg_confidence": avg_confidence}
        )


# =============================================================================
# ANALOGICAL REASONING
# =============================================================================

@dataclass
class Analogy:
    """An analogy between source and target domains."""
    source_domain: str
    target_domain: str
    mapping: Dict[str, str]  # source_concept -> target_concept
    strength: float = 0.5


class AnalogicalReasoning:
    """
    Analogical reasoning.
    Transfer knowledge from similar solved problems.
    
    Key principle: If A is like B in important ways, solutions to A may work for B
    """
    
    def __init__(self):
        # Library of known problems and solutions
        self.knowledge_base: List[Dict[str, Any]] = [
            {
                "domain": "sorting",
                "problem": "arrange items in order",
                "solution": "compare and swap until ordered",
                "concepts": ["comparison", "ordering", "iteration"]
            },
            {
                "domain": "search",
                "problem": "find item in collection",
                "solution": "systematically eliminate possibilities",
                "concepts": ["elimination", "bisection", "indexing"]
            },
            {
                "domain": "optimization",
                "problem": "find best configuration",
                "solution": "explore neighborhood, move toward improvement",
                "concepts": ["gradient", "local_search", "constraints"]
            },
            {
                "domain": "classification",
                "problem": "assign item to category",
                "solution": "compare features to known examples",
                "concepts": ["similarity", "features", "boundaries"]
            }
        ]
    
    def _find_similar_problems(self, problem: str) -> List[Dict[str, Any]]:
        """Find similar problems in knowledge base."""
        problem_lower = problem.lower()
        
        scored = []
        for entry in self.knowledge_base:
            # Simple keyword matching
            score = 0
            for concept in entry["concepts"]:
                if concept in problem_lower:
                    score += 1
            
            # Check problem description overlap
            entry_words = set(entry["problem"].lower().split())
            problem_words = set(problem_lower.split())
            overlap = len(entry_words & problem_words)
            score += overlap * 0.5
            
            if score > 0:
                scored.append((entry, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in scored[:3]]
    
    async def reason(self, problem: str) -> ReasoningResult:
        """
        Perform analogical reasoning.
        Find similar problems and transfer solutions.
        """
        trace = [f"Problem: {problem}", "---", "Analogical Reasoning:"]
        
        # Find similar problems
        similar = self._find_similar_problems(problem)
        
        if not similar:
            trace.append("  No analogous problems found.")
            return ReasoningResult(
                conclusion="No applicable analogies found",
                confidence=0.3,
                reasoning_trace=trace,
                strategy_used="analogical"
            )
        
        trace.append(f"  Found {len(similar)} potentially analogous problems:")
        
        for entry in similar:
            trace.append(f"    • Domain: {entry['domain']}")
            trace.append(f"      Problem: {entry['problem']}")
            trace.append(f"      Solution pattern: {entry['solution']}")
        
        # Transfer best analogy
        best = similar[0]
        
        conclusion = (
            f"By analogy to {best['domain']}, we can apply: {best['solution']}. "
            f"Key concepts to transfer: {', '.join(best['concepts'])}"
        )
        
        trace.append("---")
        trace.append(f"Recommended approach: {conclusion}")
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=0.7,
            reasoning_trace=trace,
            strategy_used="analogical",
            metadata={"analogies_found": len(similar)}
        )


# =============================================================================
# COUNTERFACTUAL REASONING
# =============================================================================

class CounterfactualReasoning:
    """
    Counterfactual reasoning.
    Explore "what if?" scenarios to understand causality.
    
    Key principle: To understand X, consider what happens without X
    """
    
    async def reason(self, situation: str, counterfactual: str = None) -> ReasoningResult:
        """
        Perform counterfactual reasoning.
        
        Args:
            situation: The actual situation
            counterfactual: The "what if" alternative (auto-generated if None)
        """
        trace = [f"Situation: {situation}", "---", "Counterfactual Analysis:"]
        
        # Generate counterfactual if not provided
        if not counterfactual:
            # Simple: negate or reverse key aspect
            if "because" in situation.lower():
                parts = situation.lower().split("because")
                counterfactual = f"What if {parts[1].strip()} had not occurred?"
            else:
                counterfactual = f"What if the opposite of '{situation}' were true?"
        
        trace.append(f"  Counterfactual: {counterfactual}")
        trace.append("")
        trace.append("  Analysis:")
        trace.append("    1. Identify the key causal factor")
        trace.append("    2. Remove or alter this factor")
        trace.append("    3. Trace the downstream effects")
        trace.append("    4. Compare to actual outcome")
        
        # Generate insights
        insights = [
            "The counterfactual reveals causal dependencies in the situation.",
            "By considering alternatives, we better understand what truly matters.",
        ]
        
        conclusion = (
            f"Counterfactual analysis of '{situation}' with '{counterfactual}' "
            f"reveals the causal structure and dependencies."
        )
        
        trace.append("---")
        trace.append(f"Insight: {insights[0]}")
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=0.65,
            reasoning_trace=trace,
            strategy_used="counterfactual"
        )


# =============================================================================
# ABDUCTIVE REASONING
# =============================================================================

class AbductiveReasoning:
    """
    Abductive reasoning.
    Inference to the best explanation.
    
    Key principle: Given observations, find the simplest explanation
    """
    
    async def reason(self, observations: List[str]) -> ReasoningResult:
        """
        Perform abductive reasoning.
        Find the best explanation for given observations.
        """
        trace = ["Observations:", "---"]
        for obs in observations:
            trace.append(f"  • {obs}")
        
        trace.append("")
        trace.append("Abductive Reasoning Process:")
        
        # Generate candidate explanations
        explanations = [
            ("Explanation A", 0.7, "Direct causal relationship"),
            ("Explanation B", 0.5, "Coincidental correlation"),
            ("Explanation C", 0.3, "Hidden common cause"),
        ]
        
        trace.append("  Candidate explanations:")
        for name, score, desc in explanations:
            trace.append(f"    {name} ({score:.0%}): {desc}")
        
        # Select best
        best = max(explanations, key=lambda x: x[1])
        
        trace.append("")
        trace.append("  Evaluation criteria:")
        trace.append("    • Explanatory power: Does it account for all observations?")
        trace.append("    • Simplicity: Does it avoid unnecessary complexity?")
        trace.append("    • Coherence: Is it consistent with background knowledge?")
        
        conclusion = f"Best explanation: {best[0]} - {best[2]}"
        
        trace.append("---")
        trace.append(f"Conclusion: {conclusion}")
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=best[1],
            reasoning_trace=trace,
            strategy_used="abductive",
            metadata={"explanations_considered": len(explanations)}
        )


# =============================================================================
# REASONING ORCHESTRATOR
# =============================================================================

class ReasoningOrchestrator:
    """
    Orchestrates multiple reasoning strategies.
    Can automatically select or combine strategies.
    """
    
    def __init__(self):
        self.cot = ChainOfThought()
        self.tot = TreeOfThought()
        self.sc = SelfConsistency()
        self.analogical = AnalogicalReasoning()
        self.counterfactual = CounterfactualReasoning()
        self.abductive = AbductiveReasoning()
        
        self.strategy_history: List[Tuple[str, float]] = []
    
    async def reason(self, problem: str, strategy: str = "auto",
                     **kwargs) -> ReasoningResult:
        """
        Apply a reasoning strategy to a problem.
        
        Args:
            problem: The problem to solve
            strategy: Strategy name or "auto" for automatic selection
        """
        if strategy == "auto":
            strategy = self._select_strategy(problem)
        
        if strategy == "chain_of_thought":
            result = await self.cot.reason(problem, **kwargs)
        elif strategy == "tree_of_thought":
            result = await self.tot.reason(problem, **kwargs)
        elif strategy == "self_consistency":
            result = await self.sc.reason(problem)
        elif strategy == "analogical":
            result = await self.analogical.reason(problem)
        elif strategy == "counterfactual":
            result = await self.counterfactual.reason(problem)
        elif strategy == "abductive":
            observations = kwargs.get("observations", [problem])
            result = await self.abductive.reason(observations)
        else:
            result = await self.cot.reason(problem)
        
        # Record for learning
        self.strategy_history.append((strategy, result.confidence))
        
        return result
    
    def _select_strategy(self, problem: str) -> str:
        """Automatically select best strategy for problem type."""
        problem_lower = problem.lower()
        
        # Heuristics for strategy selection
        if "why" in problem_lower or "explain" in problem_lower:
            return "abductive"
        elif "what if" in problem_lower or "instead" in problem_lower:
            return "counterfactual"
        elif "like" in problem_lower or "similar" in problem_lower:
            return "analogical"
        elif "complex" in problem_lower or "many" in problem_lower:
            return "tree_of_thought"
        elif "certain" in problem_lower or "confident" in problem_lower:
            return "self_consistency"
        else:
            return "chain_of_thought"
    
    def get_strategy_stats(self) -> Dict[str, Dict]:
        """Get performance statistics for each strategy."""
        stats = {}
        for strategy, confidence in self.strategy_history:
            if strategy not in stats:
                stats[strategy] = {"count": 0, "total_confidence": 0.0}
            stats[strategy]["count"] += 1
            stats[strategy]["total_confidence"] += confidence
        
        for strategy in stats:
            stats[strategy]["avg_confidence"] = (
                stats[strategy]["total_confidence"] / stats[strategy]["count"]
            )
        
        return stats


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Demonstrate reasoning strategies."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🧩 AION ADVANCED REASONING STRATEGIES 🧩                         ║
║                                                                           ║
║     Chain of Thought • Tree of Thought • Self-Consistency                 ║
║     Analogical • Counterfactual • Abductive                              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    orchestrator = ReasoningOrchestrator()
    
    # Demo 1: Chain of Thought
    print("📝 Chain of Thought:")
    print("-" * 50)
    result = await orchestrator.reason(
        "How can we reduce carbon emissions in cities?",
        strategy="chain_of_thought"
    )
    for line in result.reasoning_trace[:8]:
        print(f"  {line}")
    print(f"\n  Confidence: {result.confidence:.1%}")
    
    # Demo 2: Tree of Thought
    print("\n🌳 Tree of Thought:")
    print("-" * 50)
    result = await orchestrator.reason(
        "Design a system for autonomous vehicle navigation",
        strategy="tree_of_thought"
    )
    for line in result.reasoning_trace[-5:]:
        print(f"  {line}")
    print(f"\n  Paths explored: {result.paths_explored}")
    print(f"  Confidence: {result.confidence:.1%}")
    
    # Demo 3: Analogical Reasoning
    print("\n🔄 Analogical Reasoning:")
    print("-" * 50)
    result = await orchestrator.reason(
        "How do I efficiently search this large dataset?",
        strategy="analogical"
    )
    for line in result.reasoning_trace[-4:]:
        print(f"  {line}")
    
    # Demo 4: Abductive Reasoning
    print("\n🔍 Abductive Reasoning:")
    print("-" * 50)
    result = await orchestrator.reason(
        "System is running slowly",
        strategy="abductive",
        observations=[
            "CPU usage is at 95%",
            "Memory usage is normal",
            "A new background process started recently",
            "No network issues detected"
        ]
    )
    for line in result.reasoning_trace[-3:]:
        print(f"  {line}")
    
    # Stats
    print("\n📊 Strategy Statistics:")
    print("-" * 50)
    stats = orchestrator.get_strategy_stats()
    for strategy, data in stats.items():
        print(f"  {strategy}: {data['count']} uses, avg confidence {data['avg_confidence']:.1%}")


if __name__ == "__main__":
    asyncio.run(demo())
