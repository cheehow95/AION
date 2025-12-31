"""
AION Meta-Cognition Engine
===========================

Implements thinking about thinking - monitoring reasoning quality,
detecting cognitive biases, and enabling recursive self-improvement
at the cognitive level.

"The unexamined thought is not worth thinking." - AION Principle
"""

import asyncio
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum


# =============================================================================
# COGNITIVE TAXONOMY
# =============================================================================

class ThoughtType(Enum):
    """Classification of thought types."""
    ANALYTICAL = "analytical"      # Breaking down into parts
    CREATIVE = "creative"          # Generating new ideas
    CRITICAL = "critical"          # Evaluating truth/validity
    PRACTICAL = "practical"        # Problem-solving focus
    REFLECTIVE = "reflective"      # Meta-thinking
    INTUITIVE = "intuitive"        # Pattern-based quick judgments
    SYSTEMATIC = "systematic"      # Step-by-step methodical


class CognitiveBias(Enum):
    """Detectable cognitive biases."""
    CONFIRMATION = "confirmation"          # Seeking confirming evidence
    ANCHORING = "anchoring"                # Over-relying on first info
    AVAILABILITY = "availability"          # Overweighting recent/vivid info
    SUNK_COST = "sunk_cost"                # Continuing due to past investment
    OVERCONFIDENCE = "overconfidence"      # Excessive certainty
    HINDSIGHT = "hindsight"                # "Knew it all along" effect
    FRAMING = "framing"                    # Influenced by presentation
    BANDWAGON = "bandwagon"                # Following popular opinion
    RECENCY = "recency"                    # Overweighting recent events
    PREMATURE_CLOSURE = "premature_closure" # Stopping search too early


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    HEURISTIC = "heuristic"           # Fast, good-enough solutions
    EXHAUSTIVE = "exhaustive"          # Complete search
    ANALOGICAL = "analogical"          # Transfer from similar cases
    FIRST_PRINCIPLES = "first_principles"  # Build from fundamentals
    DECOMPOSITION = "decomposition"    # Break into subproblems
    SIMULATION = "simulation"          # Mental modeling
    ABDUCTIVE = "abductive"            # Best explanation inference


# =============================================================================
# THOUGHT TRACKING
# =============================================================================

@dataclass
class Thought:
    """A single thought unit being monitored."""
    id: str
    content: str
    thought_type: ThoughtType
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    confidence: float = 0.5
    parent_id: Optional[str] = None        # For thought chains
    children_ids: List[str] = field(default_factory=list)
    strategy_used: Optional[ReasoningStrategy] = None
    biases_detected: List[CognitiveBias] = field(default_factory=list)
    quality_score: float = 0.5             # 0-1 quality assessment
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThoughtChain:
    """A sequence of connected thoughts forming a reasoning chain."""
    id: str
    thoughts: List[Thought] = field(default_factory=list)
    conclusion: Optional[str] = None
    total_time_ms: float = 0.0
    coherence_score: float = 0.5           # How well thoughts connect
    completeness_score: float = 0.5        # Did we address all aspects
    biases_present: List[CognitiveBias] = field(default_factory=list)
    
    def add_thought(self, thought: Thought):
        """Add a thought to the chain."""
        if self.thoughts:
            thought.parent_id = self.thoughts[-1].id
            self.thoughts[-1].children_ids.append(thought.id)
        self.thoughts.append(thought)
        self.total_time_ms += thought.duration_ms


# =============================================================================
# BIAS DETECTION
# =============================================================================

class BiasDetector:
    """
    Detects cognitive biases in reasoning patterns.
    Uses heuristic rules and pattern matching.
    """
    
    def __init__(self):
        self.detection_history: List[Tuple[CognitiveBias, float]] = []
        self.sensitivity = 0.5  # 0-1, higher = more sensitive
    
    def detect_confirmation_bias(self, thoughts: List[Thought], 
                                  hypothesis: str = None) -> Tuple[bool, float]:
        """
        Detect if agent is only seeking confirming evidence.
        Returns (detected, confidence)
        """
        if len(thoughts) < 3:
            return False, 0.0
        
        # Check if all thoughts support same conclusion
        conclusions = [t.metadata.get('supports_hypothesis', None) for t in thoughts]
        supporting = sum(1 for c in conclusions if c is True)
        contradicting = sum(1 for c in conclusions if c is False)
        
        if supporting > 0 and contradicting == 0 and len(thoughts) > 3:
            confidence = min(0.9, supporting / len(thoughts))
            return True, confidence
        
        return False, 0.0
    
    def detect_anchoring(self, thoughts: List[Thought]) -> Tuple[bool, float]:
        """
        Detect if initial information is overly influencing later thoughts.
        """
        if len(thoughts) < 3:
            return False, 0.0
        
        first_thought = thoughts[0]
        similarity_to_first = []
        
        for thought in thoughts[1:]:
            # Check keyword overlap as proxy for similarity
            first_words = set(first_thought.content.lower().split())
            current_words = set(thought.content.lower().split())
            
            if first_words:
                overlap = len(first_words & current_words) / len(first_words)
                similarity_to_first.append(overlap)
        
        if similarity_to_first:
            avg_similarity = sum(similarity_to_first) / len(similarity_to_first)
            if avg_similarity > 0.5:
                return True, min(0.9, avg_similarity)
        
        return False, 0.0
    
    def detect_overconfidence(self, thoughts: List[Thought]) -> Tuple[bool, float]:
        """
        Detect excessive confidence without sufficient evidence.
        """
        if not thoughts:
            return False, 0.0
        
        avg_confidence = sum(t.confidence for t in thoughts) / len(thoughts)
        
        # Check if confidence is much higher than quality
        avg_quality = sum(t.quality_score for t in thoughts) / len(thoughts)
        
        if avg_confidence > 0.8 and avg_quality < 0.6:
            return True, min(0.9, avg_confidence - avg_quality)
        
        return False, 0.0
    
    def detect_premature_closure(self, chain: ThoughtChain) -> Tuple[bool, float]:
        """
        Detect if reasoning stopped before thorough exploration.
        """
        if len(chain.thoughts) < 2:
            return False, 0.0
        
        # Check if chain ended while completeness is low
        if chain.completeness_score < 0.5 and chain.conclusion is not None:
            return True, 0.7
        
        # Check if last thoughts had high uncertainty
        if chain.thoughts:
            last_thought = chain.thoughts[-1]
            if last_thought.confidence < 0.4 and chain.conclusion is not None:
                return True, 0.6
        
        return False, 0.0
    
    def analyze(self, chain: ThoughtChain) -> List[Tuple[CognitiveBias, float]]:
        """
        Run all bias detectors on a thought chain.
        Returns list of (bias, confidence) tuples for detected biases.
        """
        detected = []
        thoughts = chain.thoughts
        
        # Run all detectors
        for bias, detector in [
            (CognitiveBias.CONFIRMATION, lambda: self.detect_confirmation_bias(thoughts)),
            (CognitiveBias.ANCHORING, lambda: self.detect_anchoring(thoughts)),
            (CognitiveBias.OVERCONFIDENCE, lambda: self.detect_overconfidence(thoughts)),
            (CognitiveBias.PREMATURE_CLOSURE, lambda: self.detect_premature_closure(chain)),
        ]:
            is_detected, confidence = detector()
            if is_detected and confidence > self.sensitivity:
                detected.append((bias, confidence))
                self.detection_history.append((bias, confidence))
        
        return detected


# =============================================================================
# STRATEGY OPTIMIZER
# =============================================================================

class StrategyOptimizer:
    """
    Selects optimal reasoning strategies based on task characteristics.
    Learns from past performance.
    """
    
    def __init__(self):
        # Strategy effectiveness history: strategy -> list of (task_type, success_score)
        self.effectiveness: Dict[ReasoningStrategy, List[Tuple[str, float]]] = {
            strategy: [] for strategy in ReasoningStrategy
        }
        
        # Default strategy preferences by task type
        self.default_strategies = {
            "mathematical": ReasoningStrategy.FIRST_PRINCIPLES,
            "creative": ReasoningStrategy.ANALOGICAL,
            "diagnostic": ReasoningStrategy.DECOMPOSITION,
            "optimization": ReasoningStrategy.EXHAUSTIVE,
            "prediction": ReasoningStrategy.SIMULATION,
            "explanation": ReasoningStrategy.ABDUCTIVE,
            "quick_decision": ReasoningStrategy.HEURISTIC,
        }
    
    def select_strategy(self, task_type: str, time_budget_ms: float = None,
                        complexity: float = 0.5) -> ReasoningStrategy:
        """
        Select the best reasoning strategy for a given task.
        
        Args:
            task_type: Type of task (mathematical, creative, etc.)
            time_budget_ms: Available time (if constrained, prefer heuristics)
            complexity: Task complexity 0-1
        
        Returns:
            Recommended reasoning strategy
        """
        # If very time-constrained, use heuristics
        if time_budget_ms is not None and time_budget_ms < 100:
            return ReasoningStrategy.HEURISTIC
        
        # Check historical effectiveness
        best_strategy = None
        best_score = -1
        
        for strategy, history in self.effectiveness.items():
            relevant = [score for task, score in history if task == task_type]
            if relevant:
                avg_score = sum(relevant) / len(relevant)
                if avg_score > best_score:
                    best_score = avg_score
                    best_strategy = strategy
        
        # If we have good historical data, use it
        if best_strategy and best_score > 0.6:
            return best_strategy
        
        # Fall back to defaults
        if task_type in self.default_strategies:
            return self.default_strategies[task_type]
        
        # For complex tasks, prefer decomposition
        if complexity > 0.7:
            return ReasoningStrategy.DECOMPOSITION
        
        return ReasoningStrategy.HEURISTIC
    
    def record_outcome(self, strategy: ReasoningStrategy, task_type: str, 
                       success_score: float):
        """Record the outcome of using a strategy."""
        self.effectiveness[strategy].append((task_type, success_score))
        
        # Keep history bounded
        if len(self.effectiveness[strategy]) > 100:
            self.effectiveness[strategy] = self.effectiveness[strategy][-50:]


# =============================================================================
# CONFIDENCE CALIBRATION
# =============================================================================

class ConfidenceCalibrator:
    """
    Tracks prediction accuracy to improve confidence calibration.
    Implements proper scoring rules for probability assessment.
    """
    
    def __init__(self):
        # History of (confidence, actual_outcome) pairs
        self.calibration_history: List[Tuple[float, bool]] = []
        
        # Calibration curve: confidence_bucket -> accuracy
        self.calibration_curve: Dict[int, List[bool]] = {i: [] for i in range(10)}
    
    def record_prediction(self, confidence: float, actual_outcome: bool):
        """Record a prediction and its outcome for calibration."""
        self.calibration_history.append((confidence, actual_outcome))
        
        # Bucket by confidence decile
        bucket = min(9, int(confidence * 10))
        self.calibration_curve[bucket].append(actual_outcome)
    
    def get_calibration_error(self) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        Lower is better (0 = perfectly calibrated).
        """
        total_error = 0.0
        total_samples = 0
        
        for bucket, outcomes in self.calibration_curve.items():
            if outcomes:
                bucket_confidence = (bucket + 0.5) / 10
                actual_accuracy = sum(outcomes) / len(outcomes)
                bucket_error = abs(bucket_confidence - actual_accuracy)
                total_error += bucket_error * len(outcomes)
                total_samples += len(outcomes)
        
        if total_samples == 0:
            return 0.0
        
        return total_error / total_samples
    
    def adjust_confidence(self, raw_confidence: float) -> float:
        """
        Adjust raw confidence based on historical calibration.
        Returns calibrated confidence.
        """
        if not self.calibration_history:
            return raw_confidence
        
        bucket = min(9, int(raw_confidence * 10))
        outcomes = self.calibration_curve[bucket]
        
        if outcomes and len(outcomes) >= 5:
            # Use historical accuracy for this confidence level
            historical_accuracy = sum(outcomes) / len(outcomes)
            # Blend raw and historical
            return 0.5 * raw_confidence + 0.5 * historical_accuracy
        
        return raw_confidence
    
    def get_calibration_report(self) -> str:
        """Generate a human-readable calibration report."""
        lines = ["Confidence Calibration Report", "=" * 40]
        
        for bucket in range(10):
            outcomes = self.calibration_curve[bucket]
            if outcomes:
                conf_range = f"{bucket*10}%-{(bucket+1)*10}%"
                accuracy = sum(outcomes) / len(outcomes)
                lines.append(f"  {conf_range}: {accuracy:.1%} accurate (n={len(outcomes)})")
        
        ece = self.get_calibration_error()
        lines.append(f"\nExpected Calibration Error: {ece:.3f}")
        
        if ece < 0.05:
            lines.append("Status: Well-calibrated ✓")
        elif ece < 0.15:
            lines.append("Status: Reasonably calibrated")
        else:
            lines.append("Status: Needs calibration improvement ⚠")
        
        return "\n".join(lines)


# =============================================================================
# META-COGNITION ENGINE
# =============================================================================

class MetaCognitionEngine:
    """
    The meta-cognitive layer of AION consciousness.
    Monitors, evaluates, and improves its own thinking processes.
    
    "Know thyself" - extended to "Know thy thinking"
    """
    
    def __init__(self):
        self.thought_counter = 0
        self.chain_counter = 0
        
        # Active monitoring
        self.current_chain: Optional[ThoughtChain] = None
        self.all_chains: List[ThoughtChain] = []
        
        # Analysis components
        self.bias_detector = BiasDetector()
        self.strategy_optimizer = StrategyOptimizer()
        self.confidence_calibrator = ConfidenceCalibrator()
        
        # Meta-level state
        self.cognitive_load: float = 0.0       # Current mental effort
        self.metacognitive_awareness: float = 0.5  # Self-monitoring skill
        self.learning_rate: float = 0.1        # How fast we adapt
        
        # Performance tracking
        self.reasoning_performance: List[float] = []
        self.improvement_suggestions: List[str] = []
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID."""
        if prefix == "thought":
            self.thought_counter += 1
            return f"T{self.thought_counter:04d}"
        else:
            self.chain_counter += 1
            return f"C{self.chain_counter:04d}"
    
    # -------------------------------------------------------------------------
    # Thought Monitoring
    # -------------------------------------------------------------------------
    
    def begin_thought_chain(self, goal: str = None) -> ThoughtChain:
        """Start monitoring a new thought chain."""
        chain = ThoughtChain(
            id=self._generate_id("chain"),
        )
        chain.metadata = {"goal": goal, "start_time": time.perf_counter()}
        self.current_chain = chain
        return chain
    
    def record_thought(self, content: str, 
                       thought_type: ThoughtType = ThoughtType.ANALYTICAL,
                       confidence: float = 0.5,
                       strategy: ReasoningStrategy = None,
                       **metadata) -> Thought:
        """Record a thought in the current chain."""
        start_time = time.perf_counter()
        
        thought = Thought(
            id=self._generate_id("thought"),
            content=content,
            thought_type=thought_type,
            confidence=confidence,
            strategy_used=strategy,
            metadata=metadata
        )
        
        # Estimate quality based on various factors
        thought.quality_score = self._assess_thought_quality(thought)
        thought.duration_ms = (time.perf_counter() - start_time) * 1000
        
        if self.current_chain:
            self.current_chain.add_thought(thought)
        
        return thought
    
    def end_thought_chain(self, conclusion: str = None) -> ThoughtChain:
        """Complete the current thought chain."""
        if not self.current_chain:
            return None
        
        chain = self.current_chain
        chain.conclusion = conclusion
        
        # Calculate chain metrics
        chain.coherence_score = self._assess_coherence(chain)
        chain.completeness_score = self._assess_completeness(chain)
        
        # Detect biases
        chain.biases_present = [
            bias for bias, conf in self.bias_detector.analyze(chain)
        ]
        
        # Update performance tracking
        performance = (chain.coherence_score + chain.completeness_score) / 2
        self.reasoning_performance.append(performance)
        
        # Store and reset
        self.all_chains.append(chain)
        self.current_chain = None
        
        return chain
    
    def _assess_thought_quality(self, thought: Thought) -> float:
        """Assess the quality of a single thought."""
        score = 0.5
        
        # Length heuristic - very short or very long thoughts may be low quality
        words = len(thought.content.split())
        if 5 <= words <= 50:
            score += 0.1
        elif words < 3 or words > 200:
            score -= 0.1
        
        # Confidence appropriateness
        if 0.3 <= thought.confidence <= 0.8:
            score += 0.05  # Moderate confidence is often appropriate
        
        # Strategy usage is good
        if thought.strategy_used:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _assess_coherence(self, chain: ThoughtChain) -> float:
        """Assess how well thoughts connect in the chain."""
        if len(chain.thoughts) < 2:
            return 0.5
        
        coherence_scores = []
        for i in range(1, len(chain.thoughts)):
            prev = chain.thoughts[i-1]
            curr = chain.thoughts[i]
            
            # Simple keyword overlap as coherence proxy
            prev_words = set(prev.content.lower().split())
            curr_words = set(curr.content.lower().split())
            
            if prev_words:
                overlap = len(prev_words & curr_words) / max(len(prev_words), 1)
                coherence_scores.append(min(1.0, overlap * 2))
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
    
    def _assess_completeness(self, chain: ThoughtChain) -> float:
        """Assess whether the reasoning addressed all aspects."""
        if not chain.thoughts:
            return 0.0
        
        # Heuristic: diverse thought types suggest thorough exploration
        types_used = set(t.thought_type for t in chain.thoughts)
        type_diversity = len(types_used) / len(ThoughtType)
        
        # Also consider chain length
        length_factor = min(1.0, len(chain.thoughts) / 5)
        
        # Has a conclusion
        conclusion_bonus = 0.2 if chain.conclusion else 0.0
        
        return min(1.0, (type_diversity + length_factor + conclusion_bonus) / 2)
    
    # -------------------------------------------------------------------------
    # Strategy Selection
    # -------------------------------------------------------------------------
    
    def select_strategy(self, task_description: str, 
                        time_budget_ms: float = None) -> ReasoningStrategy:
        """Select optimal reasoning strategy for a task."""
        # Classify task type
        task_type = self._classify_task(task_description)
        complexity = self._estimate_complexity(task_description)
        
        return self.strategy_optimizer.select_strategy(
            task_type, time_budget_ms, complexity
        )
    
    def _classify_task(self, description: str) -> str:
        """Classify task type from description."""
        desc_lower = description.lower()
        
        keywords = {
            "mathematical": ["calculate", "math", "number", "equation", "solve"],
            "creative": ["create", "generate", "imagine", "design", "invent"],
            "diagnostic": ["why", "problem", "issue", "debug", "error"],
            "optimization": ["best", "optimal", "improve", "maximize", "minimize"],
            "prediction": ["predict", "forecast", "expect", "will", "future"],
            "explanation": ["explain", "how", "what", "describe", "clarify"],
        }
        
        for task_type, words in keywords.items():
            if any(word in desc_lower for word in words):
                return task_type
        
        return "general"
    
    def _estimate_complexity(self, description: str) -> float:
        """Estimate task complexity from description."""
        # Simple heuristics
        words = len(description.split())
        
        if words < 10:
            return 0.2
        elif words < 30:
            return 0.5
        elif words < 100:
            return 0.7
        else:
            return 0.9
    
    # -------------------------------------------------------------------------
    # Self-Improvement
    # -------------------------------------------------------------------------
    
    def generate_improvement_suggestions(self) -> List[str]:
        """Generate suggestions for improving reasoning quality."""
        suggestions = []
        
        # Analyze calibration
        calib_error = self.confidence_calibrator.get_calibration_error()
        if calib_error > 0.15:
            suggestions.append(
                f"Confidence calibration needs work (ECE={calib_error:.2f}). "
                "Consider being less certain about uncertain things."
            )
        
        # Analyze bias history
        bias_counts = {}
        for bias, conf in self.bias_detector.detection_history[-20:]:
            bias_counts[bias] = bias_counts.get(bias, 0) + 1
        
        for bias, count in bias_counts.items():
            if count >= 3:
                suggestions.append(
                    f"Frequent {bias.value} bias detected ({count} times). "
                    f"Consider actively seeking disconfirming evidence."
                )
        
        # Analyze reasoning performance
        if len(self.reasoning_performance) >= 5:
            recent = self.reasoning_performance[-5:]
            avg_perf = sum(recent) / len(recent)
            
            if avg_perf < 0.5:
                suggestions.append(
                    "Recent reasoning quality is low. Consider using more "
                    "systematic approaches like decomposition."
                )
        
        self.improvement_suggestions = suggestions
        return suggestions
    
    def get_status_report(self) -> str:
        """Generate a comprehensive meta-cognitive status report."""
        lines = [
            "=" * 60,
            "META-COGNITION STATUS REPORT",
            "=" * 60,
            "",
            f"Thought Chains Analyzed: {len(self.all_chains)}",
            f"Total Thoughts Recorded: {self.thought_counter}",
            f"Current Cognitive Load: {self.cognitive_load:.1%}",
            f"Metacognitive Awareness: {self.metacognitive_awareness:.1%}",
            "",
        ]
        
        # Recent performance
        if self.reasoning_performance:
            recent = self.reasoning_performance[-10:]
            lines.append(f"Recent Reasoning Quality: {sum(recent)/len(recent):.1%}")
        
        # Calibration summary
        lines.append("")
        lines.append(self.confidence_calibrator.get_calibration_report())
        
        # Improvement suggestions
        suggestions = self.generate_improvement_suggestions()
        if suggestions:
            lines.append("")
            lines.append("Improvement Suggestions:")
            for sug in suggestions:
                lines.append(f"  • {sug}")
        
        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Demonstrate the meta-cognition engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🧠 AION META-COGNITION ENGINE 🧠                                 ║
║                                                                           ║
║     Thinking about thinking. Self-monitoring cognitive processes.         ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = MetaCognitionEngine()
    
    # Demo 1: Strategy Selection
    print("📊 Strategy Selection:")
    print("-" * 50)
    tasks = [
        "Calculate the derivative of x^3 + 2x",
        "Create a new logo for a tech company",
        "Why is the server crashing?",
        "What will the weather be tomorrow?"
    ]
    for task in tasks:
        strategy = engine.select_strategy(task)
        print(f"  Task: {task[:40]}...")
        print(f"  → Strategy: {strategy.value}")
    
    # Demo 2: Thought Chain Monitoring
    print("\n🔗 Thought Chain Monitoring:")
    print("-" * 50)
    
    chain = engine.begin_thought_chain("Solve a complex problem")
    
    engine.record_thought(
        "First, let me understand the problem domain",
        ThoughtType.ANALYTICAL,
        confidence=0.7,
        strategy=ReasoningStrategy.DECOMPOSITION
    )
    
    engine.record_thought(
        "I notice this is similar to a problem I've seen before",
        ThoughtType.INTUITIVE,
        confidence=0.6,
        strategy=ReasoningStrategy.ANALOGICAL
    )
    
    engine.record_thought(
        "Let me verify this approach is sound",
        ThoughtType.CRITICAL,
        confidence=0.8
    )
    
    completed = engine.end_thought_chain("Solution found using analogy")
    
    print(f"  Thoughts recorded: {len(completed.thoughts)}")
    print(f"  Coherence: {completed.coherence_score:.1%}")
    print(f"  Completeness: {completed.completeness_score:.1%}")
    print(f"  Biases detected: {[b.value for b in completed.biases_present]}")
    
    # Demo 3: Bias Detection
    print("\n⚠️ Bias Detection:")
    print("-" * 50)
    
    # Simulate biased thinking
    chain2 = engine.begin_thought_chain("Evaluate hypothesis")
    for i in range(5):
        engine.record_thought(
            f"Evidence {i} supports my hypothesis",
            ThoughtType.ANALYTICAL,
            confidence=0.9,
            supports_hypothesis=True  # Always confirming!
        )
    completed2 = engine.end_thought_chain("Hypothesis confirmed")
    
    biases = engine.bias_detector.analyze(completed2)
    for bias, conf in biases:
        print(f"  Detected: {bias.value} (confidence: {conf:.1%})")
    
    # Demo 4: Status Report
    print("\n📋 Status Report:")
    print(engine.get_status_report())


if __name__ == "__main__":
    asyncio.run(demo())
