"""
AION Optimizers
===============

Automatic prompt optimization algorithms inspired by DSPy.
Optimizes prompts based on training data and evaluation metrics.
"""

import random
import asyncio
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json

from .signatures import Signature


@dataclass
class Example:
    """A training/evaluation example."""
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Example':
        return cls(
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    best_prompt: str
    best_score: float
    trials: List[Dict[str, Any]]
    examples_used: int
    iterations: int
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Optimizer(ABC):
    """Base class for prompt optimizers."""
    
    def __init__(
        self,
        metric: Callable[[Any, Any], float],
        max_iterations: int = 10,
        num_candidates: int = 5
    ):
        self.metric = metric
        self.max_iterations = max_iterations
        self.num_candidates = num_candidates
        self.history: List[Dict[str, Any]] = []
    
    @abstractmethod
    async def optimize(
        self,
        signature: Type[Signature],
        train_data: List[Example],
        eval_data: List[Example]
    ) -> OptimizationResult:
        """Optimize a signature using training data."""
        pass
    
    def _evaluate_prompt(
        self,
        prompt: str,
        predictions: List[Any],
        ground_truth: List[Any]
    ) -> float:
        """Evaluate a prompt using the metric."""
        if not predictions or not ground_truth:
            return 0.0
        
        scores = []
        for pred, truth in zip(predictions, ground_truth):
            try:
                score = self.metric(pred, truth)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0


class BootstrapFewShot(Optimizer):
    """
    Bootstrap few-shot examples to optimize prompts.
    
    Generates few-shot examples from successful predictions and
    iteratively improves the prompt.
    """
    
    def __init__(
        self,
        metric: Callable[[Any, Any], float],
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        max_rounds: int = 1,
        max_errors: int = 5
    ):
        super().__init__(metric)
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
    
    async def optimize(
        self,
        signature: Type[Signature],
        train_data: List[Example],
        eval_data: List[Example]
    ) -> OptimizationResult:
        """Optimize using bootstrap few-shot learning."""
        start_time = datetime.now()
        trials = []
        
        # Start with labeled demos from training data
        labeled_demos = train_data[:self.max_labeled_demos]
        bootstrapped_demos = []
        
        best_prompt = ""
        best_score = 0.0
        
        for round_idx in range(self.max_rounds):
            # Create prompt with current demos
            prompt = self._create_prompt(signature, labeled_demos, bootstrapped_demos)
            
            # Evaluate on eval data
            predictions = []
            ground_truth = []
            
            for example in eval_data:
                # Simulate prediction (in real impl, would call LLM)
                pred = await self._predict(prompt, example.inputs)
                predictions.append(pred)
                ground_truth.append(example.outputs)
            
            # Calculate score
            score = self._evaluate_prompt(prompt, predictions, ground_truth)
            
            trials.append({
                "round": round_idx,
                "prompt_length": len(prompt),
                "num_demos": len(labeled_demos) + len(bootstrapped_demos),
                "score": score
            })
            
            if score > best_score:
                best_score = score
                best_prompt = prompt
            
            # Bootstrap new demos from successful predictions
            if round_idx < self.max_rounds - 1:
                successful = [
                    Example(inputs=eval_data[i].inputs, outputs=predictions[i])
                    for i, (p, t) in enumerate(zip(predictions, ground_truth))
                    if self.metric(p, t) > 0.8
                ][:self.max_bootstrapped_demos]
                bootstrapped_demos = successful
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            trials=trials,
            examples_used=len(train_data) + len(eval_data),
            iterations=self.max_rounds,
            duration_seconds=duration
        )
    
    def _create_prompt(
        self,
        signature: Type[Signature],
        labeled_demos: List[Example],
        bootstrapped_demos: List[Example]
    ) -> str:
        """Create a prompt with demonstrations."""
        parts = []
        
        # Instruction
        parts.append(f"# Task: {signature.get_instruction()}")
        parts.append("")
        
        # Demonstrations
        all_demos = labeled_demos + bootstrapped_demos
        if all_demos:
            parts.append("## Examples")
            for i, demo in enumerate(all_demos):
                parts.append(f"\n### Example {i+1}")
                for key, value in demo.inputs.items():
                    parts.append(f"Input ({key}): {value}")
                for key, value in demo.outputs.items():
                    parts.append(f"Output ({key}): {value}")
        
        parts.append("")
        parts.append("## Now solve:")
        
        return "\n".join(parts)
    
    async def _predict(self, prompt: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction (placeholder for actual LLM call)."""
        # In real implementation, this would call the LLM
        return {"predicted": True}


class MIPRO(Optimizer):
    """
    Multi-Instruction Prompt Optimization.
    
    Generates multiple instruction candidates and evaluates them
    to find the optimal instruction set.
    """
    
    def __init__(
        self,
        metric: Callable[[Any, Any], float],
        num_candidates: int = 10,
        num_trials: int = 100,
        temperature: float = 0.7
    ):
        super().__init__(metric, num_candidates=num_candidates)
        self.num_trials = num_trials
        self.temperature = temperature
    
    async def optimize(
        self,
        signature: Type[Signature],
        train_data: List[Example],
        eval_data: List[Example]
    ) -> OptimizationResult:
        """Optimize using MIPRO algorithm."""
        start_time = datetime.now()
        trials = []
        
        # Generate instruction candidates
        candidates = await self._generate_candidates(signature, train_data)
        
        best_prompt = ""
        best_score = 0.0
        
        for trial_idx in range(min(self.num_trials, len(candidates))):
            candidate = candidates[trial_idx % len(candidates)]
            
            # Combine with few-shot examples
            demos = random.sample(train_data, min(3, len(train_data)))
            prompt = self._create_prompt(candidate, demos, signature)
            
            # Evaluate
            predictions = []
            ground_truth = []
            
            for example in eval_data:
                pred = await self._predict(prompt, example.inputs)
                predictions.append(pred)
                ground_truth.append(example.outputs)
            
            score = self._evaluate_prompt(prompt, predictions, ground_truth)
            
            trials.append({
                "trial": trial_idx,
                "instruction": candidate[:50] + "...",
                "score": score
            })
            
            if score > best_score:
                best_score = score
                best_prompt = prompt
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            trials=trials,
            examples_used=len(train_data) + len(eval_data),
            iterations=len(trials),
            duration_seconds=duration
        )
    
    async def _generate_candidates(
        self,
        signature: Type[Signature],
        train_data: List[Example]
    ) -> List[str]:
        """Generate instruction candidates."""
        base_instruction = signature.get_instruction()
        
        # Generate variations (in real impl, would use LLM)
        candidates = [base_instruction]
        
        # Add variations
        variations = [
            f"You are an expert. {base_instruction}",
            f"Think carefully and {base_instruction.lower()}",
            f"Step by step, {base_instruction.lower()}",
            f"Given the context, {base_instruction.lower()}",
            f"Analyze the input and {base_instruction.lower()}"
        ]
        candidates.extend(variations)
        
        return candidates
    
    def _create_prompt(
        self,
        instruction: str,
        demos: List[Example],
        signature: Type[Signature]
    ) -> str:
        """Create a prompt with instruction and demos."""
        parts = [f"# {instruction}", ""]
        
        if demos:
            parts.append("## Examples")
            for i, demo in enumerate(demos):
                parts.append(f"\n### Example {i+1}")
                for key, value in demo.inputs.items():
                    parts.append(f"Input ({key}): {value}")
                for key, value in demo.outputs.items():
                    parts.append(f"Output ({key}): {value}")
        
        parts.append("\n## Your turn:")
        
        return "\n".join(parts)
    
    async def _predict(self, prompt: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction (placeholder)."""
        return {"predicted": True}


class RandomSearch(Optimizer):
    """Simple random search over prompt variations."""
    
    async def optimize(
        self,
        signature: Type[Signature],
        train_data: List[Example],
        eval_data: List[Example]
    ) -> OptimizationResult:
        """Random search optimization."""
        start_time = datetime.now()
        trials = []
        
        best_prompt = signature.to_prompt_template()
        best_score = 0.0
        
        for i in range(self.max_iterations):
            # Generate random variation
            num_demos = random.randint(0, min(5, len(train_data)))
            demos = random.sample(train_data, num_demos) if train_data else []
            
            prompt = self._create_prompt(signature, demos)
            
            # Evaluate
            score = random.random()  # Placeholder
            
            trials.append({
                "iteration": i,
                "num_demos": num_demos,
                "score": score
            })
            
            if score > best_score:
                best_score = score
                best_prompt = prompt
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            trials=trials,
            examples_used=len(train_data) + len(eval_data),
            iterations=self.max_iterations,
            duration_seconds=duration
        )
    
    def _create_prompt(
        self,
        signature: Type[Signature],
        demos: List[Example]
    ) -> str:
        """Create prompt with random demo selection."""
        parts = [signature.to_prompt_template(), ""]
        
        if demos:
            parts.append("## Examples")
            for demo in demos:
                for key, value in demo.inputs.items():
                    parts.append(f"{key}: {value}")
                parts.append("---")
        
        return "\n".join(parts)


class BayesianOptimizer(Optimizer):
    """Bayesian optimization for prompt selection."""
    
    def __init__(
        self,
        metric: Callable[[Any, Any], float],
        exploration_weight: float = 0.1
    ):
        super().__init__(metric)
        self.exploration_weight = exploration_weight
        self.observations: List[tuple] = []
    
    async def optimize(
        self,
        signature: Type[Signature],
        train_data: List[Example],
        eval_data: List[Example]
    ) -> OptimizationResult:
        """Bayesian optimization."""
        start_time = datetime.now()
        trials = []
        
        best_prompt = signature.to_prompt_template()
        best_score = 0.0
        
        # Define parameter space
        num_demos_options = list(range(min(5, len(train_data) + 1)))
        instruction_styles = ["formal", "casual", "step_by_step", "expert"]
        
        for i in range(self.max_iterations):
            # Sample parameters (with exploration)
            if random.random() < self.exploration_weight or not self.observations:
                # Explore
                num_demos = random.choice(num_demos_options) if num_demos_options else 0
                style = random.choice(instruction_styles)
            else:
                # Exploit best known
                best_obs = max(self.observations, key=lambda x: x[2])
                num_demos, style = best_obs[0], best_obs[1]
            
            # Create and evaluate prompt
            demos = train_data[:num_demos] if train_data else []
            prompt = self._create_styled_prompt(signature, demos, style)
            
            score = random.random()  # Placeholder
            
            self.observations.append((num_demos, style, score))
            
            trials.append({
                "iteration": i,
                "num_demos": num_demos,
                "style": style,
                "score": score
            })
            
            if score > best_score:
                best_score = score
                best_prompt = prompt
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            trials=trials,
            examples_used=len(train_data) + len(eval_data),
            iterations=self.max_iterations,
            duration_seconds=duration
        )
    
    def _create_styled_prompt(
        self,
        signature: Type[Signature],
        demos: List[Example],
        style: str
    ) -> str:
        """Create prompt with specific style."""
        instruction = signature.get_instruction()
        
        style_prefixes = {
            "formal": "Please complete the following task: ",
            "casual": "Hey, can you help me with this? ",
            "step_by_step": "Think step by step and ",
            "expert": "As an expert in this field, "
        }
        
        prefix = style_prefixes.get(style, "")
        parts = [f"# {prefix}{instruction}"]
        
        if demos:
            parts.append("\n## Examples")
            for demo in demos:
                parts.append(json.dumps(demo.to_dict(), indent=2))
        
        return "\n".join(parts)
