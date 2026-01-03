"""
AION Teleprompter
=================

Prompt compilation and optimization orchestration.
Compiles signatures with optimizers to create optimized prompts.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import json

from .signatures import Signature
from .optimizers import Optimizer, BootstrapFewShot, Example, OptimizationResult
from .evaluators import Evaluator, Metric, MetricRegistry


@dataclass
class CompiledPrompt:
    """A compiled and optimized prompt."""
    signature: Type[Signature]
    prompt_template: str
    few_shot_examples: List[Example]
    optimizer_used: str
    optimization_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    compiled_at: datetime = field(default_factory=datetime.now)
    
    def format(self, **inputs) -> str:
        """Format the prompt with inputs."""
        prompt = self.prompt_template
        
        for key, value in inputs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        
        return prompt
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature.__name__,
            "prompt_template": self.prompt_template,
            "examples": [e.to_dict() for e in self.few_shot_examples],
            "optimizer": self.optimizer_used,
            "score": self.optimization_score,
            "compiled_at": self.compiled_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], signature: Type[Signature]) -> 'CompiledPrompt':
        return cls(
            signature=signature,
            prompt_template=data["prompt_template"],
            few_shot_examples=[Example.from_dict(e) for e in data.get("examples", [])],
            optimizer_used=data.get("optimizer", "unknown"),
            optimization_score=data.get("score", 0.0)
        )


class Teleprompter:
    """
    Prompt compiler and optimizer orchestrator.
    
    Compiles signatures into optimized prompts using training data
    and evaluation metrics.
    """
    
    def __init__(
        self,
        optimizer: Optional[Optimizer] = None,
        metric: Union[str, Metric, Callable] = "exact_match"
    ):
        self.metric_name = metric if isinstance(metric, str) else "custom"
        
        # Get metric function
        if isinstance(metric, str):
            registry = MetricRegistry()
            self.metric = registry.get(metric)
            if not self.metric:
                raise ValueError(f"Unknown metric: {metric}")
        elif isinstance(metric, Metric):
            self.metric = metric
        else:
            self.metric = metric
        
        # Default to BootstrapFewShot optimizer
        self.optimizer = optimizer or BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            max_rounds=3
        )
        
        # Cache of compiled prompts
        self.compiled_cache: Dict[str, CompiledPrompt] = {}
    
    async def compile(
        self,
        signature: Type[Signature],
        train_data: List[Example] = None,
        eval_data: List[Example] = None,
        teacher: 'Teleprompter' = None
    ) -> CompiledPrompt:
        """
        Compile a signature into an optimized prompt.
        
        Args:
            signature: The signature to compile
            train_data: Training examples for optimization
            eval_data: Evaluation examples for scoring
            teacher: Optional teacher teleprompter for distillation
        
        Returns:
            Compiled and optimized prompt
        """
        train_data = train_data or []
        eval_data = eval_data or train_data  # Use train for eval if not provided
        
        # Check cache
        cache_key = f"{signature.__name__}:{len(train_data)}:{self.metric_name}"
        if cache_key in self.compiled_cache:
            return self.compiled_cache[cache_key]
        
        # Run optimization
        if train_data:
            result = await self.optimizer.optimize(signature, train_data, eval_data)
            prompt_template = result.best_prompt
            score = result.best_score
        else:
            # No training data, use base template
            prompt_template = signature.to_prompt_template()
            score = 0.0
        
        # Create compiled prompt
        compiled = CompiledPrompt(
            signature=signature,
            prompt_template=prompt_template,
            few_shot_examples=train_data[:4],  # Keep top examples
            optimizer_used=self.optimizer.__class__.__name__,
            optimization_score=score
        )
        
        # Cache result
        self.compiled_cache[cache_key] = compiled
        
        return compiled
    
    def compile_sync(
        self,
        signature: Type[Signature],
        train_data: List[Example] = None,
        eval_data: List[Example] = None
    ) -> CompiledPrompt:
        """Synchronous version of compile."""
        return asyncio.run(self.compile(signature, train_data, eval_data))
    
    async def optimize_module(
        self,
        module: 'Module',
        train_data: List[Example],
        eval_data: List[Example] = None
    ) -> 'Module':
        """
        Optimize an AION module's prompts.
        
        Args:
            module: The module to optimize
            train_data: Training examples
            eval_data: Evaluation examples
        
        Returns:
            Optimized module
        """
        eval_data = eval_data or train_data
        
        # Get module's signature
        if hasattr(module, 'signature'):
            compiled = await self.compile(module.signature, train_data, eval_data)
            module.set_compiled_prompt(compiled)
        
        return module
    
    def clear_cache(self):
        """Clear the compiled prompt cache."""
        self.compiled_cache.clear()


async def compile_prompt(
    signature: Type[Signature],
    train_data: List[Example] = None,
    metric: str = "exact_match",
    optimizer: str = "bootstrap_few_shot"
) -> CompiledPrompt:
    """
    Convenience function to compile a prompt.
    
    Args:
        signature: Signature to compile
        train_data: Training examples
        metric: Metric to optimize for
        optimizer: Optimizer to use
    
    Returns:
        Compiled prompt
    """
    registry = MetricRegistry()
    metric_func = registry.get(metric)
    
    if optimizer == "bootstrap_few_shot":
        opt = BootstrapFewShot(metric=metric_func)
    else:
        opt = BootstrapFewShot(metric=metric_func)
    
    teleprompter = Teleprompter(optimizer=opt, metric=metric)
    return await teleprompter.compile(signature, train_data or [])


class Module:
    """
    Base class for AION modules that can be optimized.
    """
    
    signature: Type[Signature] = Signature
    compiled_prompt: Optional[CompiledPrompt] = None
    
    def __init__(self, signature: Type[Signature] = None):
        if signature:
            self.signature = signature
    
    def set_compiled_prompt(self, compiled: CompiledPrompt):
        """Set the compiled prompt."""
        self.compiled_prompt = compiled
    
    def get_prompt(self, **inputs) -> str:
        """Get the formatted prompt."""
        if self.compiled_prompt:
            return self.compiled_prompt.format(**inputs)
        return self.signature.to_prompt_template()
    
    async def forward(self, **inputs) -> Dict[str, Any]:
        """
        Execute the module (to be overridden).
        
        Args:
            **inputs: Input values matching signature
        
        Returns:
            Output values matching signature
        """
        prompt = self.get_prompt(**inputs)
        # In real implementation, would call LLM
        return {"prompt_used": prompt}


class ChainOfThoughtModule(Module):
    """Module that uses chain of thought reasoning."""
    
    async def forward(self, **inputs) -> Dict[str, Any]:
        prompt = self.get_prompt(**inputs)
        prompt += "\n\nLet's think step by step:"
        return {"prompt_used": prompt, "reasoning": "Step by step..."}


class ProgramOfThoughtModule(Module):
    """Module that generates and executes code."""
    
    async def forward(self, **inputs) -> Dict[str, Any]:
        prompt = self.get_prompt(**inputs)
        prompt += "\n\nWrite Python code to solve this:"
        return {"prompt_used": prompt, "code": "# Solution code..."}


class ParallelModule:
    """Execute multiple modules in parallel and aggregate."""
    
    def __init__(self, modules: List[Module]):
        self.modules = modules
    
    async def forward(self, **inputs) -> List[Dict[str, Any]]:
        tasks = [m.forward(**inputs) for m in self.modules]
        results = await asyncio.gather(*tasks)
        return results


class SequentialModule:
    """Execute modules sequentially, passing outputs as inputs."""
    
    def __init__(self, modules: List[Module]):
        self.modules = modules
    
    async def forward(self, **inputs) -> Dict[str, Any]:
        current_inputs = inputs
        
        for module in self.modules:
            result = await module.forward(**current_inputs)
            # Merge result into inputs for next module
            current_inputs = {**current_inputs, **result}
        
        return current_inputs
