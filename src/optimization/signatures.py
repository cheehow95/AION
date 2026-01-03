"""
AION Signatures
===============

DSPy-style signature definitions for declarative prompt specification.
Signatures define the input/output contract for an agent module.
"""

from typing import Any, Dict, List, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect


@dataclass
class Field:
    """Base class for signature fields."""
    name: str = ""
    description: str = ""
    prefix: str = ""
    format: str = ""
    
    def __set_name__(self, owner, name):
        self.name = name


@dataclass
class InputField(Field):
    """Defines an input field in a signature."""
    required: bool = True
    default: Any = None
    
    def __repr__(self):
        return f"InputField(name='{self.name}', desc='{self.description[:30]}...')" if len(self.description) > 30 else f"InputField(name='{self.name}', desc='{self.description}')"


@dataclass
class OutputField(Field):
    """Defines an output field in a signature."""
    
    def __repr__(self):
        return f"OutputField(name='{self.name}', desc='{self.description[:30]}...')" if len(self.description) > 30 else f"OutputField(name='{self.name}', desc='{self.description}')"


class SignatureMeta(type):
    """Metaclass for Signature that collects field definitions."""
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Collect input and output fields
        inputs = {}
        outputs = {}
        
        for key, value in namespace.items():
            if isinstance(value, InputField):
                value.name = key
                inputs[key] = value
            elif isinstance(value, OutputField):
                value.name = key
                outputs[key] = value
        
        # Also check parent classes
        for base in bases:
            if hasattr(base, '_inputs'):
                inputs = {**base._inputs, **inputs}
            if hasattr(base, '_outputs'):
                outputs = {**base._outputs, **outputs}
        
        cls._inputs = inputs
        cls._outputs = outputs
        
        return cls


class Signature(metaclass=SignatureMeta):
    """
    Base class for signatures that define I/O contracts.
    
    Example:
        class QuestionAnswer(Signature):
            '''Answer questions based on context.'''
            
            question = InputField(description="The question to answer")
            context = InputField(description="Relevant context", required=False)
            answer = OutputField(description="The answer")
            confidence = OutputField(description="Confidence score 0-1")
    """
    
    _inputs: Dict[str, InputField] = {}
    _outputs: Dict[str, OutputField] = {}
    
    def __init__(self, **kwargs):
        # Store input values
        for name, field in self._inputs.items():
            if name in kwargs:
                setattr(self, name, kwargs[name])
            elif field.required and field.default is None:
                raise ValueError(f"Missing required input: {name}")
            else:
                setattr(self, name, field.default)
        
        # Initialize output values
        for name in self._outputs:
            setattr(self, name, None)
    
    @classmethod
    def get_instruction(cls) -> str:
        """Get the signature's instruction from docstring."""
        return cls.__doc__ or ""
    
    @classmethod
    def get_inputs(cls) -> Dict[str, InputField]:
        """Get all input field definitions."""
        return cls._inputs.copy()
    
    @classmethod
    def get_outputs(cls) -> Dict[str, OutputField]:
        """Get all output field definitions."""
        return cls._outputs.copy()
    
    @classmethod
    def to_prompt_template(cls) -> str:
        """Convert signature to a prompt template."""
        lines = []
        
        # Instruction
        if cls.__doc__:
            lines.append(cls.__doc__.strip())
            lines.append("")
        
        # Input section
        lines.append("## Inputs")
        for name, field in cls._inputs.items():
            prefix = f"{field.prefix}: " if field.prefix else f"{name}: "
            lines.append(f"{prefix}{{{name}}}")
            if field.description:
                lines.append(f"  ({field.description})")
        lines.append("")
        
        # Output section
        lines.append("## Expected Outputs")
        for name, field in cls._outputs.items():
            prefix = f"{field.prefix}: " if field.prefix else f"{name}: "
            lines.append(f"{prefix}")
            if field.description:
                lines.append(f"  ({field.description})")
        
        return "\n".join(lines)
    
    def format_prompt(self) -> str:
        """Format the prompt with actual input values."""
        template = self.to_prompt_template()
        
        # Substitute input values
        for name in self._inputs:
            value = getattr(self, name, "")
            template = template.replace(f"{{{name}}}", str(value))
        
        return template
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "inputs": {name: getattr(self, name) for name in self._inputs},
            "outputs": {name: getattr(self, name) for name in self._outputs}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signature':
        """Create from dictionary."""
        instance = cls(**data.get("inputs", {}))
        for name, value in data.get("outputs", {}).items():
            if name in cls._outputs:
                setattr(instance, name, value)
        return instance


# ============ Pre-built Signatures ============

class Predict(Signature):
    """Generate a prediction based on input."""
    
    input = InputField(description="The input to process")
    output = OutputField(description="The predicted output")


class ChainOfThought(Signature):
    """Think step by step before answering."""
    
    question = InputField(description="The question to answer")
    reasoning = OutputField(description="Step-by-step reasoning")
    answer = OutputField(description="The final answer")


class ReAct(Signature):
    """Reason and act in an interleaved fashion."""
    
    task = InputField(description="The task to complete")
    thought = OutputField(description="Current reasoning")
    action = OutputField(description="Action to take")
    observation = OutputField(description="Result of action")
    final_answer = OutputField(description="Final answer")


class ProgramOfThought(Signature):
    """Generate code to solve the problem."""
    
    problem = InputField(description="The problem to solve")
    code = OutputField(description="Python code solution")
    output = OutputField(description="Code execution result")


class SelfConsistency(Signature):
    """Generate multiple answers and aggregate."""
    
    question = InputField(description="The question to answer")
    answers = OutputField(description="List of possible answers")
    consensus = OutputField(description="Most consistent answer")


class MultiHopQA(Signature):
    """Answer questions requiring multiple reasoning hops."""
    
    question = InputField(description="The question requiring multiple hops")
    context = InputField(description="Available context", required=False)
    hops = OutputField(description="Intermediate reasoning steps")
    answer = OutputField(description="The final answer")


# ============ Signature Composition ============

def compose_signatures(*signatures: Type[Signature]) -> Type[Signature]:
    """
    Compose multiple signatures into a pipeline.
    
    The outputs of each signature become inputs to the next.
    """
    if len(signatures) < 2:
        return signatures[0] if signatures else Signature
    
    # Collect all inputs and outputs
    all_inputs = {}
    all_outputs = {}
    
    for sig in signatures:
        all_inputs.update(sig.get_inputs())
        all_outputs.update(sig.get_outputs())
    
    # Outputs that are also inputs are intermediate (remove from final inputs)
    intermediate = set(all_inputs.keys()) & set(all_outputs.keys())
    final_inputs = {k: v for k, v in all_inputs.items() if k not in intermediate}
    
    # Create composed class
    class ComposedSignature(Signature):
        pass
    
    ComposedSignature._inputs = final_inputs
    ComposedSignature._outputs = all_outputs
    ComposedSignature.__doc__ = " → ".join(sig.__name__ for sig in signatures)
    
    return ComposedSignature


def parallel_signatures(*signatures: Type[Signature]) -> Type[Signature]:
    """
    Run multiple signatures in parallel and merge results.
    """
    all_inputs = {}
    all_outputs = {}
    
    for sig in signatures:
        all_inputs.update(sig.get_inputs())
        all_outputs.update(sig.get_outputs())
    
    class ParallelSignature(Signature):
        pass
    
    ParallelSignature._inputs = all_inputs
    ParallelSignature._outputs = all_outputs
    ParallelSignature.__doc__ = " | ".join(sig.__name__ for sig in signatures)
    
    return ParallelSignature
