"""
AION Optimization Module
========================

Provides DSPy-style automatic optimization for AION agents:
- Signature-based declarative prompts
- Automatic few-shot example generation
- Prompt optimization based on metrics
- A/B testing framework
"""

from .signatures import Signature, InputField, OutputField
from .optimizers import Optimizer, BootstrapFewShot, MIPRO
from .evaluators import Evaluator, MetricRegistry
from .teleprompter import Teleprompter, compile_prompt

__all__ = [
    'Signature',
    'InputField',
    'OutputField',
    'Optimizer',
    'BootstrapFewShot',
    'MIPRO',
    'Evaluator',
    'MetricRegistry',
    'Teleprompter',
    'compile_prompt'
]
