"""
AION Evaluators
===============

Evaluation metrics and frameworks for measuring agent performance.
Used by optimizers to score prompt candidates.
"""

import re
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""
    score: float
    metric_name: str
    predictions: List[Any]
    ground_truth: List[Any]
    per_example_scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def accuracy(self) -> float:
        """Percentage of perfect scores."""
        if not self.per_example_scores:
            return 0.0
        return sum(1 for s in self.per_example_scores if s >= 1.0) / len(self.per_example_scores)


class Metric:
    """Base class for evaluation metrics."""
    
    name: str = "base_metric"
    
    def __call__(self, prediction: Any, ground_truth: Any) -> float:
        """Compute metric score between prediction and ground truth."""
        raise NotImplementedError


class ExactMatch(Metric):
    """Exact string match metric."""
    
    name = "exact_match"
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def __call__(self, prediction: Any, ground_truth: Any) -> float:
        pred_str = str(prediction)
        truth_str = str(ground_truth)
        
        if self.normalize:
            pred_str = pred_str.strip().lower()
            truth_str = truth_str.strip().lower()
        
        return 1.0 if pred_str == truth_str else 0.0


class F1Score(Metric):
    """Token-level F1 score."""
    
    name = "f1"
    
    def __call__(self, prediction: Any, ground_truth: Any) -> float:
        pred_tokens = set(str(prediction).lower().split())
        truth_tokens = set(str(ground_truth).lower().split())
        
        if not pred_tokens or not truth_tokens:
            return 0.0 if pred_tokens != truth_tokens else 1.0
        
        common = pred_tokens & truth_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(truth_tokens) if truth_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)


class ContainsMatch(Metric):
    """Check if prediction contains ground truth."""
    
    name = "contains"
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def __call__(self, prediction: Any, ground_truth: Any) -> float:
        pred_str = str(prediction)
        truth_str = str(ground_truth)
        
        if self.normalize:
            pred_str = pred_str.lower()
            truth_str = truth_str.lower()
        
        return 1.0 if truth_str in pred_str else 0.0


class NumericAccuracy(Metric):
    """Numeric comparison with tolerance."""
    
    name = "numeric_accuracy"
    
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
    
    def __call__(self, prediction: Any, ground_truth: Any) -> float:
        try:
            pred_num = self._extract_number(prediction)
            truth_num = self._extract_number(ground_truth)
            
            if pred_num is None or truth_num is None:
                return 0.0
            
            if truth_num == 0:
                return 1.0 if abs(pred_num) < self.tolerance else 0.0
            
            error = abs(pred_num - truth_num) / abs(truth_num)
            return 1.0 if error <= self.tolerance else max(0, 1 - error)
            
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_number(self, value: Any) -> Optional[float]:
        """Extract a number from various formats."""
        if isinstance(value, (int, float)):
            return float(value)
        
        text = str(value)
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[-1])  # Take last number found
        return None


class ListMatch(Metric):
    """Compare lists with order sensitivity option."""
    
    name = "list_match"
    
    def __init__(self, order_sensitive: bool = False):
        self.order_sensitive = order_sensitive
    
    def __call__(self, prediction: Any, ground_truth: Any) -> float:
        pred_list = prediction if isinstance(prediction, list) else [prediction]
        truth_list = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        
        if self.order_sensitive:
            if len(pred_list) != len(truth_list):
                return 0.0
            matches = sum(1 for p, t in zip(pred_list, truth_list) if str(p) == str(t))
            return matches / len(truth_list)
        else:
            pred_set = set(str(x) for x in pred_list)
            truth_set = set(str(x) for x in truth_list)
            
            if not truth_set:
                return 1.0 if not pred_set else 0.0
            
            return len(pred_set & truth_set) / len(truth_set)


class SemanticSimilarity(Metric):
    """Semantic similarity using embeddings (simplified)."""
    
    name = "semantic_similarity"
    
    def __call__(self, prediction: Any, ground_truth: Any) -> float:
        # Simplified: use Jaccard similarity on n-grams
        pred_ngrams = self._get_ngrams(str(prediction), 3)
        truth_ngrams = self._get_ngrams(str(ground_truth), 3)
        
        if not pred_ngrams or not truth_ngrams:
            return 1.0 if not pred_ngrams and not truth_ngrams else 0.0
        
        intersection = len(pred_ngrams & truth_ngrams)
        union = len(pred_ngrams | truth_ngrams)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_ngrams(self, text: str, n: int) -> set:
        """Get character n-grams from text."""
        text = text.lower().strip()
        return {text[i:i+n] for i in range(len(text) - n + 1)}


class CompositeMetric(Metric):
    """Combine multiple metrics with weights."""
    
    def __init__(self, metrics: List[tuple[Metric, float]]):
        self.metrics = metrics
        total_weight = sum(w for _, w in metrics)
        self.metrics = [(m, w/total_weight) for m, w in metrics]
        self.name = "+".join(m.name for m, _ in self.metrics)
    
    def __call__(self, prediction: Any, ground_truth: Any) -> float:
        total = 0.0
        for metric, weight in self.metrics:
            total += metric(prediction, ground_truth) * weight
        return total


class MetricRegistry:
    """Registry of available metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register built-in metrics."""
        self.register(ExactMatch())
        self.register(F1Score())
        self.register(ContainsMatch())
        self.register(NumericAccuracy())
        self.register(ListMatch())
        self.register(SemanticSimilarity())
    
    def register(self, metric: Metric):
        """Register a new metric."""
        self.metrics[metric.name] = metric
    
    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self.metrics.get(name)
    
    def list_metrics(self) -> List[str]:
        """List all registered metrics."""
        return list(self.metrics.keys())


class Evaluator:
    """
    Evaluator for running metrics on predictions.
    """
    
    def __init__(self, metric: Union[str, Metric, Callable] = "exact_match"):
        self.registry = MetricRegistry()
        
        if isinstance(metric, str):
            self.metric = self.registry.get(metric)
            if not self.metric:
                raise ValueError(f"Unknown metric: {metric}")
        elif isinstance(metric, Metric):
            self.metric = metric
        else:
            # Wrap callable as metric
            self.metric = self._wrap_callable(metric)
    
    def _wrap_callable(self, func: Callable) -> Metric:
        """Wrap a callable as a Metric."""
        class WrappedMetric(Metric):
            name = "custom"
            def __call__(self, pred, truth):
                return func(pred, truth)
        return WrappedMetric()
    
    def evaluate(
        self,
        predictions: List[Any],
        ground_truth: List[Any]
    ) -> EvaluationResult:
        """Evaluate predictions against ground truth."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        scores = []
        for pred, truth in zip(predictions, ground_truth):
            try:
                score = self.metric(pred, truth)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return EvaluationResult(
            score=avg_score,
            metric_name=self.metric.name,
            predictions=predictions,
            ground_truth=ground_truth,
            per_example_scores=scores
        )
    
    def evaluate_single(self, prediction: Any, ground_truth: Any) -> float:
        """Evaluate a single prediction."""
        return self.metric(prediction, ground_truth)


class ABTestEvaluator:
    """
    A/B testing for comparing prompt variants.
    """
    
    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ["exact_match", "f1"]
        self.registry = MetricRegistry()
        self.results: Dict[str, List[Dict]] = defaultdict(list)
    
    def record_result(
        self,
        variant: str,
        prediction: Any,
        ground_truth: Any
    ):
        """Record a result for a variant."""
        scores = {}
        for metric_name in self.metrics:
            metric = self.registry.get(metric_name)
            if metric:
                scores[metric_name] = metric(prediction, ground_truth)
        
        self.results[variant].append(scores)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all variants."""
        stats = {}
        
        for variant, results in self.results.items():
            variant_stats = {}
            for metric_name in self.metrics:
                scores = [r.get(metric_name, 0) for r in results]
                if scores:
                    variant_stats[f"{metric_name}_mean"] = sum(scores) / len(scores)
                    variant_stats[f"{metric_name}_count"] = len(scores)
            stats[variant] = variant_stats
        
        return stats
    
    def get_winner(self, metric: str = "exact_match") -> Optional[str]:
        """Get the winning variant based on a metric."""
        stats = self.get_statistics()
        
        best_variant = None
        best_score = -1
        
        for variant, variant_stats in stats.items():
            score = variant_stats.get(f"{metric}_mean", 0)
            if score > best_score:
                best_score = score
                best_variant = variant
        
        return best_variant
    
    def is_significant(
        self,
        variant_a: str,
        variant_b: str,
        metric: str = "exact_match",
        threshold: float = 0.05
    ) -> bool:
        """Check if difference between variants is statistically significant."""
        results_a = [r.get(metric, 0) for r in self.results.get(variant_a, [])]
        results_b = [r.get(metric, 0) for r in self.results.get(variant_b, [])]
        
        if len(results_a) < 30 or len(results_b) < 30:
            return False  # Not enough data
        
        mean_a = sum(results_a) / len(results_a)
        mean_b = sum(results_b) / len(results_b)
        
        # Simplified significance check
        diff = abs(mean_a - mean_b)
        return diff > threshold
