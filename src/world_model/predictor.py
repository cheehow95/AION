"""
AION Outcome Predictor
======================

Predictive modeling for agent actions.
Uses world model to forecast outcomes with confidence.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

from .state_graph import StateGraph, Entity, EntityType
from .causal_engine import CausalEngine, CausalResult
from .simulator import WorldSimulator, Scenario, SimulationResult


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class Prediction:
    """A prediction about future state."""
    target: str  # What is being predicted
    predicted_value: Any
    current_value: Any
    confidence: float
    reasoning: str
    supporting_evidence: List[str] = field(default_factory=list)
    alternatives: List[Tuple[Any, float]] = field(default_factory=list)
    time_horizon: Optional[str] = None  # "immediate", "short", "medium", "long"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "predicted_value": self.predicted_value,
            "current_value": self.current_value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": self.supporting_evidence,
            "alternatives": [{"value": v, "probability": p} for v, p in self.alternatives],
            "time_horizon": self.time_horizon
        }
    
    @property
    def confidence_level(self) -> PredictionConfidence:
        """Get categorical confidence level."""
        for level in reversed(list(PredictionConfidence)):
            if self.confidence >= level.value:
                return level
        return PredictionConfidence.VERY_LOW


@dataclass
class PredictionPlan:
    """A plan of predicted outcomes for action sequence."""
    actions: List[Dict[str, Any]]
    predictions: List[Prediction]
    overall_success_probability: float
    critical_points: List[int]  # Indices of high-risk steps
    recommended_checkpoints: List[int]
    estimated_duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": self.actions,
            "predictions": [p.to_dict() for p in self.predictions],
            "success_probability": self.overall_success_probability,
            "critical_points": self.critical_points,
            "checkpoints": self.recommended_checkpoints,
            "estimated_duration": self.estimated_duration
        }


class OutcomePredictor:
    """
    Predicts outcomes of actions before execution.
    
    Uses:
    - Causal engine for effect prediction
    - Simulator for multi-step forecasting
    - Historical patterns for confidence
    """
    
    def __init__(
        self,
        causal_engine: CausalEngine = None,
        simulator: WorldSimulator = None
    ):
        self.causal_engine = causal_engine or CausalEngine()
        self.simulator = simulator or WorldSimulator(self.causal_engine)
        
        # Historical predictions for calibration
        self.prediction_history: List[Dict[str, Any]] = []
        self.calibration_data: Dict[str, List[Tuple[float, bool]]] = {}
    
    def predict_action(
        self,
        action: Dict[str, Any],
        current_state: Dict[str, Any],
        targets: List[str] = None
    ) -> List[Prediction]:
        """
        Predict outcomes of a single action.
        
        Args:
            action: The action to predict
            current_state: Current world state
            targets: Specific state variables to predict (None = all affected)
        
        Returns:
            List of predictions
        """
        predictions = []
        
        # Get causal predictions
        result = self.causal_engine.predict(action, current_state)
        
        # Build predictions for each effect
        affected_keys = set()
        for rule, effect in result.effects:
            for key, value in effect.items():
                if targets and key not in targets:
                    continue
                
                affected_keys.add(key)
                
                predictions.append(Prediction(
                    target=key,
                    predicted_value=value,
                    current_value=current_state.get(key),
                    confidence=self._calculate_confidence(rule, current_state),
                    reasoning=f"Based on rule '{rule.name}': {rule.description or str(rule.condition)}",
                    supporting_evidence=[str(rule.condition)],
                    time_horizon="immediate"
                ))
        
        # Handle targets not affected by any rule
        if targets:
            for target in targets:
                if target not in affected_keys:
                    predictions.append(Prediction(
                        target=target,
                        predicted_value=current_state.get(target),
                        current_value=current_state.get(target),
                        confidence=0.9,
                        reasoning="No causal effects predicted - value unchanged",
                        time_horizon="immediate"
                    ))
        
        return predictions
    
    def predict_sequence(
        self,
        actions: List[Dict[str, Any]],
        initial_state: Dict[str, Any],
        targets: List[str] = None
    ) -> PredictionPlan:
        """
        Predict outcomes of an action sequence.
        
        Args:
            actions: Sequence of actions
            initial_state: Starting state
            targets: Variables to track
        
        Returns:
            PredictionPlan with full forecast
        """
        predictions = []
        critical_points = []
        checkpoints = [0]  # Always checkpoint start
        
        current_state = initial_state.copy()
        cumulative_confidence = 1.0
        
        for i, action in enumerate(actions):
            # Predict this action
            action_predictions = self.predict_action(action, current_state, targets)
            
            # Track lowest confidence prediction
            min_confidence = min((p.confidence for p in action_predictions), default=1.0)
            cumulative_confidence *= min_confidence
            
            # Mark critical points (low confidence)
            if min_confidence < 0.5:
                critical_points.append(i)
                checkpoints.append(i)
            
            # Update state with predicted values
            for pred in action_predictions:
                current_state[pred.target] = pred.predicted_value
            
            predictions.extend(action_predictions)
        
        checkpoints.append(len(actions) - 1)  # Checkpoint end
        
        return PredictionPlan(
            actions=actions,
            predictions=predictions,
            overall_success_probability=cumulative_confidence,
            critical_points=critical_points,
            recommended_checkpoints=sorted(set(checkpoints))
        )
    
    async def predict_with_simulation(
        self,
        action: Dict[str, Any],
        current_state: Dict[str, Any],
        num_simulations: int = 10
    ) -> Prediction:
        """
        Use simulation for more accurate prediction.
        
        Runs multiple simulations to estimate outcome distribution.
        """
        outcomes: Dict[str, List[Any]] = {}
        
        scenario = Scenario(
            name="prediction_simulation",
            initial_state=current_state,
            actions=[action],
            max_steps=1
        )
        
        for _ in range(num_simulations):
            result = await self.simulator.simulate(scenario)
            
            for key, value in result.final_state.items():
                if key not in outcomes:
                    outcomes[key] = []
                outcomes[key].append(value)
        
        # Aggregate predictions
        primary_target = list(action.keys())[0] if action else "state"
        values = outcomes.get(primary_target, [])
        
        if values:
            # Find most common value
            value_counts = {}
            for v in values:
                v_str = str(v)
                value_counts[v_str] = value_counts.get(v_str, 0) + 1
            
            most_common = max(value_counts, key=value_counts.get)
            confidence = value_counts[most_common] / len(values)
            
            # Build alternatives
            alternatives = [
                (v, value_counts[v] / len(values))
                for v in value_counts
                if v != most_common
            ]
            
            return Prediction(
                target=primary_target,
                predicted_value=most_common,
                current_value=current_state.get(primary_target),
                confidence=confidence,
                reasoning=f"Based on {num_simulations} simulations",
                alternatives=alternatives,
                time_horizon="immediate"
            )
        
        return Prediction(
            target=primary_target,
            predicted_value=None,
            current_value=current_state.get(primary_target),
            confidence=0.0,
            reasoning="No simulation data available",
            time_horizon="immediate"
        )
    
    def compare_actions(
        self,
        actions: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        optimization_target: str
    ) -> Dict[str, Any]:
        """
        Compare multiple possible actions.
        
        Args:
            actions: List of possible actions
            current_state: Current state
            optimization_target: Variable to optimize
        
        Returns:
            Comparison with recommendation
        """
        comparisons = []
        
        for action in actions:
            predictions = self.predict_action(action, current_state, [optimization_target])
            
            target_pred = None
            for p in predictions:
                if p.target == optimization_target:
                    target_pred = p
                    break
            
            comparisons.append({
                "action": action,
                "predicted_value": target_pred.predicted_value if target_pred else None,
                "confidence": target_pred.confidence if target_pred else 0.0,
                "expected_value": self._calculate_expected_value(target_pred) if target_pred else 0
            })
        
        # Sort by expected value
        comparisons.sort(key=lambda x: x["expected_value"], reverse=True)
        
        return {
            "comparisons": comparisons,
            "recommended_action": comparisons[0]["action"] if comparisons else None,
            "optimization_target": optimization_target
        }
    
    def record_outcome(
        self,
        prediction: Prediction,
        actual_value: Any
    ):
        """
        Record actual outcome for calibration.
        
        Helps improve future confidence estimates.
        """
        was_correct = prediction.predicted_value == actual_value
        
        # Store for calibration
        if prediction.target not in self.calibration_data:
            self.calibration_data[prediction.target] = []
        
        self.calibration_data[prediction.target].append((
            prediction.confidence,
            was_correct
        ))
        
        self.prediction_history.append({
            "target": prediction.target,
            "predicted": prediction.predicted_value,
            "actual": actual_value,
            "confidence": prediction.confidence,
            "correct": was_correct,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_calibration_metrics(self) -> Dict[str, float]:
        """
        Get prediction calibration metrics.
        
        Good calibration means: 80% confidence predictions are right 80% of the time.
        """
        if not self.prediction_history:
            return {"total_predictions": 0}
        
        total = len(self.prediction_history)
        correct = sum(1 for p in self.prediction_history if p["correct"])
        
        # Calculate calibration error
        bins = {i/10: [] for i in range(11)}
        for p in self.prediction_history:
            bin_key = round(p["confidence"], 1)
            bins[bin_key].append(1 if p["correct"] else 0)
        
        calibration_error = 0.0
        bins_used = 0
        for conf, results in bins.items():
            if results:
                actual_accuracy = sum(results) / len(results)
                calibration_error += abs(conf - actual_accuracy)
                bins_used += 1
        
        avg_calibration_error = calibration_error / bins_used if bins_used else 0
        
        return {
            "total_predictions": total,
            "accuracy": correct / total,
            "avg_confidence": sum(p["confidence"] for p in self.prediction_history) / total,
            "calibration_error": avg_calibration_error,
            "well_calibrated": avg_calibration_error < 0.1
        }
    
    def _calculate_confidence(
        self,
        rule: Any,
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence for a prediction."""
        base_confidence = rule.strength.value if hasattr(rule, 'strength') else 0.5
        
        # Adjust based on calibration data
        if rule.name in self.calibration_data:
            calibration = self.calibration_data[rule.name]
            if len(calibration) > 10:
                historical_accuracy = sum(1 for _, correct in calibration if correct) / len(calibration)
                base_confidence = (base_confidence + historical_accuracy) / 2
        
        return min(0.99, max(0.01, base_confidence))
    
    def _calculate_expected_value(self, prediction: Prediction) -> float:
        """Calculate expected value considering alternatives."""
        if prediction.predicted_value is None:
            return 0.0
        
        try:
            main_value = float(prediction.predicted_value) * prediction.confidence
            
            alt_value = sum(
                float(v) * p
                for v, p in prediction.alternatives
            )
            
            return main_value + alt_value
        except (ValueError, TypeError):
            return prediction.confidence  # Non-numeric, use confidence as proxy
