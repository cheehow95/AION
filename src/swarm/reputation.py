"""
AION Swarm Intelligence 2.0 - Agent Reputation System
======================================================

Implements agent reputation scoring:
- Performance Tracking: Success/failure rate per task type
- Trust Networks: Web-of-trust style reputation propagation
- Decay Functions: Time-weighted reputation decay
- Anti-Sybil Measures: Prevention of reputation manipulation

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict


class ReputationDimension(Enum):
    """Dimensions of reputation to track."""
    RELIABILITY = "reliability"
    QUALITY = "quality"
    SPEED = "speed"
    COOPERATION = "cooperation"
    HONESTY = "honesty"
    EXPERTISE = "expertise"


@dataclass
class ReputationScore:
    """Multi-dimensional reputation score."""
    agent_id: str
    dimensions: Dict[ReputationDimension, float] = field(default_factory=dict)
    overall: float = 0.5
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    data_points: int = 0
    
    def __post_init__(self):
        for dim in ReputationDimension:
            if dim not in self.dimensions:
                self.dimensions[dim] = 0.5
    
    def get_dimension(self, dim: ReputationDimension) -> float:
        return self.dimensions.get(dim, 0.5)
    
    def update_dimension(self, dim: ReputationDimension, value: float, weight: float = 1.0):
        current = self.dimensions.get(dim, 0.5)
        alpha = weight / (self.data_points + weight)
        self.dimensions[dim] = current * (1 - alpha) + value * alpha
        self._recalculate_overall()
    
    def _recalculate_overall(self):
        if self.dimensions:
            self.overall = sum(self.dimensions.values()) / len(self.dimensions)
        self.data_points += 1
        self.confidence = min(1.0, self.data_points / 100)
        self.last_updated = datetime.now()


@dataclass
class ReputationEvent:
    """An event that affects reputation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    event_type: str = ""
    dimension: ReputationDimension = ReputationDimension.RELIABILITY
    value: float = 0.5
    weight: float = 1.0
    source: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ReputationSystem:
    """Comprehensive agent reputation system."""
    
    def __init__(self, decay_rate: float = 0.001):
        self.scores: Dict[str, ReputationScore] = {}
        self.events: List[ReputationEvent] = []
        self.decay_rate = decay_rate
        self.event_weights = {
            'task_success': 1.0, 'task_failure': 1.5,
            'peer_rating': 0.5, 'system_rating': 2.0, 'violation': 3.0,
        }
        self.history: Dict[str, List[float]] = defaultdict(list)
    
    def register_agent(self, agent_id: str, initial_score: float = 0.5):
        if agent_id not in self.scores:
            self.scores[agent_id] = ReputationScore(agent_id=agent_id, overall=initial_score)
    
    def get_score(self, agent_id: str) -> Optional[ReputationScore]:
        if agent_id not in self.scores:
            return None
        score = self.scores[agent_id]
        self._apply_decay(score)
        return score
    
    def _apply_decay(self, score: ReputationScore):
        hours = (datetime.now() - score.last_updated).total_seconds() / 3600
        decay = math.exp(-self.decay_rate * hours)
        for dim in score.dimensions:
            score.dimensions[dim] = 0.5 + (score.dimensions[dim] - 0.5) * decay
        score._recalculate_overall()
    
    def record_event(self, event: ReputationEvent):
        self.events.append(event)
        if event.agent_id not in self.scores:
            self.register_agent(event.agent_id)
        score = self.scores[event.agent_id]
        weight = event.weight * self.event_weights.get(event.event_type, 1.0)
        score.update_dimension(event.dimension, event.value, weight)
        self.history[event.agent_id].append(score.overall)
    
    def task_completed(self, agent_id: str, success: bool, quality: float = 0.5, speed: float = 0.5):
        event_type = 'task_success' if success else 'task_failure'
        self.record_event(ReputationEvent(agent_id=agent_id, event_type=event_type,
            dimension=ReputationDimension.RELIABILITY, value=1.0 if success else 0.0))
        if success:
            self.record_event(ReputationEvent(agent_id=agent_id, event_type=event_type,
                dimension=ReputationDimension.QUALITY, value=quality))
            self.record_event(ReputationEvent(agent_id=agent_id, event_type=event_type,
                dimension=ReputationDimension.SPEED, value=speed))
    
    def peer_rating(self, rater: str, rated: str, dimension: ReputationDimension,
                    rating: float, weight: float = 1.0):
        rater_score = self.get_score(rater)
        if rater_score:
            weight *= rater_score.overall
        self.record_event(ReputationEvent(agent_id=rated, event_type='peer_rating',
            dimension=dimension, value=rating, weight=weight, source=rater))
    
    def get_ranking(self, dimension: Optional[ReputationDimension] = None, top_n: int = 10):
        scores = [(a, s.get_dimension(dimension) if dimension else s.overall) 
                  for a, s in self.scores.items()]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_statistics(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self.scores:
            return {}
        score = self.get_score(agent_id)
        history = self.history.get(agent_id, [])
        trend = "stable"
        if len(history) >= 2:
            recent = history[-10:]
            slope = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0
            trend = "improving" if slope > 0.01 else ("declining" if slope < -0.01 else "stable")
        return {'current_score': score.overall, 'confidence': score.confidence,
                'dimensions': {d.value: v for d, v in score.dimensions.items()},
                'trend': trend, 'data_points': score.data_points}


@dataclass
class TrustRelation:
    truster: str
    trustee: str
    trust_level: float = 0.5
    evidence_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class TrustNetwork:
    """Web-of-trust style trust propagation network."""
    
    def __init__(self, propagation_decay: float = 0.5):
        self.relations: Dict[Tuple[str, str], TrustRelation] = {}
        self.agents: Set[str] = set()
        self.propagation_decay = propagation_decay
    
    def add_agent(self, agent_id: str):
        self.agents.add(agent_id)
    
    def set_trust(self, truster: str, trustee: str, trust_level: float):
        key = (truster, trustee)
        if key in self.relations:
            self.relations[key].trust_level = trust_level
            self.relations[key].evidence_count += 1
        else:
            self.relations[key] = TrustRelation(truster, trustee, trust_level, 1)
        self.agents.update([truster, trustee])
    
    def get_direct_trust(self, truster: str, trustee: str) -> float:
        return self.relations.get((truster, trustee), TrustRelation(truster, trustee)).trust_level
    
    def get_transitive_trust(self, truster: str, trustee: str, max_hops: int = 3) -> float:
        if truster == trustee:
            return 1.0
        if (truster, trustee) in self.relations:
            return self.relations[(truster, trustee)].trust_level
        visited, queue = {truster}, [(truster, 1.0, 0)]
        best = 0.5
        while queue:
            current, acc, hops = queue.pop(0)
            if hops >= max_hops:
                continue
            for (t, e), r in self.relations.items():
                if t != current or e in visited:
                    continue
                prop = acc * r.trust_level * self.propagation_decay
                if e == trustee:
                    best = max(best, prop)
                else:
                    visited.add(e)
                    queue.append((e, prop, hops + 1))
        return best
    
    def get_trustees(self, agent_id: str) -> List[Tuple[str, float]]:
        return sorted([(e, r.trust_level) for (t, e), r in self.relations.items() if t == agent_id],
                     key=lambda x: x[1], reverse=True)


class AntiSybilGuard:
    """Protection against Sybil attacks and reputation manipulation."""
    
    def __init__(self, reputation: ReputationSystem, trust: TrustNetwork):
        self.reputation = reputation
        self.trust = trust
        self.agent_creation: Dict[str, datetime] = {}
        self.rating_history: Dict[str, List[datetime]] = defaultdict(list)
        self.flagged_agents: Set[str] = set()
    
    def register_agent(self, agent_id: str):
        self.agent_creation[agent_id] = datetime.now()
    
    def check_rating_allowed(self, rater: str, rated: str) -> Tuple[bool, str]:
        if rater in self.agent_creation:
            if datetime.now() - self.agent_creation[rater] < timedelta(hours=24):
                return False, "Account too new"
        now = datetime.now()
        recent = [t for t in self.rating_history[rater] if now - t < timedelta(hours=1)]
        if len(recent) >= 10:
            return False, "Rate limit exceeded"
        if rater == rated:
            return False, "Cannot rate self"
        if rater in self.flagged_agents:
            return False, "Agent flagged"
        self.rating_history[rater].append(now)
        return True, "OK"
    
    def run_analysis(self) -> Dict[str, Any]:
        return {'flagged': list(self.flagged_agents), 'total': len(self.reputation.scores)}


async def demo_reputation():
    print("⭐ Reputation System Demo")
    rep = ReputationSystem()
    trust = TrustNetwork()
    for a in ["Alpha", "Beta", "Gamma"]:
        rep.register_agent(a)
        trust.add_agent(a)
    rep.task_completed("Alpha", True, 0.9, 0.8)
    rep.task_completed("Beta", False)
    trust.set_trust("Alpha", "Beta", 0.8)
    print(f"Rankings: {rep.get_ranking()}")
    print("✅ Done!")

if __name__ == "__main__":
    asyncio.run(demo_reputation())
