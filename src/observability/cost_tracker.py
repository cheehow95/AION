"""
AION Cost Tracker
=================

Track and analyze costs of agent operations.
Provides insights into token usage, API costs, and budgeting.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


class CostCategory(Enum):
    """Categories of costs."""
    MODEL_INPUT = "model_input"
    MODEL_OUTPUT = "model_output"
    EMBEDDING = "embedding"
    TOOL_CALL = "tool_call"
    STORAGE = "storage"
    COMPUTE = "compute"


@dataclass
class CostEntry:
    """A single cost entry."""
    timestamp: datetime
    category: CostCategory
    amount: float  # In cents
    tokens: int = 0
    model: str = ""
    agent: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostReport:
    """A cost report for a time period."""
    start_time: datetime
    end_time: datetime
    total_cost: float
    entries: List[CostEntry]
    by_category: Dict[str, float]
    by_model: Dict[str, float]
    by_agent: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat()
            },
            "total_cost_cents": self.total_cost,
            "total_cost_dollars": self.total_cost / 100,
            "by_category": self.by_category,
            "by_model": self.by_model,
            "by_agent": self.by_agent,
            "entry_count": len(self.entries)
        }


@dataclass
class Budget:
    """A budget configuration."""
    name: str
    limit_cents: float
    period: str  # "hourly", "daily", "weekly", "monthly"
    alert_threshold: float = 0.8  # Alert at 80% of limit
    hard_limit: bool = False  # Block when exceeded


class CostTracker:
    """
    Track and manage agent operation costs.
    
    Provides:
    - Real-time cost tracking
    - Historical cost analysis
    - Budget management
    - Alerts and limits
    """
    
    # Pricing per 1K tokens (in cents) - example rates
    MODEL_PRICING = {
        "gpt-4": {"input": 3.0, "output": 6.0},
        "gpt-4-turbo": {"input": 1.0, "output": 3.0},
        "gpt-3.5-turbo": {"input": 0.05, "output": 0.15},
        "claude-3-opus": {"input": 1.5, "output": 7.5},
        "claude-3-sonnet": {"input": 0.3, "output": 1.5},
        "claude-3-haiku": {"input": 0.025, "output": 0.125},
        "ollama": {"input": 0.0, "output": 0.0},  # Local models
        "default": {"input": 0.1, "output": 0.3}
    }
    
    def __init__(self):
        self.entries: List[CostEntry] = []
        self.budgets: Dict[str, Budget] = {}
        self.spending: Dict[str, float] = defaultdict(float)  # budget_name -> current spend
        self._alert_callbacks: List[callable] = []
    
    def record_tokens(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent: str = "",
        description: str = ""
    ) -> float:
        """
        Record token usage and calculate cost.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            agent: Agent that made the request
            description: Optional description
        
        Returns:
            Total cost in cents
        """
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["default"])
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        # Record input cost
        if input_tokens > 0:
            self.entries.append(CostEntry(
                timestamp=datetime.now(),
                category=CostCategory.MODEL_INPUT,
                amount=input_cost,
                tokens=input_tokens,
                model=model,
                agent=agent,
                description=description
            ))
        
        # Record output cost
        if output_tokens > 0:
            self.entries.append(CostEntry(
                timestamp=datetime.now(),
                category=CostCategory.MODEL_OUTPUT,
                amount=output_cost,
                tokens=output_tokens,
                model=model,
                agent=agent,
                description=description
            ))
        
        total_cost = input_cost + output_cost
        
        # Update budget spending
        self._update_budgets(total_cost)
        
        return total_cost
    
    def record_cost(
        self,
        category: CostCategory,
        amount: float,
        agent: str = "",
        model: str = "",
        description: str = "",
        metadata: Dict[str, Any] = None
    ):
        """Record a custom cost entry."""
        self.entries.append(CostEntry(
            timestamp=datetime.now(),
            category=category,
            amount=amount,
            agent=agent,
            model=model,
            description=description,
            metadata=metadata or {}
        ))
        
        self._update_budgets(amount)
    
    def get_report(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        agent: str = None,
        model: str = None
    ) -> CostReport:
        """Generate a cost report."""
        end_time = end_time or datetime.now()
        start_time = start_time or (end_time - timedelta(days=30))
        
        # Filter entries
        filtered = [
            e for e in self.entries
            if start_time <= e.timestamp <= end_time
        ]
        
        if agent:
            filtered = [e for e in filtered if e.agent == agent]
        if model:
            filtered = [e for e in filtered if e.model == model]
        
        # Aggregate
        by_category: Dict[str, float] = defaultdict(float)
        by_model: Dict[str, float] = defaultdict(float)
        by_agent: Dict[str, float] = defaultdict(float)
        
        for entry in filtered:
            by_category[entry.category.value] += entry.amount
            if entry.model:
                by_model[entry.model] += entry.amount
            if entry.agent:
                by_agent[entry.agent] += entry.amount
        
        return CostReport(
            start_time=start_time,
            end_time=end_time,
            total_cost=sum(e.amount for e in filtered),
            entries=filtered,
            by_category=dict(by_category),
            by_model=dict(by_model),
            by_agent=dict(by_agent)
        )
    
    def set_budget(
        self,
        name: str,
        limit_cents: float,
        period: str = "daily",
        alert_threshold: float = 0.8,
        hard_limit: bool = False
    ):
        """Set a budget limit."""
        self.budgets[name] = Budget(
            name=name,
            limit_cents=limit_cents,
            period=period,
            alert_threshold=alert_threshold,
            hard_limit=hard_limit
        )
        self.spending[name] = 0
    
    def check_budget(self, name: str) -> Dict[str, Any]:
        """Check budget status."""
        if name not in self.budgets:
            return {"error": "Budget not found"}
        
        budget = self.budgets[name]
        spent = self.spending[name]
        remaining = budget.limit_cents - spent
        
        return {
            "name": name,
            "limit_cents": budget.limit_cents,
            "spent_cents": spent,
            "remaining_cents": max(0, remaining),
            "usage_percent": (spent / budget.limit_cents) * 100,
            "exceeded": spent > budget.limit_cents,
            "near_limit": spent >= budget.limit_cents * budget.alert_threshold
        }
    
    def add_alert_callback(self, callback: callable):
        """Add callback for budget alerts."""
        self._alert_callbacks.append(callback)
    
    def _update_budgets(self, amount: float):
        """Update all budgets with new spending."""
        for name, budget in self.budgets.items():
            self.spending[name] += amount
            
            # Check for alerts
            usage_ratio = self.spending[name] / budget.limit_cents
            
            if usage_ratio >= budget.alert_threshold:
                for callback in self._alert_callbacks:
                    try:
                        callback(name, self.spending[name], budget.limit_cents)
                    except Exception:
                        pass
    
    def reset_period_spending(self, budget_name: str = None):
        """Reset spending for a new period."""
        if budget_name:
            self.spending[budget_name] = 0
        else:
            for name in self.spending:
                self.spending[name] = 0
    
    def get_forecasted_cost(
        self,
        days: int = 30,
        based_on_days: int = 7
    ) -> Dict[str, float]:
        """Forecast future costs based on recent usage."""
        now = datetime.now()
        recent_start = now - timedelta(days=based_on_days)
        
        recent_report = self.get_report(start_time=recent_start, end_time=now)
        daily_average = recent_report.total_cost / based_on_days
        
        return {
            "daily_average_cents": daily_average,
            "forecast_days": days,
            "forecasted_total_cents": daily_average * days,
            "forecasted_total_dollars": (daily_average * days) / 100,
            "based_on_days": based_on_days
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall cost statistics."""
        if not self.entries:
            return {"total_entries": 0}
        
        total = sum(e.amount for e in self.entries)
        total_tokens = sum(e.tokens for e in self.entries)
        
        return {
            "total_entries": len(self.entries),
            "total_cost_cents": total,
            "total_cost_dollars": total / 100,
            "total_tokens": total_tokens,
            "avg_cost_per_entry_cents": total / len(self.entries),
            "active_budgets": len(self.budgets),
            "earliest_entry": min(e.timestamp for e in self.entries).isoformat(),
            "latest_entry": max(e.timestamp for e in self.entries).isoformat()
        }
    
    def export_csv(self, filepath: str = None) -> str:
        """Export entries to CSV format."""
        lines = ["timestamp,category,amount,tokens,model,agent,description"]
        
        for entry in self.entries:
            lines.append(
                f"{entry.timestamp.isoformat()},"
                f"{entry.category.value},"
                f"{entry.amount},"
                f"{entry.tokens},"
                f"{entry.model},"
                f"{entry.agent},"
                f'"{entry.description}"'
            )
        
        csv_content = "\n".join(lines)
        
        if filepath:
            with open(filepath, "w") as f:
                f.write(csv_content)
        
        return csv_content


# Global cost tracker
_global_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker
