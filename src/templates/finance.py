"""
AION Industry Templates - Finance
==================================

Finance-specific agent templates:
- Risk Analysis: Portfolio and market risk assessment
- Compliance: Regulatory compliance checking
- Market Data: Real-time data processing
- Trading Support: Analysis and recommendations

Auto-generated for Phase 5: Scale
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import random
import statistics


class RiskLevel(Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AssetClass(Enum):
    """Asset classes."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    DERIVATIVE = "derivative"
    CRYPTO = "crypto"


@dataclass
class Position:
    """A portfolio position."""
    symbol: str = ""
    asset_class: AssetClass = AssetClass.EQUITY
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    currency: str = "USD"
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def pnl_percent(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class Portfolio:
    """Investment portfolio."""
    id: str = ""
    name: str = ""
    positions: List[Position] = field(default_factory=list)
    cash: float = 0.0
    currency: str = "USD"
    
    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions)
    
    @property
    def total_pnl(self) -> float:
        return sum(p.pnl for p in self.positions)


class RiskAnalyzer:
    """Portfolio risk analysis."""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% annual
        self.volatility_window = 252  # Trading days
    
    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if not returns:
            return 0.0
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        return sorted_returns[max(0, index)]
    
    def calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = statistics.mean(returns)
        std_dev = statistics.stdev(returns)
        
        if std_dev == 0:
            return 0.0
        
        excess_return = mean_return - (self.risk_free_rate / 252)
        return (excess_return / std_dev) * (252 ** 0.5)
    
    def analyze_concentration(self, portfolio: Portfolio) -> Dict[str, float]:
        """Analyze portfolio concentration."""
        total = portfolio.total_value
        if total == 0:
            return {}
        
        by_class = {}
        for pos in portfolio.positions:
            cls = pos.asset_class.value
            by_class[cls] = by_class.get(cls, 0) + pos.market_value
        
        return {k: v / total for k, v in by_class.items()}
    
    def assess_risk(self, portfolio: Portfolio, 
                    historical_returns: List[float] = None) -> Dict[str, Any]:
        """Comprehensive risk assessment."""
        returns = historical_returns or [random.gauss(0.001, 0.02) for _ in range(252)]
        
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        sharpe = self.calculate_sharpe_ratio(returns)
        concentration = self.analyze_concentration(portfolio)
        
        # Determine overall risk level
        if var_95 < -0.05 or max(concentration.values(), default=0) > 0.5:
            risk_level = RiskLevel.HIGH
        elif var_95 < -0.02:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'sharpe_ratio': sharpe,
            'concentration': concentration,
            'risk_level': risk_level.value,
            'total_value': portfolio.total_value,
            'total_pnl': portfolio.total_pnl
        }


class ComplianceChecker:
    """Regulatory compliance checking."""
    
    def __init__(self):
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.violations: List[Dict[str, Any]] = []
    
    def add_rule(self, rule_id: str, name: str, 
                 check_func: callable, severity: str = "medium"):
        """Add a compliance rule."""
        self.rules[rule_id] = {
            'name': name,
            'check': check_func,
            'severity': severity
        }
    
    def check_compliance(self, portfolio: Portfolio,
                         trade: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check compliance with all rules."""
        results = {'passed': True, 'violations': [], 'warnings': []}
        
        context = {'portfolio': portfolio, 'trade': trade}
        
        for rule_id, rule in self.rules.items():
            try:
                passed = rule['check'](context)
                if not passed:
                    violation = {
                        'rule_id': rule_id,
                        'rule_name': rule['name'],
                        'severity': rule['severity'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if rule['severity'] == 'critical':
                        results['passed'] = False
                        results['violations'].append(violation)
                    else:
                        results['warnings'].append(violation)
                    
                    self.violations.append(violation)
            except Exception as e:
                results['warnings'].append({
                    'rule_id': rule_id,
                    'error': str(e)
                })
        
        return results
    
    def setup_default_rules(self):
        """Setup default compliance rules."""
        # Position limit rule
        self.add_rule(
            "POS_LIMIT",
            "Single position limit",
            lambda ctx: all(
                p.market_value / ctx['portfolio'].total_value < 0.25
                for p in ctx['portfolio'].positions
            ) if ctx['portfolio'].total_value > 0 else True,
            severity="critical"
        )
        
        # Cash reserve rule
        self.add_rule(
            "CASH_RESERVE",
            "Minimum cash reserve",
            lambda ctx: ctx['portfolio'].cash / ctx['portfolio'].total_value >= 0.05
            if ctx['portfolio'].total_value > 0 else True,
            severity="medium"
        )


class MarketDataProcessor:
    """Real-time market data processing."""
    
    def __init__(self):
        self.price_cache: Dict[str, float] = {}
        self.last_update: Dict[str, datetime] = {}
        self.price_history: Dict[str, List[tuple]] = {}
    
    def update_price(self, symbol: str, price: float):
        """Update price for a symbol."""
        self.price_cache[symbol] = price
        self.last_update[symbol] = datetime.now()
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append((datetime.now(), price))
        
        # Keep last 1000 prices
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price."""
        return self.price_cache.get(symbol)
    
    def get_change(self, symbol: str) -> Optional[float]:
        """Get price change percentage."""
        history = self.price_history.get(symbol, [])
        if len(history) < 2:
            return None
        
        old_price = history[0][1]
        new_price = history[-1][1]
        
        return ((new_price - old_price) / old_price) * 100 if old_price else 0
    
    def calculate_vwap(self, symbol: str, volumes: List[float] = None) -> Optional[float]:
        """Calculate volume-weighted average price."""
        history = self.price_history.get(symbol, [])
        if not history:
            return None
        
        prices = [p[1] for p in history[-100:]]
        
        if volumes and len(volumes) == len(prices):
            total_volume = sum(volumes)
            if total_volume == 0:
                return statistics.mean(prices)
            return sum(p * v for p, v in zip(prices, volumes)) / total_volume
        
        return statistics.mean(prices)


class FinanceAgent:
    """Finance-specialized AION agent."""
    
    def __init__(self, agent_id: str = "finance-agent"):
        self.agent_id = agent_id
        self.risk_analyzer = RiskAnalyzer()
        self.compliance = ComplianceChecker()
        self.market_data = MarketDataProcessor()
        self.portfolios: Dict[str, Portfolio] = {}
        
        # Setup default compliance rules
        self.compliance.setup_default_rules()
    
    def add_portfolio(self, portfolio: Portfolio):
        """Add a portfolio to manage."""
        self.portfolios[portfolio.id] = portfolio
    
    def update_prices(self, prices: Dict[str, float]):
        """Update market prices."""
        for symbol, price in prices.items():
            self.market_data.update_price(symbol, price)
        
        # Update portfolio positions
        for portfolio in self.portfolios.values():
            for position in portfolio.positions:
                if position.symbol in prices:
                    position.current_price = prices[position.symbol]
    
    async def analyze_portfolio(self, portfolio_id: str) -> Dict[str, Any]:
        """Analyze a portfolio."""
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            return {'error': 'Portfolio not found'}
        
        risk_assessment = self.risk_analyzer.assess_risk(portfolio)
        compliance_check = self.compliance.check_compliance(portfolio)
        
        return {
            'portfolio_id': portfolio_id,
            'risk': risk_assessment,
            'compliance': compliance_check,
            'positions': len(portfolio.positions),
            'total_value': portfolio.total_value,
            'total_pnl': portfolio.total_pnl
        }
    
    async def evaluate_trade(self, portfolio_id: str,
                             trade: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a proposed trade."""
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            return {'error': 'Portfolio not found'}
        
        # Check compliance
        compliance_result = self.compliance.check_compliance(portfolio, trade)
        
        # Risk impact
        recommendation = "approve" if compliance_result['passed'] else "reject"
        
        return {
            'trade': trade,
            'compliance': compliance_result,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            'agent_id': self.agent_id,
            'portfolios_managed': len(self.portfolios),
            'compliance_rules': len(self.compliance.rules),
            'price_symbols': len(self.market_data.price_cache)
        }


async def demo_finance():
    """Demonstrate finance template."""
    print("💰 Finance Template Demo")
    print("=" * 50)
    
    agent = FinanceAgent()
    
    # Create portfolio
    portfolio = Portfolio(
        id="P001",
        name="Growth Portfolio",
        cash=50000,
        positions=[
            Position("AAPL", AssetClass.EQUITY, 100, 150.0, 175.0),
            Position("MSFT", AssetClass.EQUITY, 50, 300.0, 380.0),
            Position("BTC", AssetClass.CRYPTO, 1, 40000.0, 45000.0),
        ]
    )
    agent.add_portfolio(portfolio)
    
    print(f"\n📊 Portfolio: {portfolio.name}")
    print(f"  Total Value: ${portfolio.total_value:,.2f}")
    print(f"  Total P&L: ${portfolio.total_pnl:,.2f}")
    
    # Analyze portfolio
    analysis = await agent.analyze_portfolio("P001")
    
    print(f"\n⚠️ Risk Analysis:")
    print(f"  Risk Level: {analysis['risk']['risk_level']}")
    print(f"  VaR (95%): {analysis['risk']['var_95']:.2%}")
    print(f"  Sharpe Ratio: {analysis['risk']['sharpe_ratio']:.2f}")
    
    print(f"\n✅ Compliance:")
    print(f"  Passed: {analysis['compliance']['passed']}")
    print(f"  Warnings: {len(analysis['compliance']['warnings'])}")
    
    # Market data
    agent.update_prices({'AAPL': 180.0, 'MSFT': 390.0, 'BTC': 47000.0})
    
    updated_analysis = await agent.analyze_portfolio("P001")
    print(f"\n📈 After Price Update:")
    print(f"  Total Value: ${updated_analysis['total_value']:,.2f}")
    print(f"  Total P&L: ${updated_analysis['total_pnl']:,.2f}")
    
    print(f"\n📊 Status: {agent.get_status()}")
    print("\n✅ Finance template demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_finance())
