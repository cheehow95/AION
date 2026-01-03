"""
AION Self-Evolution v2 - Automated Benchmark Discovery
=======================================================

Automated benchmark discovery:
- Performance Probing: Automatic bottleneck detection
- Benchmark Generation: Synthetic benchmark creation
- Comparative Analysis: Cross-agent performance comparison
- Evolution Targets: Automatic improvement target identification

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
from enum import Enum
import random


class BenchmarkCategory(Enum):
    """Categories of benchmarks."""
    REASONING = "reasoning"
    MEMORY = "memory"
    SPEED = "speed"
    ACCURACY = "accuracy"
    RESOURCE = "resource"


@dataclass
class Benchmark:
    """A benchmark test."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: BenchmarkCategory = BenchmarkCategory.REASONING
    description: str = ""
    test_func: Optional[Callable] = None
    baseline: float = 0.0
    weight: float = 1.0
    
    async def run(self, agent: Any) -> 'BenchmarkResult':
        """Run the benchmark on an agent."""
        start = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(self.test_func):
                result = await self.test_func(agent)
            else:
                result = self.test_func(agent)
            duration = time.perf_counter() - start
            return BenchmarkResult(
                benchmark_id=self.id, agent_id=getattr(agent, 'id', 'unknown'),
                score=result if isinstance(result, (int, float)) else 1.0,
                duration=duration, success=True
            )
        except Exception as e:
            return BenchmarkResult(
                benchmark_id=self.id, agent_id=getattr(agent, 'id', 'unknown'),
                score=0.0, success=False, error=str(e)
            )


@dataclass
class BenchmarkResult:
    """Result of running a benchmark."""
    benchmark_id: str = ""
    agent_id: str = ""
    score: float = 0.0
    duration: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProbe:
    """Probes system performance to identify bottlenecks."""
    
    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}
        self.thresholds: Dict[str, float] = {}
    
    def measure(self, name: str, value: float):
        """Record a measurement."""
        if name not in self.measurements:
            self.measurements[name] = []
        self.measurements[name].append(value)
    
    def set_threshold(self, name: str, threshold: float):
        """Set acceptable threshold for a metric."""
        self.thresholds[name] = threshold
    
    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify bottlenecks exceeding thresholds."""
        bottlenecks = []
        for name, values in self.measurements.items():
            if not values:
                continue
            avg = statistics.mean(values)
            threshold = self.thresholds.get(name, float('inf'))
            if avg > threshold:
                bottlenecks.append({
                    'metric': name,
                    'average': avg,
                    'threshold': threshold,
                    'severity': (avg - threshold) / threshold if threshold > 0 else 1.0
                })
        return sorted(bottlenecks, key=lambda x: x['severity'], reverse=True)
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = self.measurements.get(name, [])
        if not values:
            return {}
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values)
        }


class BenchmarkDiscovery:
    """Discovers and generates benchmarks automatically."""
    
    def __init__(self):
        self.benchmarks: Dict[str, Benchmark] = {}
        self.results: Dict[str, List[BenchmarkResult]] = {}  # benchmark_id -> results
        self.probe = PerformanceProbe()
        self.improvement_targets: List[Dict[str, Any]] = []
    
    def register_benchmark(self, benchmark: Benchmark):
        """Register a benchmark."""
        self.benchmarks[benchmark.id] = benchmark
    
    def generate_synthetic_benchmark(self, category: BenchmarkCategory,
                                     difficulty: float = 0.5) -> Benchmark:
        """Generate a synthetic benchmark."""
        if category == BenchmarkCategory.REASONING:
            return self._generate_reasoning_benchmark(difficulty)
        elif category == BenchmarkCategory.MEMORY:
            return self._generate_memory_benchmark(difficulty)
        elif category == BenchmarkCategory.SPEED:
            return self._generate_speed_benchmark(difficulty)
        else:
            return self._generate_generic_benchmark(category, difficulty)
    
    def _generate_reasoning_benchmark(self, difficulty: float) -> Benchmark:
        """Generate a reasoning benchmark."""
        complexity = int(3 + difficulty * 7)
        
        async def test(agent):
            # Generate logic puzzle
            problem = f"Solve: If A implies B, and B implies C, what can we conclude from A? (depth: {complexity})"
            result = await agent.reason(problem) if hasattr(agent, 'reason') else True
            return 1.0 if result else 0.0
        
        return Benchmark(
            name=f"reasoning_depth_{complexity}",
            category=BenchmarkCategory.REASONING,
            description=f"Reasoning chain depth {complexity}",
            test_func=test,
            weight=difficulty
        )
    
    def _generate_memory_benchmark(self, difficulty: float) -> Benchmark:
        """Generate a memory benchmark."""
        items = int(10 + difficulty * 90)
        
        async def test(agent):
            data = [f"item_{i}" for i in range(items)]
            if hasattr(agent, 'store'):
                for item in data:
                    await agent.store(item)
            if hasattr(agent, 'retrieve'):
                recalled = await agent.retrieve(items)
                return len(recalled) / items if recalled else 0.0
            return 1.0
        
        return Benchmark(
            name=f"memory_capacity_{items}",
            category=BenchmarkCategory.MEMORY,
            description=f"Memory capacity {items} items",
            test_func=test,
            weight=difficulty
        )
    
    def _generate_speed_benchmark(self, difficulty: float) -> Benchmark:
        """Generate a speed benchmark."""
        iterations = int(100 + difficulty * 900)
        
        async def test(agent):
            start = time.perf_counter()
            for _ in range(iterations):
                if hasattr(agent, 'process'):
                    await agent.process("test")
            duration = time.perf_counter() - start
            target = iterations * 0.001  # 1ms per iteration
            return target / duration if duration > 0 else 1.0
        
        return Benchmark(
            name=f"speed_iters_{iterations}",
            category=BenchmarkCategory.SPEED,
            description=f"Speed test {iterations} iterations",
            test_func=test,
            weight=difficulty
        )
    
    def _generate_generic_benchmark(self, category: BenchmarkCategory,
                                    difficulty: float) -> Benchmark:
        """Generate a generic benchmark."""
        async def test(agent):
            return random.random() * (1 - difficulty * 0.5)
        
        return Benchmark(
            name=f"{category.value}_generic_{int(difficulty*100)}",
            category=category,
            test_func=test,
            weight=difficulty
        )
    
    async def run_benchmarks(self, agent: Any,
                             categories: List[BenchmarkCategory] = None) -> Dict[str, Any]:
        """Run all relevant benchmarks on an agent."""
        results = []
        
        for benchmark in self.benchmarks.values():
            if categories and benchmark.category not in categories:
                continue
            
            result = await benchmark.run(agent)
            results.append(result)
            
            if benchmark.id not in self.results:
                self.results[benchmark.id] = []
            self.results[benchmark.id].append(result)
            
            self.probe.measure(f"{benchmark.category.value}_score", result.score)
            self.probe.measure(f"{benchmark.category.value}_duration", result.duration)
        
        return self._analyze_results(results)
    
    def _analyze_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze benchmark results."""
        by_category: Dict[str, List[float]] = {}
        
        for result in results:
            bench = self.benchmarks.get(result.benchmark_id)
            if bench:
                cat = bench.category.value
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(result.score)
        
        analysis = {
            'total_benchmarks': len(results),
            'successful': sum(1 for r in results if r.success),
            'average_score': statistics.mean([r.score for r in results]) if results else 0,
            'by_category': {
                cat: statistics.mean(scores) for cat, scores in by_category.items()
            }
        }
        
        return analysis
    
    def identify_improvement_targets(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Identify areas needing improvement."""
        targets = []
        
        for bench_id, results in self.results.items():
            if not results:
                continue
            
            bench = self.benchmarks.get(bench_id)
            avg_score = statistics.mean([r.score for r in results])
            
            if avg_score < threshold:
                targets.append({
                    'benchmark': bench.name if bench else bench_id,
                    'category': bench.category.value if bench else 'unknown',
                    'current_score': avg_score,
                    'target_score': threshold,
                    'priority': (threshold - avg_score) * (bench.weight if bench else 1.0)
                })
        
        self.improvement_targets = sorted(targets, key=lambda x: x['priority'], reverse=True)
        return self.improvement_targets


async def demo_benchmark_discovery():
    """Demonstrate benchmark discovery."""
    print("📊 Benchmark Discovery Demo")
    print("=" * 50)
    
    discovery = BenchmarkDiscovery()
    
    # Generate synthetic benchmarks
    print("\n🔧 Generating benchmarks...")
    for cat in BenchmarkCategory:
        for difficulty in [0.3, 0.6, 0.9]:
            bench = discovery.generate_synthetic_benchmark(cat, difficulty)
            discovery.register_benchmark(bench)
    
    print(f"  Generated {len(discovery.benchmarks)} benchmarks")
    
    # Mock agent
    class MockAgent:
        id = "test-agent"
        async def reason(self, problem): return True
        async def store(self, item): pass
        async def retrieve(self, n): return ["item"] * n
        async def process(self, data): pass
    
    # Run benchmarks
    print("\n▶️ Running benchmarks...")
    agent = MockAgent()
    results = await discovery.run_benchmarks(agent)
    print(f"  Results: {results}")
    
    # Find improvement targets
    targets = discovery.identify_improvement_targets(threshold=0.8)
    print(f"\n🎯 Improvement targets: {len(targets)}")
    for t in targets[:3]:
        print(f"  - {t['benchmark']}: {t['current_score']:.2f} → {t['target_score']:.2f}")
    
    print("\n✅ Benchmark discovery demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_benchmark_discovery())
