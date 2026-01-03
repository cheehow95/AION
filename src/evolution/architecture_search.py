"""
AION Self-Evolution v2 - Architecture Search
=============================================

Neural architecture and code structure search:
- Search Space Definition: Configurable architecture options
- Evolution Strategy: Genetic algorithm for architecture
- Performance Evaluation: Multi-objective optimization
- Architecture Pruning: Automatic simplification

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
from enum import Enum
from copy import deepcopy


class ComponentType(Enum):
    """Types of architecture components."""
    LAYER = "layer"
    MODULE = "module"
    CONNECTION = "connection"
    HYPERPARAMETER = "hyperparameter"


@dataclass
class ArchitectureGene:
    """A gene representing one aspect of architecture."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    component_type: ComponentType = ComponentType.MODULE
    name: str = ""
    value: Any = None
    mutable: bool = True
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Architecture:
    """An architecture configuration."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    genes: List[ArchitectureGene] = field(default_factory=list)
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mutate(self, rate: float = 0.1) -> 'Architecture':
        """Create a mutated copy."""
        new_arch = deepcopy(self)
        new_arch.id = str(uuid.uuid4())
        new_arch.parent_ids = [self.id]
        new_arch.generation = self.generation + 1
        
        for gene in new_arch.genes:
            if not gene.mutable or random.random() > rate:
                continue
            gene.value = self._mutate_gene(gene)
        
        return new_arch
    
    def _mutate_gene(self, gene: ArchitectureGene) -> Any:
        """Mutate a single gene."""
        if isinstance(gene.value, int):
            min_v = gene.constraints.get('min', 0)
            max_v = gene.constraints.get('max', 100)
            return random.randint(min_v, max_v)
        elif isinstance(gene.value, float):
            min_v = gene.constraints.get('min', 0.0)
            max_v = gene.constraints.get('max', 1.0)
            return random.uniform(min_v, max_v)
        elif isinstance(gene.value, bool):
            return random.random() > 0.5
        elif isinstance(gene.value, list) and gene.constraints.get('options'):
            return random.choice(gene.constraints['options'])
        return gene.value
    
    def crossover(self, other: 'Architecture') -> 'Architecture':
        """Create child from two parents."""
        child = Architecture(
            generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.id, other.id]
        )
        
        for i, (g1, g2) in enumerate(zip(self.genes, other.genes)):
            # Random crossover point
            gene = deepcopy(g1 if random.random() > 0.5 else g2)
            child.genes.append(gene)
        
        return child
    
    def to_config(self) -> Dict[str, Any]:
        """Convert to configuration dict."""
        return {gene.name: gene.value for gene in self.genes}


class ArchitectureSpace:
    """Defines the space of possible architectures."""
    
    def __init__(self):
        self.gene_templates: List[ArchitectureGene] = []
    
    def add_integer(self, name: str, min_val: int, max_val: int, 
                    default: int = None) -> 'ArchitectureSpace':
        """Add integer parameter."""
        self.gene_templates.append(ArchitectureGene(
            component_type=ComponentType.HYPERPARAMETER,
            name=name,
            value=default or (min_val + max_val) // 2,
            constraints={'min': min_val, 'max': max_val}
        ))
        return self
    
    def add_float(self, name: str, min_val: float, max_val: float,
                  default: float = None) -> 'ArchitectureSpace':
        """Add float parameter."""
        self.gene_templates.append(ArchitectureGene(
            component_type=ComponentType.HYPERPARAMETER,
            name=name,
            value=default or (min_val + max_val) / 2,
            constraints={'min': min_val, 'max': max_val}
        ))
        return self
    
    def add_choice(self, name: str, options: List[Any],
                   default: Any = None) -> 'ArchitectureSpace':
        """Add categorical parameter."""
        self.gene_templates.append(ArchitectureGene(
            component_type=ComponentType.HYPERPARAMETER,
            name=name,
            value=default or options[0],
            constraints={'options': options}
        ))
        return self
    
    def add_boolean(self, name: str, default: bool = False) -> 'ArchitectureSpace':
        """Add boolean parameter."""
        self.gene_templates.append(ArchitectureGene(
            component_type=ComponentType.HYPERPARAMETER,
            name=name,
            value=default
        ))
        return self
    
    def sample(self) -> Architecture:
        """Sample a random architecture from the space."""
        arch = Architecture()
        for template in self.gene_templates:
            gene = deepcopy(template)
            gene.id = str(uuid.uuid4())
            gene.value = arch._mutate_gene(gene)
            arch.genes.append(gene)
        return arch
    
    def default(self) -> Architecture:
        """Get default architecture."""
        arch = Architecture()
        arch.genes = [deepcopy(t) for t in self.gene_templates]
        return arch


class EvolutionStrategy:
    """Genetic algorithm for architecture optimization."""
    
    def __init__(self, space: ArchitectureSpace, population_size: int = 20,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7,
                 elite_ratio: float = 0.1):
        self.space = space
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = max(1, int(population_size * elite_ratio))
        
        self.population: List[Architecture] = []
        self.generation = 0
        self.best_ever: Optional[Architecture] = None
        self.history: List[Dict[str, Any]] = []
    
    def initialize(self):
        """Initialize population."""
        self.population = [self.space.sample() for _ in range(self.population_size)]
        self.generation = 0
    
    async def evaluate(self, fitness_func: Callable[[Architecture], float]):
        """Evaluate fitness of all individuals."""
        for arch in self.population:
            if asyncio.iscoroutinefunction(fitness_func):
                arch.fitness = await fitness_func(arch)
            else:
                arch.fitness = fitness_func(arch)
            
            if self.best_ever is None or arch.fitness > self.best_ever.fitness:
                self.best_ever = deepcopy(arch)
    
    def select_parents(self) -> List[Architecture]:
        """Tournament selection."""
        parents = []
        for _ in range(self.population_size):
            # Tournament of 3
            candidates = random.sample(self.population, min(3, len(self.population)))
            winner = max(candidates, key=lambda x: x.fitness)
            parents.append(winner)
        return parents
    
    def evolve(self):
        """Evolve to next generation."""
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep elite
        new_population = [deepcopy(a) for a in self.population[:self.elite_count]]
        
        # Select parents
        parents = self.select_parents()
        
        # Create offspring
        while len(new_population) < self.population_size:
            p1, p2 = random.sample(parents, 2)
            
            if random.random() < self.crossover_rate:
                child = p1.crossover(p2)
            else:
                child = deepcopy(p1)
            
            child = child.mutate(self.mutation_rate)
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        # Record history
        fitnesses = [a.fitness for a in self.population]
        self.history.append({
            'generation': self.generation,
            'best': max(fitnesses),
            'mean': sum(fitnesses) / len(fitnesses),
            'worst': min(fitnesses)
        })
    
    async def run(self, fitness_func: Callable, max_generations: int = 50,
                  target_fitness: float = None) -> Architecture:
        """Run evolution until termination."""
        self.initialize()
        
        for gen in range(max_generations):
            await self.evaluate(fitness_func)
            
            if target_fitness and self.best_ever.fitness >= target_fitness:
                break
            
            self.evolve()
        
        return self.best_ever


class ArchitectureSearch:
    """High-level architecture search manager."""
    
    def __init__(self):
        self.searches: Dict[str, EvolutionStrategy] = {}
        self.results: List[Architecture] = []
    
    def create_search(self, name: str, space: ArchitectureSpace,
                      **kwargs) -> EvolutionStrategy:
        """Create a new search."""
        strategy = EvolutionStrategy(space, **kwargs)
        self.searches[name] = strategy
        return strategy
    
    async def run_search(self, name: str, evaluator: Callable,
                         generations: int = 50) -> Architecture:
        """Run a search."""
        if name not in self.searches:
            raise ValueError(f"Unknown search: {name}")
        
        result = await self.searches[name].run(evaluator, generations)
        self.results.append(result)
        return result
    
    def prune_architecture(self, arch: Architecture, 
                           threshold: float = 0.1) -> Architecture:
        """Remove low-impact components."""
        pruned = deepcopy(arch)
        pruned.id = str(uuid.uuid4())
        
        # Simple pruning: reset near-zero values
        for gene in pruned.genes:
            if isinstance(gene.value, float) and abs(gene.value) < threshold:
                gene.value = 0.0
        
        return pruned
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        return {
            'active_searches': len(self.searches),
            'total_results': len(self.results),
            'best_fitness': max((r.fitness for r in self.results), default=0)
        }


async def demo_architecture_search():
    """Demonstrate architecture search."""
    print("🔬 Architecture Search Demo")
    print("=" * 50)
    
    # Define search space
    space = ArchitectureSpace()
    space.add_integer("num_layers", 1, 10, default=3)
    space.add_integer("hidden_size", 32, 512, default=128)
    space.add_float("learning_rate", 0.0001, 0.1, default=0.01)
    space.add_float("dropout", 0.0, 0.5, default=0.1)
    space.add_choice("activation", ["relu", "gelu", "tanh"], default="relu")
    space.add_boolean("use_attention", default=True)
    
    # Create search
    search = ArchitectureSearch()
    strategy = search.create_search("test", space, population_size=10)
    
    # Mock fitness function
    def evaluate(arch: Architecture) -> float:
        config = arch.to_config()
        # Simulated fitness based on config
        score = 0.5
        score += (config['hidden_size'] / 512) * 0.2
        score += (1 - config['learning_rate'] * 10) * 0.1
        score += 0.1 if config['use_attention'] else 0
        return min(1.0, score + random.random() * 0.1)
    
    print("\n🧬 Running evolution...")
    best = await strategy.run(evaluate, max_generations=10)
    
    print(f"\n📊 Best architecture (fitness: {best.fitness:.3f}):")
    for gene in best.genes:
        print(f"  {gene.name}: {gene.value}")
    
    print(f"\n📈 Evolution history:")
    for h in strategy.history[-3:]:
        print(f"  Gen {h['generation']}: best={h['best']:.3f}, mean={h['mean']:.3f}")
    
    print("\n✅ Architecture search demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_architecture_search())
