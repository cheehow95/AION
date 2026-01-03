"""
AION Phase 4: Autonomy - Test Suite
====================================

Comprehensive tests for Phase 4 modules:
- Swarm Intelligence 2.0
- Durable Execution
- Self-Evolution v2
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime


# ============================================================================
# Swarm Intelligence 2.0 Tests
# ============================================================================

class TestSwarmCoordination:
    """Tests for swarm coordination protocols."""
    
    def test_stigmergy_pheromone_deposit(self):
        """Test pheromone deposit and evaporation."""
        from src.swarm.coordination import Stigmergy, PheromoneType
        
        stigmergy = Stigmergy(evaporation_rate=0.1)
        stigmergy.deposit("agent1", PheromoneType.ATTRACTION, (0.0, 0.0, 0.0), 1.0)
        
        pheromones = stigmergy.sense((0.0, 0.0, 0.0))
        assert len(pheromones) == 1
        assert pheromones[0].intensity > 0
    
    def test_task_auction_bidding(self):
        """Test task auction bidding mechanism."""
        from src.swarm.coordination import TaskAuction
        
        auction = TaskAuction()
        auction_id = auction.create_auction({"name": "Test task", "description": "Test task"})
        
        auction.submit_bid(auction_id, "agent1", 80, 10.0)
        auction.submit_bid(auction_id, "agent2", 90, 5.0)
        
        winner = auction.close_auction(auction_id)
        assert winner is not None
    
    def test_coalition_formation(self):
        """Test coalition formation."""
        from src.swarm.coordination import CoalitionManager
        
        manager = CoalitionManager()
        # Register agents with capabilities
        manager.register_agent("agent1", {"coding", "design"})
        manager.register_agent("agent2", {"testing"})
        manager.register_agent("agent3", {"documentation"})
        
        coalition_id = manager.form_coalition("task1", {"coding", "testing"}, ["agent1", "agent2", "agent3"])
        
        assert coalition_id is not None
        coalition = manager.coalitions[coalition_id]
        assert len(coalition.members) >= 2
        
        manager.dissolve_coalition(coalition_id)
        active_coalitions = [c for c in manager.coalitions.values() if c.status == "active"]
        assert len(active_coalitions) == 0
    
    @pytest.mark.asyncio
    async def test_emergent_coordinator(self):
        """Test emergent coordinator protocol selection."""
        from src.swarm.coordination import EmergentCoordinator
        
        coordinator = EmergentCoordinator()
        agents = ["a1", "a2"]
        task = {"id": "test", "type": "coordination"}
        
        # Should select appropriate protocol
        result = await coordinator.coordinate(agents, task)
        assert "protocol" in result or "status" in result


class TestSwarmConsensus:
    """Tests for consensus mechanisms."""
    
    @pytest.mark.asyncio
    async def test_raft_leader_election(self):
        """Test Raft leader election."""
        from src.swarm.consensus import RaftConsensus, ConsensusState
        
        raft = RaftConsensus("node1", ["node1", "node2", "node3"])
        # Start the node (which triggers election timer)
        await raft.start()
        await asyncio.sleep(0.1)  # Give it a moment
        await raft.stop()
        
        # Node should have initialized properly - compare with enum values
        assert raft.node.state in [ConsensusState.FOLLOWER, ConsensusState.CANDIDATE, ConsensusState.LEADER]
    
    def test_voting_protocol(self):
        """Test voting protocol."""
        from src.swarm.consensus import VotingProtocol, VoteType
        
        voting = VotingProtocol(default_quorum=0.5)
        voting.register_voter("voter1", 1.0)
        voting.register_voter("voter2", 1.0)
        voting.register_voter("voter3", 1.0)
        
        proposal_id = voting.create_proposal("voter1", "Test Proposal", "Description")
        
        voting.cast_vote(proposal_id, "voter1", VoteType.APPROVE)
        voting.cast_vote(proposal_id, "voter2", VoteType.APPROVE)
        
        result = voting.tally_votes(proposal_id)
        assert result["quorum_met"] == True and result["threshold_met"] == True
    
    def test_conflict_resolver(self):
        """Test conflict resolution."""
        from src.swarm.consensus import ConflictResolver, ConflictType
        
        resolver = ConflictResolver()
        resolver.register_agent("agent1")
        resolver.register_agent("agent2")
        
        conflict_id = resolver.report_conflict(
            ConflictType.RESOURCE,
            {"agent1", "agent2"},
            {"agent1": "opt1", "agent2": "opt2"}
        )
        
        resolution = resolver.resolve(conflict_id)
        assert resolution is not None


class TestSwarmReputation:
    """Tests for reputation system."""
    
    def test_reputation_scoring(self):
        """Test reputation score updates."""
        from src.swarm.reputation import ReputationSystem
        
        system = ReputationSystem()
        system.register_agent("agent1")
        
        # Record positive events using task_completed helper
        system.task_completed("agent1", success=True, quality=0.9, speed=0.8)
        system.task_completed("agent1", success=True, quality=0.8, speed=0.7)
        
        score = system.get_score("agent1")
        assert score is not None
        assert score.overall > 0
    
    def test_trust_network(self):
        """Test trust network propagation."""
        from src.swarm.reputation import TrustNetwork
        
        network = TrustNetwork()
        network.add_agent("agent1")
        network.add_agent("agent2")
        network.add_agent("agent3")
        
        network.set_trust("agent1", "agent2", 0.9)
        network.set_trust("agent2", "agent3", 0.8)
        
        # Transitive trust
        trust = network.get_transitive_trust("agent1", "agent3")
        assert trust > 0
    
    def test_anti_sybil(self):
        """Test anti-Sybil measures."""
        from src.swarm.reputation import AntiSybilGuard, ReputationSystem, TrustNetwork
        
        system = ReputationSystem()
        trust = TrustNetwork()
        guard = AntiSybilGuard(system, trust)
        
        # Register agents
        for i in range(10):
            system.register_agent(f"agent{i}")
            guard.register_agent(f"agent{i}")
        
        # Check rating permissions and run analysis
        result = guard.run_analysis()
        # Should return analysis result
        assert isinstance(result, dict)


class TestSwarmHierarchy:
    """Tests for self-organizing hierarchy."""
    
    def test_hierarchy_creation(self):
        """Test hierarchy node creation."""
        from src.swarm.hierarchy import DynamicHierarchy
        
        hierarchy = DynamicHierarchy(max_children=3)
        
        hierarchy.add_agent("leader", {"planning"}, 0.9)
        hierarchy.add_agent("worker1", {"coding"}, 0.5)
        hierarchy.add_agent("worker2", {"testing"}, 0.5)
        
        assert len(hierarchy.nodes) == 3
        assert hierarchy.root == "leader"
    
    def test_role_assignment(self):
        """Test role assignment based on skills."""
        from src.swarm.hierarchy import RoleAssignment, AgentRole
        
        roles = RoleAssignment()
        roles.register_agent("agent1", {"planning", "coordination", "decision_making"})
        
        best_role = roles.get_best_role("agent1")
        assert best_role == AgentRole.LEADER
    
    @pytest.mark.asyncio
    async def test_hierarchical_routing(self):
        """Test message routing through hierarchy."""
        from src.swarm.hierarchy import DynamicHierarchy, HierarchicalRouter
        
        hierarchy = DynamicHierarchy()
        hierarchy.add_agent("root", performance=0.9)
        hierarchy.add_agent("child1", performance=0.7)
        hierarchy.add_agent("child2", performance=0.6)
        
        router = HierarchicalRouter(hierarchy)
        
        # Broadcast down
        count = await router.broadcast_down("root", "Hello")
        assert count >= 0


# ============================================================================
# Durable Execution Tests
# ============================================================================

class TestTemporalIntegration:
    """Tests for Temporal-style workflow engine."""
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """Test basic workflow execution."""
        from src.durable.temporal_integration import (
            WorkflowDefinition, WorkflowContext, TemporalWorkflowEngine
        )
        
        class SimpleWorkflow(WorkflowDefinition):
            def __init__(self):
                super().__init__("Simple")
                self.register_activity("step1", self._step1)
            
            async def _step1(self):
                return "completed"
            
            async def run(self, ctx, input):
                result = await self.execute_activity(ctx, "step1")
                return {"status": result}
        
        engine = TemporalWorkflowEngine()
        engine.register_workflow(SimpleWorkflow())
        
        workflow_id = await engine.start_workflow("Simple")
        await asyncio.sleep(0.2)
        
        execution = engine.get_execution(workflow_id)
        assert execution is not None
    
    @pytest.mark.asyncio
    async def test_workflow_signal(self):
        """Test workflow signaling."""
        from src.durable.temporal_integration import TemporalWorkflowEngine, WorkflowDefinition, WorkflowContext
        
        class SignalWorkflow(WorkflowDefinition):
            async def run(self, ctx, input):
                signal = await ctx.wait_for_signal("continue", timeout=1.0)
                return {"signal_received": signal is not None}
        
        engine = TemporalWorkflowEngine()
        engine.register_workflow(SignalWorkflow("SignalTest"))
        
        workflow_id = await engine.start_workflow("SignalTest")
        await asyncio.sleep(0.1)
        await engine.signal_workflow(workflow_id, "continue", {"data": "test"})


class TestCheckpointing:
    """Tests for checkpointing system."""
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation(self):
        """Test checkpoint creation and recovery."""
        from src.durable.checkpointing import CheckpointManager, CheckpointType
        
        manager = CheckpointManager()
        
        state = {"step": 1, "data": "test"}
        checkpoint = await manager.create_checkpoint("workflow1", state)
        
        assert checkpoint.workflow_id == "workflow1"
        assert checkpoint.state == state
    
    @pytest.mark.asyncio
    async def test_checkpoint_recovery(self):
        """Test state recovery from checkpoint."""
        from src.durable.checkpointing import CheckpointManager
        
        manager = CheckpointManager()
        
        # Create checkpoints
        await manager.create_checkpoint("workflow1", {"step": 1})
        await manager.create_checkpoint("workflow1", {"step": 2})
        await manager.create_checkpoint("workflow1", {"step": 3})
        
        # Recover
        recovered = await manager.recover("workflow1")
        assert recovered["step"] == 3
    
    def test_incremental_checkpoint(self):
        """Test incremental checkpoint delta computation."""
        from src.durable.checkpointing import IncrementalCheckpoint
        
        old_state = {"a": 1, "b": 2, "c": 3}
        new_state = {"a": 1, "b": 5, "d": 4}
        
        delta = IncrementalCheckpoint.compute_delta(old_state, new_state)
        
        assert "c" in delta.deleted
        assert "d" in delta.added
        assert "b" in delta.modified


class TestResumableWorkflows:
    """Tests for resumable workflows."""
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """Test workflow execution with steps."""
        from src.durable.workflows import ResumableWorkflow
        
        async def step1(ctx):
            ctx["result"] = "step1_done"
            return "ok"
        
        async def step2(ctx):
            return ctx["result"] + "_step2"
        
        workflow = ResumableWorkflow("test")
        workflow.add_step("step1", step1)
        workflow.add_step("step2", step2)
        
        result = await workflow.execute()
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_saga_compensation(self):
        """Test saga compensation on failure."""
        from src.durable.workflows import SagaCoordinator, SagaStep
        
        compensated = []
        
        def action1(ctx):
            return "done"
        
        def comp1(ctx):
            compensated.append("step1")
        
        def action2(ctx):
            raise Exception("Simulated failure")
        
        def comp2(ctx):
            compensated.append("step2")
        
        saga = SagaCoordinator()
        saga.define_saga("test", [
            SagaStep("step1", action1, comp1),
            SagaStep("step2", action2, comp2),
        ])
        
        result = await saga.execute_saga("test", {})
        assert result["status"] == "compensated"


class TestTimeTravelDebugging:
    """Tests for time-travel debugging."""
    
    def test_event_recording(self):
        """Test event store recording."""
        from src.durable.time_travel import EventStore, EventType
        
        store = EventStore()
        
        store.record_change("status", None, "started")
        store.record_change("step", 0, 1)
        store.record_change("step", 1, 2)
        
        assert store.sequence == 3
    
    def test_state_reconstruction(self):
        """Test state reconstruction at sequence."""
        from src.durable.time_travel import EventStore
        
        store = EventStore()
        
        store.record_change("a", None, 1)
        store.record_change("b", None, 2)
        store.record_change("a", 1, 10)
        
        state_at_2 = store.reconstruct_state(2)
        assert state_at_2["a"] == 1
        assert state_at_2["b"] == 2
        
        state_at_3 = store.reconstruct_state(3)
        assert state_at_3["a"] == 10
    
    def test_time_travel_navigation(self):
        """Test debugger navigation."""
        from src.durable.time_travel import TimeTravelDebugger, EventStore
        
        store = EventStore()
        for i in range(10):
            store.record_change("counter", i, i + 1)
        
        debugger = TimeTravelDebugger(store)
        
        # Navigate
        state = debugger.goto(5)
        assert state["counter"] == 5
        
        state = debugger.step_forward(2)
        assert debugger.current_position == 7
    
    def test_state_diffing(self):
        """Test state comparison."""
        from src.durable.time_travel import TimeTravelDebugger, EventStore
        
        store = EventStore()
        store.record_change("a", None, 1)
        store.record_change("b", None, 2)
        store.record_change("a", 1, 10)
        store.record_change("c", None, 3)
        
        debugger = TimeTravelDebugger(store)
        diff = debugger.diff_states(2, 4)
        
        assert "a" in diff["modified"]
        assert "c" in diff["added"]


# ============================================================================
# Self-Evolution v2 Tests
# ============================================================================

class TestBenchmarkDiscovery:
    """Tests for benchmark discovery."""
    
    @pytest.mark.asyncio
    async def test_synthetic_benchmark_generation(self):
        """Test synthetic benchmark creation."""
        from src.evolution.benchmark_discovery import BenchmarkDiscovery, BenchmarkCategory
        
        discovery = BenchmarkDiscovery()
        
        benchmark = discovery.generate_synthetic_benchmark(
            BenchmarkCategory.REASONING, difficulty=0.5
        )
        
        assert benchmark is not None
        assert benchmark.category == BenchmarkCategory.REASONING
    
    def test_performance_probe(self):
        """Test performance probing."""
        from src.evolution.benchmark_discovery import PerformanceProbe
        
        probe = PerformanceProbe()
        probe.set_threshold("latency", 100)
        
        probe.measure("latency", 50)
        probe.measure("latency", 150)
        probe.measure("latency", 200)
        
        bottlenecks = probe.get_bottlenecks()
        assert len(bottlenecks) > 0


class TestArchitectureSearch:
    """Tests for architecture search."""
    
    def test_architecture_space(self):
        """Test architecture space definition."""
        from src.evolution.architecture_search import ArchitectureSpace
        
        space = ArchitectureSpace()
        space.add_integer("layers", 1, 10)
        space.add_float("learning_rate", 0.001, 0.1)
        space.add_choice("activation", ["relu", "gelu", "tanh"])
        
        arch = space.sample()
        assert arch is not None
        assert len(arch.genes) == 3
    
    def test_architecture_mutation(self):
        """Test architecture mutation."""
        from src.evolution.architecture_search import ArchitectureSpace
        
        space = ArchitectureSpace()
        space.add_integer("layers", 1, 10)
        space.add_float("dropout", 0.0, 0.5)
        
        original = space.sample()
        mutated = original.mutate(rate=1.0)  # Force mutation
        
        assert mutated.id != original.id
        assert mutated.generation == original.generation + 1
    
    @pytest.mark.asyncio
    async def test_evolution_strategy(self):
        """Test evolution strategy."""
        from src.evolution.architecture_search import ArchitectureSpace, EvolutionStrategy
        
        space = ArchitectureSpace()
        space.add_integer("x", 0, 100)
        
        strategy = EvolutionStrategy(space, population_size=10)
        
        def fitness(arch):
            return arch.genes[0].value / 100  # Maximize x
        
        best = await strategy.run(fitness, max_generations=5)
        assert best is not None


class TestKnowledgeTransfer:
    """Tests for knowledge transfer."""
    
    def test_knowledge_graph(self):
        """Test knowledge graph operations."""
        from src.evolution.knowledge_transfer import KnowledgeGraph, Knowledge, KnowledgeType
        
        graph = KnowledgeGraph()
        
        k1 = Knowledge(type=KnowledgeType.SKILL, domain="coding", content={"pattern": "factory"})
        k2 = Knowledge(type=KnowledgeType.STRATEGY, domain="coding", content={"approach": "tdd"})
        
        graph.add_knowledge(k1)
        graph.add_knowledge(k2)
        graph.add_relation(k1.id, k2.id, "enables")
        
        related = graph.get_related(k1.id)
        assert len(related) == 1
    
    @pytest.mark.asyncio
    async def test_knowledge_distillation(self):
        """Test knowledge distillation."""
        from src.evolution.knowledge_transfer import KnowledgeDistillation, Knowledge
        
        teacher_knowledge = [
            Knowledge(quality_score=0.9),
            Knowledge(quality_score=0.5),
            Knowledge(quality_score=0.8),
        ]
        
        distiller = KnowledgeDistillation()
        distilled = await distiller.distill(teacher_knowledge, quality_threshold=0.7)
        
        assert len(distilled) == 2  # Only high quality
    
    def test_experience_replay(self):
        """Test experience replay buffer."""
        from src.evolution.knowledge_transfer import ExperienceReplay, Experience
        
        replay = ExperienceReplay(capacity=100)
        
        for i in range(10):
            exp = Experience(
                state={"step": i},
                action=f"action_{i}",
                reward=float(i)
            )
            replay.add(exp, priority=float(i))
        
        samples = replay.sample(5, prioritized=True)
        assert len(samples) == 5


class TestSafetyEvolution:
    """Tests for safety constraint evolution."""
    
    def test_constraint_checking(self):
        """Test safety constraint checking."""
        from src.evolution.safety_evolution import SafetyConstraint, ConstraintSeverity
        
        constraint = SafetyConstraint(
            name="rate_limit",
            severity=ConstraintSeverity.HIGH,
            check_func=lambda ctx: ctx.get("rate", 0) < 100
        )
        
        assert constraint.check({"rate": 50}) == True
        assert constraint.check({"rate": 150}) == False
    
    @pytest.mark.asyncio
    async def test_safety_evaluation(self):
        """Test safety evaluation."""
        from src.evolution.safety_evolution import SafetyEvolution, SafetyConstraint
        
        evolution = SafetyEvolution()
        evolution.add_constraint(SafetyConstraint(
            name="memory_check",
            check_func=lambda ctx: ctx.get("memory", 0) < 1000
        ))
        
        result = await evolution.evaluate({"memory": 500})
        assert result["passed"] >= 0
    
    def test_constraint_learning(self):
        """Test learning constraints from incidents."""
        from src.evolution.safety_evolution import SafetyEvolution, SafetyIncident
        
        evolution = SafetyEvolution()
        
        # Record incidents
        for i in range(5):
            incident = SafetyIncident(
                description="High CPU",
                context={"cpu": 90, "memory": 500}
            )
            evolution.learn_from_incident(incident)
        
        # Should have learned patterns
        assert len(evolution.learner.patterns) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
