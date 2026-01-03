"""
AION Durable Execution - Resumable Workflows
=============================================

Resumable workflow engine:
- Workflow State Machine: Explicit state transitions
- Compensation Logic: Automatic rollback on failure
- Retry Policies: Configurable retry with backoff
- Saga Pattern: Multi-step transaction coordination

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum


class WorkflowState(Enum):
    """States a workflow step can be in."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    action: Optional[Callable] = None
    compensation: Optional[Callable] = None  # Undo action
    state: WorkflowState = WorkflowState.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retryable_exceptions: tuple = (Exception,)


class ResumableWorkflow:
    """A workflow that can be paused and resumed."""
    
    def __init__(self, name: str, retry_config: RetryConfig = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.steps: List[WorkflowStep] = []
        self.current_step: int = 0
        self.state = WorkflowState.PENDING
        self.context: Dict[str, Any] = {}
        self.retry_config = retry_config or RetryConfig()
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None
    
    def add_step(self, name: str, action: Callable, 
                 compensation: Callable = None) -> 'ResumableWorkflow':
        """Add a step to the workflow (fluent interface)."""
        step = WorkflowStep(
            name=name,
            action=action,
            compensation=compensation,
            max_retries=self.retry_config.max_attempts
        )
        self.steps.append(step)
        return self
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the workflow from current position."""
        self.state = WorkflowState.RUNNING
        
        try:
            while self.current_step < len(self.steps):
                step = self.steps[self.current_step]
                
                success = await self._execute_step(step)
                
                if not success:
                    await self._compensate()
                    return {"status": "failed", "step": step.name, "error": step.error}
                
                self.current_step += 1
            
            self.state = WorkflowState.COMPLETED
            self.completed_at = datetime.now()
            
            return {
                "status": "completed",
                "results": {s.name: s.result for s in self.steps}
            }
            
        except Exception as e:
            self.state = WorkflowState.FAILED
            return {"status": "error", "error": str(e)}
    
    async def _execute_step(self, step: WorkflowStep) -> bool:
        """Execute a single step with retries."""
        step.state = WorkflowState.RUNNING
        step.started_at = datetime.now()
        
        delay = self.retry_config.initial_delay
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(step.action):
                    step.result = await step.action(self.context)
                else:
                    step.result = step.action(self.context)
                
                step.state = WorkflowState.COMPLETED
                step.completed_at = datetime.now()
                return True
                
            except self.retry_config.retryable_exceptions as e:
                step.retry_count = attempt + 1
                step.error = str(e)
                
                if attempt < self.retry_config.max_attempts - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * self.retry_config.backoff_factor,
                               self.retry_config.max_delay)
        
        step.state = WorkflowState.FAILED
        return False
    
    async def _compensate(self):
        """Run compensation for completed steps in reverse."""
        for i in range(self.current_step, -1, -1):
            step = self.steps[i]
            if step.state == WorkflowState.COMPLETED and step.compensation:
                step.state = WorkflowState.COMPENSATING
                try:
                    if asyncio.iscoroutinefunction(step.compensation):
                        await step.compensation(self.context, step.result)
                    else:
                        step.compensation(self.context, step.result)
                    step.state = WorkflowState.COMPENSATED
                except Exception:
                    pass  # Log but continue compensating
    
    def get_status(self) -> Dict[str, Any]:
        """Get workflow status."""
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "steps": [{"name": s.name, "state": s.state.value} for s in self.steps]
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize workflow state for persistence."""
        return {
            "id": self.id,
            "name": self.name,
            "current_step": self.current_step,
            "state": self.state.value,
            "context": self.context,
            "steps": [
                {"name": s.name, "state": s.state.value, "result": s.result}
                for s in self.steps
            ]
        }


class WorkflowExecutor:
    """Manages workflow execution and persistence."""
    
    def __init__(self):
        self.workflows: Dict[str, ResumableWorkflow] = {}
        self.running: Dict[str, asyncio.Task] = {}
    
    def register(self, workflow: ResumableWorkflow):
        """Register a workflow."""
        self.workflows[workflow.id] = workflow
    
    async def run(self, workflow_id: str) -> Dict[str, Any]:
        """Run a workflow."""
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.workflows[workflow_id]
        task = asyncio.create_task(workflow.execute())
        self.running[workflow_id] = task
        
        return await task
    
    async def pause(self, workflow_id: str):
        """Pause a running workflow."""
        if workflow_id in self.running:
            self.running[workflow_id].cancel()
            del self.running[workflow_id]
    
    async def resume(self, workflow_id: str) -> Dict[str, Any]:
        """Resume a paused workflow."""
        return await self.run(workflow_id)


@dataclass
class SagaStep:
    """A step in a saga transaction."""
    name: str
    action: Callable
    compensation: Callable
    timeout: float = 30.0


class SagaCoordinator:
    """Coordinates saga transactions across distributed steps."""
    
    def __init__(self):
        self.sagas: Dict[str, List[SagaStep]] = {}
        self.completed_steps: Dict[str, List[str]] = {}
    
    def define_saga(self, saga_id: str, steps: List[SagaStep]):
        """Define a new saga."""
        self.sagas[saga_id] = steps
        self.completed_steps[saga_id] = []
    
    async def execute_saga(self, saga_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a saga with compensation on failure."""
        if saga_id not in self.sagas:
            return {"error": "Saga not found"}
        
        steps = self.sagas[saga_id]
        results = {}
        
        for step in steps:
            try:
                if asyncio.iscoroutinefunction(step.action):
                    result = await asyncio.wait_for(
                        step.action(context), timeout=step.timeout
                    )
                else:
                    result = step.action(context)
                
                results[step.name] = result
                self.completed_steps[saga_id].append(step.name)
                
            except Exception as e:
                # Compensate
                await self._compensate_saga(saga_id, context)
                return {"status": "compensated", "failed_at": step.name, "error": str(e)}
        
        return {"status": "completed", "results": results}
    
    async def _compensate_saga(self, saga_id: str, context: Dict[str, Any]):
        """Run compensations in reverse order."""
        steps = self.sagas[saga_id]
        completed = self.completed_steps[saga_id]
        
        for step in reversed(steps):
            if step.name in completed:
                try:
                    if asyncio.iscoroutinefunction(step.compensation):
                        await step.compensation(context)
                    else:
                        step.compensation(context)
                except Exception:
                    pass  # Log but continue


async def demo_workflows():
    """Demonstrate resumable workflows."""
    print("🔄 Resumable Workflows Demo")
    print("=" * 50)
    
    # Create a workflow
    async def step1(ctx):
        ctx['order_id'] = 'ORD-123'
        return 'Order created'
    
    async def step2(ctx):
        ctx['payment_id'] = 'PAY-456'
        return 'Payment processed'
    
    async def step3(ctx):
        ctx['shipment_id'] = 'SHIP-789'
        return 'Order shipped'
    
    async def comp1(ctx, result):
        print(f"  Compensating: Cancelling order {ctx.get('order_id')}")
    
    workflow = ResumableWorkflow("order-processing")
    workflow.add_step("create_order", step1, comp1)
    workflow.add_step("process_payment", step2)
    workflow.add_step("ship_order", step3)
    
    executor = WorkflowExecutor()
    executor.register(workflow)
    
    print("\n▶️ Executing workflow...")
    result = await executor.run(workflow.id)
    print(f"  Result: {result}")
    print(f"  Status: {workflow.get_status()}")
    
    # Demo saga
    print("\n📋 Saga Transaction Demo:")
    saga = SagaCoordinator()
    
    saga.define_saga("order-saga", [
        SagaStep("reserve", lambda ctx: "reserved", lambda ctx: print("  Undo reserve")),
        SagaStep("charge", lambda ctx: "charged", lambda ctx: print("  Refund")),
        SagaStep("ship", lambda ctx: "shipped", lambda ctx: print("  Cancel shipment")),
    ])
    
    result = await saga.execute_saga("order-saga", {})
    print(f"  Saga result: {result}")
    
    print("\n✅ Workflows demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_workflows())
