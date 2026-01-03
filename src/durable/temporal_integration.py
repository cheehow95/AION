"""
AION Durable Execution - Temporal.io Integration
=================================================

Temporal.io-inspired workflow orchestration:
- Workflow Definitions: Declarative workflow specifications
- Activity Execution: Side-effect isolated task execution
- Signal Handling: External event injection into workflows
- Query Interface: Runtime workflow state inspection

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from enum import Enum
import functools
import json
import traceback


T = TypeVar('T')


class WorkflowStatus(Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    PAUSED = "paused"


class ActivityStatus(Enum):
    """Status of an activity execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class ActivityOptions:
    """Options for activity execution."""
    timeout: float = 60.0
    retry_policy: Optional['RetryPolicy'] = None
    heartbeat_timeout: float = 30.0


@dataclass
class RetryPolicy:
    """Retry policy for activities."""
    max_attempts: int = 3
    initial_interval: float = 1.0
    backoff_coefficient: float = 2.0
    max_interval: float = 60.0


@dataclass
class Activity:
    """An activity that can be executed within a workflow."""
    name: str
    handler: Callable
    options: ActivityOptions = field(default_factory=ActivityOptions)
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the activity with timeout."""
        try:
            return await asyncio.wait_for(
                self.handler(*args, **kwargs),
                timeout=self.options.timeout
            )
        except asyncio.TimeoutError:
            raise ActivityTimeoutError(f"Activity {self.name} timed out")


class ActivityTimeoutError(Exception):
    pass


@dataclass
class WorkflowContext:
    """Context for workflow execution."""
    workflow_id: str
    run_id: str
    attempt: int = 1
    start_time: datetime = field(default_factory=datetime.now)
    memo: Dict[str, Any] = field(default_factory=dict)
    
    # Signal handling
    pending_signals: Dict[str, asyncio.Queue] = field(default_factory=dict)
    
    # Query handling
    query_handlers: Dict[str, Callable] = field(default_factory=dict)
    
    # State
    state: Dict[str, Any] = field(default_factory=dict)
    
    async def wait_for_signal(self, signal_name: str, timeout: float = None) -> Any:
        """Wait for an external signal."""
        if signal_name not in self.pending_signals:
            self.pending_signals[signal_name] = asyncio.Queue()
        
        try:
            return await asyncio.wait_for(
                self.pending_signals[signal_name].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
    
    def set_query_handler(self, name: str, handler: Callable):
        """Register a query handler."""
        self.query_handlers[name] = handler
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get workflow state."""
        return self.state.get(key, default)
    
    def set_state(self, key: str, value: Any):
        """Set workflow state."""
        self.state[key] = value


@dataclass
class WorkflowExecution:
    """A single workflow execution instance."""
    workflow_id: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_type: str = ""
    status: WorkflowStatus = WorkflowStatus.PENDING
    input: Any = None
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    context: Optional[WorkflowContext] = None
    
    # History
    events: List[Dict[str, Any]] = field(default_factory=list)


class WorkflowDefinition(ABC):
    """Base class for workflow definitions."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.activities: Dict[str, Activity] = {}
    
    def register_activity(self, name: str, handler: Callable, 
                          options: ActivityOptions = None):
        """Register an activity with the workflow."""
        self.activities[name] = Activity(
            name=name,
            handler=handler,
            options=options or ActivityOptions()
        )
    
    @abstractmethod
    async def run(self, ctx: WorkflowContext, input: Any) -> Any:
        """Main workflow logic."""
        pass
    
    async def execute_activity(self, ctx: WorkflowContext, 
                               name: str, *args, **kwargs) -> Any:
        """Execute a registered activity."""
        if name not in self.activities:
            raise ValueError(f"Unknown activity: {name}")
        
        activity = self.activities[name]
        retry_policy = activity.options.retry_policy or RetryPolicy()
        
        last_error = None
        for attempt in range(retry_policy.max_attempts):
            try:
                result = await activity.execute(*args, **kwargs)
                
                # Record event
                ctx.memo[f"activity_{name}_{len(ctx.memo)}"] = {
                    'activity': name,
                    'status': 'completed',
                    'result': str(result)[:100],
                    'attempt': attempt + 1
                }
                
                return result
                
            except Exception as e:
                last_error = e
                if attempt < retry_policy.max_attempts - 1:
                    delay = retry_policy.initial_interval * (
                        retry_policy.backoff_coefficient ** attempt
                    )
                    delay = min(delay, retry_policy.max_interval)
                    await asyncio.sleep(delay)
        
        raise last_error


class TemporalWorkflowEngine:
    """Workflow execution engine inspired by Temporal.io."""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a workflow definition."""
        self.workflows[workflow.name] = workflow
    
    async def start_workflow(self, workflow_name: str, 
                             workflow_id: str = None,
                             input: Any = None) -> str:
        """Start a new workflow execution."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow_id = workflow_id or str(uuid.uuid4())
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_type=workflow_name,
            input=input
        )
        
        self.executions[workflow_id] = execution
        
        # Start execution
        task = asyncio.create_task(
            self._run_workflow(workflow_id)
        )
        self.running_tasks[workflow_id] = task
        
        return workflow_id
    
    async def _run_workflow(self, workflow_id: str):
        """Internal workflow execution."""
        execution = self.executions[workflow_id]
        workflow = self.workflows[execution.workflow_type]
        
        # Create context
        ctx = WorkflowContext(
            workflow_id=workflow_id,
            run_id=execution.run_id
        )
        execution.context = ctx
        execution.status = WorkflowStatus.RUNNING
        execution.start_time = datetime.now()
        
        try:
            result = await workflow.run(ctx, execution.input)
            execution.result = result
            execution.status = WorkflowStatus.COMPLETED
        except asyncio.CancelledError:
            execution.status = WorkflowStatus.CANCELLED
        except Exception as e:
            execution.error = str(e)
            execution.status = WorkflowStatus.FAILED
        finally:
            execution.end_time = datetime.now()
    
    async def signal_workflow(self, workflow_id: str, 
                              signal_name: str, data: Any = None):
        """Send a signal to a running workflow."""
        execution = self.executions.get(workflow_id)
        if not execution or not execution.context:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        if signal_name not in execution.context.pending_signals:
            execution.context.pending_signals[signal_name] = asyncio.Queue()
        
        await execution.context.pending_signals[signal_name].put(data)
    
    def query_workflow(self, workflow_id: str, 
                       query_name: str, *args) -> Any:
        """Query a workflow's state."""
        execution = self.executions.get(workflow_id)
        if not execution or not execution.context:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        handler = execution.context.query_handlers.get(query_name)
        if not handler:
            raise ValueError(f"Unknown query: {query_name}")
        
        return handler(*args)
    
    async def cancel_workflow(self, workflow_id: str):
        """Cancel a running workflow."""
        task = self.running_tasks.get(workflow_id)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    def get_execution(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution details."""
        return self.executions.get(workflow_id)
    
    def get_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """Get workflow status."""
        execution = self.executions.get(workflow_id)
        return execution.status if execution else None


# Decorator for defining activities
def activity(name: str = None, timeout: float = 60.0, 
             max_retries: int = 3):
    """Decorator to define an activity."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        
        wrapper._activity_name = name or func.__name__
        wrapper._activity_options = ActivityOptions(
            timeout=timeout,
            retry_policy=RetryPolicy(max_attempts=max_retries)
        )
        return wrapper
    return decorator


async def demo_temporal():
    """Demonstrate Temporal-style workflow execution."""
    print("⏱️ Temporal Workflow Engine Demo")
    print("=" * 50)
    
    # Define a sample workflow
    class OrderProcessingWorkflow(WorkflowDefinition):
        def __init__(self):
            super().__init__("OrderProcessing")
            
            # Register activities
            self.register_activity("validate_order", self._validate)
            self.register_activity("charge_payment", self._charge)
            self.register_activity("ship_order", self._ship)
        
        async def _validate(self, order: Dict) -> bool:
            await asyncio.sleep(0.1)
            return order.get('amount', 0) > 0
        
        async def _charge(self, amount: float) -> str:
            await asyncio.sleep(0.1)
            return f"txn_{uuid.uuid4().hex[:8]}"
        
        async def _ship(self, address: str) -> str:
            await asyncio.sleep(0.1)
            return f"track_{uuid.uuid4().hex[:8]}"
        
        async def run(self, ctx: WorkflowContext, input: Dict) -> Dict:
            # Set up query handler
            ctx.set_query_handler("get_status", lambda: ctx.get_state("current_step"))
            
            ctx.set_state("current_step", "validating")
            valid = await self.execute_activity(ctx, "validate_order", input)
            if not valid:
                return {"status": "failed", "reason": "validation"}
            
            ctx.set_state("current_step", "charging")
            txn = await self.execute_activity(ctx, "charge_payment", input['amount'])
            
            ctx.set_state("current_step", "shipping")
            tracking = await self.execute_activity(ctx, "ship_order", input['address'])
            
            return {"status": "completed", "transaction": txn, "tracking": tracking}
    
    # Create engine and register workflow
    engine = TemporalWorkflowEngine()
    engine.register_workflow(OrderProcessingWorkflow())
    
    # Start workflow
    order = {"amount": 99.99, "address": "123 Main St"}
    workflow_id = await engine.start_workflow("OrderProcessing", input=order)
    print(f"  Started workflow: {workflow_id}")
    
    # Wait for completion
    await asyncio.sleep(0.5)
    
    # Check result
    execution = engine.get_execution(workflow_id)
    print(f"  Status: {execution.status.value}")
    print(f"  Result: {execution.result}")
    
    print("\n✅ Temporal demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_temporal())
