"""
AION Pulse Task Automation - Scheduler
=======================================

Automated prompt scheduling:
- Cron-like task definitions
- Priority queue management
- Task lifecycle management
- Execution history

Matches GPT-5.2 Pulse feature.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
import heapq


class ScheduleType(Enum):
    """Types of schedules."""
    ONCE = "once"           # Run once at specified time
    RECURRING = "recurring"  # Run on schedule
    INTERVAL = "interval"    # Run every N seconds
    CRON = "cron"           # Cron-like schedule


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ScheduledTask:
    """A scheduled task."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    prompt: str = ""
    schedule_type: ScheduleType = ScheduleType.ONCE
    next_run: datetime = field(default_factory=datetime.now)
    interval_seconds: int = 0
    cron_expression: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10, lower is higher priority
    max_retries: int = 3
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    last_result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        # For heap ordering - earlier time and lower priority number first
        return (self.next_run, self.priority) < (other.next_run, other.priority)


@dataclass
class ExecutionResult:
    """Result of task execution."""
    task_id: str = ""
    success: bool = False
    output: str = ""
    error: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0


class TaskScheduler:
    """Schedules and manages automated tasks."""
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_queue: List[ScheduledTask] = []  # Min-heap by next_run
        self.execution_history: List[ExecutionResult] = []
        self.running = False
        self._executor: Optional[Callable] = None
    
    def set_executor(self, executor: Callable):
        """Set the task executor function."""
        self._executor = executor
    
    def schedule(self, task: ScheduledTask) -> str:
        """Schedule a task."""
        self.tasks[task.id] = task
        heapq.heappush(self.task_queue, task)
        return task.id
    
    def schedule_prompt(self, name: str, prompt: str,
                        run_at: datetime = None,
                        interval_seconds: int = None,
                        priority: int = 5) -> str:
        """Convenience method to schedule a prompt."""
        schedule_type = ScheduleType.ONCE
        if interval_seconds:
            schedule_type = ScheduleType.INTERVAL
        
        task = ScheduledTask(
            name=name,
            prompt=prompt,
            schedule_type=schedule_type,
            next_run=run_at or datetime.now(),
            interval_seconds=interval_seconds or 0,
            priority=priority
        )
        
        return self.schedule(task)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.CANCELLED
            return True
        return False
    
    def pause_task(self, task_id: str) -> bool:
        """Pause a scheduled task."""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.PAUSED
            return True
        return False
    
    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.PAUSED:
            self.tasks[task_id].status = TaskStatus.PENDING
            return True
        return False
    
    async def run_task(self, task: ScheduledTask) -> ExecutionResult:
        """Execute a single task."""
        result = ExecutionResult(task_id=task.id)
        task.status = TaskStatus.RUNNING
        task.last_run = datetime.now()
        
        try:
            if self._executor:
                if asyncio.iscoroutinefunction(self._executor):
                    output = await self._executor(task.prompt)
                else:
                    output = self._executor(task.prompt)
                result.output = str(output)
            else:
                result.output = f"[DRY RUN] Would execute: {task.prompt[:50]}..."
            
            result.success = True
            task.status = TaskStatus.COMPLETED
            task.last_result = result.output
        except Exception as e:
            result.error = str(e)
            result.success = False
            task.status = TaskStatus.FAILED
            task.retry_count += 1
        
        result.completed_at = datetime.now()
        result.duration_ms = (result.completed_at - result.started_at).total_seconds() * 1000
        
        self.execution_history.append(result)
        
        # Reschedule if recurring
        if task.schedule_type == ScheduleType.INTERVAL and result.success:
            task.next_run = datetime.now() + timedelta(seconds=task.interval_seconds)
            task.status = TaskStatus.PENDING
            heapq.heappush(self.task_queue, task)
        
        return result
    
    async def run_due_tasks(self) -> List[ExecutionResult]:
        """Run all tasks that are due."""
        results = []
        now = datetime.now()
        
        while self.task_queue and self.task_queue[0].next_run <= now:
            task = heapq.heappop(self.task_queue)
            
            if task.status in [TaskStatus.CANCELLED, TaskStatus.PAUSED]:
                continue
            
            result = await self.run_task(task)
            results.append(result)
        
        return results
    
    async def start(self, check_interval: float = 1.0):
        """Start the scheduler loop."""
        self.running = True
        while self.running:
            await self.run_due_tasks()
            await asyncio.sleep(check_interval)
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
    
    def get_upcoming(self, limit: int = 10) -> List[ScheduledTask]:
        """Get upcoming scheduled tasks."""
        pending = [t for t in self.tasks.values() 
                   if t.status == TaskStatus.PENDING]
        pending.sort(key=lambda t: t.next_run)
        return pending[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        by_status = {}
        for task in self.tasks.values():
            by_status[task.status.value] = by_status.get(task.status.value, 0) + 1
        
        success_rate = (
            sum(1 for r in self.execution_history if r.success) /
            len(self.execution_history)
            if self.execution_history else 0
        )
        
        return {
            'total_tasks': len(self.tasks),
            'by_status': by_status,
            'queue_size': len(self.task_queue),
            'executions': len(self.execution_history),
            'success_rate': success_rate,
            'running': self.running
        }


async def demo_scheduler():
    """Demonstrate task scheduler."""
    print("⏰ Pulse Task Scheduler Demo")
    print("=" * 50)
    
    scheduler = TaskScheduler()
    
    # Set executor
    scheduler.set_executor(lambda prompt: f"Executed: {prompt}")
    
    # Schedule some tasks
    now = datetime.now()
    
    scheduler.schedule_prompt(
        "Daily Summary",
        "Generate a summary of today's activities",
        run_at=now + timedelta(seconds=1),
        priority=1
    )
    
    scheduler.schedule_prompt(
        "Check Emails",
        "Check and summarize new emails",
        run_at=now + timedelta(seconds=2),
        interval_seconds=60  # Recurring
    )
    
    scheduler.schedule_prompt(
        "Low Priority Task",
        "Do something less important",
        run_at=now + timedelta(seconds=1),
        priority=10
    )
    
    print(f"\n📋 Scheduled Tasks: {len(scheduler.tasks)}")
    print(f"📊 Stats: {scheduler.get_stats()}")
    
    # Get upcoming
    print("\n📅 Upcoming Tasks:")
    for task in scheduler.get_upcoming():
        print(f"   • {task.name} at {task.next_run.strftime('%H:%M:%S')} (priority: {task.priority})")
    
    # Wait and run due tasks
    print("\n⏳ Waiting for tasks to become due...")
    await asyncio.sleep(1.5)
    
    results = await scheduler.run_due_tasks()
    print(f"\n✅ Executed {len(results)} tasks")
    for result in results:
        task = scheduler.tasks[result.task_id]
        print(f"   • {task.name}: {result.success} ({result.duration_ms:.0f}ms)")
    
    print(f"\n📊 Final Stats: {scheduler.get_stats()}")
    print("\n✅ Scheduler demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_scheduler())
