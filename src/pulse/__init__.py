"""
AION Pulse Task Automation - Package Initialization
====================================================

Automated prompt scheduling like GPT-5.2 Pulse:
- Task scheduling and triggers
- Recurring task management
- Centralized dashboard
"""

from .scheduler import (
    ScheduledTask,
    TaskScheduler,
    ScheduleType
)

from .triggers import (
    Trigger,
    TimeTrigger,
    EventTrigger,
    ConditionTrigger,
    TriggerManager
)

__all__ = [
    # Scheduler
    'ScheduledTask',
    'TaskScheduler',
    'ScheduleType',
    # Triggers
    'Trigger',
    'TimeTrigger',
    'EventTrigger',
    'ConditionTrigger',
    'TriggerManager',
]
