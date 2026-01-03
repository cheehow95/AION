"""
AION Actuator Control Protocol
==============================

Actuator command interface with safety limits,
state monitoring, and emergency stop capabilities.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import uuid


# =============================================================================
# ACTUATOR TYPES
# =============================================================================

class ActuatorType(Enum):
    """Types of actuators."""
    MOTOR = "motor"
    SERVO = "servo"
    STEPPER = "stepper"
    LINEAR = "linear"
    GRIPPER = "gripper"
    PNEUMATIC = "pneumatic"
    HYDRAULIC = "hydraulic"
    LED = "led"
    SPEAKER = "speaker"
    DISPLAY = "display"
    RELAY = "relay"
    VALVE = "valve"


class ControlMode(Enum):
    """Control modes for actuators."""
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    EFFORT = "effort"
    PWM = "pwm"
    OPEN_LOOP = "open_loop"


class ActuatorStatus(Enum):
    """Status of an actuator."""
    IDLE = "idle"
    MOVING = "moving"
    HOLDING = "holding"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"
    DISABLED = "disabled"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SafetyLimits:
    """Safety limits for an actuator."""
    # Position limits
    min_position: float = -float("inf")
    max_position: float = float("inf")
    
    # Velocity limits
    max_velocity: float = float("inf")
    max_acceleration: float = float("inf")
    
    # Effort/torque limits
    max_effort: float = float("inf")
    max_torque: float = float("inf")
    
    # Temperature
    max_temperature: float = 80.0  # Celsius
    
    # Current
    max_current: float = float("inf")  # Amperes
    
    # Soft limits (warning zone)
    soft_limit_margin: float = 0.1  # 10% before hard limit
    
    def check_position(self, position: float) -> bool:
        """Check if position is within limits."""
        return self.min_position <= position <= self.max_position
    
    def check_velocity(self, velocity: float) -> bool:
        """Check if velocity is within limits."""
        return abs(velocity) <= self.max_velocity
    
    def clamp_position(self, position: float) -> float:
        """Clamp position to limits."""
        return max(self.min_position, min(self.max_position, position))


@dataclass
class ActuatorState:
    """Current state of an actuator."""
    actuator_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: ActuatorStatus = ActuatorStatus.IDLE
    
    # Position/velocity
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    
    # Effort
    effort: float = 0.0
    torque: float = 0.0
    
    # Sensor feedback
    temperature: float = 25.0
    current: float = 0.0
    
    # Target
    target_position: Optional[float] = None
    target_velocity: Optional[float] = None
    
    # Error
    error_code: int = 0
    error_message: str = ""
    
    # Quality
    is_calibrated: bool = True
    
    @property
    def at_target(self) -> bool:
        """Check if actuator is at target position."""
        if self.target_position is None:
            return True
        return abs(self.position - self.target_position) < 0.01


@dataclass
class ActuatorCommand:
    """Command to send to an actuator."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    actuator_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Control mode
    mode: ControlMode = ControlMode.POSITION
    
    # Target values
    position: Optional[float] = None
    velocity: Optional[float] = None
    effort: Optional[float] = None
    
    # Motion profile
    max_velocity: Optional[float] = None
    max_acceleration: Optional[float] = None
    
    # Duration/timing
    duration_ms: Optional[int] = None
    
    # Metadata
    priority: int = 0
    source: str = ""


@dataclass
class CommandResult:
    """Result of executing a command."""
    command_id: str
    success: bool
    final_state: Optional[ActuatorState] = None
    error_message: str = ""
    execution_time_ms: float = 0.0


# =============================================================================
# ACTUATOR CONTROLLER
# =============================================================================

class ActuatorController:
    """
    Controller for managing actuators with safety limits and state monitoring.
    """
    
    def __init__(self, actuator_id: str, actuator_type: ActuatorType = ActuatorType.MOTOR):
        self.actuator_id = actuator_id
        self.actuator_type = actuator_type
        self._state = ActuatorState(actuator_id=actuator_id)
        self._limits = SafetyLimits()
        self._enabled = False
        self._emergency_stop = False
        self._command_queue: List[ActuatorCommand] = []
        self._state_callbacks: List[Callable[[ActuatorState], None]] = []
        self._running = False
    
    @property
    def state(self) -> ActuatorState:
        return self._state
    
    @property
    def limits(self) -> SafetyLimits:
        return self._limits
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled and not self._emergency_stop
    
    def set_safety_limits(self, limits: SafetyLimits) -> None:
        """Set safety limits for this actuator."""
        self._limits = limits
    
    def enable(self) -> bool:
        """Enable the actuator."""
        if self._emergency_stop:
            return False
        self._enabled = True
        self._state.status = ActuatorStatus.IDLE
        return True
    
    def disable(self) -> None:
        """Disable the actuator."""
        self._enabled = False
        self._state.status = ActuatorStatus.DISABLED
    
    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self._emergency_stop = True
        self._enabled = False
        self._state.status = ActuatorStatus.EMERGENCY_STOP
        self._command_queue.clear()
    
    def reset_emergency_stop(self) -> None:
        """Reset emergency stop (requires re-enabling)."""
        self._emergency_stop = False
        self._state.status = ActuatorStatus.DISABLED
    
    async def send_command(self, command: ActuatorCommand) -> CommandResult:
        """
        Send a command to the actuator.
        
        Args:
            command: Command to execute
            
        Returns:
            CommandResult with execution status
        """
        start_time = datetime.now()
        
        # Check if enabled
        if not self.is_enabled:
            return CommandResult(
                command_id=command.id,
                success=False,
                error_message="Actuator not enabled"
            )
        
        # Validate command against safety limits
        if command.position is not None:
            if not self._limits.check_position(command.position):
                return CommandResult(
                    command_id=command.id,
                    success=False,
                    error_message=f"Position {command.position} out of limits"
                )
        
        if command.velocity is not None:
            if not self._limits.check_velocity(command.velocity):
                command.velocity = min(abs(command.velocity), self._limits.max_velocity)
                if command.velocity < 0:
                    command.velocity = -command.velocity
        
        # Execute command
        try:
            await self._execute_command(command)
            
            exec_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return CommandResult(
                command_id=command.id,
                success=True,
                final_state=self._state,
                execution_time_ms=exec_time
            )
            
        except Exception as e:
            return CommandResult(
                command_id=command.id,
                success=False,
                error_message=str(e)
            )
    
    async def move_to(
        self, 
        position: float, 
        velocity: Optional[float] = None
    ) -> CommandResult:
        """Move to a position."""
        command = ActuatorCommand(
            actuator_id=self.actuator_id,
            mode=ControlMode.POSITION,
            position=position,
            max_velocity=velocity
        )
        return await self.send_command(command)
    
    async def set_velocity(self, velocity: float) -> CommandResult:
        """Set velocity (continuous motion)."""
        command = ActuatorCommand(
            actuator_id=self.actuator_id,
            mode=ControlMode.VELOCITY,
            velocity=velocity
        )
        return await self.send_command(command)
    
    async def set_effort(self, effort: float) -> CommandResult:
        """Set effort/torque."""
        command = ActuatorCommand(
            actuator_id=self.actuator_id,
            mode=ControlMode.EFFORT,
            effort=effort
        )
        return await self.send_command(command)
    
    async def stop(self) -> CommandResult:
        """Stop the actuator."""
        command = ActuatorCommand(
            actuator_id=self.actuator_id,
            mode=ControlMode.VELOCITY,
            velocity=0.0
        )
        return await self.send_command(command)
    
    async def get_state(self) -> ActuatorState:
        """Get current state."""
        return self._state
    
    def subscribe_state(self, callback: Callable[[ActuatorState], None]) -> None:
        """Subscribe to state updates."""
        self._state_callbacks.append(callback)
    
    async def calibrate(self) -> bool:
        """Run calibration routine."""
        # Simulated calibration
        await asyncio.sleep(0.1)
        self._state.is_calibrated = True
        return True
    
    async def _execute_command(self, command: ActuatorCommand) -> None:
        """Execute a command (simulated)."""
        self._state.status = ActuatorStatus.MOVING
        
        if command.mode == ControlMode.POSITION and command.position is not None:
            # Simulate motion
            target = command.position
            self._state.target_position = target
            
            # Simple motion simulation
            steps = 10
            start = self._state.position
            for i in range(steps):
                progress = (i + 1) / steps
                self._state.position = start + (target - start) * progress
                self._state.velocity = (target - start) / 0.1  # Simulated velocity
                
                self._notify_state()
                await asyncio.sleep(0.01)
            
            self._state.position = target
            self._state.velocity = 0.0
        
        elif command.mode == ControlMode.VELOCITY and command.velocity is not None:
            self._state.velocity = command.velocity
            self._state.target_velocity = command.velocity
        
        elif command.mode == ControlMode.EFFORT and command.effort is not None:
            self._state.effort = command.effort
        
        self._state.status = ActuatorStatus.IDLE if self._state.velocity == 0 else ActuatorStatus.MOVING
        self._state.timestamp = datetime.now()
        self._notify_state()
    
    def _notify_state(self) -> None:
        """Notify subscribers of state change."""
        for callback in self._state_callbacks:
            try:
                callback(self._state)
            except Exception:
                pass


# =============================================================================
# MULTI-ACTUATOR CONTROLLER
# =============================================================================

class MultiActuatorController:
    """
    Controller for managing multiple actuators simultaneously.
    """
    
    def __init__(self):
        self._actuators: Dict[str, ActuatorController] = {}
    
    def add_actuator(
        self, 
        actuator_id: str, 
        actuator_type: ActuatorType = ActuatorType.MOTOR,
        limits: SafetyLimits = None
    ) -> ActuatorController:
        """Add a new actuator."""
        controller = ActuatorController(actuator_id, actuator_type)
        if limits:
            controller.set_safety_limits(limits)
        self._actuators[actuator_id] = controller
        return controller
    
    def get_actuator(self, actuator_id: str) -> Optional[ActuatorController]:
        """Get actuator by ID."""
        return self._actuators.get(actuator_id)
    
    def enable_all(self) -> None:
        """Enable all actuators."""
        for actuator in self._actuators.values():
            actuator.enable()
    
    def disable_all(self) -> None:
        """Disable all actuators."""
        for actuator in self._actuators.values():
            actuator.disable()
    
    def emergency_stop_all(self) -> None:
        """Emergency stop all actuators."""
        for actuator in self._actuators.values():
            actuator.emergency_stop()
    
    async def send_synchronized_commands(
        self, 
        commands: List[ActuatorCommand]
    ) -> List[CommandResult]:
        """Send commands to multiple actuators simultaneously."""
        tasks = []
        for command in commands:
            actuator = self._actuators.get(command.actuator_id)
            if actuator:
                tasks.append(actuator.send_command(command))
        
        if tasks:
            return await asyncio.gather(*tasks)
        return []
    
    async def get_all_states(self) -> Dict[str, ActuatorState]:
        """Get states of all actuators."""
        states = {}
        for actuator_id, actuator in self._actuators.items():
            states[actuator_id] = await actuator.get_state()
        return states


# =============================================================================
# DEMO
# =============================================================================

async def demo_actuators():
    """Demonstrate actuator control."""
    print("🦾 Actuator Control Demo")
    print("-" * 40)
    
    # Create actuator with limits
    limits = SafetyLimits(
        min_position=-180.0,
        max_position=180.0,
        max_velocity=90.0,  # deg/s
        max_effort=10.0  # Nm
    )
    
    controller = ActuatorController("joint_1", ActuatorType.SERVO)
    controller.set_safety_limits(limits)
    
    # Enable
    controller.enable()
    print(f"Actuator enabled: {controller.is_enabled}")
    
    # Move to position
    result = await controller.move_to(45.0)
    print(f"Move to 45°: {result.success}")
    print(f"  Position: {result.final_state.position:.1f}°")
    
    # Try out-of-limits
    result = await controller.move_to(200.0)  # Out of limits
    print(f"Move to 200° (out of limits): {result.success}")
    print(f"  Error: {result.error_message}")
    
    # Emergency stop
    controller.emergency_stop()
    print(f"Emergency stop triggered")
    
    result = await controller.move_to(0.0)
    print(f"Move after e-stop: {result.success}")
    print(f"  Error: {result.error_message}")
    
    # Reset
    controller.reset_emergency_stop()
    controller.enable()
    result = await controller.move_to(0.0)
    print(f"Move after reset: {result.success}")
    
    print("-" * 40)
    print("✅ Actuator demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_actuators())
