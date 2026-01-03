"""
AION Embodied AI Module
========================

Provides sensor/actuator interfaces, ROS2 integration, and
simulation environments for embodied AI and robotics research.
"""

from .sensors import SensorStream, SensorFusion, SensorType, SensorReading, SensorConfig
from .actuators import ActuatorController, ActuatorCommand, ActuatorState, ActuatorType, SafetyLimits
from .ros2_bridge import ROS2Bridge, ROS2Message, ROS2Topic, ROS2Service
from .simulation import SimulationEnvironment, GenericSimulator, Episode, SimulationState

__all__ = [
    # Sensors
    "SensorStream",
    "SensorFusion",
    "SensorType",
    "SensorReading",
    "SensorConfig",
    # Actuators
    "ActuatorController",
    "ActuatorCommand",
    "ActuatorState",
    "ActuatorType",
    "SafetyLimits",
    # ROS2
    "ROS2Bridge",
    "ROS2Message",
    "ROS2Topic",
    "ROS2Service",
    # Simulation
    "SimulationEnvironment",
    "GenericSimulator",
    "Episode",
    "SimulationState",
]
