"""Tests for AION Embodied AI Module"""

import pytest
import asyncio
import sys
sys.path.insert(0, 'd:/Time/aion')

from src.embodied.sensors import (
    SensorStream, SensorFusion, SensorType, SensorReading, SensorConfig
)
from src.embodied.actuators import (
    ActuatorController, ActuatorCommand, ActuatorState, ActuatorType,
    SafetyLimits, ControlMode, MultiActuatorController
)
from src.embodied.ros2_bridge import (
    ROS2Bridge, ROS2Message, ROS2Topic, QoSProfile
)
from src.embodied.simulation import (
    GenericSimulator, SimulationConfig, Episode, ActionSpace, ObservationSpace, EpisodeManager
)


class TestSensors:
    """Test sensor interface."""
    
    def test_sensor_config_creation(self):
        """Test SensorConfig creation."""
        config = SensorConfig(
            sensor_id="imu_0",
            sensor_type=SensorType.IMU,
            name="body_imu",
            sample_rate_hz=100.0
        )
        assert config.sensor_id == "imu_0"
        assert config.sensor_type == SensorType.IMU
        assert config.sample_rate_hz == 100.0
    
    def test_sensor_reading_creation(self):
        """Test SensorReading creation."""
        reading = SensorReading(
            sensor_id="gps_0",
            sensor_type=SensorType.GPS,
            data={"latitude": 37.7749}
        )
        assert reading.sensor_id == "gps_0"
        assert reading.valid == True
    
    @pytest.mark.asyncio
    async def test_sensor_stream(self):
        """Test sensor streaming."""
        config = SensorConfig(
            sensor_id="test_sensor",
            sensor_type=SensorType.IMU,
            sample_rate_hz=10.0
        )
        stream = SensorStream(config)
        
        await stream.start()
        await asyncio.sleep(0.15)
        
        reading = await stream.get_reading()
        assert reading is not None
        
        await stream.stop()
    
    @pytest.mark.asyncio
    async def test_sensor_fusion(self):
        """Test sensor fusion."""
        imu_config = SensorConfig(
            sensor_id="imu",
            sensor_type=SensorType.IMU,
            sample_rate_hz=50.0
        )
        
        fusion = SensorFusion()
        fusion.add_sensor(SensorStream(imu_config))
        
        await fusion.start()
        await asyncio.sleep(0.1)
        
        state = await fusion.get_state()
        assert state is not None
        
        await fusion.stop()


class TestActuators:
    """Test actuator control."""
    
    def test_safety_limits(self):
        """Test SafetyLimits."""
        limits = SafetyLimits(
            min_position=-180.0,
            max_position=180.0,
            max_velocity=90.0
        )
        
        assert limits.check_position(0.0) == True
        assert limits.check_position(200.0) == False
        assert limits.clamp_position(200.0) == 180.0
    
    def test_actuator_state(self):
        """Test ActuatorState."""
        state = ActuatorState(
            actuator_id="joint_1",
            position=45.0,
            target_position=45.0
        )
        assert state.at_target == True
    
    @pytest.mark.asyncio
    async def test_actuator_controller(self):
        """Test ActuatorController."""
        controller = ActuatorController("test_motor", ActuatorType.MOTOR)
        controller.set_safety_limits(SafetyLimits(min_position=-100, max_position=100))
        
        assert controller.enable() == True
        
        result = await controller.move_to(50.0)
        assert result.success == True
        assert abs(result.final_state.position - 50.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_actuator_safety_limits(self):
        """Test safety limit enforcement."""
        controller = ActuatorController("test", ActuatorType.SERVO)
        controller.set_safety_limits(SafetyLimits(min_position=0, max_position=90))
        controller.enable()
        
        result = await controller.move_to(100.0)  # Out of limits
        assert result.success == False
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self):
        """Test emergency stop."""
        controller = ActuatorController("test", ActuatorType.MOTOR)
        controller.enable()
        
        controller.emergency_stop()
        
        result = await controller.move_to(10.0)
        assert result.success == False


class TestROS2Bridge:
    """Test ROS2 bridge."""
    
    @pytest.mark.asyncio
    async def test_bridge_connect(self):
        """Test bridge connection."""
        bridge = ROS2Bridge("test_node")
        
        connected = await bridge.connect()
        
        assert connected == True
        assert bridge.is_connected == True
        
        await bridge.disconnect()
        assert bridge.is_connected == False
    
    @pytest.mark.asyncio
    async def test_publisher_subscriber(self):
        """Test pub/sub."""
        bridge = ROS2Bridge("test_node")
        await bridge.connect()
        
        messages = []
        def callback(msg):
            messages.append(msg)
        
        sub_id = bridge.subscribe("test_topic", "std_msgs/String", callback)
        pub_id = bridge.create_publisher("test_topic", "std_msgs/String")
        
        await bridge.publish(pub_id, {"data": "Hello"})
        await asyncio.sleep(0.05)
        
        assert len(messages) > 0
        assert messages[0].data["data"] == "Hello"
        
        await bridge.disconnect()
    
    @pytest.mark.asyncio
    async def test_service_call(self):
        """Test service call."""
        bridge = ROS2Bridge("test_node")
        await bridge.connect()
        
        client_id = bridge.create_service_client("test_service", "std_srvs/Trigger")
        
        response = await bridge.call_service(client_id, {})
        
        assert response.success == True
        assert response.latency_ms > 0
        
        await bridge.disconnect()
    
    @pytest.mark.asyncio
    async def test_topic_discovery(self):
        """Test topic discovery."""
        bridge = ROS2Bridge("test_node")
        await bridge.connect()
        
        topics = await bridge.list_topics()
        
        assert len(topics) > 0
        assert all(isinstance(t, ROS2Topic) for t in topics)
        
        await bridge.disconnect()


class TestSimulation:
    """Test simulation environment."""
    
    def test_action_space(self):
        """Test ActionSpace creation."""
        discrete = ActionSpace.discrete(4)
        assert discrete.type == "discrete"
        assert discrete.n == 4
        
        continuous = ActionSpace.continuous((2,), [-1.0, -1.0], [1.0, 1.0])
        assert continuous.type == "continuous"
        assert continuous.shape == (2,)
    
    def test_observation_space(self):
        """Test ObservationSpace creation."""
        box = ObservationSpace.box((10,))
        assert box.type == "box"
        
        image = ObservationSpace.image(480, 640, 3)
        assert image.shape == (480, 640, 3)
    
    @pytest.mark.asyncio
    async def test_simulator_reset(self):
        """Test simulator reset."""
        sim = GenericSimulator()
        
        observation, info = await sim.reset(seed=42)
        
        assert "position" in observation
        assert "goal_relative" in observation
    
    @pytest.mark.asyncio
    async def test_simulator_step(self):
        """Test simulator step."""
        sim = GenericSimulator()
        await sim.reset()
        
        result = await sim.step([0.1, 0.1])
        
        assert isinstance(result.reward, float)
        assert isinstance(result.terminated, bool)
        assert "distance_to_goal" in result.observation
    
    @pytest.mark.asyncio
    async def test_episode_tracking(self):
        """Test episode tracking."""
        sim = GenericSimulator()
        await sim.reset()
        
        episode = sim.current_episode
        assert episode is not None
        
        for _ in range(5):
            await sim.step([0.1, 0.0])
        
        assert episode.total_steps == 5
    
    @pytest.mark.asyncio
    async def test_episode_manager(self):
        """Test episode manager."""
        sim = GenericSimulator(SimulationConfig(max_episode_steps=20))
        manager = EpisodeManager(sim)
        
        episode = await manager.run_episode()
        
        assert episode.total_steps > 0
        assert episode.final_state is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
