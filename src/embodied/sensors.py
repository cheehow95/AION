"""
AION Sensor Interface
=====================

Real-time sensor data streaming and multi-sensor fusion
for embodied AI applications.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncIterator, Callable, Tuple
from enum import Enum
from datetime import datetime
import uuid
import math


# =============================================================================
# SENSOR TYPES
# =============================================================================

class SensorType(Enum):
    """Types of sensors."""
    CAMERA = "camera"
    DEPTH_CAMERA = "depth_camera"
    LIDAR = "lidar"
    RADAR = "radar"
    ULTRASONIC = "ultrasonic"
    IMU = "imu"
    GYROSCOPE = "gyroscope"
    ACCELEROMETER = "accelerometer"
    MAGNETOMETER = "magnetometer"
    GPS = "gps"
    FORCE = "force"
    TORQUE = "torque"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PROXIMITY = "proximity"
    TOUCH = "touch"
    JOINT_ENCODER = "joint_encoder"
    WHEEL_ENCODER = "wheel_encoder"
    MICROPHONE = "microphone"
    INFRARED = "infrared"


class DataFormat(Enum):
    """Sensor data formats."""
    RAW = "raw"
    NUMPY = "numpy"
    IMAGE = "image"
    POINTCLOUD = "pointcloud"
    POSE = "pose"
    VECTOR3 = "vector3"
    QUATERNION = "quaternion"
    SCALAR = "scalar"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SensorConfig:
    """Configuration for a sensor."""
    sensor_id: str
    sensor_type: SensorType
    name: str = ""
    enabled: bool = True
    
    # Timing
    sample_rate_hz: float = 30.0
    latency_ms: float = 0.0
    
    # Position/orientation in robot frame
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # quaternion
    
    # Data format
    data_format: DataFormat = DataFormat.RAW
    
    # Type-specific config
    resolution: Optional[Tuple[int, int]] = None  # For cameras
    fov_degrees: Optional[float] = None
    range_meters: Optional[Tuple[float, float]] = None  # min, max
    
    # Additional params
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorReading:
    """A single sensor reading."""
    sensor_id: str
    sensor_type: SensorType
    timestamp: datetime = field(default_factory=datetime.now)
    sequence: int = 0
    
    # Data
    data: Any = None
    data_format: DataFormat = DataFormat.RAW
    
    # Quality
    valid: bool = True
    confidence: float = 1.0
    noise_estimate: float = 0.0
    
    # Metadata
    frame_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_ms(self) -> float:
        """Get age of reading in milliseconds."""
        return (datetime.now() - self.timestamp).total_seconds() * 1000


@dataclass
class FusedState:
    """State estimate from sensor fusion."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Position (x, y, z) in meters
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_covariance: List[float] = field(default_factory=lambda: [0.0] * 9)
    
    # Orientation as quaternion (x, y, z, w)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    orientation_covariance: List[float] = field(default_factory=lambda: [0.0] * 9)
    
    # Velocity (vx, vy, vz) in m/s
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_covariance: List[float] = field(default_factory=lambda: [0.0] * 9)
    
    # Angular velocity (wx, wy, wz) in rad/s
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Source sensors
    sources: List[str] = field(default_factory=list)
    
    # Quality
    confidence: float = 1.0


# =============================================================================
# SENSOR STREAM
# =============================================================================

class SensorStream:
    """
    Async sensor data stream.
    
    Provides real-time sensor data with subscription-based updates.
    """
    
    def __init__(self, config: SensorConfig):
        self.config = config
        self._running = False
        self._sequence = 0
        self._callbacks: List[Callable[[SensorReading], None]] = []
        self._buffer: List[SensorReading] = []
        self._buffer_size = 100
        self._last_reading: Optional[SensorReading] = None
    
    @property
    def sensor_id(self) -> str:
        return self.config.sensor_id
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    async def start(self) -> None:
        """Start the sensor stream."""
        if self._running:
            return
        
        self._running = True
        asyncio.create_task(self._stream_loop())
    
    async def stop(self) -> None:
        """Stop the sensor stream."""
        self._running = False
    
    def subscribe(self, callback: Callable[[SensorReading], None]) -> None:
        """Subscribe to sensor updates."""
        self._callbacks.append(callback)
    
    def unsubscribe(self, callback: Callable[[SensorReading], None]) -> None:
        """Unsubscribe from sensor updates."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def get_reading(self) -> Optional[SensorReading]:
        """Get the latest reading."""
        return self._last_reading
    
    async def stream(self) -> AsyncIterator[SensorReading]:
        """Async iterator for sensor readings."""
        if not self._running:
            await self.start()
        
        last_seq = -1
        while self._running:
            if self._last_reading and self._last_reading.sequence > last_seq:
                last_seq = self._last_reading.sequence
                yield self._last_reading
            await asyncio.sleep(1.0 / self.config.sample_rate_hz)
    
    def get_buffer(self, count: int = 10) -> List[SensorReading]:
        """Get recent readings from buffer."""
        return self._buffer[-count:]
    
    async def _stream_loop(self) -> None:
        """Internal streaming loop."""
        interval = 1.0 / self.config.sample_rate_hz
        
        while self._running:
            reading = self._generate_reading()
            self._last_reading = reading
            
            # Add to buffer
            self._buffer.append(reading)
            if len(self._buffer) > self._buffer_size:
                self._buffer = self._buffer[-self._buffer_size:]
            
            # Notify subscribers
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(reading)
                    else:
                        callback(reading)
                except Exception:
                    pass
            
            await asyncio.sleep(interval)
    
    def _generate_reading(self) -> SensorReading:
        """Generate a sensor reading (simulated)."""
        self._sequence += 1
        
        # Generate type-specific data
        data = self._generate_sensor_data()
        
        return SensorReading(
            sensor_id=self.config.sensor_id,
            sensor_type=self.config.sensor_type,
            sequence=self._sequence,
            data=data,
            data_format=self.config.data_format,
            frame_id=self.config.name
        )
    
    def _generate_sensor_data(self) -> Any:
        """Generate simulated sensor data."""
        t = self.config.sensor_type
        
        if t == SensorType.IMU:
            return {
                "acceleration": (0.0, 0.0, 9.81),  # m/s^2
                "angular_velocity": (0.0, 0.0, 0.0),  # rad/s
                "orientation": (0.0, 0.0, 0.0, 1.0)  # quaternion
            }
        
        elif t == SensorType.GPS:
            return {
                "latitude": 37.7749,  # degrees
                "longitude": -122.4194,
                "altitude": 10.0,  # meters
                "hdop": 1.0
            }
        
        elif t in (SensorType.CAMERA, SensorType.DEPTH_CAMERA):
            res = self.config.resolution or (640, 480)
            return {
                "width": res[0],
                "height": res[1],
                "encoding": "rgb8" if t == SensorType.CAMERA else "32FC1",
                "data": b""  # Would contain actual image data
            }
        
        elif t == SensorType.LIDAR:
            return {
                "num_points": 1000,
                "points": [],  # Would contain point cloud
                "intensity": []
            }
        
        elif t == SensorType.JOINT_ENCODER:
            return {
                "position": 0.0,  # radians
                "velocity": 0.0,  # rad/s
                "effort": 0.0  # Nm
            }
        
        else:
            return {"value": 0.0}


# =============================================================================
# SENSOR FUSION
# =============================================================================

class SensorFusion:
    """
    Multi-sensor fusion for state estimation.
    
    Combines data from multiple sensors to produce
    accurate state estimates.
    """
    
    def __init__(self):
        self._sensors: Dict[str, SensorStream] = {}
        self._state = FusedState()
        self._running = False
    
    def add_sensor(self, stream: SensorStream) -> None:
        """Add a sensor stream to fusion."""
        self._sensors[stream.sensor_id] = stream
    
    def remove_sensor(self, sensor_id: str) -> None:
        """Remove a sensor from fusion."""
        if sensor_id in self._sensors:
            del self._sensors[sensor_id]
    
    async def start(self) -> None:
        """Start sensor fusion."""
        self._running = True
        
        # Start all sensor streams
        for stream in self._sensors.values():
            await stream.start()
        
        # Start fusion loop
        asyncio.create_task(self._fusion_loop())
    
    async def stop(self) -> None:
        """Stop sensor fusion."""
        self._running = False
        
        for stream in self._sensors.values():
            await stream.stop()
    
    async def get_state(self) -> FusedState:
        """Get current fused state estimate."""
        return self._state
    
    async def _fusion_loop(self) -> None:
        """Main fusion loop."""
        while self._running:
            # Collect readings from all sensors
            readings = {}
            for sensor_id, stream in self._sensors.items():
                reading = await stream.get_reading()
                if reading and reading.valid:
                    readings[sensor_id] = reading
            
            # Update state estimate
            if readings:
                self._update_state(readings)
            
            await asyncio.sleep(0.01)  # 100 Hz fusion rate
    
    def _update_state(self, readings: Dict[str, SensorReading]) -> None:
        """Update state from readings (simplified EKF-style fusion)."""
        sources = list(readings.keys())
        
        # Process IMU for orientation
        for sensor_id, reading in readings.items():
            if reading.sensor_type == SensorType.IMU:
                data = reading.data
                if isinstance(data, dict) and "orientation" in data:
                    self._state.orientation = tuple(data["orientation"])
                if isinstance(data, dict) and "angular_velocity" in data:
                    self._state.angular_velocity = tuple(data["angular_velocity"])
        
        # Process GPS for position
        for sensor_id, reading in readings.items():
            if reading.sensor_type == SensorType.GPS:
                data = reading.data
                if isinstance(data, dict):
                    # Convert lat/lon to local coordinates (simplified)
                    lat = data.get("latitude", 0.0)
                    lon = data.get("longitude", 0.0)
                    alt = data.get("altitude", 0.0)
                    # Very simplified conversion
                    x = (lon + 180) * 111000 * math.cos(math.radians(lat))
                    y = (lat + 90) * 111000
                    self._state.position = (x, y, alt)
        
        self._state.timestamp = datetime.now()
        self._state.sources = sources
        self._state.confidence = min(1.0, len(sources) / 3.0)


# =============================================================================
# DEMO
# =============================================================================

async def demo_sensors():
    """Demonstrate sensor streaming."""
    print("📡 Sensor Interface Demo")
    print("-" * 40)
    
    # Create IMU sensor
    imu_config = SensorConfig(
        sensor_id="imu_0",
        sensor_type=SensorType.IMU,
        name="body_imu",
        sample_rate_hz=100.0
    )
    imu_stream = SensorStream(imu_config)
    
    # Create GPS sensor
    gps_config = SensorConfig(
        sensor_id="gps_0",
        sensor_type=SensorType.GPS,
        name="main_gps",
        sample_rate_hz=10.0
    )
    gps_stream = SensorStream(gps_config)
    
    # Setup fusion
    fusion = SensorFusion()
    fusion.add_sensor(imu_stream)
    fusion.add_sensor(gps_stream)
    
    # Start fusion
    await fusion.start()
    
    # Get some state estimates
    for i in range(5):
        await asyncio.sleep(0.1)
        state = await fusion.get_state()
        print(f"State {i+1}:")
        print(f"  Position: {state.position}")
        print(f"  Orientation: {state.orientation}")
        print(f"  Sources: {state.sources}")
    
    # Stop
    await fusion.stop()
    
    print("-" * 40)
    print("✅ Sensor demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_sensors())
