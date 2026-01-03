"""
AION ROS2 Bridge
================

Integration bridge for ROS2 (Robot Operating System 2).
Provides topic pub/sub, service calls, and action interfaces.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, AsyncIterator
from enum import Enum
from datetime import datetime
import uuid


# =============================================================================
# ROS2 MESSAGE TYPES
# =============================================================================

class QoSReliability(Enum):
    """Quality of Service reliability settings."""
    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"


class QoSDurability(Enum):
    """Quality of Service durability settings."""
    VOLATILE = "volatile"
    TRANSIENT_LOCAL = "transient_local"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QoSProfile:
    """ROS2 Quality of Service profile."""
    reliability: QoSReliability = QoSReliability.RELIABLE
    durability: QoSDurability = QoSDurability.VOLATILE
    depth: int = 10
    deadline_ms: int = 0
    lifespan_ms: int = 0


@dataclass
class ROS2Message:
    """A ROS2 message."""
    topic: str
    message_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    frame_id: str = ""
    sequence: int = 0


@dataclass
class ROS2Topic:
    """Information about a ROS2 topic."""
    name: str
    message_type: str
    publishers: int = 0
    subscribers: int = 0
    qos: QoSProfile = field(default_factory=QoSProfile)


@dataclass
class ROS2Service:
    """A ROS2 service definition."""
    name: str
    service_type: str
    available: bool = True


@dataclass
class ROS2Action:
    """A ROS2 action definition."""
    name: str
    action_type: str
    available: bool = True


@dataclass
class ServiceRequest:
    """A service call request."""
    service_name: str
    request_data: Dict[str, Any]
    timeout_ms: int = 5000


@dataclass
class ServiceResponse:
    """Response from a service call."""
    success: bool
    response_data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    latency_ms: float = 0.0


@dataclass
class ActionGoal:
    """An action goal."""
    action_name: str
    goal_data: Dict[str, Any]
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ActionFeedback:
    """Feedback from an action."""
    goal_id: str
    feedback_data: Dict[str, Any]
    progress: float = 0.0


@dataclass
class ActionResult:
    """Result of an action."""
    goal_id: str
    success: bool
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""


# =============================================================================
# ROS2 BRIDGE
# =============================================================================

class ROS2Bridge:
    """
    Bridge for ROS2 integration.
    
    Provides publishers, subscribers, service clients/servers,
    and action clients for ROS2 interoperability.
    """
    
    def __init__(self, node_name: str = "aion_bridge"):
        self.node_name = node_name
        self._connected = False
        self._namespace = ""
        
        # Publishers and subscribers
        self._publishers: Dict[str, Dict[str, Any]] = {}
        self._subscribers: Dict[str, Dict[str, Any]] = {}
        self._subscription_callbacks: Dict[str, List[Callable]] = {}
        
        # Services
        self._service_clients: Dict[str, ROS2Service] = {}
        self._service_servers: Dict[str, Callable] = {}
        
        # Actions
        self._action_clients: Dict[str, ROS2Action] = {}
        self._active_goals: Dict[str, ActionGoal] = {}
        
        # Message buffers
        self._message_buffers: Dict[str, List[ROS2Message]] = {}
        self._buffer_size = 100
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def connect(self, namespace: str = "") -> bool:
        """
        Connect to ROS2 network.
        
        Args:
            namespace: ROS2 namespace for this node
            
        Returns:
            True if connected successfully
        """
        await asyncio.sleep(0.1)  # Simulated connection
        
        self._namespace = namespace
        self._connected = True
        
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from ROS2 network."""
        # Clean up subscriptions
        self._subscribers.clear()
        self._subscription_callbacks.clear()
        
        # Clean up publishers
        self._publishers.clear()
        
        self._connected = False
    
    # -------------------------------------------------------------------------
    # Topic Publishing
    # -------------------------------------------------------------------------
    
    def create_publisher(
        self, 
        topic: str, 
        message_type: str,
        qos: QoSProfile = None
    ) -> str:
        """
        Create a publisher for a topic.
        
        Args:
            topic: Topic name
            message_type: ROS2 message type (e.g., "std_msgs/String")
            qos: Quality of Service profile
            
        Returns:
            Publisher ID
        """
        full_topic = f"{self._namespace}/{topic}" if self._namespace else topic
        
        pub_id = str(uuid.uuid4())
        self._publishers[pub_id] = {
            "topic": full_topic,
            "message_type": message_type,
            "qos": qos or QoSProfile(),
            "sequence": 0
        }
        
        return pub_id
    
    async def publish(
        self, 
        publisher_id: str, 
        data: Dict[str, Any],
        frame_id: str = ""
    ) -> bool:
        """
        Publish a message.
        
        Args:
            publisher_id: ID of the publisher
            data: Message data
            frame_id: Optional frame ID
            
        Returns:
            True if published successfully
        """
        if publisher_id not in self._publishers:
            return False
        
        pub = self._publishers[publisher_id]
        pub["sequence"] += 1
        
        message = ROS2Message(
            topic=pub["topic"],
            message_type=pub["message_type"],
            data=data,
            frame_id=frame_id,
            sequence=pub["sequence"]
        )
        
        # Simulate network transmission
        await asyncio.sleep(0.001)
        
        # Add to buffer for local subscribers
        topic = pub["topic"]
        if topic in self._message_buffers:
            self._message_buffers[topic].append(message)
            if len(self._message_buffers[topic]) > self._buffer_size:
                self._message_buffers[topic] = self._message_buffers[topic][-self._buffer_size:]
        
        # Notify local subscribers
        if topic in self._subscription_callbacks:
            for callback in self._subscription_callbacks[topic]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception:
                    pass
        
        return True
    
    # -------------------------------------------------------------------------
    # Topic Subscription
    # -------------------------------------------------------------------------
    
    def subscribe(
        self, 
        topic: str, 
        message_type: str,
        callback: Callable[[ROS2Message], None],
        qos: QoSProfile = None
    ) -> str:
        """
        Subscribe to a topic.
        
        Args:
            topic: Topic name
            message_type: Expected message type
            callback: Callback for received messages
            qos: Quality of Service profile
            
        Returns:
            Subscription ID
        """
        full_topic = f"{self._namespace}/{topic}" if self._namespace else topic
        
        sub_id = str(uuid.uuid4())
        self._subscribers[sub_id] = {
            "topic": full_topic,
            "message_type": message_type,
            "qos": qos or QoSProfile()
        }
        
        # Register callback
        if full_topic not in self._subscription_callbacks:
            self._subscription_callbacks[full_topic] = []
            self._message_buffers[full_topic] = []
        self._subscription_callbacks[full_topic].append(callback)
        
        return sub_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a topic."""
        if subscription_id not in self._subscribers:
            return False
        
        del self._subscribers[subscription_id]
        return True
    
    async def get_latest_message(self, topic: str) -> Optional[ROS2Message]:
        """Get the latest message on a topic."""
        full_topic = f"{self._namespace}/{topic}" if self._namespace else topic
        
        if full_topic in self._message_buffers and self._message_buffers[full_topic]:
            return self._message_buffers[full_topic][-1]
        return None
    
    # -------------------------------------------------------------------------
    # Service Calls
    # -------------------------------------------------------------------------
    
    def create_service_client(
        self, 
        service_name: str, 
        service_type: str
    ) -> str:
        """
        Create a service client.
        
        Args:
            service_name: Name of the service
            service_type: ROS2 service type
            
        Returns:
            Client ID
        """
        full_name = f"{self._namespace}/{service_name}" if self._namespace else service_name
        
        service = ROS2Service(
            name=full_name,
            service_type=service_type
        )
        
        client_id = str(uuid.uuid4())
        self._service_clients[client_id] = service
        
        return client_id
    
    async def call_service(
        self, 
        client_id: str, 
        request_data: Dict[str, Any],
        timeout_ms: int = 5000
    ) -> ServiceResponse:
        """
        Call a service.
        
        Args:
            client_id: Service client ID
            request_data: Request data
            timeout_ms: Timeout in milliseconds
            
        Returns:
            ServiceResponse with result
        """
        start_time = datetime.now()
        
        if client_id not in self._service_clients:
            return ServiceResponse(
                success=False,
                error_message="Service client not found"
            )
        
        service = self._service_clients[client_id]
        
        # Check if service is available
        if not service.available:
            return ServiceResponse(
                success=False,
                error_message="Service not available"
            )
        
        # Simulate service call
        await asyncio.sleep(0.01)
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        # Simulated response
        return ServiceResponse(
            success=True,
            response_data={"result": f"Processed request for {service.name}"},
            latency_ms=latency
        )
    
    def create_service(
        self, 
        service_name: str, 
        service_type: str,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> str:
        """
        Create a service server.
        
        Args:
            service_name: Name of the service
            service_type: ROS2 service type
            handler: Handler function for requests
            
        Returns:
            Service server ID
        """
        full_name = f"{self._namespace}/{service_name}" if self._namespace else service_name
        
        server_id = str(uuid.uuid4())
        self._service_servers[full_name] = handler
        
        return server_id
    
    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------
    
    def create_action_client(
        self, 
        action_name: str, 
        action_type: str
    ) -> str:
        """Create an action client."""
        full_name = f"{self._namespace}/{action_name}" if self._namespace else action_name
        
        action = ROS2Action(
            name=full_name,
            action_type=action_type
        )
        
        client_id = str(uuid.uuid4())
        self._action_clients[client_id] = action
        
        return client_id
    
    async def send_goal(
        self, 
        client_id: str, 
        goal_data: Dict[str, Any]
    ) -> ActionGoal:
        """
        Send a goal to an action server.
        
        Args:
            client_id: Action client ID
            goal_data: Goal data
            
        Returns:
            ActionGoal with goal ID
        """
        if client_id not in self._action_clients:
            raise ValueError("Action client not found")
        
        action = self._action_clients[client_id]
        
        goal = ActionGoal(
            action_name=action.name,
            goal_data=goal_data
        )
        
        self._active_goals[goal.goal_id] = goal
        
        return goal
    
    async def get_action_feedback(
        self, 
        goal_id: str
    ) -> AsyncIterator[ActionFeedback]:
        """
        Stream feedback for an action goal.
        
        Args:
            goal_id: Goal ID
            
        Yields:
            ActionFeedback updates
        """
        if goal_id not in self._active_goals:
            return
        
        # Simulate feedback stream
        for i in range(10):
            await asyncio.sleep(0.1)
            
            yield ActionFeedback(
                goal_id=goal_id,
                feedback_data={"step": i + 1},
                progress=(i + 1) / 10
            )
    
    async def wait_for_action_result(
        self, 
        goal_id: str,
        timeout_ms: int = 30000
    ) -> ActionResult:
        """
        Wait for an action to complete.
        
        Args:
            goal_id: Goal ID
            timeout_ms: Timeout in milliseconds
            
        Returns:
            ActionResult
        """
        if goal_id not in self._active_goals:
            return ActionResult(
                goal_id=goal_id,
                success=False,
                error_message="Goal not found"
            )
        
        # Simulate action execution
        await asyncio.sleep(0.5)
        
        # Remove from active goals
        del self._active_goals[goal_id]
        
        return ActionResult(
            goal_id=goal_id,
            success=True,
            result_data={"message": "Action completed successfully"}
        )
    
    async def cancel_goal(self, goal_id: str) -> bool:
        """Cancel an action goal."""
        if goal_id in self._active_goals:
            del self._active_goals[goal_id]
            return True
        return False
    
    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------
    
    async def list_topics(self) -> List[ROS2Topic]:
        """List available topics."""
        # Simulated topic discovery
        return [
            ROS2Topic(name="/cmd_vel", message_type="geometry_msgs/Twist"),
            ROS2Topic(name="/odom", message_type="nav_msgs/Odometry"),
            ROS2Topic(name="/scan", message_type="sensor_msgs/LaserScan"),
            ROS2Topic(name="/camera/image_raw", message_type="sensor_msgs/Image"),
        ]
    
    async def list_services(self) -> List[ROS2Service]:
        """List available services."""
        return [
            ROS2Service(name="/get_map", service_type="nav_msgs/GetMap"),
            ROS2Service(name="/set_parameters", service_type="rcl_interfaces/SetParameters"),
        ]
    
    async def list_actions(self) -> List[ROS2Action]:
        """List available actions."""
        return [
            ROS2Action(name="/navigate_to_pose", action_type="nav2_msgs/NavigateToPose"),
            ROS2Action(name="/follow_path", action_type="nav2_msgs/FollowPath"),
        ]


# =============================================================================
# DEMO
# =============================================================================

async def demo_ros2():
    """Demonstrate ROS2 bridge."""
    print("🤖 ROS2 Bridge Demo")
    print("-" * 40)
    
    bridge = ROS2Bridge("aion_demo")
    
    # Connect
    await bridge.connect(namespace="robot")
    print(f"Connected: {bridge.is_connected}")
    
    # List available topics
    topics = await bridge.list_topics()
    print(f"Available topics: {len(topics)}")
    for topic in topics[:2]:
        print(f"  - {topic.name} ({topic.message_type})")
    
    # Create publisher
    pub_id = bridge.create_publisher("status", "std_msgs/String")
    print(f"Created publisher: {pub_id[:8]}...")
    
    # Create subscriber
    messages_received = []
    def on_message(msg):
        messages_received.append(msg)
    
    sub_id = bridge.subscribe("status", "std_msgs/String", on_message)
    print(f"Created subscriber: {sub_id[:8]}...")
    
    # Publish
    await bridge.publish(pub_id, {"data": "Hello from AION!"})
    await asyncio.sleep(0.05)
    print(f"Published message, received: {len(messages_received)}")
    
    # Service call
    client_id = bridge.create_service_client("get_status", "std_srvs/Trigger")
    response = await bridge.call_service(client_id, {})
    print(f"Service call: {response.success}, latency: {response.latency_ms:.1f}ms")
    
    # Action
    action_id = bridge.create_action_client("navigate", "nav2_msgs/NavigateToPose")
    goal = await bridge.send_goal(action_id, {"target_x": 1.0, "target_y": 2.0})
    print(f"Sent goal: {goal.goal_id[:8]}...")
    
    result = await bridge.wait_for_action_result(goal.goal_id)
    print(f"Action result: {result.success}")
    
    # Disconnect
    await bridge.disconnect()
    print(f"Disconnected")
    
    print("-" * 40)
    print("✅ ROS2 demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_ros2())
