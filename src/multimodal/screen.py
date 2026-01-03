"""
AION Screen/UI Understanding
=============================

Screen capture and UI element detection for automation
and visual understanding of user interfaces.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import uuid


# =============================================================================
# UI ELEMENT TYPES
# =============================================================================

class UIElementType(Enum):
    """Types of UI elements."""
    BUTTON = "button"
    INPUT = "input"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    DROPDOWN = "dropdown"
    LINK = "link"
    TEXT = "text"
    LABEL = "label"
    IMAGE = "image"
    ICON = "icon"
    MENU = "menu"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    SLIDER = "slider"
    TOGGLE = "toggle"
    MODAL = "modal"
    TOOLTIP = "tooltip"
    CARD = "card"
    LIST = "list"
    TABLE = "table"
    FORM = "form"
    NAVBAR = "navbar"
    SIDEBAR = "sidebar"
    HEADER = "header"
    FOOTER = "footer"
    CONTAINER = "container"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Types of UI actions."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    PRESS_KEY = "press_key"
    SCROLL = "scroll"
    DRAG = "drag"
    HOVER = "hover"
    FOCUS = "focus"
    SELECT = "select"
    WAIT = "wait"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UIElement:
    """A detected UI element."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: UIElementType = UIElementType.UNKNOWN
    text: str = ""
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, width, height
    confidence: float = 1.0
    
    # State
    is_enabled: bool = True
    is_visible: bool = True
    is_focused: bool = False
    is_selected: bool = False
    
    # Hierarchy
    parent_id: Optional[str] = None
    children: List["UIElement"] = field(default_factory=list)
    
    # Semantic info
    role: str = ""
    aria_label: str = ""
    placeholder: str = ""
    value: str = ""
    
    # Style hints
    color: str = ""
    background_color: str = ""
    
    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center coordinates."""
        x, y, w, h = self.bounding_box
        return (x + w // 2, y + h // 2)
    
    @property
    def clickable(self) -> bool:
        """Check if element is likely clickable."""
        return self.type in {
            UIElementType.BUTTON,
            UIElementType.LINK,
            UIElementType.CHECKBOX,
            UIElementType.RADIO,
            UIElementType.TAB,
            UIElementType.MENU_ITEM,
            UIElementType.TOGGLE,
        }


@dataclass
class ScreenCapture:
    """A screen capture for analysis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: bytes = b""
    width: int = 0
    height: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Capture region
    region: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    
    # Source info
    window_title: str = ""
    application: str = ""
    url: str = ""  # For browser captures
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get capture dimensions."""
        return (self.width, self.height)


@dataclass
class ScreenAnalysis:
    """Complete analysis of a screen capture."""
    capture_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Detected elements
    elements: List[UIElement] = field(default_factory=list)
    
    # Layout info
    layout_type: str = ""  # "desktop", "mobile", "web", etc.
    has_navbar: bool = False
    has_sidebar: bool = False
    has_modal: bool = False
    
    # Text content
    all_text: List[str] = field(default_factory=list)
    
    # Interactable elements
    buttons: List[UIElement] = field(default_factory=list)
    inputs: List[UIElement] = field(default_factory=list)
    links: List[UIElement] = field(default_factory=list)
    
    # Description
    description: str = ""
    
    # Raw response
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    def find_element(self, text: str) -> Optional[UIElement]:
        """Find element by text content."""
        text_lower = text.lower()
        for elem in self.elements:
            if text_lower in elem.text.lower():
                return elem
        return None
    
    def find_elements_by_type(self, type: UIElementType) -> List[UIElement]:
        """Find all elements of a type."""
        return [e for e in self.elements if e.type == type]


@dataclass
class UIAction:
    """A planned UI action."""
    type: ActionType
    target: Optional[UIElement] = None
    coordinates: Optional[Tuple[int, int]] = None
    text: str = ""
    key: str = ""
    scroll_amount: int = 0
    duration_ms: int = 100
    description: str = ""


@dataclass
class ActionPlan:
    """A sequence of actions to achieve a goal."""
    goal: str
    actions: List[UIAction] = field(default_factory=list)
    success_condition: str = ""
    estimated_duration_ms: int = 0


# =============================================================================
# SCREEN PROCESSOR
# =============================================================================

class ScreenProcessor:
    """
    Process screen captures for UI understanding.
    """
    
    def __init__(self, model: str = "default"):
        self.model = model
        self._cache: Dict[str, ScreenAnalysis] = {}
    
    async def capture_screen(self) -> ScreenCapture:
        """
        Capture the entire screen.
        
        Returns:
            ScreenCapture with screen data
        """
        await asyncio.sleep(0.01)
        
        # Simulated capture - in production, use platform-specific APIs
        return ScreenCapture(
            width=1920,
            height=1080,
            data=b"",  # Would contain actual screenshot
            metadata={"method": "full_screen"}
        )
    
    async def capture_region(
        self, 
        x: int, 
        y: int, 
        width: int, 
        height: int
    ) -> ScreenCapture:
        """
        Capture a specific screen region.
        
        Args:
            x, y: Top-left coordinates
            width, height: Region dimensions
            
        Returns:
            ScreenCapture of the region
        """
        await asyncio.sleep(0.01)
        
        return ScreenCapture(
            width=width,
            height=height,
            region=(x, y, width, height),
            data=b"",
            metadata={"method": "region"}
        )
    
    async def capture_window(self, window_title: str = "") -> ScreenCapture:
        """
        Capture a specific window.
        
        Args:
            window_title: Title of the window to capture
            
        Returns:
            ScreenCapture of the window
        """
        await asyncio.sleep(0.01)
        
        return ScreenCapture(
            width=800,
            height=600,
            window_title=window_title,
            data=b"",
            metadata={"method": "window"}
        )
    
    async def detect_elements(
        self, 
        capture: ScreenCapture
    ) -> List[UIElement]:
        """
        Detect UI elements in a screen capture.
        
        Args:
            capture: ScreenCapture to analyze
            
        Returns:
            List of detected UI elements
        """
        await asyncio.sleep(0.05)  # Simulate processing
        
        # Simulated detection - in production, use computer vision models
        elements = [
            UIElement(
                type=UIElementType.NAVBAR,
                text="Navigation",
                bounding_box=(0, 0, capture.width, 60),
                role="navigation"
            ),
            UIElement(
                type=UIElementType.BUTTON,
                text="Submit",
                bounding_box=(100, 200, 100, 40),
                is_enabled=True
            ),
            UIElement(
                type=UIElementType.INPUT,
                text="",
                placeholder="Enter text...",
                bounding_box=(100, 150, 200, 30)
            ),
            UIElement(
                type=UIElementType.LINK,
                text="Learn more",
                bounding_box=(100, 260, 80, 20)
            )
        ]
        
        return elements
    
    async def analyze(self, capture: ScreenCapture) -> ScreenAnalysis:
        """
        Perform full analysis on a screen capture.
        
        Args:
            capture: ScreenCapture to analyze
            
        Returns:
            ScreenAnalysis with all detected elements
        """
        if capture.id in self._cache:
            return self._cache[capture.id]
        
        elements = await self.detect_elements(capture)
        
        # Categorize elements
        buttons = [e for e in elements if e.type == UIElementType.BUTTON]
        inputs = [e for e in elements if e.type == UIElementType.INPUT]
        links = [e for e in elements if e.type == UIElementType.LINK]
        
        # Extract text
        all_text = [e.text for e in elements if e.text]
        
        # Detect layout features
        has_navbar = any(e.type == UIElementType.NAVBAR for e in elements)
        has_sidebar = any(e.type == UIElementType.SIDEBAR for e in elements)
        has_modal = any(e.type == UIElementType.MODAL for e in elements)
        
        analysis = ScreenAnalysis(
            capture_id=capture.id,
            elements=elements,
            layout_type="desktop",
            has_navbar=has_navbar,
            has_sidebar=has_sidebar,
            has_modal=has_modal,
            all_text=all_text,
            buttons=buttons,
            inputs=inputs,
            links=links,
            description=f"Screen with {len(elements)} UI elements"
        )
        
        self._cache[capture.id] = analysis
        return analysis
    
    async def find_element_at(
        self, 
        capture: ScreenCapture, 
        x: int, 
        y: int
    ) -> Optional[UIElement]:
        """Find element at specific coordinates."""
        analysis = await self.analyze(capture)
        
        for elem in analysis.elements:
            ex, ey, ew, eh = elem.bounding_box
            if ex <= x <= ex + ew and ey <= y <= ey + eh:
                return elem
        
        return None


# =============================================================================
# ACTION PLANNER
# =============================================================================

class ActionPlanner:
    """
    Plan UI actions to achieve goals.
    """
    
    def __init__(self, screen_processor: ScreenProcessor = None):
        self.processor = screen_processor or ScreenProcessor()
    
    async def plan_action(
        self, 
        goal: str, 
        capture: ScreenCapture
    ) -> ActionPlan:
        """
        Plan actions to achieve a goal on the given screen.
        
        Args:
            goal: Natural language goal description
            capture: Current screen state
            
        Returns:
            ActionPlan with sequence of actions
        """
        analysis = await self.processor.analyze(capture)
        actions = []
        
        goal_lower = goal.lower()
        
        # Simple goal parsing
        if "click" in goal_lower:
            # Find matching element
            for word in goal.split():
                elem = analysis.find_element(word)
                if elem and elem.clickable:
                    actions.append(UIAction(
                        type=ActionType.CLICK,
                        target=elem,
                        coordinates=elem.center,
                        description=f"Click on '{elem.text}'"
                    ))
                    break
        
        elif "type" in goal_lower or "enter" in goal_lower:
            # Find text to type (after "type" or between quotes)
            import re
            text_match = re.search(r'"([^"]+)"', goal)
            if text_match:
                text = text_match.group(1)
                
                # Find input field
                if analysis.inputs:
                    actions.append(UIAction(
                        type=ActionType.CLICK,
                        target=analysis.inputs[0],
                        description=f"Focus on input field"
                    ))
                    actions.append(UIAction(
                        type=ActionType.TYPE,
                        text=text,
                        description=f"Type '{text}'"
                    ))
        
        elif "scroll" in goal_lower:
            direction = -300 if "up" in goal_lower else 300
            actions.append(UIAction(
                type=ActionType.SCROLL,
                scroll_amount=direction,
                description=f"Scroll {'up' if direction < 0 else 'down'}"
            ))
        
        # Calculate estimated duration
        total_duration = sum(a.duration_ms for a in actions) + len(actions) * 50
        
        return ActionPlan(
            goal=goal,
            actions=actions,
            success_condition=f"Goal '{goal}' achieved",
            estimated_duration_ms=total_duration
        )
    
    def generate_click(self, element: UIElement) -> UIAction:
        """Generate a click action for an element."""
        return UIAction(
            type=ActionType.CLICK,
            target=element,
            coordinates=element.center,
            description=f"Click on '{element.text}'"
        )
    
    def generate_type(
        self, 
        text: str, 
        element: Optional[UIElement] = None
    ) -> UIAction:
        """Generate a type action."""
        return UIAction(
            type=ActionType.TYPE,
            target=element,
            text=text,
            description=f"Type '{text}'"
        )
    
    def generate_scroll(
        self, 
        amount: int, 
        coordinates: Tuple[int, int] = None
    ) -> UIAction:
        """Generate a scroll action."""
        return UIAction(
            type=ActionType.SCROLL,
            coordinates=coordinates,
            scroll_amount=amount,
            description=f"Scroll by {amount}"
        )
    
    def generate_key_press(self, key: str) -> UIAction:
        """Generate a key press action."""
        return UIAction(
            type=ActionType.PRESS_KEY,
            key=key,
            description=f"Press '{key}'"
        )
    
    async def execute_plan(
        self, 
        plan: ActionPlan,
        executor = None
    ) -> Dict[str, Any]:
        """
        Execute an action plan.
        
        Args:
            plan: ActionPlan to execute
            executor: Optional executor for real actions
            
        Returns:
            Execution result
        """
        results = []
        
        for action in plan.actions:
            # Simulate execution
            await asyncio.sleep(action.duration_ms / 1000)
            results.append({
                "action": action.type.value,
                "success": True,
                "description": action.description
            })
        
        return {
            "goal": plan.goal,
            "actions_executed": len(results),
            "results": results,
            "success": all(r["success"] for r in results)
        }


# =============================================================================
# DEMO
# =============================================================================

async def demo_screen():
    """Demonstrate screen/UI processing."""
    print("🖥️ Screen/UI Processing Demo")
    print("-" * 40)
    
    processor = ScreenProcessor()
    planner = ActionPlanner(processor)
    
    # Capture screen
    capture = await processor.capture_screen()
    print(f"Captured: {capture.width}x{capture.height}")
    
    # Analyze
    analysis = await processor.analyze(capture)
    print(f"Elements detected: {len(analysis.elements)}")
    print(f"Buttons: {len(analysis.buttons)}")
    print(f"Inputs: {len(analysis.inputs)}")
    print(f"Links: {len(analysis.links)}")
    print(f"Has navbar: {analysis.has_navbar}")
    
    # Plan an action
    plan = await planner.plan_action("click on Submit button", capture)
    print(f"\nAction plan for: {plan.goal}")
    for action in plan.actions:
        print(f"  - {action.description}")
    
    # Execute plan
    result = await planner.execute_plan(plan)
    print(f"\nExecution: {result['success']}")
    
    print("-" * 40)
    print("✅ Screen/UI demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_screen())
