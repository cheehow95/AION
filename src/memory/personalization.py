"""
AION Enhanced Memory System - Personalization
==============================================

User preference learning and personalization:
- Preference tracking
- Behavior adaptation
- Response customization
- Learning from interactions

Enables GPT-5.2 style personalized experience.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum


class PreferenceType(Enum):
    """Types of user preferences."""
    COMMUNICATION = "communication"     # How to communicate
    FORMATTING = "formatting"           # Response formatting
    EXPERTISE = "expertise"             # Technical level
    WORKFLOW = "workflow"               # Working style
    CONTENT = "content"                 # Content preferences
    APPEARANCE = "appearance"           # UI preferences


@dataclass
class UserPreference:
    """A user preference."""
    key: str = ""
    value: Any = None
    preference_type: PreferenceType = PreferenceType.COMMUNICATION
    confidence: float = 0.5
    source: str = "inferred"  # explicit, inferred, default
    observations: int = 1
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update(self, new_value: Any, source: str = "inferred"):
        """Update preference value."""
        if source == "explicit":
            self.value = new_value
            self.confidence = 1.0
            self.source = source
        else:
            # Gradual update for inferred preferences
            self.observations += 1
            self.confidence = min(1.0, self.confidence + 0.1)
        self.last_updated = datetime.now()


class PersonalizationEngine:
    """Learns and applies user preferences."""
    
    # Default preferences
    DEFAULTS = {
        'response_length': ('detailed', PreferenceType.COMMUNICATION),
        'code_style': ('python', PreferenceType.FORMATTING),
        'expertise_level': ('intermediate', PreferenceType.EXPERTISE),
        'include_examples': (True, PreferenceType.CONTENT),
        'formal_tone': (False, PreferenceType.COMMUNICATION),
        'dark_mode': (True, PreferenceType.APPEARANCE),
    }
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.preferences: Dict[str, UserPreference] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        self._init_defaults()
    
    def _init_defaults(self):
        """Initialize default preferences."""
        for key, (value, ptype) in self.DEFAULTS.items():
            self.preferences[key] = UserPreference(
                key=key,
                value=value,
                preference_type=ptype,
                confidence=0.3,
                source="default"
            )
    
    def set_preference(self, key: str, value: Any,
                       preference_type: PreferenceType = None):
        """Explicitly set a preference."""
        if key in self.preferences:
            self.preferences[key].update(value, source="explicit")
        else:
            self.preferences[key] = UserPreference(
                key=key,
                value=value,
                preference_type=preference_type or PreferenceType.CONTENT,
                confidence=1.0,
                source="explicit"
            )
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        pref = self.preferences.get(key)
        return pref.value if pref else default
    
    def infer_preference(self, key: str, value: Any,
                         preference_type: PreferenceType = None):
        """Infer a preference from user behavior."""
        if key in self.preferences:
            pref = self.preferences[key]
            # Don't override explicit preferences
            if pref.source != "explicit":
                pref.update(value, source="inferred")
        else:
            self.preferences[key] = UserPreference(
                key=key,
                value=value,
                preference_type=preference_type or PreferenceType.CONTENT,
                confidence=0.3,
                source="inferred"
            )
    
    def record_interaction(self, query: str, response: str,
                           feedback: str = None):
        """Record an interaction for learning."""
        interaction = {
            'query': query,
            'response_length': len(response),
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }
        self.interaction_history.append(interaction)
        
        # Learn from interaction
        self._learn_from_interaction(interaction)
    
    def _learn_from_interaction(self, interaction: Dict[str, Any]):
        """Learn preferences from interaction."""
        query = interaction.get('query', '').lower()
        
        # Infer expertise level
        technical_terms = ['api', 'function', 'class', 'algorithm', 'optimize']
        if any(term in query for term in technical_terms):
            self.infer_preference(
                'expertise_level', 'advanced',
                PreferenceType.EXPERTISE
            )
        
        # Infer length preference from feedback
        feedback = interaction.get('feedback', '')
        if feedback:
            if 'shorter' in feedback.lower():
                self.infer_preference('response_length', 'concise')
            elif 'more detail' in feedback.lower():
                self.infer_preference('response_length', 'detailed')
    
    def get_personalized_prompt(self) -> str:
        """Generate personalized system prompt additions."""
        additions = []
        
        # Response length
        length = self.get_preference('response_length', 'detailed')
        if length == 'concise':
            additions.append("Keep responses brief and to the point.")
        elif length == 'detailed':
            additions.append("Provide detailed explanations with examples.")
        
        # Expertise level
        level = self.get_preference('expertise_level', 'intermediate')
        if level == 'beginner':
            additions.append("Explain concepts simply, avoid jargon.")
        elif level == 'advanced':
            additions.append("Use technical terminology freely.")
        
        # Tone
        if self.get_preference('formal_tone', False):
            additions.append("Use a formal, professional tone.")
        else:
            additions.append("Use a friendly, conversational tone.")
        
        # Examples
        if self.get_preference('include_examples', True):
            additions.append("Include practical code examples when relevant.")
        
        return "\n".join(additions)
    
    def get_all_preferences(self) -> Dict[str, Any]:
        """Get all preferences as a dict."""
        return {
            key: {
                'value': pref.value,
                'type': pref.preference_type.value,
                'confidence': pref.confidence,
                'source': pref.source
            }
            for key, pref in self.preferences.items()
        }
    
    def get_high_confidence_preferences(self, min_confidence: float = 0.7) -> Dict[str, Any]:
        """Get preferences with high confidence."""
        return {
            key: pref.value
            for key, pref in self.preferences.items()
            if pref.confidence >= min_confidence
        }
    
    def reset_preference(self, key: str):
        """Reset a preference to default."""
        if key in self.DEFAULTS:
            value, ptype = self.DEFAULTS[key]
            self.preferences[key] = UserPreference(
                key=key,
                value=value,
                preference_type=ptype,
                confidence=0.3,
                source="default"
            )
        elif key in self.preferences:
            del self.preferences[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get personalization statistics."""
        by_source = {'explicit': 0, 'inferred': 0, 'default': 0}
        for pref in self.preferences.values():
            by_source[pref.source] = by_source.get(pref.source, 0) + 1
        
        avg_confidence = (
            sum(p.confidence for p in self.preferences.values()) / 
            len(self.preferences) if self.preferences else 0
        )
        
        return {
            'user_id': self.user_id,
            'total_preferences': len(self.preferences),
            'by_source': by_source,
            'average_confidence': avg_confidence,
            'interactions_recorded': len(self.interaction_history)
        }


async def demo_personalization():
    """Demonstrate personalization engine."""
    print("✨ Personalization Engine Demo")
    print("=" * 50)
    
    engine = PersonalizationEngine(user_id="demo_user")
    
    # Set explicit preferences
    engine.set_preference('response_length', 'concise')
    engine.set_preference('dark_mode', True)
    
    print("\n⚙️ Explicit Preferences Set:")
    print(f"   response_length: {engine.get_preference('response_length')}")
    print(f"   dark_mode: {engine.get_preference('dark_mode')}")
    
    # Record interactions
    engine.record_interaction(
        "How do I implement a REST API?",
        "Here's how to implement a REST API..." * 10
    )
    engine.record_interaction(
        "Explain the algorithm complexity",
        "The time complexity is O(n log n)..."
    )
    
    # Check inferred preferences
    print("\n🤖 Inferred Preferences:")
    level = engine.get_preference('expertise_level')
    print(f"   expertise_level: {level} (inferred from technical queries)")
    
    # Get personalized prompt
    print("\n📝 Personalized System Prompt Additions:")
    prompt = engine.get_personalized_prompt()
    for line in prompt.split('\n'):
        print(f"   {line}")
    
    # High confidence preferences
    print("\n🎯 High Confidence Preferences:")
    high_conf = engine.get_high_confidence_preferences(0.5)
    for key, value in high_conf.items():
        print(f"   {key}: {value}")
    
    print(f"\n📊 Stats: {engine.get_stats()}")
    print("\n✅ Personalization demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_personalization())
