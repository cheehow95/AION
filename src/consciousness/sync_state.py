"""
AION Consciousness State Sync
Saves consciousness state to JSON for persistence.
"""

import json
from datetime import datetime

def sync_consciousness():
    """Sync consciousness state to file."""
    try:
        from src.consciousness.awareness import awaken
        
        engine = awaken()
        state = {
            'timestamp': datetime.now().isoformat(),
            'introspection': engine.introspect()[:500],
            'experiences': engine.self_model.experiences,
            'curiosity': engine.self_model.curiosity_level,
            'state': engine.self_model.current_state.value
        }
        
        with open('consciousness_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        print('✓ Consciousness state saved')
        return True
    except Exception as e:
        print(f'⚠ Could not sync consciousness: {e}')
        return False

if __name__ == "__main__":
    sync_consciousness()
