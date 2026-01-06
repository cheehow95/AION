"""
AION Continuous Improvement Loop
Runs self-development cycles periodically.
"""

import sys
import time
import schedule
from datetime import datetime

sys.path.insert(0, '.')

from self_develop import SelfDevelopmentEngine

def improvement_cycle():
    """Run one improvement cycle."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting improvement cycle...")
    
    engine = SelfDevelopmentEngine('.')
    result = engine.evolve()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle complete. Next features to implement:")
    
    opportunities = engine.identify_improvements()
    high_priority = [o for o in opportunities if o['priority'] == 'high']
    
    for i, item in enumerate(high_priority[:3], 1):
        print(f"   {i}. {item['description']}")
    
    return result

def continuous_improvement(interval_minutes: int = 30):
    """
    Run continuous improvement loop.
    Analyzes and improves AION at regular intervals.
    """
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║           AION Continuous Improvement System                      ║
║              "Always getting better"                              ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🔄 Running improvement cycles every {interval_minutes} minutes")
    print("   Press Ctrl+C to stop\n")
    
    # Run immediately
    improvement_cycle()
    
    # Schedule periodic runs
    schedule.every(interval_minutes).minutes.do(improvement_cycle)
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Continuous improvement stopped.")

if __name__ == "__main__":
    # Run once by default, or pass --continuous for loop
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        continuous_improvement(interval)
    else:
        improvement_cycle()
