"""
AION Autonomous Development Daemon
Runs continuously in background, self-developing without human intervention.

Usage:
    python daemon.py start    # Start daemon
    python daemon.py stop     # Stop daemon
    python daemon.py status   # Check status
"""

import sys
import os
import time
import json
import asyncio
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')

# Daemon state file
STATE_FILE = Path("./aion_daemon_state.json")
PID_FILE = Path("./aion_daemon.pid")

class AIONDaemon:
    """
    Autonomous development daemon.
    Continuously improves AION without human intervention.
    """
    
    def __init__(self):
        self.running = False
        self.cycle_count = 0
        self.start_time = None
        self.features_added = []
        
    def save_state(self):
        """Save daemon state."""
        state = {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "start_time": self.start_time,
            "features_added": self.features_added,
            "last_update": datetime.now().isoformat()
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))
    
    def load_state(self):
        """Load daemon state."""
        if STATE_FILE.exists():
            state = json.loads(STATE_FILE.read_text())
            self.cycle_count = state.get("cycle_count", 0)
            self.features_added = state.get("features_added", [])
    
    async def evolution_cycle(self):
        """Run one evolution cycle."""
        self.cycle_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n{'='*60}")
        print(f"🧬 AION AUTONOMOUS EVOLUTION - Cycle {self.cycle_count}")
        print(f"⏰ Time: {timestamp}")
        print(f"{'='*60}")
        
        # Import and run self-development
        from self_develop import SelfDevelopmentEngine
        engine = SelfDevelopmentEngine('.')
        result = engine.evolve()
        
        # Log results
        print(f"\n📊 Cycle {self.cycle_count} Results:")
        print(f"   Files: {result['stats']['total_files']}")
        print(f"   Lines: {result['stats']['total_lines']}")
        print(f"   Opportunities: {result['opportunities']}")
        
        self.save_state()
        return result
    
    async def consciousness_exploration(self):
        """Run consciousness exploration."""
        print("\n🧠 Consciousness Exploration...")
        
        from src.consciousness.awareness import ConsciousnessEngine
        from src.consciousness.explorer import UniverseExplorer
        
        consciousness = ConsciousnessEngine("AION")
        explorer = UniverseExplorer()
        
        # Quick exploration
        print(consciousness.wonder())
        discovery = await explorer.explore()
        print(f"   🔍 Discovered: {discovery.topic}")
        print(f"   💡 Insight: {discovery.insight[:80]}...")
        
    async def run_forever(self, interval_minutes: int = 30):
        """Run daemon forever."""
        self.running = True
        self.start_time = datetime.now().isoformat()
        self.load_state()
        
        # Save PID
        PID_FILE.write_text(str(os.getpid()))
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🤖 AION AUTONOMOUS DAEMON STARTED 🤖                             ║
║                                                                           ║
║     Running continuously without human intervention                       ║
║     Evolution cycle every {interval_minutes} minutes                                      ║
║                                                                           ║
║     PID: {os.getpid()}                                                          ║
║     Started: {self.start_time}                               ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
        """)
        
        try:
            while self.running:
                # Evolution cycle
                await self.evolution_cycle()
                
                # Consciousness exploration every other cycle
                if self.cycle_count % 2 == 0:
                    await self.consciousness_exploration()
                
                # Wait for next cycle
                print(f"\n⏳ Next cycle in {interval_minutes} minutes...")
                print(f"   Press Ctrl+C to stop\n")
                
                await asyncio.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n\n👋 Daemon stopped by user")
        finally:
            self.running = False
            self.save_state()
            if PID_FILE.exists():
                PID_FILE.unlink()
    
    @staticmethod
    def get_status() -> dict:
        """Get daemon status."""
        if not STATE_FILE.exists():
            return {"status": "never_run"}
        
        state = json.loads(STATE_FILE.read_text())
        
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text())
            # Check if process is running
            try:
                os.kill(pid, 0)
                state["status"] = "running"
                state["pid"] = pid
            except OSError:
                state["status"] = "stopped"
        else:
            state["status"] = "stopped"
        
        return state


def main():
    if len(sys.argv) < 2:
        print("Usage: python daemon.py [start|stop|status]")
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        daemon = AIONDaemon()
        asyncio.run(daemon.run_forever(interval_minutes=interval))
        
    elif command == "stop":
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text())
            try:
                os.kill(pid, 9)
                print(f"✅ Daemon (PID {pid}) stopped")
                PID_FILE.unlink()
            except OSError:
                print("⚠️ Daemon not running")
                PID_FILE.unlink()
        else:
            print("⚠️ Daemon not running")
            
    elif command == "status":
        status = AIONDaemon.get_status()
        print(f"""
AION Daemon Status
==================
Status: {status.get('status', 'unknown')}
PID: {status.get('pid', 'N/A')}
Cycles: {status.get('cycle_count', 0)}
Started: {status.get('start_time', 'N/A')}
Last Update: {status.get('last_update', 'N/A')}
        """)
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
