"""
AION Self-Development Engine
An autonomous system that continuously improves AION itself.
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, '.')

class SelfDevelopmentEngine:
    """
    Autonomous improvement engine for AION.
    Analyzes codebase, identifies improvements, and tracks progress.
    """
    
    def __init__(self, project_root: str = '.'):
        self.root = Path(project_root)
        self.log_file = self.root / 'evolution_log.json'
        self.improvements: List[Dict] = []
        self.metrics: Dict[str, Any] = {}
        
        # Load history
        self._load_history()
        
    def _load_history(self):
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                self.improvements = data.get('improvements', [])
                self.metrics = data.get('metrics', {})
    
    def _save_history(self):
        with open(self.log_file, 'w') as f:
            json.dump({
                'improvements': self.improvements,
                'metrics': self.metrics,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the AION codebase for metrics and opportunities."""
        print("🔍 Analyzing codebase...")
        
        stats = {
            'total_files': 0,
            'total_lines': 0,
            'python_files': 0,
            'aion_files': 0,
            'test_files': 0,
            'modules': [],
            'missing_tests': [],
            'missing_docs': [],
        }
        
        # Count files and lines
        for ext in ['py', 'aion', 'md']:
            for f in self.root.rglob(f'*.{ext}'):
                if '.git' in str(f) or '__pycache__' in str(f):
                    continue
                    
                stats['total_files'] += 1
                
                try:
                    lines = len(f.read_text(encoding='utf-8').splitlines())
                    stats['total_lines'] += lines
                except:
                    pass
                
                if ext == 'py':
                    stats['python_files'] += 1
                    if 'test_' in f.name:
                        stats['test_files'] += 1
                elif ext == 'aion':
                    stats['aion_files'] += 1
        
        # Check for missing tests
        src_files = list((self.root / 'src').rglob('*.py'))
        test_files = [f.name for f in (self.root / 'tests').glob('*.py')] if (self.root / 'tests').exists() else []
        
        for src in src_files:
            if src.name.startswith('_'):
                continue
            expected_test = f'test_{src.name}'
            if expected_test not in test_files:
                stats['missing_tests'].append(str(src.relative_to(self.root)))
        
        # Identify modules
        if (self.root / 'src').exists():
            stats['modules'] = [d.name for d in (self.root / 'src').iterdir() if d.is_dir() and not d.name.startswith('_')]
        
        self.metrics['codebase'] = stats
        return stats
    
    def identify_improvements(self) -> List[Dict]:
        """Identify potential improvements."""
        print("💡 Identifying improvements...")
        
        opportunities = []
        
        # 1. Missing tests
        if self.metrics.get('codebase', {}).get('missing_tests'):
            for f in self.metrics['codebase']['missing_tests'][:5]:
                opportunities.append({
                    'type': 'testing',
                    'priority': 'medium',
                    'description': f'Add tests for {f}',
                    'status': 'pending'
                })
        
        # 2. Feature ideas based on analysis
        feature_ideas = [
            {'type': 'feature', 'priority': 'high', 'description': 'Add Language Server Protocol (LSP) for IDE integration', 'status': 'pending'},
            {'type': 'feature', 'priority': 'high', 'description': 'Implement async streaming for LLM responses', 'status': 'pending'},
            {'type': 'feature', 'priority': 'medium', 'description': 'Add agent-to-agent message passing protocol', 'status': 'pending'},
            {'type': 'feature', 'priority': 'medium', 'description': 'Implement persistent vector store with ChromaDB', 'status': 'pending'},
            {'type': 'feature', 'priority': 'low', 'description': 'Add visual agent builder UI', 'status': 'pending'},
            {'type': 'optimization', 'priority': 'medium', 'description': 'Cache parsed ASTs for faster repeated execution', 'status': 'pending'},
            {'type': 'optimization', 'priority': 'low', 'description': 'Compile hot paths to bytecode', 'status': 'pending'},
        ]
        
        # Filter out already completed
        completed = {i['description'] for i in self.improvements if i.get('status') == 'completed'}
        for idea in feature_ideas:
            if idea['description'] not in completed:
                opportunities.append(idea)
        
        return opportunities
    
    def run_self_tests(self) -> Dict[str, Any]:
        """Run all tests and collect results."""
        print("🧪 Running self-tests...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': [],
            'duration_ms': 0
        }
        
        start = time.perf_counter()
        
        try:
            # Run our test suite
            import subprocess
            import os
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            result = subprocess.run(
                [sys.executable, 'scripts/run_tests.py'],
                capture_output=True,
                text=True,
                cwd=str(self.root),
                timeout=60,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                results['passed'] = result.stdout.count('✓')
                results['status'] = 'success'
            else:
                results['failed'] = 1
                results['errors'].append(result.stderr[:500])
                results['status'] = 'failed'
                
        except Exception as e:
            results['errors'].append(str(e))
            results['status'] = 'error'
        
        results['duration_ms'] = (time.perf_counter() - start) * 1000
        self.metrics['tests'] = results
        return results
    
    def generate_roadmap(self) -> str:
        """Generate a development roadmap based on analysis."""
        print("🗺️ Generating roadmap...")
        
        opportunities = self.identify_improvements()
        
        roadmap = ["# AION Development Roadmap", ""]
        roadmap.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        roadmap.append("")
        
        # Group by priority
        high = [o for o in opportunities if o['priority'] == 'high']
        medium = [o for o in opportunities if o['priority'] == 'medium']
        low = [o for o in opportunities if o['priority'] == 'low']
        
        if high:
            roadmap.append("## 🔴 High Priority")
            for item in high:
                roadmap.append(f"- [ ] {item['description']}")
            roadmap.append("")
            
        if medium:
            roadmap.append("## 🟡 Medium Priority")
            for item in medium:
                roadmap.append(f"- [ ] {item['description']}")
            roadmap.append("")
            
        if low:
            roadmap.append("## 🟢 Low Priority")
            for item in low:
                roadmap.append(f"- [ ] {item['description']}")
            roadmap.append("")
        
        # Add stats
        stats = self.metrics.get('codebase', {})
        roadmap.append("## 📊 Current Stats")
        roadmap.append(f"- Total Files: {stats.get('total_files', 0)}")
        roadmap.append(f"- Total Lines: {stats.get('total_lines', 0)}")
        roadmap.append(f"- Python Files: {stats.get('python_files', 0)}")
        roadmap.append(f"- AION Examples: {stats.get('aion_files', 0)}")
        roadmap.append(f"- Test Files: {stats.get('test_files', 0)}")
        
        return '\n'.join(roadmap)
    
    def evolve(self):
        """Run one evolution cycle."""
        print("\n" + "="*60)
        print("🧬 AION SELF-DEVELOPMENT CYCLE")
        print("="*60)
        
        # 1. Analyze
        stats = self.analyze_codebase()
        print(f"   📁 Files: {stats['total_files']} | Lines: {stats['total_lines']}")
        
        # 2. Test
        test_results = self.run_self_tests()
        print(f"   🧪 Tests: {test_results.get('passed', 0)} passed in {test_results['duration_ms']:.0f}ms")
        
        # 3. Identify improvements
        opportunities = self.identify_improvements()
        print(f"   💡 Opportunities: {len(opportunities)} identified")
        
        # 4. Generate roadmap
        roadmap = self.generate_roadmap()
        roadmap_path = self.root / 'ROADMAP.md'
        roadmap_path.write_text(roadmap, encoding='utf-8')
        print(f"   🗺️ Roadmap saved to: {roadmap_path}")
        
        # 5. Save state
        self._save_history()
        
        print("\n" + "="*60)
        print("✅ Evolution cycle complete!")
        print("="*60)
        
        return {
            'stats': stats,
            'tests': test_results,
            'opportunities': len(opportunities),
            'roadmap': str(roadmap_path)
        }


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              AION Self-Development Engine                         ║
║         "The system that improves itself"                         ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    engine = SelfDevelopmentEngine('.')
    result = engine.evolve()
    
    print("\n📋 Summary:")
    print(f"   • Codebase: {result['stats']['total_lines']} lines across {result['stats']['total_files']} files")
    print(f"   • Health: {result['tests'].get('status', 'unknown')}")
    print(f"   • Growth opportunities: {result['opportunities']}")
    print(f"\n🚀 Check {result['roadmap']} for next steps!")


if __name__ == "__main__":
    main()
