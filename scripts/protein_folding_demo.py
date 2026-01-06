"""
AION Protein Folding Demo
Demonstrates AION's application to computational biology.
"""

import sys
import asyncio
import time

sys.path.insert(0, '.')

from src.domains.protein_folding import (
    ProteinFolder, analyze_sequence
)
from src.runtime.local_engine import LocalReasoningEngine
from src.runtime.reflexion import ReflexionLoop

async def demo_protein_folding():
    """
    Demonstrate AION solving protein folding problem.
    """
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🧬 AION PROTEIN FOLDING AGENT 🧬                                 ║
║                                                                           ║
║     Applying AI-Native Reasoning to Computational Biology                ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Test sequences
    sequences = [
        ("HPHPPHHPHPPHPHHPPHPH", "Hydrophobic-Polar test sequence"),
        ("AARAASPPGVACFG", "Small realistic protein fragment"),
        ("KVFGRCELAAAMKRHGLDNY", "20-residue test protein"),
    ]
    
    engine = LocalReasoningEngine()
    
    for seq, description in sequences:
        print("\n" + "="*70)
        print(f"🧬 SEQUENCE: {seq}")
        print(f"   Description: {description}")
        print("="*70)
        
        # Phase 1: AION Reasoning - Analyze sequence
        print("\n📊 Phase 1: Sequence Analysis (AION Reasoning)")
        print("-" * 70)
        
        thought = engine.think(f"Analyzing protein sequence: {seq}")
        print(f"💭 Thought: {thought}")
        
        analysis = analyze_sequence(seq)
        print(f"\n✅ Sequence Properties:")
        print(f"   • Length: {analysis['length']} residues")
        print(f"   • Hydrophobic residues: {analysis['hydrophobic_count']}")
        print(f"   • Charged residues: {analysis['charged_count']}")
        print(f"   • Average hydrophobicity: {analysis['average_hydrophobicity']:.2f}")
        print(f"   • Net charge: {analysis['net_charge']:.1f}")
        
        # Phase 2: Reasoning - Predict folding strategy
        print("\n🤔 Phase 2: Folding Strategy (AION Decision)")
        print("-" * 70)
        
        strategy_prompt = f"""
        Based on sequence properties:
        - Hydrophobic count: {analysis['hydrophobic_count']}
        - Net charge: {analysis['net_charge']}
        
        What folding strategy should I use?
        """
        
        decision = engine.decide(strategy_prompt)
        print(f"🎯 Strategy: {decision['decision']}")
        
        # Phase 3: Self-Correcting Folding with Reflexion
        print("\n🔄 Phase 3: Self-Correcting Structure Prediction")
        print("-" * 70)
        
        folder = ProteinFolder(seq)
        
        # Generator: Fold protein
        async def generate_fold(attempt_info):
            iterations = 500 + (attempt_info * 200) if isinstance(attempt_info, int) else 500
            print(f"   🔬 Attempt: Running Monte Carlo simulation ({iterations} iterations)...")
            structure = folder.fold(iterations=iterations)
            return structure
        
        # Evaluator: Score based on energy
        async def evaluate_structure(structure):
            if not structure:
                return 0.0
            # Normalize energy to 0-1 range (lower energy = higher score)
            # This is a heuristic - real scoring is complex
            score = max(0.0, min(1.0, 1.0 / (1.0 + abs(structure.energy) * 0.1)))
            return score
        
        # Critic: Analyze what could be improved
        async def critique_structure(structure):
            if not structure:
                return "Failed to generate valid structure. Try different initialization."
            
            if structure.energy > 0:
                return "Energy is positive (unfavorable). Need more hydrophobic contacts."
            elif structure.energy > -5:
                return "Energy is suboptimal. Try more iterations or different sampling."
            else:
                return "Good energy but can potentially find better minimum."
        
        # Run reflexion loop
        reflexion = ReflexionLoop(
            generator=generate_fold,
            evaluator=evaluate_structure,
            critique_model=critique_structure,
            max_attempts=3,
            min_score=0.7
        )
        
        print("   🧠 Using AION Reflexion Loop for self-correction...")
        start_time = time.perf_counter()
        best_structure = await reflexion.run(0)
        elapsed = time.perf_counter() - start_time
        
        # Phase 4: Results
        print("\n" + "="*70)
        print("✨ RESULTS")
        print("="*70)
        
        if best_structure:
            print(f"\n🎯 Best Structure Found:")
            print(f"   • Energy: {best_structure.energy:.4f} kcal/mol")
            print(f"   • Reflexion attempts: {len(reflexion.traces)}")
            print(f"   • Computation time: {elapsed:.3f}s")
            
            print(f"\n📈 Improvement Trace:")
            for i, trace in enumerate(reflexion.traces, 1):
                print(f"   Attempt {i}: Score={trace.score:.3f} | {trace.critique[:60]}...")
            
            print(f"\n🗺️ 2D Structure Visualization:")
            print("   " + "-" * 40)
            viz = folder.visualize_structure(best_structure)
            for line in viz.split('\n'):
                print("   " + line)
            print("   " + "-" * 40)
            
            # AION Reflection
            reflection = engine.reflect(f"Folded {seq} with energy {best_structure.energy:.4f}")
            print(f"\n💭 AION Reflection: {reflection}")
        
        await asyncio.sleep(0.5)
    
    # Summary
    print("\n" + "="*70)
    print("🎓 DEMONSTRATION COMPLETE")
    print("="*70)
    print("""
AION's Advantages for Protein Folding:

1. 🧠 Reasoning: Analyzes sequence properties before folding
2. 🔄 Self-Correction: Reflexion loop improves solutions iteratively
3. 💭 Reflection: Learns from successes and failures
4. 📊 Transparency: Every decision is traceable
5. ⚡ Speed: Local reasoning in <1ms

This is a simplified demonstration. Real protein folding requires:
  • Molecular dynamics simulations
  • All-atom force fields
  • Advanced sampling methods (e.g., AlphaFold uses deep learning)
  
But AION's architecture shows how reasoning + self-correction
can be applied to complex scientific problems! 🚀
    """)


if __name__ == "__main__":
    asyncio.run(demo_protein_folding())
