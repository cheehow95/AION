"""
AION Comprehensive Protein Analysis Demo
==========================================

Demonstrates all protein analysis capabilities:
1. Final 3D Structure
2. Folding Process & Dynamics
3. Drug Binding Sites
4. Disordered Regions
5. Cellular Environment Effects
"""

import sys
sys.path.insert(0, 'src')

from domains.protein_dynamics import simulate_folding, MisfoldingDetector
from domains.drug_binding import analyze_drug_binding
from domains.disorder import analyze_disorder
from domains.cellular import analyze_cellular_environment
from domains.protein_engine import UltimateProteinFolder

import math


def generate_backbone_coords(sequence: str) -> list:
    """Generate simple backbone coordinates for testing."""
    coords = []
    n = len(sequence)
    
    for i in range(n):
        if i < n // 3:
            x = 2.3 * math.cos(i * math.radians(100))
            y = 2.3 * math.sin(i * math.radians(100))
            z = i * 1.5
        elif i < 2 * n // 3:
            x = (i - n // 3) * 3.5
            y = 5 + math.sin(i * 0.5) * 2
            z = n // 3 * 1.5
        else:
            j = i - 2 * n // 3
            x = 2 * n // 3 * 3.5 / 3 + 2.3 * math.cos(j * math.radians(100))
            y = 5 + 2.3 * math.sin(j * math.radians(100))
            z = n // 3 * 1.5 + j * 1.5
        
        coords.append((x, y, z))
    
    return coords


def main():
    print("=" * 70)
    print("AION COMPREHENSIVE PROTEIN FOLDING SOLUTION")
    print("=" * 70)
    
    test_seq = "MQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    
    print(f"\n📋 Sequence: {test_seq[:30]}... ({len(test_seq)} residues)\n")
    
    # 1. Structure
    print("📐 PHASE 1: 3D STRUCTURE")
    folder = UltimateProteinFolder(test_seq)
    structure = folder.fold()
    print(f"   Energy: {structure.total_energy:.2f} kcal/mol\n")
    
    # 2. Folding
    print("🔄 PHASE 2: FOLDING DYNAMICS")
    folding = simulate_folding(test_seq)
    print(f"   Pathway: {folding['pathway']['pathway_type']}")
    print(f"   Aggregation: {folding['aggregation_score']:.2f}\n")
    
    # 3. Drug Binding
    print("💊 PHASE 3: DRUG BINDING")
    coords = generate_backbone_coords(test_seq)
    binding = analyze_drug_binding(test_seq, coords)
    print(f"   Druggable sites: {binding['num_druggable_pockets']}\n")
    
    # 4. Disorder
    print("🌊 PHASE 4: DISORDER")
    disorder = analyze_disorder(test_seq)
    print(f"   Classification: {disorder['classification']}")
    print(f"   LLPS: {disorder['llps']['propensity']:.2f}\n")
    
    # 5. Cellular Environment
    print("🏠 PHASE 5: CELLULAR ENVIRONMENT")
    cellular = analyze_cellular_environment(test_seq)
    print(f"   Net charge: {cellular['ph_effects']['net_charge']:.1f}")
    print(f"   Chaperones: {len(cellular['chaperones'])}")
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
