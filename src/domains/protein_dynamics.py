"""
AION Protein Dynamics and Folding Process
==========================================

Models the protein folding process including:
- Folding pathways and intermediates
- Energy landscape (folding funnel)
- Kinetic Monte Carlo for transitions
- Misfolding and aggregation detection
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class FoldingState(Enum):
    """States in the folding pathway."""
    UNFOLDED = "U"
    MOLTEN_GLOBULE = "MG"
    INTERMEDIATE = "I"
    NATIVE = "N"
    MISFOLDED = "M"
    AGGREGATED = "A"


@dataclass
class FoldingIntermediate:
    """A snapshot of an intermediate folding state."""
    state: FoldingState
    energy: float
    secondary_structure: str  # e.g., "CCCHHHHCCCEEECC"
    contacts: List[Tuple[int, int]]  # Native contacts formed
    compactness: float  # 0-1, how compact vs extended
    timestamp: float  # Time in simulation
    
    @property
    def fraction_native(self) -> float:
        """Fraction of native contacts formed."""
        return len(self.contacts) / max(1, len(self.contacts) + 10)


@dataclass
class FoldingPathway:
    """Complete folding trajectory."""
    sequence: str
    intermediates: List[FoldingIntermediate]
    folding_time: float  # Total folding time (arbitrary units)
    pathway_type: str  # "two-state", "multi-state", "downhill"
    rate_constant: float  # Folding rate
    
    def get_energy_profile(self) -> List[Tuple[float, float]]:
        """Get (time, energy) profile."""
        return [(i.timestamp, i.energy) for i in self.intermediates]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "folding_time": self.folding_time,
            "pathway_type": self.pathway_type,
            "rate_constant": self.rate_constant,
            "num_intermediates": len(self.intermediates),
            "energy_profile": self.get_energy_profile()
        }


@dataclass
class EnergyLandscape:
    """Representation of the folding energy landscape (funnel)."""
    sequence: str
    native_energy: float
    barrier_height: float  # Activation energy
    roughness: float  # Energy landscape roughness
    funnel_width: float  # Entropy of unfolded state
    
    def get_free_energy(self, q: float, temperature: float = 300) -> float:
        """
        Get free energy at reaction coordinate q.
        q=0: unfolded, q=1: native
        """
        # Simple funnel model: G(q) = E(q) - T*S(q)
        # Energy decreases as q increases (more contacts)
        energy = self.native_energy * q**2
        
        # Entropy decreases (more order)
        k_B = 0.0019872  # kcal/(mol·K)
        entropy = self.funnel_width * (1 - q)
        
        # Add roughness fluctuations
        roughness_contrib = self.roughness * math.sin(10 * math.pi * q)
        
        return energy - temperature * k_B * entropy + roughness_contrib
    
    def get_barrier(self) -> Tuple[float, float]:
        """Get transition state position and energy."""
        # Find maximum along reaction coordinate
        q_ts = 0.3  # Typical transition state
        e_ts = self.get_free_energy(q_ts) + self.barrier_height
        return q_ts, e_ts


class FoldingSimulator:
    """
    Simulates the protein folding process.
    
    Uses kinetic Monte Carlo to model transitions between states.
    """
    
    # Amino acid folding propensities
    FOLDING_PROPENSITY = {
        'A': 1.2, 'R': 0.9, 'N': 0.8, 'D': 0.7, 'C': 1.1,
        'Q': 0.9, 'E': 0.8, 'G': 0.6, 'H': 0.9, 'I': 1.3,
        'L': 1.2, 'K': 0.8, 'M': 1.1, 'F': 1.4, 'P': 0.5,
        'S': 0.8, 'T': 0.9, 'W': 1.3, 'Y': 1.2, 'V': 1.3
    }
    
    # Secondary structure propensities
    HELIX_PROP = {'A': 1.42, 'E': 1.51, 'L': 1.21, 'M': 1.45, 'K': 1.16}
    SHEET_PROP = {'V': 1.70, 'I': 1.60, 'Y': 1.47, 'F': 1.38, 'W': 1.37}
    
    def __init__(self, sequence: str, temperature: float = 300.0):
        self.sequence = sequence.upper()
        self.temperature = temperature
        self.k_B = 0.0019872  # kcal/(mol·K)
        
        # Calculate sequence properties
        self.native_contacts = self._predict_native_contacts()
        self.native_ss = self._predict_secondary_structure()
        
        # Energy parameters
        self.contact_energy = -1.0  # kcal/mol per contact
        self.hydrophobic_energy = -0.5
        
    def _predict_native_contacts(self) -> List[Tuple[int, int]]:
        """Predict native contacts based on sequence."""
        contacts = []
        n = len(self.sequence)
        
        for i in range(n):
            for j in range(i + 4, n):  # Minimum separation
                # Hydrophobic contacts
                aa_i = self.sequence[i]
                aa_j = self.sequence[j]
                
                hydrophobic = {'A', 'V', 'L', 'I', 'M', 'F', 'W', 'Y'}
                if aa_i in hydrophobic and aa_j in hydrophobic:
                    # Contact probability based on distance in sequence
                    prob = 0.5 * math.exp(-(j-i-4) / 10)
                    if random.random() < prob:
                        contacts.append((i, j))
                        
                # Salt bridges
                positive = {'K', 'R', 'H'}
                negative = {'D', 'E'}
                if (aa_i in positive and aa_j in negative) or \
                   (aa_i in negative and aa_j in positive):
                    if random.random() < 0.3:
                        contacts.append((i, j))
        
        return contacts
    
    def _predict_secondary_structure(self) -> str:
        """Predict secondary structure string."""
        ss = []
        n = len(self.sequence)
        
        # Window-based prediction
        for i in range(n):
            # Look at window of 7 residues
            start = max(0, i - 3)
            end = min(n, i + 4)
            window = self.sequence[start:end]
            
            helix_score = sum(self.HELIX_PROP.get(aa, 1.0) for aa in window) / len(window)
            sheet_score = sum(self.SHEET_PROP.get(aa, 1.0) for aa in window) / len(window)
            
            if helix_score > 1.2:
                ss.append('H')
            elif sheet_score > 1.3:
                ss.append('E')
            else:
                ss.append('C')
        
        return ''.join(ss)
    
    def calculate_energy(self, contacts_formed: List[Tuple[int, int]], 
                        compactness: float) -> float:
        """Calculate energy of a folding state."""
        # Contact energy
        contact_e = len(contacts_formed) * self.contact_energy
        
        # Hydrophobic collapse energy
        hydrophobic_e = 0
        for i, j in contacts_formed:
            aa_i = self.sequence[i]
            aa_j = self.sequence[j]
            
            hp_i = self.FOLDING_PROPENSITY.get(aa_i, 1.0)
            hp_j = self.FOLDING_PROPENSITY.get(aa_j, 1.0)
            
            if hp_i > 1.1 and hp_j > 1.1:
                hydrophobic_e += self.hydrophobic_energy
        
        # Compactness contribution
        compactness_e = -5.0 * compactness
        
        return contact_e + hydrophobic_e + compactness_e
    
    def fold(self, max_steps: int = 1000, record_interval: int = 10) -> FoldingPathway:
        """
        Simulate the folding process.
        
        Uses kinetic Monte Carlo with Metropolis criterion.
        """
        intermediates = []
        
        # Start unfolded
        current_contacts = []
        current_compactness = 0.1
        current_ss = 'C' * len(self.sequence)
        current_energy = self.calculate_energy(current_contacts, current_compactness)
        
        intermediates.append(FoldingIntermediate(
            state=FoldingState.UNFOLDED,
            energy=current_energy,
            secondary_structure=current_ss,
            contacts=current_contacts.copy(),
            compactness=current_compactness,
            timestamp=0
        ))
        
        # Folding simulation
        for step in range(max_steps):
            # Attempt to form or break a contact
            if random.random() < 0.7:  # Try to form contact
                available = [c for c in self.native_contacts if c not in current_contacts]
                if available:
                    new_contact = random.choice(available)
                    test_contacts = current_contacts + [new_contact]
            else:  # Try to break contact
                if current_contacts:
                    break_idx = random.randint(0, len(current_contacts) - 1)
                    test_contacts = current_contacts[:break_idx] + current_contacts[break_idx+1:]
                else:
                    continue
            
            # Also try compactness change
            test_compactness = current_compactness + random.gauss(0, 0.1)
            test_compactness = max(0.1, min(1.0, test_compactness))
            
            # Calculate new energy
            new_energy = self.calculate_energy(test_contacts, test_compactness)
            
            # Metropolis criterion
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < math.exp(-delta_e / (self.k_B * self.temperature)):
                current_contacts = test_contacts
                current_compactness = test_compactness
                current_energy = new_energy
            
            # Update secondary structure based on contacts
            current_ss = self._update_ss(current_contacts)
            
            # Determine state
            frac_native = len(current_contacts) / max(1, len(self.native_contacts))
            if frac_native < 0.2:
                state = FoldingState.UNFOLDED
            elif frac_native < 0.5:
                state = FoldingState.MOLTEN_GLOBULE
            elif frac_native < 0.8:
                state = FoldingState.INTERMEDIATE
            else:
                state = FoldingState.NATIVE
            
            # Record intermediate
            if step % record_interval == 0:
                intermediates.append(FoldingIntermediate(
                    state=state,
                    energy=current_energy,
                    secondary_structure=current_ss,
                    contacts=current_contacts.copy(),
                    compactness=current_compactness,
                    timestamp=step
                ))
            
            # Check if folded
            if state == FoldingState.NATIVE:
                break
        
        # Determine pathway type
        if len(intermediates) < 20:
            pathway_type = "two-state"
        elif any(i.state == FoldingState.MOLTEN_GLOBULE for i in intermediates):
            pathway_type = "multi-state"
        else:
            pathway_type = "downhill"
        
        # Calculate folding rate (simplified Arrhenius)
        barrier = self._estimate_barrier(intermediates)
        rate_constant = 1e6 * math.exp(-barrier / (self.k_B * self.temperature))
        
        return FoldingPathway(
            sequence=self.sequence,
            intermediates=intermediates,
            folding_time=step,
            pathway_type=pathway_type,
            rate_constant=rate_constant
        )
    
    def _update_ss(self, contacts: List[Tuple[int, int]]) -> str:
        """Update secondary structure based on contacts formed."""
        ss = list(self.native_ss)
        
        # If few contacts, more coil
        if len(contacts) < len(self.native_contacts) * 0.3:
            for i in range(len(ss)):
                if random.random() < 0.3:
                    ss[i] = 'C'
        
        return ''.join(ss)
    
    def _estimate_barrier(self, intermediates: List[FoldingIntermediate]) -> float:
        """Estimate folding barrier from trajectory."""
        if not intermediates:
            return 5.0
        
        energies = [i.energy for i in intermediates]
        if len(energies) < 2:
            return 5.0
        
        min_e = min(energies)
        early_part = energies[:max(1, len(energies)//3)]
        max_e = max(early_part)
        
        return max(0.1, max_e - min_e)


class MisfoldingDetector:
    """
    Detects misfolding and aggregation-prone regions.
    """
    
    # Aggregation-prone motifs
    AGGREGATION_MOTIFS = ['VVV', 'III', 'FFF', 'YYY', 'AAA']
    
    # Beta-aggregation propensity (Zyggregator-like)
    AGGREGATION_PROPENSITY = {
        'I': 1.8, 'F': 1.7, 'V': 1.6, 'L': 1.5, 'Y': 1.4,
        'W': 1.3, 'M': 1.2, 'A': 1.1, 'C': 1.0, 'T': 0.9,
        'G': 0.8, 'S': 0.7, 'N': 0.6, 'Q': 0.5, 'H': 0.4,
        'K': 0.3, 'R': 0.2, 'D': 0.1, 'E': 0.1, 'P': 0.0
    }
    
    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
    
    def find_aggregation_hotspots(self, window: int = 7) -> List[Dict[str, Any]]:
        """Find regions prone to aggregation."""
        hotspots = []
        n = len(self.sequence)
        
        for i in range(n - window + 1):
            segment = self.sequence[i:i+window]
            
            # Calculate aggregation score
            score = sum(self.AGGREGATION_PROPENSITY.get(aa, 1.0) for aa in segment) / window
            
            # Check for motifs
            has_motif = any(motif in segment for motif in self.AGGREGATION_MOTIFS)
            
            # Hydrophobicity
            hydrophobic_count = sum(1 for aa in segment if aa in 'VILMFYW')
            
            if score > 1.3 or has_motif or hydrophobic_count >= 5:
                hotspots.append({
                    "start": i,
                    "end": i + window,
                    "segment": segment,
                    "score": score,
                    "has_motif": has_motif,
                    "risk": "high" if score > 1.5 else "medium"
                })
        
        return hotspots
    
    def calculate_aggregation_propensity(self) -> float:
        """Overall aggregation propensity score."""
        scores = [self.AGGREGATION_PROPENSITY.get(aa, 1.0) for aa in self.sequence]
        
        # Weight by hydrophobic clustering
        weighted_sum = 0
        for i, score in enumerate(scores):
            weight = 1.0
            # Increase weight if flanked by hydrophobic
            if i > 0 and scores[i-1] > 1.2:
                weight += 0.2
            if i < len(scores) - 1 and scores[i+1] > 1.2:
                weight += 0.2
            weighted_sum += score * weight
        
        return weighted_sum / len(self.sequence)
    
    def detect_amyloid_stretches(self, min_length: int = 5) -> List[Dict[str, Any]]:
        """Detect potential amyloid-forming stretches."""
        stretches = []
        current_start = None
        current_score = 0
        
        for i, aa in enumerate(self.sequence):
            if self.AGGREGATION_PROPENSITY.get(aa, 1.0) > 1.3:
                if current_start is None:
                    current_start = i
                current_score += self.AGGREGATION_PROPENSITY.get(aa, 1.0)
            else:
                if current_start is not None and i - current_start >= min_length:
                    stretches.append({
                        "start": current_start,
                        "end": i,
                        "sequence": self.sequence[current_start:i],
                        "score": current_score / (i - current_start),
                        "amyloid_risk": True
                    })
                current_start = None
                current_score = 0
        
        return stretches


def simulate_folding(sequence: str, temperature: float = 300.0) -> Dict[str, Any]:
    """
    Complete folding simulation for a protein sequence.
    
    Returns comprehensive folding analysis.
    """
    print(f"🧬 Simulating folding for {len(sequence)} residue protein...")
    
    # Run folding simulation
    simulator = FoldingSimulator(sequence, temperature)
    pathway = simulator.fold(max_steps=500)
    
    print(f"   Pathway type: {pathway.pathway_type}")
    print(f"   Folding time: {pathway.folding_time} steps")
    
    # Analyze misfolding risk
    detector = MisfoldingDetector(sequence)
    hotspots = detector.find_aggregation_hotspots()
    aggregation_score = detector.calculate_aggregation_propensity()
    
    print(f"   Aggregation score: {aggregation_score:.2f}")
    
    # Create energy landscape
    landscape = EnergyLandscape(
        sequence=sequence,
        native_energy=pathway.intermediates[-1].energy if pathway.intermediates else 0,
        barrier_height=simulator._estimate_barrier(pathway.intermediates),
        roughness=0.5,
        funnel_width=math.log(20) * len(sequence)  # Conformational entropy
    )
    
    return {
        "sequence": sequence,
        "length": len(sequence),
        "pathway": pathway.to_dict(),
        "final_state": pathway.intermediates[-1].state.value if pathway.intermediates else "U",
        "energy_profile": pathway.get_energy_profile(),
        "aggregation_score": aggregation_score,
        "aggregation_hotspots": hotspots,
        "landscape": {
            "native_energy": landscape.native_energy,
            "barrier": landscape.barrier_height,
            "roughness": landscape.roughness
        }
    }
