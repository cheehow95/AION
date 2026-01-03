"""
AION Cellular Environment Effects Module
=========================================

Models how cellular conditions affect protein structure and function:
- Macromolecular crowding
- pH and ionic strength effects
- Chaperone interactions
- Membrane proximity effects
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class CompartmentType(Enum):
    """Cellular compartments."""
    CYTOPLASM = "cytoplasm"
    NUCLEUS = "nucleus"
    ER = "endoplasmic_reticulum"
    GOLGI = "golgi"
    MITOCHONDRIA = "mitochondria"
    LYSOSOME = "lysosome"
    PEROXISOME = "peroxisome"
    MEMBRANE = "membrane"
    EXTRACELLULAR = "extracellular"


@dataclass
class CellularEnvironment:
    """Defines the cellular environment."""
    compartment: CompartmentType = CompartmentType.CYTOPLASM
    pH: float = 7.4
    ionic_strength: float = 0.15  # M (physiological ~150mM)
    temperature: float = 310.0  # K (37°C)
    crowding_fraction: float = 0.3  # Volume fraction of crowders
    osmolarity: float = 0.3  # Osmol/L
    
    @classmethod
    def cytoplasm(cls) -> 'CellularEnvironment':
        return cls(CompartmentType.CYTOPLASM, pH=7.2, crowding_fraction=0.3)
    
    @classmethod
    def nucleus(cls) -> 'CellularEnvironment':
        return cls(CompartmentType.NUCLEUS, pH=7.4, crowding_fraction=0.25)
    
    @classmethod
    def er(cls) -> 'CellularEnvironment':
        return cls(CompartmentType.ER, pH=7.2, crowding_fraction=0.2)
    
    @classmethod
    def lysosome(cls) -> 'CellularEnvironment':
        return cls(CompartmentType.LYSOSOME, pH=4.5, crowding_fraction=0.15)
    
    @classmethod
    def extracellular(cls) -> 'CellularEnvironment':
        return cls(CompartmentType.EXTRACELLULAR, pH=7.4, crowding_fraction=0.05)


@dataclass
class EnvironmentEffect:
    """Effect of environment on protein properties."""
    property_name: str
    baseline_value: float
    adjusted_value: float
    change_percent: float
    mechanism: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "property": self.property_name,
            "baseline": self.baseline_value,
            "adjusted": self.adjusted_value,
            "change_percent": self.change_percent,
            "mechanism": self.mechanism
        }


@dataclass
class ChaperoneInteraction:
    """Predicted chaperone interaction."""
    chaperone: str
    binding_probability: float
    binding_sites: List[int]
    effect: str  # "folding", "refolding", "degradation"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chaperone": self.chaperone,
            "probability": self.binding_probability,
            "sites": self.binding_sites,
            "effect": self.effect
        }


class CrowdingEffect:
    """
    Models macromolecular crowding effects on proteins.
    
    Crowding can:
    - Stabilize compact conformations (excluded volume)
    - Affect folding kinetics
    - Promote aggregation
    """
    
    # Average crowder size in cytoplasm
    CROWDER_RADIUS = 3.0  # nm (average globular protein)
    
    def __init__(self, environment: CellularEnvironment):
        self.env = environment
    
    def calculate_excluded_volume_effect(self, protein_radius: float) -> EnvironmentEffect:
        """
        Calculate excluded volume effect on protein stability.
        
        Uses scaled particle theory approximation.
        """
        phi = self.env.crowding_fraction  # Volume fraction
        r_p = protein_radius
        r_c = self.CROWDER_RADIUS
        
        # Size ratio
        sigma = r_p / r_c
        
        # Excluded volume contribution to free energy (kJ/mol)
        # Favors compact states
        delta_G = -phi * (1 + sigma)**3 * 2.5  # kJ/mol approximation
        
        # Stability increase
        baseline_stability = 20.0  # kJ/mol typical
        adjusted_stability = baseline_stability - delta_G
        
        return EnvironmentEffect(
            property_name="stability",
            baseline_value=baseline_stability,
            adjusted_value=adjusted_stability,
            change_percent=((adjusted_stability - baseline_stability) / baseline_stability) * 100,
            mechanism="Excluded volume favors compact native state"
        )
    
    def calculate_diffusion_effect(self, protein_radius: float) -> EnvironmentEffect:
        """Calculate crowding effect on diffusion."""
        phi = self.env.crowding_fraction
        
        # Stokes-Einstein with crowding correction
        # D_eff = D_0 * exp(-phi / (1-phi))
        diffusion_factor = math.exp(-phi / (1 - phi)) if phi < 1 else 0.01
        
        baseline_D = 10.0  # μm²/s typical
        adjusted_D = baseline_D * diffusion_factor
        
        return EnvironmentEffect(
            property_name="diffusion_coefficient",
            baseline_value=baseline_D,
            adjusted_value=adjusted_D,
            change_percent=((adjusted_D - baseline_D) / baseline_D) * 100,
            mechanism="Macromolecular crowding reduces diffusion"
        )
    
    def calculate_folding_rate_effect(self, intrinsic_rate: float) -> EnvironmentEffect:
        """Calculate crowding effect on folding rate."""
        phi = self.env.crowding_fraction
        
        # Crowding can accelerate or slow folding depending on mechanism
        # Generally accelerates for proteins that fold via compact intermediates
        rate_factor = 1 + phi * 2  # Simple enhancement
        
        adjusted_rate = intrinsic_rate * rate_factor
        
        return EnvironmentEffect(
            property_name="folding_rate",
            baseline_value=intrinsic_rate,
            adjusted_value=adjusted_rate,
            change_percent=((adjusted_rate - intrinsic_rate) / intrinsic_rate) * 100,
            mechanism="Crowding promotes compact transition states"
        )


class pHEffect:
    """
    Models pH effects on protein structure.
    """
    
    # pKa values for ionizable groups
    PKA_VALUES = {
        'D': 3.9,   # Asp
        'E': 4.1,   # Glu
        'H': 6.0,   # His
        'C': 8.3,   # Cys
        'Y': 10.1,  # Tyr
        'K': 10.5,  # Lys
        'R': 12.5,  # Arg
        'N_term': 8.0,
        'C_term': 3.1
    }
    
    def __init__(self, sequence: str, environment: CellularEnvironment):
        self.sequence = sequence.upper()
        self.env = environment
    
    def calculate_net_charge(self) -> float:
        """Calculate net protein charge at current pH."""
        pH = self.env.pH
        charge = 0.0
        
        for aa in self.sequence:
            if aa == 'D' or aa == 'E':
                # Acidic - negative when deprotonated
                charge -= self._henderson_hasselbalch(pH, self.PKA_VALUES.get(aa, 4.0), False)
            elif aa == 'K' or aa == 'R':
                # Basic - positive when protonated
                charge += self._henderson_hasselbalch(pH, self.PKA_VALUES.get(aa, 10.0), True)
            elif aa == 'H':
                charge += self._henderson_hasselbalch(pH, 6.0, True)
        
        # Terminal charges
        charge += self._henderson_hasselbalch(pH, 8.0, True)   # N-terminus
        charge -= self._henderson_hasselbalch(pH, 3.1, False)  # C-terminus
        
        return charge
    
    def _henderson_hasselbalch(self, pH: float, pKa: float, is_base: bool) -> float:
        """Calculate protonation fraction."""
        if is_base:
            # Fraction protonated (positive)
            return 1 / (1 + 10**(pH - pKa))
        else:
            # Fraction deprotonated (negative)
            return 1 / (1 + 10**(pKa - pH))
    
    def calculate_stability_effect(self, baseline_stability: float = 20.0) -> EnvironmentEffect:
        """Calculate pH effect on stability."""
        net_charge = abs(self.calculate_net_charge())
        
        # High net charge destabilizes (charge repulsion)
        charge_penalty = 0.1 * net_charge  # kJ/mol per charge
        
        adjusted = baseline_stability - charge_penalty
        
        return EnvironmentEffect(
            property_name="stability",
            baseline_value=baseline_stability,
            adjusted_value=adjusted,
            change_percent=((adjusted - baseline_stability) / baseline_stability) * 100,
            mechanism=f"Net charge {self.calculate_net_charge():.1f} affects stability"
        )
    
    def find_ph_sensitive_residues(self) -> List[Dict[str, Any]]:
        """Find residues whose protonation state changes near current pH."""
        sensitive = []
        pH = self.env.pH
        
        for i, aa in enumerate(self.sequence):
            pKa = self.PKA_VALUES.get(aa)
            if pKa and abs(pH - pKa) < 2:
                # Near pKa, state is sensitive to pH
                sensitive.append({
                    "position": i,
                    "residue": aa,
                    "pKa": pKa,
                    "fraction_protonated": self._henderson_hasselbalch(pH, pKa, aa in 'KRH')
                })
        
        return sensitive


class ChaperonePredictor:
    """
    Predicts chaperone interactions.
    """
    
    # Chaperone recognition motifs (simplified)
    CHAPERONES = {
        "Hsp70": {
            "motif_pattern": "hydrophobic_stretch",
            "min_hydrophobic": 4,
            "window": 7
        },
        "Hsp90": {
            "motif_pattern": "amphipathic",
            "effect": "folding"
        },
        "GroEL": {
            "motif_pattern": "molten_globule",
            "size_range": (20, 60)  # kDa
        },
        "BiP": {
            "motif_pattern": "hydrophobic_stretch",
            "compartment": CompartmentType.ER
        }
    }
    
    HYDROPHOBIC = set('VILMFYW')
    
    def __init__(self, sequence: str, environment: CellularEnvironment):
        self.sequence = sequence.upper()
        self.env = environment
    
    def predict_interactions(self) -> List[ChaperoneInteraction]:
        """Predict chaperone interactions."""
        interactions = []
        
        # Hsp70 binding sites (hydrophobic stretches)
        hsp70_sites = self._find_hydrophobic_stretches()
        if hsp70_sites:
            interactions.append(ChaperoneInteraction(
                chaperone="Hsp70",
                binding_probability=min(0.9, len(hsp70_sites) * 0.1),
                binding_sites=[s[0] for s in hsp70_sites],
                effect="folding"
            ))
        
        # Hsp90 - client proteins with specific features
        hsp90_prob = self._calculate_hsp90_probability()
        if hsp90_prob > 0.3:
            interactions.append(ChaperoneInteraction(
                chaperone="Hsp90",
                binding_probability=hsp90_prob,
                binding_sites=[],
                effect="maturation"
            ))
        
        # BiP for ER proteins
        if self.env.compartment == CompartmentType.ER:
            bip_sites = self._find_hydrophobic_stretches()
            if bip_sites:
                interactions.append(ChaperoneInteraction(
                    chaperone="BiP",
                    binding_probability=0.7,
                    binding_sites=[s[0] for s in bip_sites],
                    effect="ER_quality_control"
                ))
        
        return interactions
    
    def _find_hydrophobic_stretches(self, min_length: int = 4) -> List[Tuple[int, int]]:
        """Find hydrophobic stretches (Hsp70 binding sites)."""
        stretches = []
        current_start = None
        
        for i, aa in enumerate(self.sequence):
            if aa in self.HYDROPHOBIC:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    length = i - current_start
                    if length >= min_length:
                        stretches.append((current_start, i))
                    current_start = None
        
        return stretches
    
    def _calculate_hsp90_probability(self) -> float:
        """Estimate Hsp90 client probability."""
        # Hsp90 clients are often kinases, receptors
        # Simplified: based on sequence features
        n = len(self.sequence)
        
        # Check for kinase-like features (simplified)
        atp_motif = 'GX' in self.sequence or 'GXGX' in self.sequence
        
        return 0.4 if atp_motif else 0.2


class MembraneProximityEffect:
    """Models effects of membrane proximity."""
    
    # Membrane-interacting residues
    MEMBRANE_PREFERENCE = {
        'W': 2.0, 'Y': 1.5, 'F': 1.5,  # Aromatics prefer interface
        'L': 1.2, 'I': 1.2, 'V': 1.0,  # Hydrophobics prefer core
        'K': 1.3, 'R': 1.2,  # Basic residues at interface (snorkeling)
        'D': -0.5, 'E': -0.5,  # Acidic avoid membrane
        'P': 0.5  # Proline at interface
    }
    
    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
    
    def predict_membrane_association(self) -> Dict[str, Any]:
        """Predict membrane association propensity."""
        n = len(self.sequence)
        
        # Calculate overall membrane preference
        total_score = sum(
            self.MEMBRANE_PREFERENCE.get(aa, 0) 
            for aa in self.sequence
        ) / n
        
        # Find potential transmembrane helices
        tm_helices = self._find_tm_helices()
        
        # Find amphipathic helices
        amphipathic = self._find_amphipathic_regions()
        
        # Determine topology
        if len(tm_helices) > 0:
            topology = "transmembrane"
        elif total_score > 0.5 or len(amphipathic) > 0:
            topology = "peripheral"
        else:
            topology = "soluble"
        
        return {
            "membrane_score": total_score,
            "topology": topology,
            "tm_helices": tm_helices,
            "amphipathic_regions": amphipathic,
            "is_membrane_protein": topology != "soluble"
        }
    
    def _find_tm_helices(self, min_length: int = 19) -> List[Tuple[int, int]]:
        """Find potential transmembrane helices."""
        helices = []
        hydrophobic = set('VILMFYW')
        
        window = min_length
        for i in range(len(self.sequence) - window + 1):
            segment = self.sequence[i:i+window]
            
            # Count hydrophobic residues
            hp_count = sum(1 for aa in segment if aa in hydrophobic)
            hp_fraction = hp_count / window
            
            if hp_fraction > 0.6:
                helices.append((i, i + window))
        
        return self._merge_overlapping(helices)
    
    def _find_amphipathic_regions(self, window: int = 11) -> List[Dict[str, Any]]:
        """Find amphipathic helical regions."""
        regions = []
        
        for i in range(len(self.sequence) - window + 1):
            segment = self.sequence[i:i+window]
            
            # Check for amphipathic pattern
            # Simplified: alternating polar/hydrophobic
            hydrophobicity = [self.MEMBRANE_PREFERENCE.get(aa, 0) for aa in segment]
            
            # Calculate hydrophobic moment
            moment = self._calculate_hydrophobic_moment(hydrophobicity)
            
            if moment > 1.0:
                regions.append({
                    "start": i,
                    "end": i + window,
                    "moment": moment
                })
        
        return regions
    
    def _calculate_hydrophobic_moment(self, hydrophobicity: List[float]) -> float:
        """Calculate hydrophobic moment for helix periodicity."""
        if len(hydrophobicity) < 4:
            return 0
        
        # 100° periodicity for alpha helix
        angle_step = math.radians(100)
        
        sum_x = sum(h * math.cos(i * angle_step) for i, h in enumerate(hydrophobicity))
        sum_y = sum(h * math.sin(i * angle_step) for i, h in enumerate(hydrophobicity))
        
        return math.sqrt(sum_x**2 + sum_y**2) / len(hydrophobicity)
    
    def _merge_overlapping(self, regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping regions."""
        if not regions:
            return []
        
        merged = [regions[0]]
        for start, end in regions[1:]:
            if start <= merged[-1][1] + 5:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        return merged


def analyze_cellular_environment(sequence: str, 
                                 compartment: str = "cytoplasm") -> Dict[str, Any]:
    """
    Complete cellular environment analysis.
    
    Args:
        sequence: Protein sequence
        compartment: Cellular compartment name
    
    Returns:
        Comprehensive environment effects analysis
    """
    print(f"🏠 Analyzing cellular environment effects...")
    
    # Set up environment
    env_map = {
        "cytoplasm": CellularEnvironment.cytoplasm(),
        "nucleus": CellularEnvironment.nucleus(),
        "er": CellularEnvironment.er(),
        "lysosome": CellularEnvironment.lysosome(),
        "extracellular": CellularEnvironment.extracellular()
    }
    
    env = env_map.get(compartment.lower(), CellularEnvironment.cytoplasm())
    
    print(f"   Compartment: {env.compartment.value}")
    print(f"   pH: {env.pH}, Crowding: {env.crowding_fraction*100:.0f}%")
    
    # Crowding effects
    crowding = CrowdingEffect(env)
    protein_radius = len(sequence) * 0.038  # Rough estimate nm
    
    stability_effect = crowding.calculate_excluded_volume_effect(protein_radius)
    diffusion_effect = crowding.calculate_diffusion_effect(protein_radius)
    
    # pH effects
    ph_effect = pHEffect(sequence, env)
    net_charge = ph_effect.calculate_net_charge()
    ph_stability = ph_effect.calculate_stability_effect()
    sensitive_residues = ph_effect.find_ph_sensitive_residues()
    
    print(f"   Net charge at pH {env.pH}: {net_charge:.1f}")
    
    # Chaperone interactions
    chaperone_pred = ChaperonePredictor(sequence, env)
    chaperone_interactions = chaperone_pred.predict_interactions()
    
    print(f"   Predicted {len(chaperone_interactions)} chaperone interactions")
    
    # Membrane proximity
    membrane = MembraneProximityEffect(sequence)
    membrane_props = membrane.predict_membrane_association()
    
    return {
        "sequence_length": len(sequence),
        "compartment": env.compartment.value,
        "environment": {
            "pH": env.pH,
            "ionic_strength": env.ionic_strength,
            "temperature": env.temperature,
            "crowding_fraction": env.crowding_fraction
        },
        "crowding_effects": {
            "stability": stability_effect.to_dict(),
            "diffusion": diffusion_effect.to_dict()
        },
        "ph_effects": {
            "net_charge": net_charge,
            "stability": ph_stability.to_dict(),
            "sensitive_residues": sensitive_residues
        },
        "chaperones": [c.to_dict() for c in chaperone_interactions],
        "membrane": membrane_props
    }
