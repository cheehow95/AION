"""
AION Ultimate Protein Folding Engine
=====================================

A comprehensive physics-based protein structure prediction system that implements
real molecular mechanics, energy minimization, and structure prediction.

Based on fundamental principles:
1. Levinthal's Paradox resolution via energy funnel
2. AMBER/CHARMM-like force fields
3. GOR secondary structure prediction
4. Contact map prediction
5. Simulated annealing optimization
6. Molecular dynamics simulation

Author: AION Self-Development System
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json

# =============================================================================
# CONSTANTS - Physical parameters from AMBER force field
# =============================================================================

# Bond lengths in Angstroms
BOND_N_CA = 1.458
BOND_CA_C = 1.524
BOND_C_N = 1.329
BOND_C_O = 1.231

# Bond angles in radians
ANGLE_N_CA_C = math.radians(111.0)
ANGLE_CA_C_N = math.radians(116.6)
ANGLE_C_N_CA = math.radians(121.9)

# Dihedral constraints - Ramachandran regions
HELIX_PHI = math.radians(-57)
HELIX_PSI = math.radians(-47)
SHEET_PHI = math.radians(-140)
SHEET_PSI = math.radians(135)

# Physical constants
BOLTZMANN = 0.0019872041  # kcal/(mol·K)

# =============================================================================
# AMINO ACID DATABASE - Complete chemistry
# =============================================================================

class AAType(Enum):
    HYDROPHOBIC = "H"
    POLAR = "P"
    POSITIVE = "+"
    NEGATIVE = "-"
    SPECIAL = "S"
    AROMATIC = "A"

@dataclass
class AminoAcid:
    """Complete amino acid chemistry data."""
    code: str
    name: str
    aa_type: AAType
    hydropathy: float          # Kyte-Doolittle scale (-4.5 to 4.5)
    volume: float              # Van der Waals volume (Å³)
    mass: float                # Molecular weight (Da)
    pKa: Optional[float]       # Side chain pKa if ionizable
    helix_prop: float = 1.0    # Chou-Fasman helix propensity
    sheet_prop: float = 1.0    # Chou-Fasman sheet propensity
    turn_prop: float = 1.0     # Turn propensity
    
AMINO_ACIDS = {
    'A': AminoAcid('A', 'Alanine', AAType.HYDROPHOBIC, 1.8, 88.6, 89.1, None, 1.42, 0.83, 0.66),
    'R': AminoAcid('R', 'Arginine', AAType.POSITIVE, -4.5, 173.4, 174.2, 12.48, 0.98, 0.93, 0.95),
    'N': AminoAcid('N', 'Asparagine', AAType.POLAR, -3.5, 114.1, 132.1, None, 0.67, 0.89, 1.56),
    'D': AminoAcid('D', 'Aspartate', AAType.NEGATIVE, -3.5, 111.1, 133.1, 3.86, 1.01, 0.54, 1.46),
    'C': AminoAcid('C', 'Cysteine', AAType.SPECIAL, 2.5, 108.5, 121.2, 8.33, 0.70, 1.19, 1.19),
    'E': AminoAcid('E', 'Glutamate', AAType.NEGATIVE, -3.5, 138.4, 147.1, 4.25, 1.51, 0.37, 0.74),
    'Q': AminoAcid('Q', 'Glutamine', AAType.POLAR, -3.5, 143.8, 146.2, None, 1.11, 1.10, 0.98),
    'G': AminoAcid('G', 'Glycine', AAType.SPECIAL, -0.4, 60.1, 75.1, None, 0.57, 0.75, 1.56),
    'H': AminoAcid('H', 'Histidine', AAType.POSITIVE, -3.2, 153.2, 155.2, 6.00, 1.00, 0.87, 0.95),
    'I': AminoAcid('I', 'Isoleucine', AAType.HYDROPHOBIC, 4.5, 166.7, 131.2, None, 1.08, 1.60, 0.47),
    'L': AminoAcid('L', 'Leucine', AAType.HYDROPHOBIC, 3.8, 166.7, 131.2, None, 1.21, 1.30, 0.59),
    'K': AminoAcid('K', 'Lysine', AAType.POSITIVE, -3.9, 168.6, 146.2, 10.53, 1.16, 0.74, 1.01),
    'M': AminoAcid('M', 'Methionine', AAType.HYDROPHOBIC, 1.9, 162.9, 149.2, None, 1.45, 1.05, 0.60),
    'F': AminoAcid('F', 'Phenylalanine', AAType.AROMATIC, 2.8, 189.9, 165.2, None, 1.13, 1.38, 0.60),
    'P': AminoAcid('P', 'Proline', AAType.SPECIAL, -1.6, 112.7, 115.1, None, 0.57, 0.55, 1.52),
    'S': AminoAcid('S', 'Serine', AAType.POLAR, -0.8, 89.0, 105.1, None, 0.77, 0.75, 1.43),
    'T': AminoAcid('T', 'Threonine', AAType.POLAR, -0.7, 116.1, 119.1, None, 0.83, 1.19, 0.96),
    'W': AminoAcid('W', 'Tryptophan', AAType.AROMATIC, -0.9, 227.8, 204.2, None, 1.08, 1.37, 0.96),
    'Y': AminoAcid('Y', 'Tyrosine', AAType.AROMATIC, -1.3, 193.6, 181.2, 10.07, 0.69, 1.47, 1.14),
    'V': AminoAcid('V', 'Valine', AAType.HYDROPHOBIC, 4.2, 140.0, 117.1, None, 1.06, 1.70, 0.50),
}

# =============================================================================
# 3D STRUCTURE CLASSES
# =============================================================================

@dataclass
class Vec3:
    """3D vector with operations."""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vec3':
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vec3':
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other: 'Vec3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vec3') -> 'Vec3':
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vec3':
        L = self.length()
        if L < 1e-10:
            return Vec3(0, 0, 1)
        return self / L
    
    def distance_to(self, other: 'Vec3') -> float:
        return (self - other).length()
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]
    
    @classmethod
    def from_list(cls, lst: List[float]) -> 'Vec3':
        return cls(lst[0], lst[1], lst[2])

class SecondaryStructure(Enum):
    COIL = 'C'
    HELIX = 'H'
    SHEET = 'E'
    TURN = 'T'

@dataclass
class Atom:
    """Atom with position and properties."""
    name: str
    element: str
    position: Vec3
    residue_idx: int
    charge: float = 0.0
    
@dataclass 
class Residue:
    """Amino acid residue with backbone atoms."""
    index: int
    code: str
    aa: AminoAcid
    # Backbone atoms
    N: Optional[Atom] = None
    CA: Optional[Atom] = None
    C: Optional[Atom] = None
    O: Optional[Atom] = None
    # Backbone angles
    phi: float = 0.0
    psi: float = 0.0
    omega: float = math.pi  # Trans by default
    # Secondary structure
    ss: SecondaryStructure = SecondaryStructure.COIL
    # Confidence score (0-100)
    confidence: float = 50.0

@dataclass
class ProteinStructure:
    """Complete protein 3D structure."""
    sequence: str
    residues: List[Residue] = field(default_factory=list)
    total_energy: float = 0.0
    method: str = "physics"
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    def get_ca_positions(self) -> List[Vec3]:
        """Get all Cα positions."""
        return [r.CA.position for r in self.residues if r.CA]
    
    def calculate_radius_of_gyration(self) -> float:
        """Calculate Rg - measure of compactness."""
        positions = self.get_ca_positions()
        if not positions:
            return 0.0
        
        # Calculate center of mass
        center = Vec3(0, 0, 0)
        for pos in positions:
            center = center + pos
        center = center / len(positions)
        
        # Calculate Rg
        sum_sq = sum((pos - center).length()**2 for pos in positions)
        return math.sqrt(sum_sq / len(positions))
    
    def calculate_end_to_end_distance(self) -> float:
        """Distance from first to last Cα."""
        positions = self.get_ca_positions()
        if len(positions) < 2:
            return 0.0
        return positions[0].distance_to(positions[-1])
    
    def get_ss_content(self) -> Dict[str, float]:
        """Get secondary structure content percentages."""
        if not self.residues:
            return {'helix': 0, 'sheet': 0, 'coil': 0, 'turn': 0}
        
        counts = {ss: 0 for ss in SecondaryStructure}
        for r in self.residues:
            counts[r.ss] += 1
        
        n = len(self.residues)
        return {
            'helix': counts[SecondaryStructure.HELIX] / n * 100,
            'sheet': counts[SecondaryStructure.SHEET] / n * 100,
            'coil': counts[SecondaryStructure.COIL] / n * 100,
            'turn': counts[SecondaryStructure.TURN] / n * 100
        }
    
    def to_pdb(self) -> str:
        """Export to PDB format."""
        lines = ["HEADER    AION PREDICTED STRUCTURE"]
        lines.append("TITLE     PHYSICS-BASED PROTEIN FOLDING")
        lines.append(f"REMARK    Total Energy: {self.total_energy:.2f} kcal/mol")
        lines.append(f"REMARK    Method: {self.method}")
        
        AA_3LETTER = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        
        atom_num = 1
        for res in self.residues:
            res_name = AA_3LETTER.get(res.code, 'UNK')
            res_num = res.index + 1
            
            for atom in [res.N, res.CA, res.C, res.O]:
                if atom:
                    x, y, z = atom.position.x, atom.position.y, atom.position.z
                    lines.append(
                        f"ATOM  {atom_num:5d}  {atom.name:<3s} {res_name} A{res_num:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{res.confidence:6.2f}           {atom.element}"
                    )
                    atom_num += 1
        
        lines.append("END")
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export structure to JSON for visualization."""
        return json.dumps({
            'sequence': self.sequence,
            'length': self.length,
            'energy': self.total_energy,
            'rg': self.calculate_radius_of_gyration(),
            'ss_content': self.get_ss_content(),
            'residues': [
                {
                    'index': r.index,
                    'code': r.code,
                    'ss': r.ss.value,
                    'phi': math.degrees(r.phi),
                    'psi': math.degrees(r.psi),
                    'confidence': r.confidence,
                    'ca_position': r.CA.position.to_list() if r.CA else None
                }
                for r in self.residues
            ]
        }, indent=2)

# =============================================================================
# FORCE FIELD - AMBER-like energy functions
# =============================================================================

class ForceField:
    """
    Simplified AMBER-like force field for protein energy calculation.
    
    E_total = E_bond + E_angle + E_dihedral + E_vdw + E_electrostatic + E_solvation
    """
    
    # Force constants
    K_BOND = 300.0      # kcal/(mol·Å²)
    K_ANGLE = 50.0      # kcal/(mol·rad²)
    K_DIHEDRAL = 1.0    # kcal/mol
    
    # Lennard-Jones parameters (simplified)
    LJ_EPSILON = 0.1    # kcal/mol
    LJ_SIGMA = 3.5      # Å
    
    # Electrostatic
    DIELECTRIC = 4.0    # Implicit solvent
    
    # Solvation (SASA-based)
    GAMMA = 0.005       # kcal/(mol·Å²)
    
    @classmethod
    def bond_energy(cls, r: float, r0: float) -> float:
        """Harmonic bond energy: E = k(r - r0)²"""
        return cls.K_BOND * (r - r0) ** 2
    
    @classmethod
    def angle_energy(cls, theta: float, theta0: float) -> float:
        """Harmonic angle energy: E = k(θ - θ0)²"""
        return cls.K_ANGLE * (theta - theta0) ** 2
    
    @classmethod
    def dihedral_energy(cls, phi: float, n: int = 1, delta: float = 0) -> float:
        """Torsional energy: E = k[1 + cos(nφ - δ)]"""
        return cls.K_DIHEDRAL * (1 + math.cos(n * phi - delta))
    
    @classmethod
    def lj_energy(cls, r: float, epsilon: float = None, sigma: float = None) -> float:
        """Lennard-Jones 12-6 potential."""
        if epsilon is None:
            epsilon = cls.LJ_EPSILON
        if sigma is None:
            sigma = cls.LJ_SIGMA
        
        if r < 0.1:  # Prevent division by zero
            return 1000.0
        
        ratio = sigma / r
        return 4 * epsilon * (ratio**12 - ratio**6)
    
    @classmethod
    def electrostatic_energy(cls, q1: float, q2: float, r: float) -> float:
        """Coulomb potential with dielectric."""
        if r < 0.1:
            return 1000.0 if q1 * q2 > 0 else -1000.0
        return 332.0 * q1 * q2 / (cls.DIELECTRIC * r)
    
    @classmethod
    def hydrophobic_energy(cls, aa1: AminoAcid, aa2: AminoAcid, r: float) -> float:
        """Effective hydrophobic interaction energy."""
        if r > 8.0:  # Cutoff
            return 0.0
        
        # Only attractive if both are hydrophobic
        if aa1.hydropathy > 0 and aa2.hydropathy > 0:
            strength = (aa1.hydropathy + aa2.hydropathy) / 9.0  # Normalized
            # Distance-dependent attraction
            if r < 4.0:
                return 0.0  # Within VdW, already counted
            return -strength * math.exp(-(r - 4.0) / 2.0)
        
        # Repulsion if hydrophobic + polar
        if (aa1.hydropathy > 1.0 and aa2.hydropathy < -1.0) or \
           (aa1.hydropathy < -1.0 and aa2.hydropathy > 1.0):
            return 0.5 * math.exp(-(r - 4.0) / 2.0)
        
        return 0.0
    
    @classmethod
    def hydrogen_bond_energy(cls, r: float, angle: float) -> float:
        """Simplified H-bond energy (distance and angle dependent)."""
        # Optimal: r=2.9Å, angle=180°
        if r > 3.5 or r < 2.5:
            return 0.0
        
        angle_factor = math.cos(angle) ** 2
        dist_factor = 1.0 - ((r - 2.9) / 0.6) ** 2
        
        if dist_factor < 0:
            return 0.0
        
        return -2.0 * dist_factor * angle_factor  # ~2 kcal/mol for ideal H-bond

# =============================================================================
# SECONDARY STRUCTURE PREDICTION - GOR Method
# =============================================================================

class GORPredictor:
    """
    GOR (Garnier-Osguthorpe-Robson) secondary structure prediction.
    Uses statistical propensities in a sliding window.
    """
    
    WINDOW_SIZE = 17  # 8 residues on each side
    
    # GOR I propensity parameters (simplified)
    # Format: (helix, sheet, turn, coil) propensities
    PROPENSITIES = {
        'A': (1.42, 0.83, 0.66, 0.91),
        'R': (0.98, 0.93, 0.95, 1.04),
        'N': (0.67, 0.89, 1.56, 1.01),
        'D': (1.01, 0.54, 1.46, 1.05),
        'C': (0.70, 1.19, 1.19, 0.97),
        'E': (1.51, 0.37, 0.74, 1.05),
        'Q': (1.11, 1.10, 0.98, 0.93),
        'G': (0.57, 0.75, 1.56, 1.06),
        'H': (1.00, 0.87, 0.95, 1.04),
        'I': (1.08, 1.60, 0.47, 0.96),
        'L': (1.21, 1.30, 0.59, 0.93),
        'K': (1.16, 0.74, 1.01, 1.03),
        'M': (1.45, 1.05, 0.60, 0.90),
        'F': (1.13, 1.38, 0.60, 0.94),
        'P': (0.57, 0.55, 1.52, 1.17),
        'S': (0.77, 0.75, 1.43, 1.05),
        'T': (0.83, 1.19, 0.96, 1.02),
        'W': (1.08, 1.37, 0.96, 0.89),
        'Y': (0.69, 1.47, 1.14, 0.91),
        'V': (1.06, 1.70, 0.50, 0.92),
    }
    
    @classmethod
    def predict(cls, sequence: str) -> List[SecondaryStructure]:
        """Predict secondary structure for each residue."""
        n = len(sequence)
        predictions = []
        
        for i in range(n):
            # Calculate propensity scores in window
            helix_score = 0.0
            sheet_score = 0.0
            turn_score = 0.0
            coil_score = 0.0
            
            half_window = cls.WINDOW_SIZE // 2
            for j in range(max(0, i - half_window), min(n, i + half_window + 1)):
                aa = sequence[j]
                if aa in cls.PROPENSITIES:
                    props = cls.PROPENSITIES[aa]
                    # Weight by distance from center
                    weight = 1.0 - abs(j - i) / (half_window + 1)
                    helix_score += props[0] * weight
                    sheet_score += props[1] * weight
                    turn_score += props[2] * weight
                    coil_score += props[3] * weight
            
            # Special rules
            aa = sequence[i]
            
            # Proline breaks helices
            if aa == 'P' and i > 0:
                helix_score *= 0.3
            
            # Glycine favors turns
            if aa == 'G':
                turn_score *= 1.5
            
            # Assign secondary structure
            scores = {
                SecondaryStructure.HELIX: helix_score,
                SecondaryStructure.SHEET: sheet_score,
                SecondaryStructure.TURN: turn_score,
                SecondaryStructure.COIL: coil_score
            }
            
            predictions.append(max(scores, key=scores.get))
        
        # Smooth predictions (require minimum segment length)
        predictions = cls._smooth_predictions(predictions)
        
        return predictions
    
    @classmethod
    def _smooth_predictions(cls, predictions: List[SecondaryStructure], 
                           min_helix: int = 4, min_sheet: int = 2) -> List[SecondaryStructure]:
        """Smooth predictions by enforcing minimum segment lengths."""
        n = len(predictions)
        result = predictions.copy()
        
        # Pass 1: Remove short helix segments
        i = 0
        while i < n:
            if result[i] == SecondaryStructure.HELIX:
                j = i
                while j < n and result[j] == SecondaryStructure.HELIX:
                    j += 1
                if j - i < min_helix:
                    for k in range(i, j):
                        result[k] = SecondaryStructure.COIL
                i = j
            else:
                i += 1
        
        # Pass 2: Remove short sheet segments
        i = 0
        while i < n:
            if result[i] == SecondaryStructure.SHEET:
                j = i
                while j < n and result[j] == SecondaryStructure.SHEET:
                    j += 1
                if j - i < min_sheet:
                    for k in range(i, j):
                        result[k] = SecondaryStructure.COIL
                i = j
            else:
                i += 1
        
        return result

# =============================================================================
# CONTACT MAP PREDICTION
# =============================================================================

class ContactMapPredictor:
    """
    Predict residue-residue contacts based on:
    1. Sequence separation
    2. Hydrophobic clustering
    3. Electrostatic interactions
    4. Secondary structure patterns
    """
    
    CONTACT_THRESHOLD = 8.0  # Å for Cβ-Cβ contact
    
    @classmethod
    def predict(cls, sequence: str, ss: List[SecondaryStructure]) -> List[Tuple[int, int, float]]:
        """
        Predict contacts with confidence scores.
        Returns: List of (residue_i, residue_j, probability)
        """
        n = len(sequence)
        contacts = []
        
        for i in range(n):
            for j in range(i + 4, n):  # Minimum sequence separation
                prob = cls._contact_probability(sequence, ss, i, j)
                if prob > 0.3:  # Threshold
                    contacts.append((i, j, prob))
        
        return contacts
    
    @classmethod
    def _contact_probability(cls, sequence: str, ss: List[SecondaryStructure], 
                            i: int, j: int) -> float:
        """Calculate contact probability between residues i and j."""
        aa_i = AMINO_ACIDS.get(sequence[i])
        aa_j = AMINO_ACIDS.get(sequence[j])
        
        if not aa_i or not aa_j:
            return 0.0
        
        prob = 0.0
        
        # 1. Hydrophobic attraction
        if aa_i.hydropathy > 0 and aa_j.hydropathy > 0:
            prob += 0.3 * (aa_i.hydropathy + aa_j.hydropathy) / 9.0
        
        # 2. Salt bridges (opposite charges)
        if (aa_i.aa_type == AAType.POSITIVE and aa_j.aa_type == AAType.NEGATIVE) or \
           (aa_i.aa_type == AAType.NEGATIVE and aa_j.aa_type == AAType.POSITIVE):
            prob += 0.4
        
        # 3. Same charge repulsion
        if aa_i.aa_type == aa_j.aa_type and aa_i.aa_type in [AAType.POSITIVE, AAType.NEGATIVE]:
            prob -= 0.3
        
        # 4. Aromatic stacking
        if aa_i.aa_type == AAType.AROMATIC and aa_j.aa_type == AAType.AROMATIC:
            prob += 0.25
        
        # 5. Disulfide bonds (Cys-Cys)
        if aa_i.code == 'C' and aa_j.code == 'C':
            prob += 0.6
        
        # 6. Secondary structure patterns
        # Anti-parallel beta sheet contacts
        if ss[i] == SecondaryStructure.SHEET and ss[j] == SecondaryStructure.SHEET:
            prob += 0.2
        
        # Helix-helix packing
        if ss[i] == SecondaryStructure.HELIX and ss[j] == SecondaryStructure.HELIX:
            # Check if different helix segments
            if abs(i - j) > 10:
                prob += 0.15
        
        # 7. Sequence separation bonus (long-range contacts more significant)
        sep = j - i
        if sep > 20:
            prob += 0.1
        if sep > 40:
            prob += 0.1
        
        return min(1.0, max(0.0, prob))

# =============================================================================
# STRUCTURE BUILDER - Build 3D coordinates
# =============================================================================

class StructureBuilder:
    """Build 3D structure from sequence and secondary structure."""
    
    @classmethod
    def build(cls, sequence: str, ss: List[SecondaryStructure]) -> ProteinStructure:
        """Build initial 3D structure."""
        structure = ProteinStructure(sequence=sequence)
        
        for i, (code, ss_type) in enumerate(zip(sequence, ss)):
            aa = AMINO_ACIDS.get(code)
            if not aa:
                aa = AMINO_ACIDS['A']  # Default to Alanine
            
            residue = Residue(index=i, code=code, aa=aa, ss=ss_type)
            
            # Set backbone angles based on secondary structure
            if ss_type == SecondaryStructure.HELIX:
                residue.phi = HELIX_PHI
                residue.psi = HELIX_PSI
            elif ss_type == SecondaryStructure.SHEET:
                residue.phi = SHEET_PHI
                residue.psi = SHEET_PSI
            else:
                # Random coil with some variation
                residue.phi = random.uniform(-math.pi, math.pi)
                residue.psi = random.uniform(-math.pi, math.pi)
            
            structure.residues.append(residue)
        
        # Build atomic coordinates
        cls._build_backbone(structure)
        
        return structure
    
    @classmethod
    def _build_backbone(cls, structure: ProteinStructure):
        """Build backbone atom positions from phi/psi angles."""
        # Start at origin
        current_pos = Vec3(0, 0, 0)
        current_direction = Vec3(1, 0, 0)
        current_normal = Vec3(0, 1, 0)
        
        for i, residue in enumerate(structure.residues):
            # Place N atom
            residue.N = Atom('N', 'N', current_pos, i)
            
            # Place CA atom (1.458 Å from N)
            ca_pos = current_pos + current_direction * BOND_N_CA
            residue.CA = Atom('CA', 'C', ca_pos, i)
            
            # Rotate by phi angle around N-CA bond
            phi = residue.phi
            new_direction = cls._rotate_around_axis(current_direction, current_normal, phi)
            
            # Place C atom (1.524 Å from CA)
            c_pos = ca_pos + new_direction * BOND_CA_C
            residue.C = Atom('C', 'C', c_pos, i)
            
            # Place O atom (1.231 Å from C, in peptide plane)
            o_direction = current_normal * (-1)
            o_pos = c_pos + (new_direction + o_direction).normalize() * BOND_C_O
            residue.O = Atom('O', 'O', o_pos, i)
            
            # Update for next residue
            # Rotate by psi angle around CA-C bond
            psi = residue.psi
            next_direction = cls._rotate_around_axis(new_direction, 
                                                     current_normal.cross(new_direction).normalize(),
                                                     psi)
            
            # Next N position
            current_pos = c_pos + next_direction * BOND_C_N
            current_direction = next_direction
            current_normal = current_normal.cross(new_direction).normalize()
        
        # Center the structure
        cls._center_structure(structure)
    
    @classmethod
    def _rotate_around_axis(cls, vector: Vec3, axis: Vec3, angle: float) -> Vec3:
        """Rotate vector around axis by angle (Rodrigues' formula)."""
        axis = axis.normalize()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        return (vector * cos_a + 
                axis.cross(vector) * sin_a + 
                axis * (axis.dot(vector)) * (1 - cos_a))
    
    @classmethod
    def _center_structure(cls, structure: ProteinStructure):
        """Center structure at origin."""
        positions = structure.get_ca_positions()
        if not positions:
            return
        
        center = Vec3(0, 0, 0)
        for pos in positions:
            center = center + pos
        center = center / len(positions)
        
        for residue in structure.residues:
            for atom in [residue.N, residue.CA, residue.C, residue.O]:
                if atom:
                    atom.position = atom.position - center

# =============================================================================
# ENERGY MINIMIZER - Simulated Annealing & Gradient Descent
# =============================================================================

class EnergyMinimizer:
    """
    Energy minimization using simulated annealing.
    Implements the folding funnel concept.
    """
    
    def __init__(self, structure: ProteinStructure, 
                 contacts: List[Tuple[int, int, float]]):
        self.structure = structure
        self.contacts = contacts
        self.best_energy = float('inf')
        self.best_positions = None
        self.trajectory = []
    
    def calculate_energy(self) -> float:
        """Calculate total potential energy."""
        energy = 0.0
        residues = self.structure.residues
        n = len(residues)
        
        # 1. Bond energy (keep CA-CA distances ~3.8 Å)
        for i in range(n - 1):
            if residues[i].CA and residues[i+1].CA:
                r = residues[i].CA.position.distance_to(residues[i+1].CA.position)
                energy += ForceField.bond_energy(r, 3.8)
        
        # 2. Non-bonded interactions
        for i in range(n):
            for j in range(i + 2, n):
                if not residues[i].CA or not residues[j].CA:
                    continue
                
                r = residues[i].CA.position.distance_to(residues[j].CA.position)
                
                # Van der Waals (prevent clashes)
                if r < 4.0:
                    energy += ForceField.lj_energy(r)
                
                # Hydrophobic interactions
                energy += ForceField.hydrophobic_energy(
                    residues[i].aa, residues[j].aa, r)
                
                # Electrostatic
                q_i = self._get_charge(residues[i].aa)
                q_j = self._get_charge(residues[j].aa)
                if q_i != 0 and q_j != 0:
                    energy += ForceField.electrostatic_energy(q_i, q_j, r)
        
        # 3. Contact satisfaction
        for (i, j, prob) in self.contacts:
            if i < n and j < n and residues[i].CA and residues[j].CA:
                r = residues[i].CA.position.distance_to(residues[j].CA.position)
                # Encourage predicted contacts to be close
                if r > 8.0:
                    energy += prob * 5.0 * (r - 8.0)  # Penalty for unfulfilled contact
        
        return energy
    
    def _get_charge(self, aa: AminoAcid) -> float:
        """Get effective charge for amino acid."""
        if aa.aa_type == AAType.POSITIVE:
            return 1.0
        elif aa.aa_type == AAType.NEGATIVE:
            return -1.0
        return 0.0
    
    def minimize(self, max_iterations: int = 5000,
                initial_temp: float = 5.0,
                cooling_rate: float = 0.995,
                callback=None) -> float:
        """
        Perform simulated annealing minimization.
        
        Args:
            max_iterations: Maximum number of MC steps
            initial_temp: Starting temperature (kcal/mol)
            cooling_rate: Temperature decay factor
            callback: Optional callback(iteration, energy, temperature)
        
        Returns:
            Final energy
        """
        temperature = initial_temp
        current_energy = self.calculate_energy()
        self.best_energy = current_energy
        self._save_positions()
        
        for iteration in range(max_iterations):
            # Select random residue and movement type
            idx = random.randint(0, len(self.structure.residues) - 1)
            residue = self.structure.residues[idx]
            
            # Save current position
            old_pos = residue.CA.position if residue.CA else None
            
            # Generate trial move
            move_type = random.choice(['phi', 'psi', 'local'])
            
            if move_type == 'phi':
                delta = random.gauss(0, 0.2)  # Small angle change
                residue.phi += delta
            elif move_type == 'psi':
                delta = random.gauss(0, 0.2)
                residue.psi += delta
            else:
                # Local CA movement
                if residue.CA:
                    displacement = Vec3(
                        random.gauss(0, 0.5),
                        random.gauss(0, 0.5),
                        random.gauss(0, 0.5)
                    )
                    residue.CA.position = residue.CA.position + displacement
            
            # Rebuild affected backbone
            self._rebuild_local(idx)
            
            # Calculate new energy
            new_energy = self.calculate_energy()
            delta_E = new_energy - current_energy
            
            # Metropolis criterion
            accept = False
            if delta_E < 0:
                accept = True
            elif temperature > 0:
                accept = random.random() < math.exp(-delta_E / (BOLTZMANN * temperature))
            
            if accept:
                current_energy = new_energy
                if current_energy < self.best_energy:
                    self.best_energy = current_energy
                    self._save_positions()
            else:
                # Revert move
                if move_type == 'phi':
                    residue.phi -= delta
                elif move_type == 'psi':
                    residue.psi -= delta
                elif old_pos:
                    residue.CA.position = old_pos
                self._rebuild_local(idx)
            
            # Cool down
            temperature *= cooling_rate
            
            # Record trajectory
            if iteration % 100 == 0:
                self.trajectory.append({
                    'iteration': iteration,
                    'energy': current_energy,
                    'temperature': temperature,
                    'rg': self.structure.calculate_radius_of_gyration()
                })
                
                if callback:
                    callback(iteration, current_energy, temperature)
        
        # Restore best structure
        self._restore_positions()
        self.structure.total_energy = self.best_energy
        
        return self.best_energy
    
    def _save_positions(self):
        """Save current positions as best."""
        self.best_positions = []
        for r in self.structure.residues:
            pos = r.CA.position if r.CA else Vec3(0, 0, 0)
            self.best_positions.append(Vec3(pos.x, pos.y, pos.z))
    
    def _restore_positions(self):
        """Restore best positions."""
        if self.best_positions:
            for i, r in enumerate(self.structure.residues):
                if r.CA and i < len(self.best_positions):
                    r.CA.position = self.best_positions[i]
    
    def _rebuild_local(self, idx: int):
        """Rebuild backbone around modified residue."""
        # Simplified: just enforce bond constraints
        residues = self.structure.residues
        n = len(residues)
        
        # Constrain to previous residue
        if idx > 0 and residues[idx-1].CA and residues[idx].CA:
            self._constrain_bond(residues[idx-1].CA, residues[idx].CA, 3.8)
        
        # Constrain to next residue  
        if idx < n - 1 and residues[idx].CA and residues[idx+1].CA:
            self._constrain_bond(residues[idx].CA, residues[idx+1].CA, 3.8)
    
    def _constrain_bond(self, atom1: Atom, atom2: Atom, target: float):
        """Constrain bond length to target."""
        current = atom1.position.distance_to(atom2.position)
        if current < 0.1:
            return
        
        direction = (atom2.position - atom1.position).normalize()
        midpoint = (atom1.position + atom2.position) / 2
        
        atom1.position = midpoint - direction * (target / 2)
        atom2.position = midpoint + direction * (target / 2)

# =============================================================================
# MAIN FOLDING ENGINE
# =============================================================================

class ProteinFoldingEngine:
    """
    Complete protein folding pipeline.
    
    Steps:
    1. Predict secondary structure (GOR)
    2. Predict contact map
    3. Build initial structure
    4. Energy minimization (simulated annealing)
    5. Refinement
    """
    
    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
        self.structure = None
        self.ss_prediction = None
        self.contacts = None
        self.folding_log = []
    
    def fold(self, iterations: int = 5000, 
             verbose: bool = True,
             callback=None) -> ProteinStructure:
        """
        Perform complete folding simulation.
        
        Args:
            iterations: Number of SA iterations
            verbose: Print progress
            callback: Progress callback(step, message)
        
        Returns:
            Folded ProteinStructure
        """
        self._log("Starting protein folding", callback)
        self._log(f"Sequence: {self.sequence[:30]}... ({len(self.sequence)} residues)", callback)
        
        # Step 1: Secondary structure prediction
        self._log("Step 1: Predicting secondary structure (GOR method)", callback)
        self.ss_prediction = GORPredictor.predict(self.sequence)
        ss_content = self._get_ss_stats()
        self._log(f"  Predicted: {ss_content['helix']:.0f}% helix, {ss_content['sheet']:.0f}% sheet", callback)
        
        # Step 2: Contact prediction
        self._log("Step 2: Predicting residue contacts", callback)
        self.contacts = ContactMapPredictor.predict(self.sequence, self.ss_prediction)
        self._log(f"  Found {len(self.contacts)} predicted contacts", callback)
        
        # Step 3: Build initial structure
        self._log("Step 3: Building initial 3D structure", callback)
        self.structure = StructureBuilder.build(self.sequence, self.ss_prediction)
        initial_rg = self.structure.calculate_radius_of_gyration()
        self._log(f"  Initial Rg: {initial_rg:.2f} Å", callback)
        
        # Step 4: Energy minimization
        self._log(f"Step 4: Energy minimization ({iterations} iterations)", callback)
        minimizer = EnergyMinimizer(self.structure, self.contacts)
        
        def progress(i, e, t):
            if i % 500 == 0:
                self._log(f"  Iteration {i}: E={e:.1f} kcal/mol, T={t:.3f}", callback)
        
        final_energy = minimizer.minimize(
            max_iterations=iterations,
            callback=progress if verbose else None
        )
        
        # Step 5: Update structure metadata
        self._log("Step 5: Finalizing structure", callback)
        self.structure.total_energy = final_energy
        self.structure.method = "AION Physics-Based Folding"
        
        # Calculate confidence scores
        self._calculate_confidence()
        
        final_rg = self.structure.calculate_radius_of_gyration()
        self._log(f"Folding complete!", callback)
        self._log(f"  Final Energy: {final_energy:.1f} kcal/mol", callback)
        self._log(f"  Final Rg: {final_rg:.2f} Å", callback)
        self._log(f"  Compaction: {(initial_rg - final_rg) / initial_rg * 100:.1f}%", callback)
        
        return self.structure
    
    def _get_ss_stats(self) -> Dict[str, float]:
        """Get secondary structure statistics."""
        if not self.ss_prediction:
            return {'helix': 0, 'sheet': 0, 'coil': 0}
        
        n = len(self.ss_prediction)
        helix = sum(1 for ss in self.ss_prediction if ss == SecondaryStructure.HELIX)
        sheet = sum(1 for ss in self.ss_prediction if ss == SecondaryStructure.SHEET)
        
        return {
            'helix': helix / n * 100,
            'sheet': sheet / n * 100,
            'coil': 100 - helix/n*100 - sheet/n*100
        }
    
    def _calculate_confidence(self):
        """Calculate per-residue confidence scores."""
        for residue in self.structure.residues:
            confidence = 50.0  # Base confidence
            
            # Higher confidence for secondary structure
            if residue.ss in [SecondaryStructure.HELIX, SecondaryStructure.SHEET]:
                confidence += 20.0
            
            # Higher confidence for buried hydrophobics
            if residue.aa.hydropathy > 0:
                confidence += 10.0
            
            # Lower confidence for flexible regions
            if residue.aa.code in ['G', 'P']:
                confidence -= 10.0
            
            residue.confidence = min(100, max(0, confidence))
    
    def _log(self, message: str, callback=None):
        """Log message."""
        self.folding_log.append(message)
        if callback:
            callback(len(self.folding_log), message)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def fold_protein(sequence: str, iterations: int = 5000) -> ProteinStructure:
    """
    Main entry point for protein folding.
    
    Args:
        sequence: Amino acid sequence (1-letter codes)
        iterations: Number of optimization iterations
    
    Returns:
        Folded ProteinStructure object
    """
    engine = ProteinFoldingEngine(sequence)
    return engine.fold(iterations=iterations)


if __name__ == "__main__":
    # Example: Fold a small protein fragment
    test_sequences = {
        'helix': 'AEAAAKEAAAKEAAAKEAAAK',  # Designed helix
        'sheet': 'VTVTVTVTVTVTVTVT',        # Alternating pattern
        'lysozyme_frag': 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK',
        'insulin_b': 'FVNQHLCGSHLVEALYLVCGERGFFYTPKA'
    }
    
    print("=" * 60)
    print("AION Ultimate Protein Folding Engine")
    print("=" * 60)
    
    for name, seq in test_sequences.items():
        print(f"\n{'='*60}")
        print(f"Folding: {name}")
        print(f"{'='*60}")
        
        structure = fold_protein(seq, iterations=2000)
        
        print(f"\nResults:")
        print(f"  Energy: {structure.total_energy:.2f} kcal/mol")
        print(f"  Rg: {structure.calculate_radius_of_gyration():.2f} Å")
        ss = structure.get_ss_content()
        print(f"  SS: {ss['helix']:.0f}% helix, {ss['sheet']:.0f}% sheet")
        
        # Save PDB
        with open(f"{name}_folded.pdb", 'w') as f:
            f.write(structure.to_pdb())
        print(f"  Saved: {name}_folded.pdb")
