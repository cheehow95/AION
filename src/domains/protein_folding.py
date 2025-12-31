"""
AION Protein Folding Module
Domain-specific implementation for protein structure prediction.

Protein folding is one of the grand challenges in computational biology.
AION's reasoning + self-correction capabilities can explore the conformational space.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

class AminoAcid(Enum):
    """20 standard amino acids"""
    ALA = "A"  # Alanine (hydrophobic)
    ARG = "R"  # Arginine (positive)
    ASN = "N"  # Asparagine (polar)
    ASP = "D"  # Aspartic acid (negative)
    CYS = "C"  # Cysteine (special)
    GLN = "Q"  # Glutamine (polar)
    GLU = "E"  # Glutamic acid (negative)
    GLY = "G"  # Glycine (small)
    HIS = "H"  # Histidine (polar)
    ILE = "I"  # Isoleucine (hydrophobic)
    LEU = "L"  # Leucine (hydrophobic)
    LYS = "K"  # Lysine (positive)
    MET = "M"  # Methionine (hydrophobic)
    PHE = "F"  # Phenylalanine (aromatic)
    PRO = "P"  # Proline (special)
    SER = "S"  # Serine (polar)
    THR = "T"  # Threonine (polar)
    TRP = "W"  # Tryptophan (aromatic)
    TYR = "Y"  # Tyrosine (aromatic)
    VAL = "V"  # Valine (hydrophobic)

@dataclass
class AminoAcidProperties:
    """Properties of amino acids relevant to folding"""
    code: str
    hydrophobicity: float  # Kyte-Doolittle scale
    charge: float  # At pH 7
    size: float  # Molecular weight / 100
    
# Simplified amino acid properties
AA_PROPERTIES = {
    "A": AminoAcidProperties("A", 1.8, 0, 0.89),
    "R": AminoAcidProperties("R", -4.5, +1, 1.74),
    "N": AminoAcidProperties("N", -3.5, 0, 1.32),
    "D": AminoAcidProperties("D", -3.5, -1, 1.33),
    "C": AminoAcidProperties("C", 2.5, 0, 1.21),
    "Q": AminoAcidProperties("Q", -3.5, 0, 1.46),
    "E": AminoAcidProperties("E", -3.5, -1, 1.47),
    "G": AminoAcidProperties("G", -0.4, 0, 0.75),
    "H": AminoAcidProperties("H", -3.2, 0, 1.55),
    "I": AminoAcidProperties("I", 4.5, 0, 1.31),
    "L": AminoAcidProperties("L", 3.8, 0, 1.31),
    "K": AminoAcidProperties("K", -3.9, +1, 1.46),
    "M": AminoAcidProperties("M", 1.9, 0, 1.49),
    "F": AminoAcidProperties("F", 2.8, 0, 1.65),
    "P": AminoAcidProperties("P", -1.6, 0, 1.15),
    "S": AminoAcidProperties("S", -0.8, 0, 1.05),
    "T": AminoAcidProperties("T", -0.7, 0, 1.19),
    "W": AminoAcidProperties("W", -0.9, 0, 2.04),
    "Y": AminoAcidProperties("Y", -1.3, 0, 1.81),
    "V": AminoAcidProperties("V", 4.2, 0, 1.17),
}

@dataclass
class ProteinStructure:
    """Simplified 2D lattice protein structure"""
    sequence: str
    coordinates: List[Tuple[int, int]]  # 2D positions on lattice
    energy: float = 0.0
    
    def calculate_energy(self) -> float:
        """
        Calculate simplified energy based on:
        1. Hydrophobic contacts (HP model)
        2. Electrostatic interactions
        """
        energy = 0.0
        
        for i in range(len(self.sequence)):
            for j in range(i + 2, len(self.sequence)):  # Non-adjacent
                # Check if residues are neighbors on lattice
                dist = abs(self.coordinates[i][0] - self.coordinates[j][0]) + \
                       abs(self.coordinates[i][1] - self.coordinates[j][1])
                
                if dist == 1:  # Adjacent on lattice
                    aa_i = AA_PROPERTIES[self.sequence[i]]
                    aa_j = AA_PROPERTIES[self.sequence[j]]
                    
                    # Hydrophobic interaction (favorable)
                    if aa_i.hydrophobicity > 0 and aa_j.hydrophobicity > 0:
                        energy -= aa_i.hydrophobicity * aa_j.hydrophobicity * 0.1
                    
                    # Electrostatic interaction
                    if aa_i.charge != 0 and aa_j.charge != 0:
                        if aa_i.charge * aa_j.charge < 0:  # Opposite charges
                            energy -= 2.0  # Attraction
                        else:  # Same charges
                            energy += 2.0  # Repulsion
        
        self.energy = energy
        return energy

class ProteinFolder:
    """
    Simplified protein folding algorithm using Monte Carlo simulation.
    This is a toy model for demonstration - real folding is far more complex!
    """
    
    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
        self.length = len(sequence)
        self.best_structure = None
        self.best_energy = float('inf')
        
    def generate_random_conformation(self) -> ProteinStructure:
        """Generate a random self-avoiding walk on 2D lattice"""
        coords = [(0, 0)]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left
        
        for i in range(1, self.length):
            # Try random directions until we find a valid one
            attempts = 0
            while attempts < 20:
                direction = directions[np.random.randint(0, 4)]
                new_pos = (coords[-1][0] + direction[0], coords[-1][1] + direction[1])
                
                if new_pos not in coords:  # Self-avoiding
                    coords.append(new_pos)
                    break
                attempts += 1
            
            if len(coords) <= i:  # Failed to place residue
                # Restart
                return self.generate_random_conformation()
        
        structure = ProteinStructure(self.sequence, coords)
        structure.calculate_energy()
        return structure
    
    def fold(self, iterations: int = 1000) -> ProteinStructure:
        """
        Simple Monte Carlo folding simulation.
        Returns the lowest energy structure found.
        """
        temperature = 2.0  # "Temperature" for MC acceptance
        
        for i in range(iterations):
            structure = self.generate_random_conformation()
            
            # Accept if better or probabilistically based on Boltzmann
            delta_e = structure.energy - self.best_energy
            
            if delta_e < 0 or (temperature > 0 and np.random.random() < np.exp(-delta_e / temperature)):
                if structure.energy < self.best_energy:
                    self.best_energy = structure.energy
                    self.best_structure = structure
            
            # Simulated annealing - cool down
            if i % 100 == 0:
                temperature *= 0.95
        
        return self.best_structure
    
    def visualize_structure(self, structure: ProteinStructure) -> str:
        """Create ASCII visualization of the structure"""
        if not structure:
            return "No structure"
        
        # Find bounds
        min_x = min(c[0] for c in structure.coordinates)
        max_x = max(c[0] for c in structure.coordinates)
        min_y = min(c[1] for c in structure.coordinates)
        max_y = max(c[1] for c in structure.coordinates)
        
        # Create grid
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Place residues
        for i, (x, y) in enumerate(structure.coordinates):
            grid[y - min_y][x - min_x] = structure.sequence[i]
        
        # Add connections
        viz = []
        for i in range(len(structure.coordinates) - 1):
            x1, y1 = structure.coordinates[i]
            x2, y2 = structure.coordinates[i + 1]
            
            # Draw line character
            if x1 == x2:  # Vertical
                char = '|'
            else:  # Horizontal
                char = '-'
            
            # Place connector (simplified - between residues)
            mid_x = (x1 + x2) // 2 - min_x
            mid_y = (y1 + y2) // 2 - min_y
            if 0 <= mid_y < height and 0 <= mid_x < width:
                if grid[mid_y][mid_x] == ' ':
                    grid[mid_y][mid_x] = char
        
        viz_str = '\n'.join(''.join(row) for row in grid)
        return viz_str


def analyze_sequence(sequence: str) -> Dict[str, Any]:
    """Analyze properties of a protein sequence"""
    analysis = {
        'length': len(sequence),
        'hydrophobic_count': 0,
        'charged_count': 0,
        'average_hydrophobicity': 0.0,
        'net_charge': 0.0,
    }
    
    total_hydrophobicity = 0
    for aa in sequence:
        if aa in AA_PROPERTIES:
            props = AA_PROPERTIES[aa]
            total_hydrophobicity += props.hydrophobicity
            
            if props.hydrophobicity > 1.0:
                analysis['hydrophobic_count'] += 1
            if props.charge != 0:
                analysis['charged_count'] += 1
                analysis['net_charge'] += props.charge
    
    if len(sequence) > 0:
        analysis['average_hydrophobicity'] = total_hydrophobicity / len(sequence)
    
    return analysis
