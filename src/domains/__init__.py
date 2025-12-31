"""AION Domains Package"""
from .protein_folding import (
    ProteinFolder, ProteinStructure, AminoAcid,
    AA_PROPERTIES, analyze_sequence
)

__all__ = [
    'ProteinFolder', 'ProteinStructure', 'AminoAcid',
    'AA_PROPERTIES', 'analyze_sequence'
]
