"""
AION Elements Database Engine
=============================

Complete periodic table of elements with:
- All 118 elements
- Physical and chemical properties
- Electron configurations
- Atomic physics calculations
- Spectroscopic data

Comprehensive element database for physics and chemistry.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class AtomicConstants:
    """Atomic physics constants."""
    c = 299792458           # Speed of light (m/s)
    h = 6.62607015e-34      # Planck constant (J·s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    e = 1.602176634e-19     # Elementary charge (C)
    m_e = 9.1093837015e-31  # Electron mass (kg)
    a_0 = 5.29177210903e-11 # Bohr radius (m)
    R_inf = 1.0973731568160e7  # Rydberg constant (m⁻¹)
    E_h = 4.3597447222071e-18  # Hartree energy (J)
    alpha = 7.2973525693e-3  # Fine structure constant
    mu_B = 9.2740100783e-24  # Bohr magneton (J/T)
    k_B = 1.380649e-23      # Boltzmann constant (J/K)
    N_A = 6.02214076e23     # Avogadro's number


# =============================================================================
# ELEMENT CATEGORIES
# =============================================================================

class ElementCategory(Enum):
    ALKALI_METAL = "alkali_metal"
    ALKALINE_EARTH = "alkaline_earth_metal"
    TRANSITION_METAL = "transition_metal"
    POST_TRANSITION = "post_transition_metal"
    METALLOID = "metalloid"
    NONMETAL = "nonmetal"
    HALOGEN = "halogen"
    NOBLE_GAS = "noble_gas"
    LANTHANIDE = "lanthanide"
    ACTINIDE = "actinide"


class ElementState(Enum):
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"


# =============================================================================
# ELEMENT DATA CLASS
# =============================================================================

@dataclass
class Element:
    """Chemical element with all properties."""
    atomic_number: int           # Z
    symbol: str
    name: str
    atomic_mass: float           # u (atomic mass units)
    category: ElementCategory
    period: int
    group: int
    electron_config: str         # e.g., "[He] 2s2 2p2"
    electronegativity: float = 0.0  # Pauling scale
    ionization_energy: float = 0.0  # kJ/mol (first)
    atomic_radius: float = 0.0   # pm
    density: float = 0.0         # g/cm³
    melting_point: float = 0.0   # K
    boiling_point: float = 0.0   # K
    state_at_stp: ElementState = ElementState.SOLID
    discovered: int = 0          # Year
    
    @property
    def Z(self) -> int:
        return self.atomic_number
    
    @property
    def mass_kg(self) -> float:
        """Mass of one atom in kg."""
        return self.atomic_mass * 1.66054e-27
    
    @property
    def electrons(self) -> int:
        """Number of electrons (neutral atom)."""
        return self.atomic_number
    
    @property
    def protons(self) -> int:
        """Number of protons."""
        return self.atomic_number
    
    @property
    def neutrons_typical(self) -> int:
        """Typical number of neutrons."""
        return round(self.atomic_mass) - self.atomic_number
    
    def bohr_radius_for_Z(self) -> float:
        """Bohr radius scaled by Z: a₀/Z."""
        return AtomicConstants.a_0 / self.Z
    
    def ground_state_energy_eV(self) -> float:
        """
        Approximate ground state energy for hydrogen-like ion.
        E = -13.6 × Z² eV
        """
        return -13.6 * self.Z ** 2
    
    def ionization_energy_eV(self) -> float:
        """First ionization energy in eV."""
        return self.ionization_energy / 96.485  # kJ/mol to eV (approx)


# =============================================================================
# PERIODIC TABLE DATABASE
# =============================================================================

class PeriodicTable:
    """Complete periodic table with all 118 elements."""
    
    # Complete element data (118 elements)
    _ELEMENTS: List[Tuple] = [
        # (Z, symbol, name, mass, category, period, group, config, EN, IE, radius, density, mp, bp, state)
        (1, "H", "Hydrogen", 1.008, ElementCategory.NONMETAL, 1, 1, "1s1", 2.20, 1312, 53, 0.00009, 14, 20, ElementState.GAS),
        (2, "He", "Helium", 4.003, ElementCategory.NOBLE_GAS, 1, 18, "1s2", 0, 2372, 31, 0.00018, 0.95, 4.2, ElementState.GAS),
        (3, "Li", "Lithium", 6.94, ElementCategory.ALKALI_METAL, 2, 1, "[He] 2s1", 0.98, 520, 167, 0.534, 454, 1615, ElementState.SOLID),
        (4, "Be", "Beryllium", 9.012, ElementCategory.ALKALINE_EARTH, 2, 2, "[He] 2s2", 1.57, 900, 112, 1.85, 1560, 2744, ElementState.SOLID),
        (5, "B", "Boron", 10.81, ElementCategory.METALLOID, 2, 13, "[He] 2s2 2p1", 2.04, 801, 87, 2.34, 2349, 4200, ElementState.SOLID),
        (6, "C", "Carbon", 12.011, ElementCategory.NONMETAL, 2, 14, "[He] 2s2 2p2", 2.55, 1086, 77, 2.27, 3823, 4300, ElementState.SOLID),
        (7, "N", "Nitrogen", 14.007, ElementCategory.NONMETAL, 2, 15, "[He] 2s2 2p3", 3.04, 1402, 75, 0.00125, 63, 77, ElementState.GAS),
        (8, "O", "Oxygen", 15.999, ElementCategory.NONMETAL, 2, 16, "[He] 2s2 2p4", 3.44, 1314, 73, 0.00143, 54, 90, ElementState.GAS),
        (9, "F", "Fluorine", 18.998, ElementCategory.HALOGEN, 2, 17, "[He] 2s2 2p5", 3.98, 1681, 71, 0.0017, 53, 85, ElementState.GAS),
        (10, "Ne", "Neon", 20.180, ElementCategory.NOBLE_GAS, 2, 18, "[He] 2s2 2p6", 0, 2081, 69, 0.0009, 25, 27, ElementState.GAS),
        (11, "Na", "Sodium", 22.990, ElementCategory.ALKALI_METAL, 3, 1, "[Ne] 3s1", 0.93, 496, 190, 0.97, 371, 1156, ElementState.SOLID),
        (12, "Mg", "Magnesium", 24.305, ElementCategory.ALKALINE_EARTH, 3, 2, "[Ne] 3s2", 1.31, 738, 160, 1.74, 923, 1363, ElementState.SOLID),
        (13, "Al", "Aluminum", 26.982, ElementCategory.POST_TRANSITION, 3, 13, "[Ne] 3s2 3p1", 1.61, 578, 143, 2.70, 933, 2792, ElementState.SOLID),
        (14, "Si", "Silicon", 28.086, ElementCategory.METALLOID, 3, 14, "[Ne] 3s2 3p2", 1.90, 786, 118, 2.33, 1687, 3538, ElementState.SOLID),
        (15, "P", "Phosphorus", 30.974, ElementCategory.NONMETAL, 3, 15, "[Ne] 3s2 3p3", 2.19, 1012, 110, 1.82, 317, 554, ElementState.SOLID),
        (16, "S", "Sulfur", 32.06, ElementCategory.NONMETAL, 3, 16, "[Ne] 3s2 3p4", 2.58, 1000, 103, 2.07, 388, 718, ElementState.SOLID),
        (17, "Cl", "Chlorine", 35.45, ElementCategory.HALOGEN, 3, 17, "[Ne] 3s2 3p5", 3.16, 1251, 99, 0.0032, 172, 239, ElementState.GAS),
        (18, "Ar", "Argon", 39.948, ElementCategory.NOBLE_GAS, 3, 18, "[Ne] 3s2 3p6", 0, 1521, 97, 0.0018, 84, 87, ElementState.GAS),
        (19, "K", "Potassium", 39.098, ElementCategory.ALKALI_METAL, 4, 1, "[Ar] 4s1", 0.82, 419, 243, 0.86, 337, 1032, ElementState.SOLID),
        (20, "Ca", "Calcium", 40.078, ElementCategory.ALKALINE_EARTH, 4, 2, "[Ar] 4s2", 1.00, 590, 197, 1.55, 1115, 1757, ElementState.SOLID),
        (21, "Sc", "Scandium", 44.956, ElementCategory.TRANSITION_METAL, 4, 3, "[Ar] 3d1 4s2", 1.36, 633, 162, 2.99, 1814, 3109, ElementState.SOLID),
        (22, "Ti", "Titanium", 47.867, ElementCategory.TRANSITION_METAL, 4, 4, "[Ar] 3d2 4s2", 1.54, 659, 147, 4.51, 1941, 3560, ElementState.SOLID),
        (23, "V", "Vanadium", 50.942, ElementCategory.TRANSITION_METAL, 4, 5, "[Ar] 3d3 4s2", 1.63, 651, 134, 6.11, 2183, 3680, ElementState.SOLID),
        (24, "Cr", "Chromium", 51.996, ElementCategory.TRANSITION_METAL, 4, 6, "[Ar] 3d5 4s1", 1.66, 653, 128, 7.15, 2180, 2944, ElementState.SOLID),
        (25, "Mn", "Manganese", 54.938, ElementCategory.TRANSITION_METAL, 4, 7, "[Ar] 3d5 4s2", 1.55, 717, 127, 7.44, 1519, 2334, ElementState.SOLID),
        (26, "Fe", "Iron", 55.845, ElementCategory.TRANSITION_METAL, 4, 8, "[Ar] 3d6 4s2", 1.83, 762, 126, 7.87, 1811, 3134, ElementState.SOLID),
        (27, "Co", "Cobalt", 58.933, ElementCategory.TRANSITION_METAL, 4, 9, "[Ar] 3d7 4s2", 1.88, 760, 125, 8.90, 1768, 3200, ElementState.SOLID),
        (28, "Ni", "Nickel", 58.693, ElementCategory.TRANSITION_METAL, 4, 10, "[Ar] 3d8 4s2", 1.91, 737, 124, 8.91, 1728, 3186, ElementState.SOLID),
        (29, "Cu", "Copper", 63.546, ElementCategory.TRANSITION_METAL, 4, 11, "[Ar] 3d10 4s1", 1.90, 745, 128, 8.96, 1358, 2835, ElementState.SOLID),
        (30, "Zn", "Zinc", 65.38, ElementCategory.TRANSITION_METAL, 4, 12, "[Ar] 3d10 4s2", 1.65, 906, 134, 7.13, 693, 1180, ElementState.SOLID),
        (31, "Ga", "Gallium", 69.723, ElementCategory.POST_TRANSITION, 4, 13, "[Ar] 3d10 4s2 4p1", 1.81, 579, 135, 5.91, 303, 2477, ElementState.SOLID),
        (32, "Ge", "Germanium", 72.63, ElementCategory.METALLOID, 4, 14, "[Ar] 3d10 4s2 4p2", 2.01, 762, 122, 5.32, 1211, 3106, ElementState.SOLID),
        (33, "As", "Arsenic", 74.922, ElementCategory.METALLOID, 4, 15, "[Ar] 3d10 4s2 4p3", 2.18, 947, 119, 5.78, 1090, 887, ElementState.SOLID),
        (34, "Se", "Selenium", 78.971, ElementCategory.NONMETAL, 4, 16, "[Ar] 3d10 4s2 4p4", 2.55, 941, 116, 4.81, 494, 958, ElementState.SOLID),
        (35, "Br", "Bromine", 79.904, ElementCategory.HALOGEN, 4, 17, "[Ar] 3d10 4s2 4p5", 2.96, 1140, 114, 3.12, 266, 332, ElementState.LIQUID),
        (36, "Kr", "Krypton", 83.798, ElementCategory.NOBLE_GAS, 4, 18, "[Ar] 3d10 4s2 4p6", 3.00, 1351, 112, 0.0037, 116, 120, ElementState.GAS),
        (37, "Rb", "Rubidium", 85.468, ElementCategory.ALKALI_METAL, 5, 1, "[Kr] 5s1", 0.82, 403, 265, 1.53, 312, 961, ElementState.SOLID),
        (38, "Sr", "Strontium", 87.62, ElementCategory.ALKALINE_EARTH, 5, 2, "[Kr] 5s2", 0.95, 550, 219, 2.64, 1050, 1655, ElementState.SOLID),
        (39, "Y", "Yttrium", 88.906, ElementCategory.TRANSITION_METAL, 5, 3, "[Kr] 4d1 5s2", 1.22, 600, 180, 4.47, 1799, 3609, ElementState.SOLID),
        (40, "Zr", "Zirconium", 91.224, ElementCategory.TRANSITION_METAL, 5, 4, "[Kr] 4d2 5s2", 1.33, 640, 160, 6.51, 2128, 4682, ElementState.SOLID),
        (41, "Nb", "Niobium", 92.906, ElementCategory.TRANSITION_METAL, 5, 5, "[Kr] 4d4 5s1", 1.60, 652, 146, 8.57, 2750, 5017, ElementState.SOLID),
        (42, "Mo", "Molybdenum", 95.95, ElementCategory.TRANSITION_METAL, 5, 6, "[Kr] 4d5 5s1", 2.16, 684, 139, 10.22, 2896, 4912, ElementState.SOLID),
        (43, "Tc", "Technetium", 98, ElementCategory.TRANSITION_METAL, 5, 7, "[Kr] 4d5 5s2", 1.90, 702, 136, 11.5, 2430, 4538, ElementState.SOLID),
        (44, "Ru", "Ruthenium", 101.07, ElementCategory.TRANSITION_METAL, 5, 8, "[Kr] 4d7 5s1", 2.20, 710, 134, 12.37, 2607, 4423, ElementState.SOLID),
        (45, "Rh", "Rhodium", 102.91, ElementCategory.TRANSITION_METAL, 5, 9, "[Kr] 4d8 5s1", 2.28, 720, 134, 12.41, 2237, 3968, ElementState.SOLID),
        (46, "Pd", "Palladium", 106.42, ElementCategory.TRANSITION_METAL, 5, 10, "[Kr] 4d10", 2.20, 804, 137, 12.02, 1828, 3236, ElementState.SOLID),
        (47, "Ag", "Silver", 107.87, ElementCategory.TRANSITION_METAL, 5, 11, "[Kr] 4d10 5s1", 1.93, 731, 144, 10.50, 1235, 2435, ElementState.SOLID),
        (48, "Cd", "Cadmium", 112.41, ElementCategory.TRANSITION_METAL, 5, 12, "[Kr] 4d10 5s2", 1.69, 868, 151, 8.69, 594, 1040, ElementState.SOLID),
        (49, "In", "Indium", 114.82, ElementCategory.POST_TRANSITION, 5, 13, "[Kr] 4d10 5s2 5p1", 1.78, 558, 167, 7.31, 430, 2345, ElementState.SOLID),
        (50, "Sn", "Tin", 118.71, ElementCategory.POST_TRANSITION, 5, 14, "[Kr] 4d10 5s2 5p2", 1.96, 709, 140, 7.31, 505, 2875, ElementState.SOLID),
        (51, "Sb", "Antimony", 121.76, ElementCategory.METALLOID, 5, 15, "[Kr] 4d10 5s2 5p3", 2.05, 834, 140, 6.68, 904, 1860, ElementState.SOLID),
        (52, "Te", "Tellurium", 127.60, ElementCategory.METALLOID, 5, 16, "[Kr] 4d10 5s2 5p4", 2.10, 869, 140, 6.24, 723, 1261, ElementState.SOLID),
        (53, "I", "Iodine", 126.90, ElementCategory.HALOGEN, 5, 17, "[Kr] 4d10 5s2 5p5", 2.66, 1008, 140, 4.93, 387, 457, ElementState.SOLID),
        (54, "Xe", "Xenon", 131.29, ElementCategory.NOBLE_GAS, 5, 18, "[Kr] 4d10 5s2 5p6", 2.60, 1170, 140, 0.0059, 161, 165, ElementState.GAS),
        (55, "Cs", "Cesium", 132.91, ElementCategory.ALKALI_METAL, 6, 1, "[Xe] 6s1", 0.79, 376, 298, 1.87, 302, 944, ElementState.SOLID),
        (56, "Ba", "Barium", 137.33, ElementCategory.ALKALINE_EARTH, 6, 2, "[Xe] 6s2", 0.89, 503, 253, 3.59, 1000, 2170, ElementState.SOLID),
        # Lanthanides (57-71)
        (57, "La", "Lanthanum", 138.91, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 5d1 6s2", 1.10, 538, 187, 6.15, 1193, 3737, ElementState.SOLID),
        (58, "Ce", "Cerium", 140.12, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f1 5d1 6s2", 1.12, 534, 182, 6.77, 1068, 3716, ElementState.SOLID),
        (59, "Pr", "Praseodymium", 140.91, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f3 6s2", 1.13, 527, 182, 6.77, 1208, 3793, ElementState.SOLID),
        (60, "Nd", "Neodymium", 144.24, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f4 6s2", 1.14, 533, 181, 7.01, 1297, 3347, ElementState.SOLID),
        (61, "Pm", "Promethium", 145, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f5 6s2", 1.13, 540, 183, 7.26, 1315, 3273, ElementState.SOLID),
        (62, "Sm", "Samarium", 150.36, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f6 6s2", 1.17, 545, 180, 7.52, 1345, 2067, ElementState.SOLID),
        (63, "Eu", "Europium", 151.96, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f7 6s2", 1.20, 547, 180, 5.24, 1099, 1802, ElementState.SOLID),
        (64, "Gd", "Gadolinium", 157.25, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f7 5d1 6s2", 1.20, 593, 180, 7.90, 1585, 3546, ElementState.SOLID),
        (65, "Tb", "Terbium", 158.93, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f9 6s2", 1.20, 566, 177, 8.23, 1629, 3503, ElementState.SOLID),
        (66, "Dy", "Dysprosium", 162.50, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f10 6s2", 1.22, 573, 178, 8.55, 1680, 2840, ElementState.SOLID),
        (67, "Ho", "Holmium", 164.93, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f11 6s2", 1.23, 581, 176, 8.80, 1734, 2993, ElementState.SOLID),
        (68, "Er", "Erbium", 167.26, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f12 6s2", 1.24, 589, 176, 9.07, 1802, 3141, ElementState.SOLID),
        (69, "Tm", "Thulium", 168.93, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f13 6s2", 1.25, 597, 176, 9.32, 1818, 2223, ElementState.SOLID),
        (70, "Yb", "Ytterbium", 173.05, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f14 6s2", 1.10, 603, 176, 6.90, 1097, 1469, ElementState.SOLID),
        (71, "Lu", "Lutetium", 174.97, ElementCategory.LANTHANIDE, 6, 3, "[Xe] 4f14 5d1 6s2", 1.27, 524, 174, 9.84, 1925, 3675, ElementState.SOLID),
        # Continue Period 6
        (72, "Hf", "Hafnium", 178.49, ElementCategory.TRANSITION_METAL, 6, 4, "[Xe] 4f14 5d2 6s2", 1.30, 659, 159, 13.31, 2506, 4876, ElementState.SOLID),
        (73, "Ta", "Tantalum", 180.95, ElementCategory.TRANSITION_METAL, 6, 5, "[Xe] 4f14 5d3 6s2", 1.50, 761, 146, 16.65, 3290, 5731, ElementState.SOLID),
        (74, "W", "Tungsten", 183.84, ElementCategory.TRANSITION_METAL, 6, 6, "[Xe] 4f14 5d4 6s2", 2.36, 770, 139, 19.25, 3695, 5828, ElementState.SOLID),
        (75, "Re", "Rhenium", 186.21, ElementCategory.TRANSITION_METAL, 6, 7, "[Xe] 4f14 5d5 6s2", 1.90, 760, 137, 21.02, 3459, 5869, ElementState.SOLID),
        (76, "Os", "Osmium", 190.23, ElementCategory.TRANSITION_METAL, 6, 8, "[Xe] 4f14 5d6 6s2", 2.20, 840, 135, 22.59, 3306, 5285, ElementState.SOLID),
        (77, "Ir", "Iridium", 192.22, ElementCategory.TRANSITION_METAL, 6, 9, "[Xe] 4f14 5d7 6s2", 2.20, 880, 136, 22.56, 2719, 4701, ElementState.SOLID),
        (78, "Pt", "Platinum", 195.08, ElementCategory.TRANSITION_METAL, 6, 10, "[Xe] 4f14 5d9 6s1", 2.28, 870, 139, 21.45, 2041, 4098, ElementState.SOLID),
        (79, "Au", "Gold", 196.97, ElementCategory.TRANSITION_METAL, 6, 11, "[Xe] 4f14 5d10 6s1", 2.54, 890, 144, 19.32, 1337, 3129, ElementState.SOLID),
        (80, "Hg", "Mercury", 200.59, ElementCategory.TRANSITION_METAL, 6, 12, "[Xe] 4f14 5d10 6s2", 2.00, 1007, 151, 13.55, 234, 630, ElementState.LIQUID),
        (81, "Tl", "Thallium", 204.38, ElementCategory.POST_TRANSITION, 6, 13, "[Xe] 4f14 5d10 6s2 6p1", 1.62, 589, 170, 11.85, 577, 1746, ElementState.SOLID),
        (82, "Pb", "Lead", 207.2, ElementCategory.POST_TRANSITION, 6, 14, "[Xe] 4f14 5d10 6s2 6p2", 1.87, 716, 180, 11.34, 601, 2022, ElementState.SOLID),
        (83, "Bi", "Bismuth", 208.98, ElementCategory.POST_TRANSITION, 6, 15, "[Xe] 4f14 5d10 6s2 6p3", 2.02, 703, 160, 9.78, 545, 1837, ElementState.SOLID),
        (84, "Po", "Polonium", 209, ElementCategory.METALLOID, 6, 16, "[Xe] 4f14 5d10 6s2 6p4", 2.00, 812, 190, 9.20, 527, 1235, ElementState.SOLID),
        (85, "At", "Astatine", 210, ElementCategory.HALOGEN, 6, 17, "[Xe] 4f14 5d10 6s2 6p5", 2.20, 920, 202, 7, 575, 610, ElementState.SOLID),
        (86, "Rn", "Radon", 222, ElementCategory.NOBLE_GAS, 6, 18, "[Xe] 4f14 5d10 6s2 6p6", 2.20, 1037, 220, 0.0097, 202, 211, ElementState.GAS),
        (87, "Fr", "Francium", 223, ElementCategory.ALKALI_METAL, 7, 1, "[Rn] 7s1", 0.70, 380, 348, 1.87, 300, 950, ElementState.SOLID),
        (88, "Ra", "Radium", 226, ElementCategory.ALKALINE_EARTH, 7, 2, "[Rn] 7s2", 0.90, 509, 283, 5.5, 973, 2010, ElementState.SOLID),
        # Actinides (89-103)
        (89, "Ac", "Actinium", 227, ElementCategory.ACTINIDE, 7, 3, "[Rn] 6d1 7s2", 1.10, 499, 195, 10.07, 1323, 3471, ElementState.SOLID),
        (90, "Th", "Thorium", 232.04, ElementCategory.ACTINIDE, 7, 3, "[Rn] 6d2 7s2", 1.30, 587, 180, 11.72, 2115, 5061, ElementState.SOLID),
        (91, "Pa", "Protactinium", 231.04, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f2 6d1 7s2", 1.50, 568, 163, 15.37, 1841, 4300, ElementState.SOLID),
        (92, "U", "Uranium", 238.03, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f3 6d1 7s2", 1.38, 598, 156, 19.05, 1405, 4404, ElementState.SOLID),
        (93, "Np", "Neptunium", 237, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f4 6d1 7s2", 1.36, 605, 155, 20.25, 917, 4273, ElementState.SOLID),
        (94, "Pu", "Plutonium", 244, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f6 7s2", 1.28, 585, 159, 19.84, 912, 3501, ElementState.SOLID),
        (95, "Am", "Americium", 243, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f7 7s2", 1.30, 578, 173, 13.67, 1449, 2880, ElementState.SOLID),
        (96, "Cm", "Curium", 247, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f7 6d1 7s2", 1.30, 581, 174, 13.51, 1613, 3383, ElementState.SOLID),
        (97, "Bk", "Berkelium", 247, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f9 7s2", 1.30, 601, 170, 14.78, 1259, 2900, ElementState.SOLID),
        (98, "Cf", "Californium", 251, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f10 7s2", 1.30, 608, 186, 15.1, 1173, 1743, ElementState.SOLID),
        (99, "Es", "Einsteinium", 252, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f11 7s2", 1.30, 619, 186, 8.84, 1133, 1269, ElementState.SOLID),
        (100, "Fm", "Fermium", 257, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f12 7s2", 1.30, 627, 175, 9.7, 1800, 0, ElementState.SOLID),
        (101, "Md", "Mendelevium", 258, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f13 7s2", 1.30, 635, 175, 10.3, 1100, 0, ElementState.SOLID),
        (102, "No", "Nobelium", 259, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f14 7s2", 1.30, 642, 175, 9.9, 1100, 0, ElementState.SOLID),
        (103, "Lr", "Lawrencium", 266, ElementCategory.ACTINIDE, 7, 3, "[Rn] 5f14 7s2 7p1", 1.30, 479, 175, 14.4, 1900, 0, ElementState.SOLID),
        # Period 7 continued
        (104, "Rf", "Rutherfordium", 267, ElementCategory.TRANSITION_METAL, 7, 4, "[Rn] 5f14 6d2 7s2", 0, 580, 157, 23.2, 2400, 5800, ElementState.SOLID),
        (105, "Db", "Dubnium", 268, ElementCategory.TRANSITION_METAL, 7, 5, "[Rn] 5f14 6d3 7s2", 0, 665, 149, 29.3, 0, 0, ElementState.SOLID),
        (106, "Sg", "Seaborgium", 269, ElementCategory.TRANSITION_METAL, 7, 6, "[Rn] 5f14 6d4 7s2", 0, 757, 143, 35, 0, 0, ElementState.SOLID),
        (107, "Bh", "Bohrium", 270, ElementCategory.TRANSITION_METAL, 7, 7, "[Rn] 5f14 6d5 7s2", 0, 740, 141, 37.1, 0, 0, ElementState.SOLID),
        (108, "Hs", "Hassium", 277, ElementCategory.TRANSITION_METAL, 7, 8, "[Rn] 5f14 6d6 7s2", 0, 730, 134, 40.7, 0, 0, ElementState.SOLID),
        (109, "Mt", "Meitnerium", 278, ElementCategory.TRANSITION_METAL, 7, 9, "[Rn] 5f14 6d7 7s2", 0, 800, 129, 37.4, 0, 0, ElementState.SOLID),
        (110, "Ds", "Darmstadtium", 281, ElementCategory.TRANSITION_METAL, 7, 10, "[Rn] 5f14 6d8 7s2", 0, 840, 128, 34.8, 0, 0, ElementState.SOLID),
        (111, "Rg", "Roentgenium", 282, ElementCategory.TRANSITION_METAL, 7, 11, "[Rn] 5f14 6d9 7s2", 0, 884, 121, 28.7, 0, 0, ElementState.SOLID),
        (112, "Cn", "Copernicium", 285, ElementCategory.TRANSITION_METAL, 7, 12, "[Rn] 5f14 6d10 7s2", 0, 940, 122, 23.7, 0, 357, ElementState.SOLID),
        (113, "Nh", "Nihonium", 286, ElementCategory.POST_TRANSITION, 7, 13, "[Rn] 5f14 6d10 7s2 7p1", 0, 707, 175, 16, 700, 1400, ElementState.SOLID),
        (114, "Fl", "Flerovium", 289, ElementCategory.POST_TRANSITION, 7, 14, "[Rn] 5f14 6d10 7s2 7p2", 0, 832, 175, 14, 340, 420, ElementState.SOLID),
        (115, "Mc", "Moscovium", 290, ElementCategory.POST_TRANSITION, 7, 15, "[Rn] 5f14 6d10 7s2 7p3", 0, 538, 175, 13.5, 670, 1400, ElementState.SOLID),
        (116, "Lv", "Livermorium", 293, ElementCategory.POST_TRANSITION, 7, 16, "[Rn] 5f14 6d10 7s2 7p4", 0, 664, 175, 12.9, 709, 1085, ElementState.SOLID),
        (117, "Ts", "Tennessine", 294, ElementCategory.HALOGEN, 7, 17, "[Rn] 5f14 6d10 7s2 7p5", 0, 743, 175, 7.2, 700, 883, ElementState.SOLID),
        (118, "Og", "Oganesson", 294, ElementCategory.NOBLE_GAS, 7, 18, "[Rn] 5f14 6d10 7s2 7p6", 0, 839, 175, 5, 258, 450, ElementState.SOLID),
    ]
    
    _elements_by_symbol: Dict[str, Element] = {}
    _elements_by_Z: Dict[int, Element] = {}
    
    @classmethod
    def _init_elements(cls):
        """Initialize element database."""
        if cls._elements_by_symbol:
            return
        
        for data in cls._ELEMENTS:
            Z, sym, name, mass, cat, period, group, config, en, ie, rad, dens, mp, bp, state = data
            elem = Element(
                atomic_number=Z, symbol=sym, name=name, atomic_mass=mass,
                category=cat, period=period, group=group, electron_config=config,
                electronegativity=en, ionization_energy=ie, atomic_radius=rad,
                density=dens, melting_point=mp, boiling_point=bp, state_at_stp=state
            )
            cls._elements_by_symbol[sym] = elem
            cls._elements_by_Z[Z] = elem
    
    @classmethod
    def get(cls, identifier) -> Optional[Element]:
        """Get element by symbol (str) or atomic number (int)."""
        cls._init_elements()
        if isinstance(identifier, int):
            return cls._elements_by_Z.get(identifier)
        elif isinstance(identifier, str):
            return cls._elements_by_symbol.get(identifier)
        return None
    
    @classmethod
    def all_elements(cls) -> List[Element]:
        """Get all elements."""
        cls._init_elements()
        return list(cls._elements_by_Z.values())
    
    @classmethod
    def by_category(cls, category: ElementCategory) -> List[Element]:
        """Get elements by category."""
        cls._init_elements()
        return [e for e in cls._elements_by_Z.values() if e.category == category]
    
    @classmethod
    def by_period(cls, period: int) -> List[Element]:
        """Get elements in a period."""
        cls._init_elements()
        return sorted([e for e in cls._elements_by_Z.values() if e.period == period], 
                     key=lambda x: x.group)
    
    @classmethod
    def by_group(cls, group: int) -> List[Element]:
        """Get elements in a group."""
        cls._init_elements()
        return sorted([e for e in cls._elements_by_Z.values() if e.group == group],
                     key=lambda x: x.period)


# =============================================================================
# ELEMENTS ENGINE - MAIN INTERFACE
# =============================================================================

class ElementsEngine:
    """AION Elements Database Engine."""
    
    def __init__(self):
        self.table = PeriodicTable
    
    def element(self, identifier) -> Optional[Dict]:
        """Get element information."""
        elem = self.table.get(identifier)
        if not elem:
            return None
        
        return {
            'Z': elem.Z,
            'symbol': elem.symbol,
            'name': elem.name,
            'atomic_mass': elem.atomic_mass,
            'category': elem.category.value,
            'period': elem.period,
            'group': elem.group,
            'electron_config': elem.electron_config,
            'electronegativity': elem.electronegativity,
            'ionization_energy_kJ_mol': elem.ionization_energy,
            'atomic_radius_pm': elem.atomic_radius,
            'density_g_cm3': elem.density,
            'melting_point_K': elem.melting_point,
            'boiling_point_K': elem.boiling_point,
            'state_at_STP': elem.state_at_stp.value,
            'protons': elem.protons,
            'neutrons': elem.neutrons_typical,
            'electrons': elem.electrons
        }
    
    def search(self, name: str) -> List[Dict]:
        """Search elements by partial name."""
        results = []
        for elem in self.table.all_elements():
            if name.lower() in elem.name.lower():
                results.append({'Z': elem.Z, 'symbol': elem.symbol, 'name': elem.name})
        return results
    
    def compare(self, symbols: List[str]) -> Dict:
        """Compare multiple elements."""
        elements = [self.table.get(s) for s in symbols if self.table.get(s)]
        
        return {
            'elements': [e.symbol for e in elements],
            'masses': {e.symbol: e.atomic_mass for e in elements},
            'electronegativity': {e.symbol: e.electronegativity for e in elements},
            'ionization_energy': {e.symbol: e.ionization_energy for e in elements}
        }
    
    def category_summary(self) -> Dict:
        """Summarize elements by category."""
        summary = {}
        for cat in ElementCategory:
            elements = self.table.by_category(cat)
            summary[cat.value] = {
                'count': len(elements),
                'examples': [e.symbol for e in elements[:5]]
            }
        return summary
    
    def period_summary(self, period: int) -> List[Dict]:
        """Get summary of a period."""
        elements = self.table.by_period(period)
        return [{'Z': e.Z, 'symbol': e.symbol, 'name': e.name, 'group': e.group} 
                for e in elements]


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Elements Database Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🧪 AION ELEMENTS DATABASE ENGINE 🧪                              ║
║                                                                           ║
║     All 118 Elements with Physical Properties                            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = ElementsEngine()
    
    # Sample elements
    print("🔬 Sample Elements:")
    print("-" * 50)
    for sym in ['H', 'C', 'Fe', 'Au', 'U']:
        elem = engine.element(sym)
        if elem:
            print(f"   {elem['symbol']:3} {elem['name']:12} Z={elem['Z']:3} "
                  f"mass={elem['atomic_mass']:.3f} u")
    
    # Noble gases
    print("\n💨 Noble Gases:")
    print("-" * 50)
    for elem in PeriodicTable.by_group(18):
        print(f"   {elem.symbol:3} {elem.name:12} bp={elem.boiling_point:.0f} K")
    
    # Alkali metals
    print("\n🔥 Alkali Metals:")
    print("-" * 50)
    for elem in PeriodicTable.by_category(ElementCategory.ALKALI_METAL):
        print(f"   {elem.symbol:3} {elem.name:12} IE={elem.ionization_energy:.0f} kJ/mol")
    
    # Heaviest elements
    print("\n⚗️ Heaviest Elements:")
    print("-" * 50)
    all_elems = sorted(PeriodicTable.all_elements(), key=lambda e: e.atomic_mass, reverse=True)
    for elem in all_elems[:5]:
        print(f"   {elem.symbol:3} {elem.name:15} Z={elem.Z:3} mass={elem.atomic_mass:.0f} u")
    
    # Category summary
    print("\n📊 Elements by Category:")
    print("-" * 50)
    summary = engine.category_summary()
    for cat, info in list(summary.items())[:6]:
        print(f"   {cat:25} {info['count']:3} elements: {', '.join(info['examples'])}")


if __name__ == "__main__":
    demo()
