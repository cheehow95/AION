"""
AION Chemistry Domain Engine
============================

A chemistry engine for AION agents to explore molecular structures,
chemical reactions, and fundamental chemistry concepts.

Features:
- Periodic table with element properties
- Molecular formula parsing
- Molecular weight calculation
- Reaction balancing
- Basic thermochemistry
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# PERIODIC TABLE
# =============================================================================

@dataclass
class Element:
    """Chemical element with properties."""
    atomic_number: int
    symbol: str
    name: str
    atomic_mass: float  # g/mol
    group: int
    period: int
    category: str
    electronegativity: Optional[float] = None
    electron_config: str = ""
    
    def __repr__(self):
        return f"{self.symbol} ({self.name}, Z={self.atomic_number})"


# Periodic table data (first 36 elements + common heavy elements)
PERIODIC_TABLE = {
    "H": Element(1, "H", "Hydrogen", 1.008, 1, 1, "nonmetal", 2.20, "1s1"),
    "He": Element(2, "He", "Helium", 4.003, 18, 1, "noble gas", None, "1s2"),
    "Li": Element(3, "Li", "Lithium", 6.941, 1, 2, "alkali metal", 0.98, "[He]2s1"),
    "Be": Element(4, "Be", "Beryllium", 9.012, 2, 2, "alkaline earth", 1.57, "[He]2s2"),
    "B": Element(5, "B", "Boron", 10.81, 13, 2, "metalloid", 2.04, "[He]2s2 2p1"),
    "C": Element(6, "C", "Carbon", 12.011, 14, 2, "nonmetal", 2.55, "[He]2s2 2p2"),
    "N": Element(7, "N", "Nitrogen", 14.007, 15, 2, "nonmetal", 3.04, "[He]2s2 2p3"),
    "O": Element(8, "O", "Oxygen", 15.999, 16, 2, "nonmetal", 3.44, "[He]2s2 2p4"),
    "F": Element(9, "F", "Fluorine", 18.998, 17, 2, "halogen", 3.98, "[He]2s2 2p5"),
    "Ne": Element(10, "Ne", "Neon", 20.180, 18, 2, "noble gas", None, "[He]2s2 2p6"),
    "Na": Element(11, "Na", "Sodium", 22.990, 1, 3, "alkali metal", 0.93, "[Ne]3s1"),
    "Mg": Element(12, "Mg", "Magnesium", 24.305, 2, 3, "alkaline earth", 1.31, "[Ne]3s2"),
    "Al": Element(13, "Al", "Aluminum", 26.982, 13, 3, "metal", 1.61, "[Ne]3s2 3p1"),
    "Si": Element(14, "Si", "Silicon", 28.086, 14, 3, "metalloid", 1.90, "[Ne]3s2 3p2"),
    "P": Element(15, "P", "Phosphorus", 30.974, 15, 3, "nonmetal", 2.19, "[Ne]3s2 3p3"),
    "S": Element(16, "S", "Sulfur", 32.065, 16, 3, "nonmetal", 2.58, "[Ne]3s2 3p4"),
    "Cl": Element(17, "Cl", "Chlorine", 35.453, 17, 3, "halogen", 3.16, "[Ne]3s2 3p5"),
    "Ar": Element(18, "Ar", "Argon", 39.948, 18, 3, "noble gas", None, "[Ne]3s2 3p6"),
    "K": Element(19, "K", "Potassium", 39.098, 1, 4, "alkali metal", 0.82, "[Ar]4s1"),
    "Ca": Element(20, "Ca", "Calcium", 40.078, 2, 4, "alkaline earth", 1.00, "[Ar]4s2"),
    "Sc": Element(21, "Sc", "Scandium", 44.956, 3, 4, "transition metal", 1.36, "[Ar]3d1 4s2"),
    "Ti": Element(22, "Ti", "Titanium", 47.867, 4, 4, "transition metal", 1.54, "[Ar]3d2 4s2"),
    "V": Element(23, "V", "Vanadium", 50.942, 5, 4, "transition metal", 1.63, "[Ar]3d3 4s2"),
    "Cr": Element(24, "Cr", "Chromium", 51.996, 6, 4, "transition metal", 1.66, "[Ar]3d5 4s1"),
    "Mn": Element(25, "Mn", "Manganese", 54.938, 7, 4, "transition metal", 1.55, "[Ar]3d5 4s2"),
    "Fe": Element(26, "Fe", "Iron", 55.845, 8, 4, "transition metal", 1.83, "[Ar]3d6 4s2"),
    "Co": Element(27, "Co", "Cobalt", 58.933, 9, 4, "transition metal", 1.88, "[Ar]3d7 4s2"),
    "Ni": Element(28, "Ni", "Nickel", 58.693, 10, 4, "transition metal", 1.91, "[Ar]3d8 4s2"),
    "Cu": Element(29, "Cu", "Copper", 63.546, 11, 4, "transition metal", 1.90, "[Ar]3d10 4s1"),
    "Zn": Element(30, "Zn", "Zinc", 65.38, 12, 4, "transition metal", 1.65, "[Ar]3d10 4s2"),
    "Ga": Element(31, "Ga", "Gallium", 69.723, 13, 4, "metal", 1.81, "[Ar]3d10 4s2 4p1"),
    "Ge": Element(32, "Ge", "Germanium", 72.630, 14, 4, "metalloid", 2.01, "[Ar]3d10 4s2 4p2"),
    "As": Element(33, "As", "Arsenic", 74.922, 15, 4, "metalloid", 2.18, "[Ar]3d10 4s2 4p3"),
    "Se": Element(34, "Se", "Selenium", 78.971, 16, 4, "nonmetal", 2.55, "[Ar]3d10 4s2 4p4"),
    "Br": Element(35, "Br", "Bromine", 79.904, 17, 4, "halogen", 2.96, "[Ar]3d10 4s2 4p5"),
    "Kr": Element(36, "Kr", "Krypton", 83.798, 18, 4, "noble gas", 3.00, "[Ar]3d10 4s2 4p6"),
    # Common heavier elements
    "Ag": Element(47, "Ag", "Silver", 107.868, 11, 5, "transition metal", 1.93, "[Kr]4d10 5s1"),
    "I": Element(53, "I", "Iodine", 126.904, 17, 5, "halogen", 2.66, "[Kr]4d10 5s2 5p5"),
    "Au": Element(79, "Au", "Gold", 196.967, 11, 6, "transition metal", 2.54, "[Xe]4f14 5d10 6s1"),
    "Pb": Element(82, "Pb", "Lead", 207.2, 14, 6, "metal", 2.33, "[Xe]4f14 5d10 6s2 6p2"),
    "U": Element(92, "U", "Uranium", 238.029, 0, 7, "actinide", 1.38, "[Rn]5f3 6d1 7s2"),
}


def get_element(symbol: str) -> Optional[Element]:
    """Get element by symbol."""
    return PERIODIC_TABLE.get(symbol)


def get_element_by_number(atomic_number: int) -> Optional[Element]:
    """Get element by atomic number."""
    for elem in PERIODIC_TABLE.values():
        if elem.atomic_number == atomic_number:
            return elem
    return None


# =============================================================================
# MOLECULAR FORMULA PARSING
# =============================================================================

@dataclass
class MolecularFormula:
    """Parsed molecular formula with element counts."""
    formula: str
    composition: Dict[str, int]  # Element symbol -> count
    
    @property
    def molecular_weight(self) -> float:
        """Calculate molecular weight in g/mol."""
        weight = 0.0
        for symbol, count in self.composition.items():
            elem = get_element(symbol)
            if elem:
                weight += elem.atomic_mass * count
        return weight
    
    @property
    def empirical_formula(self) -> str:
        """Get empirical formula (lowest ratio)."""
        from math import gcd
        from functools import reduce
        
        counts = list(self.composition.values())
        if not counts:
            return ""
        
        common = reduce(gcd, counts)
        parts = []
        for symbol, count in sorted(self.composition.items()):
            ratio = count // common
            if ratio == 1:
                parts.append(symbol)
            else:
                parts.append(f"{symbol}{ratio}")
        
        return "".join(parts)
    
    def __repr__(self):
        return f"{self.formula} (MW: {self.molecular_weight:.2f} g/mol)"


def parse_formula(formula: str) -> MolecularFormula:
    """
    Parse a molecular formula string into element counts.
    Supports: H2O, C6H12O6, Ca(OH)2, Mg3(PO4)2
    """
    composition = {}
    
    # Handle parentheses recursively
    def parse_group(s: str, multiplier: int = 1) -> Dict[str, int]:
        result = {}
        i = 0
        
        while i < len(s):
            if s[i] == '(':
                # Find matching closing parenthesis
                depth = 1
                j = i + 1
                while j < len(s) and depth > 0:
                    if s[j] == '(':
                        depth += 1
                    elif s[j] == ')':
                        depth -= 1
                    j += 1
                
                # Get multiplier after parenthesis
                k = j
                while k < len(s) and s[k].isdigit():
                    k += 1
                
                group_mult = int(s[j:k]) if j < k else 1
                
                # Parse inside parentheses
                inner = parse_group(s[i+1:j-1], group_mult * multiplier)
                for elem, count in inner.items():
                    result[elem] = result.get(elem, 0) + count
                
                i = k
            elif s[i].isupper():
                # Element symbol
                j = i + 1
                while j < len(s) and s[j].islower():
                    j += 1
                
                symbol = s[i:j]
                
                # Get count
                k = j
                while k < len(s) and s[k].isdigit():
                    k += 1
                
                count = int(s[j:k]) if j < k else 1
                result[symbol] = result.get(symbol, 0) + count * multiplier
                
                i = k
            else:
                i += 1
        
        return result
    
    composition = parse_group(formula)
    return MolecularFormula(formula, composition)


# =============================================================================
# CHEMICAL REACTIONS
# =============================================================================

@dataclass
class ChemicalReaction:
    """Represents a chemical reaction."""
    reactants: List[Tuple[int, str]]  # (coefficient, formula)
    products: List[Tuple[int, str]]   # (coefficient, formula)
    
    def __repr__(self):
        lhs = " + ".join(
            f"{c if c > 1 else ''}{f}" for c, f in self.reactants
        )
        rhs = " + ".join(
            f"{c if c > 1 else ''}{f}" for c, f in self.products
        )
        return f"{lhs} → {rhs}"
    
    def is_balanced(self) -> bool:
        """Check if reaction is balanced."""
        def count_elements(compounds: List[Tuple[int, str]]) -> Dict[str, int]:
            total = {}
            for coef, formula in compounds:
                mol = parse_formula(formula)
                for elem, count in mol.composition.items():
                    total[elem] = total.get(elem, 0) + count * coef
            return total
        
        reactant_counts = count_elements(self.reactants)
        product_counts = count_elements(self.products)
        
        return reactant_counts == product_counts
    
    def enthalpy_change(self, formation_enthalpies: Dict[str, float]) -> Optional[float]:
        """
        Calculate enthalpy change using Hess's law:
        ΔH_rxn = Σ ΔH_f(products) - Σ ΔH_f(reactants)
        
        Args:
            formation_enthalpies: Dict of formula -> ΔH_f in kJ/mol
        """
        products_h = 0.0
        reactants_h = 0.0
        
        for coef, formula in self.products:
            if formula not in formation_enthalpies:
                return None
            products_h += coef * formation_enthalpies[formula]
        
        for coef, formula in self.reactants:
            if formula not in formation_enthalpies:
                return None
            reactants_h += coef * formation_enthalpies[formula]
        
        return products_h - reactants_h


# Standard formation enthalpies (kJ/mol) at 298 K
FORMATION_ENTHALPIES = {
    "H2": 0,
    "O2": 0,
    "N2": 0,
    "C": 0,  # graphite
    "H2O": -285.8,  # liquid
    "H2O(g)": -241.8,  # gas
    "CO2": -393.5,
    "CO": -110.5,
    "CH4": -74.8,
    "C2H6": -84.7,
    "C6H12O6": -1274,  # glucose
    "NH3": -45.9,
    "HCl": -92.3,
    "NaCl": -411.2,
    "CaCO3": -1206.9,
    "CaO": -635.1,
}


def parse_reaction(reaction_str: str) -> ChemicalReaction:
    """
    Parse a reaction string.
    Format: "2H2 + O2 -> 2H2O" or "2H2 + O2 = 2H2O"
    """
    # Split into reactants and products
    if "->" in reaction_str:
        lhs, rhs = reaction_str.split("->")
    elif "=" in reaction_str:
        lhs, rhs = reaction_str.split("=")
    elif "→" in reaction_str:
        lhs, rhs = reaction_str.split("→")
    else:
        raise ValueError("Reaction must contain '->', '=', or '→'")
    
    def parse_side(side: str) -> List[Tuple[int, str]]:
        compounds = []
        for term in side.split("+"):
            term = term.strip()
            if not term:
                continue
            
            # Extract coefficient
            match = re.match(r'^(\d+)?(.+)$', term)
            if match:
                coef = int(match.group(1)) if match.group(1) else 1
                formula = match.group(2).strip()
                compounds.append((coef, formula))
        
        return compounds
    
    return ChemicalReaction(
        reactants=parse_side(lhs),
        products=parse_side(rhs)
    )


# =============================================================================
# REACTION BALANCING (Simple Algebraic Method)
# =============================================================================

def balance_reaction(reactants: List[str], products: List[str], 
                     max_coef: int = 10) -> Optional[ChemicalReaction]:
    """
    Balance a chemical reaction using brute force search.
    Works for simple reactions. Returns None if no balance found.
    """
    from itertools import product as cartesian_product
    
    def get_composition(compounds: List[Tuple[int, str]]) -> Dict[str, int]:
        total = {}
        for coef, formula in compounds:
            mol = parse_formula(formula)
            for elem, count in mol.composition.items():
                total[elem] = total.get(elem, 0) + count * coef
        return total
    
    n_reactants = len(reactants)
    n_products = len(products)
    
    # Try all coefficient combinations
    for coefs in cartesian_product(range(1, max_coef + 1), repeat=n_reactants + n_products):
        react_coefs = coefs[:n_reactants]
        prod_coefs = coefs[n_reactants:]
        
        react_compounds = list(zip(react_coefs, reactants))
        prod_compounds = list(zip(prod_coefs, products))
        
        if get_composition(react_compounds) == get_composition(prod_compounds):
            # Reduce coefficients to lowest terms
            from math import gcd
            from functools import reduce
            
            all_coefs = react_coefs + prod_coefs
            common = reduce(gcd, all_coefs)
            
            react_compounds = [(c // common, f) for c, f in react_compounds]
            prod_compounds = [(c // common, f) for c, f in prod_compounds]
            
            return ChemicalReaction(react_compounds, prod_compounds)
    
    return None


# =============================================================================
# STOICHIOMETRY
# =============================================================================

class Stoichiometry:
    """Stoichiometric calculations."""
    
    @staticmethod
    def moles_to_grams(moles: float, formula: str) -> float:
        """Convert moles to grams."""
        mol = parse_formula(formula)
        return moles * mol.molecular_weight
    
    @staticmethod
    def grams_to_moles(grams: float, formula: str) -> float:
        """Convert grams to moles."""
        mol = parse_formula(formula)
        return grams / mol.molecular_weight
    
    @staticmethod
    def limiting_reagent(reaction: ChemicalReaction, 
                        amounts: Dict[str, float]) -> str:
        """
        Determine the limiting reagent.
        
        Args:
            reaction: Balanced chemical reaction
            amounts: Dict of formula -> moles available
        
        Returns:
            Formula of limiting reagent
        """
        min_ratio = float('inf')
        limiting = None
        
        for coef, formula in reaction.reactants:
            if formula in amounts:
                ratio = amounts[formula] / coef
                if ratio < min_ratio:
                    min_ratio = ratio
                    limiting = formula
        
        return limiting
    
    @staticmethod
    def theoretical_yield(reaction: ChemicalReaction,
                         amounts: Dict[str, float],
                         product: str) -> float:
        """
        Calculate theoretical yield of a product.
        
        Args:
            reaction: Balanced chemical reaction
            amounts: Dict of formula -> moles available
            product: Formula of desired product
        
        Returns:
            Moles of product that can be formed
        """
        limiting = Stoichiometry.limiting_reagent(reaction, amounts)
        if not limiting:
            return 0
        
        # Find coefficients
        react_coef = next(c for c, f in reaction.reactants if f == limiting)
        prod_coef = next((c for c, f in reaction.products if f == product), 0)
        
        if prod_coef == 0:
            return 0
        
        return amounts[limiting] * (prod_coef / react_coef)


# =============================================================================
# ACIDS AND BASES
# =============================================================================

class AcidBase:
    """Acid-base chemistry calculations."""
    
    @staticmethod
    def pH(H_concentration: float) -> float:
        """Calculate pH from hydrogen ion concentration."""
        import math
        if H_concentration <= 0:
            return 14.0
        return -math.log10(H_concentration)
    
    @staticmethod
    def pOH(OH_concentration: float) -> float:
        """Calculate pOH from hydroxide ion concentration."""
        import math
        if OH_concentration <= 0:
            return 14.0
        return -math.log10(OH_concentration)
    
    @staticmethod
    def H_from_pH(pH: float) -> float:
        """Calculate [H+] from pH."""
        return 10 ** (-pH)
    
    @staticmethod
    def OH_from_pOH(pOH: float) -> float:
        """Calculate [OH-] from pOH."""
        return 10 ** (-pOH)
    
    @staticmethod
    def pOH_from_pH(pH: float, Kw: float = 1e-14) -> float:
        """Convert pH to pOH (at 25°C, pH + pOH = 14)."""
        import math
        return -math.log10(Kw) - pH
    
    @staticmethod
    def Ka_from_pKa(pKa: float) -> float:
        """Calculate Ka from pKa."""
        return 10 ** (-pKa)
    
    @staticmethod
    def buffer_pH(Ka: float, acid_conc: float, base_conc: float) -> float:
        """
        Calculate pH of buffer using Henderson-Hasselbalch equation:
        pH = pKa + log([A-]/[HA])
        """
        import math
        pKa = -math.log10(Ka)
        return pKa + math.log10(base_conc / acid_conc)


# =============================================================================
# COMMON MOLECULES DATABASE
# =============================================================================

COMMON_MOLECULES = {
    "water": {"formula": "H2O", "name": "Water", "type": "solvent"},
    "glucose": {"formula": "C6H12O6", "name": "Glucose", "type": "carbohydrate"},
    "ethanol": {"formula": "C2H5OH", "name": "Ethanol", "type": "alcohol"},
    "methane": {"formula": "CH4", "name": "Methane", "type": "hydrocarbon"},
    "carbon_dioxide": {"formula": "CO2", "name": "Carbon Dioxide", "type": "gas"},
    "ammonia": {"formula": "NH3", "name": "Ammonia", "type": "base"},
    "sodium_chloride": {"formula": "NaCl", "name": "Sodium Chloride", "type": "salt"},
    "sulfuric_acid": {"formula": "H2SO4", "name": "Sulfuric Acid", "type": "acid"},
    "nitric_acid": {"formula": "HNO3", "name": "Nitric Acid", "type": "acid"},
    "acetic_acid": {"formula": "CH3COOH", "name": "Acetic Acid", "type": "acid"},
    "benzene": {"formula": "C6H6", "name": "Benzene", "type": "aromatic"},
    "caffeine": {"formula": "C8H10N4O2", "name": "Caffeine", "type": "alkaloid"},
    "aspirin": {"formula": "C9H8O4", "name": "Aspirin", "type": "drug"},
}


# =============================================================================
# CHEMISTRY ENGINE - Main Interface
# =============================================================================

class ChemistryEngine:
    """
    AION Chemistry Engine for molecular analysis and reactions.
    """
    
    def __init__(self):
        self.history: List[Dict] = []
    
    def get_element(self, symbol: str) -> Optional[Element]:
        """Get element information."""
        return get_element(symbol)
    
    def parse_formula(self, formula: str) -> Dict:
        """Parse a molecular formula."""
        mol = parse_formula(formula)
        return {
            "formula": mol.formula,
            "composition": mol.composition,
            "molecular_weight": mol.molecular_weight,
            "empirical_formula": mol.empirical_formula
        }
    
    def molecular_weight(self, formula: str) -> float:
        """Calculate molecular weight."""
        return parse_formula(formula).molecular_weight
    
    def balance_equation(self, reactants: List[str], products: List[str]) -> Optional[str]:
        """Balance a chemical equation."""
        reaction = balance_reaction(reactants, products)
        if reaction:
            return str(reaction)
        return None
    
    def check_balance(self, reaction_str: str) -> bool:
        """Check if a reaction is balanced."""
        reaction = parse_reaction(reaction_str)
        return reaction.is_balanced()
    
    def calculate_pH(self, H_concentration: float) -> float:
        """Calculate pH from [H+]."""
        return AcidBase.pH(H_concentration)
    
    def stoichiometry(self, reaction_str: str, amounts: Dict[str, float], 
                      target_product: str) -> Dict:
        """
        Perform stoichiometric calculations.
        
        Args:
            reaction_str: Balanced reaction string
            amounts: Dict of formula -> moles
            target_product: Product to calculate yield for
        """
        reaction = parse_reaction(reaction_str)
        
        limiting = Stoichiometry.limiting_reagent(reaction, amounts)
        theoretical = Stoichiometry.theoretical_yield(reaction, amounts, target_product)
        
        # Calculate molar mass
        product_mol = parse_formula(target_product)
        
        return {
            "limiting_reagent": limiting,
            "theoretical_yield_moles": theoretical,
            "theoretical_yield_grams": theoretical * product_mol.molecular_weight
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Chemistry Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🧪 AION CHEMISTRY ENGINE 🧪                                      ║
║                                                                           ║
║     Molecular analysis, reactions, and stoichiometry                      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = ChemistryEngine()
    
    # Periodic table
    print("📊 Periodic Table:")
    print("-" * 50)
    elements = ["H", "C", "N", "O", "Fe", "Au"]
    for sym in elements:
        elem = engine.get_element(sym)
        if elem:
            print(f"   {elem.symbol:2s} {elem.name:12s} Z={elem.atomic_number:2d} "
                  f"M={elem.atomic_mass:.3f} g/mol")
    
    # Molecular formulas
    print("\n🔬 Molecular Formula Parsing:")
    print("-" * 50)
    formulas = ["H2O", "C6H12O6", "Ca(OH)2", "Mg3(PO4)2", "C8H10N4O2"]
    for f in formulas:
        result = engine.parse_formula(f)
        print(f"   {f:12s} → MW = {result['molecular_weight']:.2f} g/mol")
        print(f"              Composition: {result['composition']}")
    
    # Reaction balancing
    print("\n⚗️ Reaction Balancing:")
    print("-" * 50)
    reactions = [
        (["H2", "O2"], ["H2O"]),
        (["CH4", "O2"], ["CO2", "H2O"]),
        (["Fe", "O2"], ["Fe2O3"]),
    ]
    for reactants, products in reactions:
        balanced = engine.balance_equation(reactants, products)
        unbalanced = " + ".join(reactants) + " → " + " + ".join(products)
        print(f"   {unbalanced}")
        print(f"   → {balanced}")
    
    # Check if balanced
    print("\n✓ Balance Checking:")
    print("-" * 50)
    test_rxns = [
        "2H2 + O2 -> 2H2O",
        "H2 + O2 -> H2O",
    ]
    for rxn in test_rxns:
        balanced = engine.check_balance(rxn)
        status = "✓ Balanced" if balanced else "✗ Not balanced"
        print(f"   {rxn}: {status}")
    
    # pH calculation
    print("\n🧫 pH Calculations:")
    print("-" * 50)
    concentrations = [0.1, 0.01, 0.001, 1e-7, 1e-10]
    for conc in concentrations:
        pH = engine.calculate_pH(conc)
        print(f"   [H+] = {conc:.1e} M → pH = {pH:.2f}")
    
    # Stoichiometry
    print("\n📐 Stoichiometry:")
    print("-" * 50)
    result = engine.stoichiometry(
        "2H2 + O2 -> 2H2O",
        {"H2": 4.0, "O2": 1.0},
        "H2O"
    )
    print(f"   Reaction: 2H2 + O2 → 2H2O")
    print(f"   Given: 4 mol H2, 1 mol O2")
    print(f"   Limiting reagent: {result['limiting_reagent']}")
    print(f"   Theoretical yield: {result['theoretical_yield_moles']:.1f} mol H2O")
    print(f"                      {result['theoretical_yield_grams']:.2f} g H2O")
    
    # Common molecules
    print("\n📚 Common Molecules:")
    print("-" * 50)
    for name, info in list(COMMON_MOLECULES.items())[:5]:
        mol = parse_formula(info["formula"])
        print(f"   {info['name']:20s} {info['formula']:12s} ({mol.molecular_weight:.1f} g/mol)")


if __name__ == "__main__":
    demo()
