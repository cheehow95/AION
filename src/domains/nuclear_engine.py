"""
AION Nuclear Physics Engine
===========================

Comprehensive nuclear physics implementation covering:
- Nuclear structure and binding energy
- Radioactive decay (alpha, beta, gamma)
- Nuclear reactions and cross-sections
- Fission and fusion
- Nuclear models (liquid drop, shell)

Based on nuclear physics principles.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class NuclearConstants:
    """Nuclear physics constants."""
    c = 299792458           # Speed of light (m/s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    e = 1.602176634e-19     # Elementary charge (C)
    
    # Masses
    m_p = 1.007276466621    # Proton mass (u)
    m_n = 1.008664916       # Neutron mass (u)
    m_e = 0.000548579909    # Electron mass (u)
    m_u = 931.494           # Atomic mass unit (MeV/c²)
    
    # Masses in kg
    m_p_kg = 1.67262192369e-27
    m_n_kg = 1.67492749804e-27
    
    # Nuclear
    r_0 = 1.2e-15           # Nuclear radius constant (m)
    a_v = 15.56             # Volume term (MeV)
    a_s = 17.23             # Surface term (MeV)
    a_c = 0.697             # Coulomb term (MeV)
    a_a = 23.285            # Asymmetry term (MeV)
    a_p = 12.0              # Pairing term (MeV)
    
    # Decay constants
    ln2 = 0.693147180559945


# =============================================================================
# DECAY TYPES
# =============================================================================

class DecayType(Enum):
    ALPHA = "alpha"           # Emits He-4
    BETA_MINUS = "beta_minus" # n → p + e⁻ + ν̄ₑ
    BETA_PLUS = "beta_plus"   # p → n + e⁺ + νₑ
    GAMMA = "gamma"           # Emits high-energy photon
    ELECTRON_CAPTURE = "ec"    # p + e⁻ → n + νₑ
    NEUTRON_EMISSION = "n"
    PROTON_EMISSION = "p"
    SPONTANEOUS_FISSION = "sf"


# =============================================================================
# NUCLEUS
# =============================================================================

@dataclass
class Nucleus:
    """Represents an atomic nucleus."""
    Z: int          # Atomic number (protons)
    A: int          # Mass number (protons + neutrons)
    symbol: str = ""
    name: str = ""
    mass_u: float = 0.0  # Atomic mass in u (if known)
    
    def __post_init__(self):
        if not self.symbol:
            self.symbol = f"X-{self.A}"
    
    @property
    def N(self) -> int:
        """Neutron number."""
        return self.A - self.Z
    
    @property
    def radius(self) -> float:
        """Nuclear radius: R = r₀ × A^(1/3)."""
        return NuclearConstants.r_0 * self.A ** (1/3)
    
    @property
    def radius_fm(self) -> float:
        """Radius in femtometers."""
        return self.radius * 1e15
    
    @property
    def volume(self) -> float:
        """Nuclear volume (m³)."""
        return (4/3) * math.pi * self.radius ** 3
    
    @property
    def density(self) -> float:
        """Nuclear density (kg/m³). Nearly constant ~2×10¹⁷ kg/m³."""
        mass = self.A * NuclearConstants.m_p_kg
        return mass / self.volume
    
    def binding_energy_MeV(self) -> float:
        """
        Semi-empirical mass formula (Bethe-Weizsäcker).
        
        B(Z,A) = aᵥA - aₛA^(2/3) - aᶜZ(Z-1)/A^(1/3) - aₐ(A-2Z)²/A + δ
        """
        A, Z, N = self.A, self.Z, self.N
        
        # Volume term (attractive)
        E_v = NuclearConstants.a_v * A
        
        # Surface term (reduces binding)
        E_s = NuclearConstants.a_s * A ** (2/3)
        
        # Coulomb term (reduces binding)
        E_c = NuclearConstants.a_c * Z * (Z - 1) / A ** (1/3)
        
        # Asymmetry term (reduces binding for N ≠ Z)
        E_a = NuclearConstants.a_a * (A - 2*Z) ** 2 / A
        
        # Pairing term
        if A % 2 == 0:
            if Z % 2 == 0:
                E_p = NuclearConstants.a_p / A ** 0.5  # even-even
            else:
                E_p = -NuclearConstants.a_p / A ** 0.5  # odd-odd
        else:
            E_p = 0  # odd A
        
        return E_v - E_s - E_c - E_a + E_p
    
    @property
    def binding_energy_per_nucleon(self) -> float:
        """B/A in MeV."""
        return self.binding_energy_MeV() / self.A
    
    def mass_excess_MeV(self) -> float:
        """
        Mass excess: Δ = (M - A)c² in MeV.
        Where M is actual mass in u.
        """
        if self.mass_u > 0:
            return (self.mass_u - self.A) * NuclearConstants.m_u
        # Estimate from semi-empirical formula
        return self.Z * NuclearConstants.m_p * NuclearConstants.m_u + \
               self.N * NuclearConstants.m_n * NuclearConstants.m_u - \
               self.A * NuclearConstants.m_u - self.binding_energy_MeV()
    
    def is_stable(self) -> bool:
        """Check if nucleus is likely stable (approximately)."""
        if self.Z > 82:  # Z > Pb
            return False
        
        # Stability valley approximation
        ideal_Z = self.A / (2 + 0.015 * self.A ** (2/3))
        return abs(self.Z - ideal_Z) < 2


# =============================================================================
# RADIOACTIVE DECAY
# =============================================================================

class RadioactiveDecay:
    """Radioactive decay calculations."""
    
    @staticmethod
    def decay_constant(half_life: float) -> float:
        """
        Calculate decay constant λ from half-life.
        λ = ln(2) / t₁/₂
        """
        return NuclearConstants.ln2 / half_life
    
    @staticmethod
    def half_life(decay_constant: float) -> float:
        """Calculate half-life from decay constant."""
        return NuclearConstants.ln2 / decay_constant
    
    @staticmethod
    def mean_lifetime(decay_constant: float) -> float:
        """Mean lifetime τ = 1/λ."""
        return 1 / decay_constant
    
    @staticmethod
    def remaining_nuclei(N_0: float, decay_constant: float, t: float) -> float:
        """
        Number of nuclei remaining at time t.
        N(t) = N₀ e^(-λt)
        """
        return N_0 * math.exp(-decay_constant * t)
    
    @staticmethod
    def remaining_fraction(half_life: float, t: float) -> float:
        """Fraction remaining: N/N₀ = (1/2)^(t/t₁/₂)."""
        return 0.5 ** (t / half_life)
    
    @staticmethod
    def activity(N: float, decay_constant: float) -> float:
        """Activity A = λN (decays per second)."""
        return decay_constant * N
    
    @staticmethod
    def activity_Bq(N: float, half_life: float) -> float:
        """Activity in Becquerel (decays/s)."""
        lam = RadioactiveDecay.decay_constant(half_life)
        return lam * N
    
    @staticmethod
    def time_for_fraction(fraction: float, half_life: float) -> float:
        """Time for N/N₀ to reach given fraction."""
        return -half_life * math.log2(fraction)
    
    @staticmethod
    def alpha_decay_products(parent: Nucleus) -> Tuple[Nucleus, Nucleus]:
        """
        Alpha decay: X → Y + He-4.
        """
        daughter = Nucleus(parent.Z - 2, parent.A - 4)
        alpha = Nucleus(2, 4, "He", "Helium-4")
        return (daughter, alpha)
    
    @staticmethod
    def alpha_decay_energy(parent_mass_u: float, daughter_mass_u: float, 
                          alpha_mass_u: float = 4.002603) -> float:
        """
        Q-value for alpha decay.
        Q = (M_parent - M_daughter - M_α)c²
        """
        mass_defect = parent_mass_u - daughter_mass_u - alpha_mass_u
        return mass_defect * NuclearConstants.m_u  # MeV
    
    @staticmethod
    def beta_minus_products(parent: Nucleus) -> Tuple[Nucleus, str, str]:
        """
        Beta-minus decay: n → p + e⁻ + ν̄ₑ.
        """
        daughter = Nucleus(parent.Z + 1, parent.A)
        return (daughter, "e⁻", "ν̄ₑ")
    
    @staticmethod
    def beta_plus_products(parent: Nucleus) -> Tuple[Nucleus, str, str]:
        """
        Beta-plus decay: p → n + e⁺ + νₑ.
        """
        daughter = Nucleus(parent.Z - 1, parent.A)
        return (daughter, "e⁺", "νₑ")


# =============================================================================
# NUCLEAR REACTIONS
# =============================================================================

class NuclearReactions:
    """Nuclear reaction calculations."""
    
    @staticmethod
    def q_value(reactant_masses: List[float], product_masses: List[float]) -> float:
        """
        Calculate Q-value of reaction.
        Q = (Σm_reactants - Σm_products)c²
        
        Masses in atomic mass units, returns MeV.
        """
        delta_m = sum(reactant_masses) - sum(product_masses)
        return delta_m * NuclearConstants.m_u
    
    @staticmethod
    def threshold_energy(q_value: float, m_projectile: float, 
                         m_target: float) -> float:
        """
        Threshold kinetic energy for endothermic reaction (Q < 0).
        E_th = -Q(1 + m_a/m_b) where a is projectile, b is target.
        """
        if q_value >= 0:
            return 0.0
        return -q_value * (1 + m_projectile / m_target)
    
    @staticmethod
    def coulomb_barrier(Z1: int, Z2: int, r: float) -> float:
        """
        Coulomb barrier between two nuclei.
        V = k Z₁Z₂e² / r (in MeV)
        """
        k = 8.9875517923e9
        e = 1.602176634e-19
        V_J = k * Z1 * Z2 * e ** 2 / r
        return V_J / (1.602176634e-13)  # Convert to MeV
    
    @staticmethod
    def fusion_barrier(Z1: int, A1: int, Z2: int, A2: int) -> float:
        """
        Coulomb barrier for fusion of two nuclei.
        Uses touching distance r = r₀(A₁^(1/3) + A₂^(1/3)).
        """
        r0 = NuclearConstants.r_0
        r = r0 * (A1 ** (1/3) + A2 ** (1/3))
        return NuclearReactions.coulomb_barrier(Z1, Z2, r)


# =============================================================================
# FISSION AND FUSION
# =============================================================================

class FissionFusion:
    """Fission and fusion energy calculations."""
    
    @staticmethod
    def fission_energy_U235() -> Dict:
        """
        Energy released in U-235 fission.
        ²³⁵U + n → fission products + 2-3n + ~200 MeV
        """
        return {
            'reaction': '²³⁵U + n → fission fragments + 2.4n + energy',
            'total_energy_MeV': 200,
            'kinetic_fragments_MeV': 165,
            'neutron_energy_MeV': 5,
            'prompt_gamma_MeV': 7,
            'neutrino_MeV': 12,
            'beta_decay_MeV': 8,
            'delayed_gamma_MeV': 3
        }
    
    @staticmethod
    def fission_energy_per_mass() -> float:
        """Energy per kg of U-235 (in Joules)."""
        # 200 MeV per fission, Avogadro nuclei per 235g
        E_per_fission = 200 * 1.602176634e-13  # J
        N_per_kg = 6.022e23 / 0.235  # nuclei per kg
        return E_per_fission * N_per_kg
    
    @staticmethod
    def fusion_DT() -> Dict:
        """
        D-T fusion reaction.
        ²H + ³H → ⁴He + n + 17.6 MeV
        """
        return {
            'reaction': '²H + ³H → ⁴He + n',
            'energy_MeV': 17.6,
            'neutron_energy_MeV': 14.1,
            'alpha_energy_MeV': 3.5,
            'ignition_temp_keV': 10,
            'ignition_temp_K': 1.16e8
        }
    
    @staticmethod
    def fusion_DD() -> Dict:
        """
        D-D fusion reactions.
        """
        return {
            'reaction_1': '²H + ²H → ³He + n + 3.27 MeV',
            'reaction_2': '²H + ²H → ³H + p + 4.03 MeV',
            'average_energy_MeV': 3.65,
            'branching_ratio': '50/50'
        }
    
    @staticmethod
    def pp_chain() -> Dict:
        """
        Proton-proton chain (stellar fusion).
        4¹H → ⁴He + 2e⁺ + 2νₑ + 26.7 MeV
        """
        return {
            'net_reaction': '4¹H → ⁴He + 2e⁺ + 2νₑ',
            'energy_MeV': 26.7,
            'neutrino_energy_MeV': 0.5,
            'gamma_energy_MeV': 26.2,
            'steps': [
                '¹H + ¹H → ²H + e⁺ + νₑ (1.44 MeV)',
                '²H + ¹H → ³He + γ (5.49 MeV)',
                '³He + ³He → ⁴He + 2¹H (12.86 MeV)'
            ]
        }
    
    @staticmethod
    def cno_cycle() -> Dict:
        """CNO cycle for stellar fusion."""
        return {
            'net_reaction': '4¹H → ⁴He + 2e⁺ + 2νₑ + 3γ',
            'energy_MeV': 26.7,
            'catalysts': ['¹²C', '¹³N', '¹³C', '¹⁴N', '¹⁵O', '¹⁵N'],
            'dominant_above': '17 million K',
            'neutrino_energy_MeV': 1.7
        }
    
    @staticmethod
    def energy_per_mass_comparison() -> Dict:
        """Compare energy density of nuclear vs chemical."""
        return {
            'fission_U235_J_per_kg': 8.2e13,
            'fusion_DT_J_per_kg': 3.4e14,
            'chemical_TNT_J_per_kg': 4.6e6,
            'ratio_fission_to_chemical': 1.8e7,
            'ratio_fusion_to_chemical': 7.4e7
        }


# =============================================================================
# NUCLEAR ENGINE - MAIN INTERFACE
# =============================================================================

class NuclearEngine:
    """AION Nuclear Physics Engine."""
    
    def __init__(self):
        self.decay = RadioactiveDecay()
        self.reactions = NuclearReactions()
        self.fission_fusion = FissionFusion()
    
    def nucleus(self, Z: int, A: int, symbol: str = "", name: str = "") -> Dict:
        """Analyze a nucleus."""
        nuc = Nucleus(Z, A, symbol, name)
        
        return {
            'Z': Z,
            'N': nuc.N,
            'A': A,
            'symbol': symbol or f"Z={Z}, A={A}",
            'radius_fm': nuc.radius_fm,
            'binding_energy_MeV': nuc.binding_energy_MeV(),
            'binding_per_nucleon_MeV': nuc.binding_energy_per_nucleon,
            'likely_stable': nuc.is_stable(),
            'density_kg_m3': nuc.density
        }
    
    def binding_energy_curve(self) -> List[Dict]:
        """Generate binding energy per nucleon curve."""
        curve = []
        elements = [
            (1, 1, 'H'), (2, 4, 'He'), (6, 12, 'C'), (8, 16, 'O'),
            (26, 56, 'Fe'), (28, 62, 'Ni'), (50, 120, 'Sn'),
            (82, 208, 'Pb'), (92, 238, 'U')
        ]
        
        for Z, A, sym in elements:
            nuc = Nucleus(Z, A, sym)
            curve.append({
                'element': sym,
                'A': A,
                'B_per_A': nuc.binding_energy_per_nucleon
            })
        
        return curve
    
    def radioactive_decay_analysis(self, half_life: float, initial_atoms: float = 1e24,
                                    time: float = None) -> Dict:
        """Analyze radioactive decay."""
        lam = RadioactiveDecay.decay_constant(half_life)
        time = time or half_life * 3
        
        remaining = RadioactiveDecay.remaining_nuclei(initial_atoms, lam, time)
        activity = RadioactiveDecay.activity(remaining, lam)
        
        return {
            'half_life': half_life,
            'decay_constant': lam,
            'mean_lifetime': RadioactiveDecay.mean_lifetime(lam),
            'initial_atoms': initial_atoms,
            'time': time,
            'remaining_atoms': remaining,
            'fraction_remaining': remaining / initial_atoms,
            'activity_Bq': activity
        }
    
    def alpha_decay(self, parent_Z: int, parent_A: int) -> Dict:
        """Analyze alpha decay."""
        parent = Nucleus(parent_Z, parent_A)
        daughter, alpha = RadioactiveDecay.alpha_decay_products(parent)
        
        # Estimate Q-value from binding energy difference
        B_parent = parent.binding_energy_MeV()
        B_daughter = daughter.binding_energy_MeV()
        B_alpha = Nucleus(2, 4).binding_energy_MeV()
        
        Q = B_daughter + B_alpha - B_parent
        
        return {
            'parent': f'Z={parent_Z}, A={parent_A}',
            'daughter': f'Z={daughter.Z}, A={daughter.A}',
            'alpha': 'He-4',
            'Q_value_MeV': Q,
            'alpha_KE_MeV': Q * (daughter.A) / (daughter.A + 4)
        }
    
    def fusion_energy(self, reaction: str = 'DT') -> Dict:
        """Get fusion reaction energy."""
        if reaction.upper() == 'DT':
            return self.fission_fusion.fusion_DT()
        elif reaction.upper() == 'DD':
            return self.fission_fusion.fusion_DD()
        elif reaction.upper() == 'PP':
            return self.fission_fusion.pp_chain()
        else:
            return {'error': f'Unknown reaction: {reaction}'}
    
    def fission_energy(self) -> Dict:
        """Get U-235 fission energy."""
        return self.fission_fusion.fission_energy_U235()
    
    def energy_comparison(self) -> Dict:
        """Compare nuclear vs chemical energy."""
        return self.fission_fusion.energy_per_mass_comparison()


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Nuclear Physics Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          ☢️ AION NUCLEAR PHYSICS ENGINE ☢️                                ║
║                                                                           ║
║     Binding Energy, Radioactive Decay, Fission, Fusion                   ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = NuclearEngine()
    
    # Binding energy curve
    print("⚡ Binding Energy per Nucleon:")
    print("-" * 50)
    for item in engine.binding_energy_curve():
        bar = '█' * int(item['B_per_A'])
        print(f"   {item['element']:3} (A={item['A']:3}): {item['B_per_A']:.2f} MeV {bar}")
    
    # Radioactive decay
    print("\n☢️ Radioactive Decay (C-14, t₁/₂ = 5730 years):")
    print("-" * 50)
    result = engine.radioactive_decay_analysis(5730 * 365.25 * 24 * 3600, 1e24, 
                                                 5730 * 365.25 * 24 * 3600)
    print(f"   Half-life: 5730 years")
    print(f"   After 1 half-life: {result['fraction_remaining']*100:.1f}% remaining")
    
    # Alpha decay
    print("\n🔴 Alpha Decay (U-238):")
    print("-" * 50)
    result = engine.alpha_decay(92, 238)
    print(f"   U-238 → Th-234 + α")
    print(f"   Q-value: ~{result['Q_value_MeV']:.1f} MeV")
    
    # Fusion
    print("\n☀️ D-T Fusion:")
    print("-" * 50)
    result = engine.fusion_energy('DT')
    print(f"   Reaction: {result['reaction']}")
    print(f"   Energy: {result['energy_MeV']} MeV")
    print(f"   Ignition: {result['ignition_temp_K']:.2e} K")
    
    # Fission
    print("\n💥 U-235 Fission:")
    print("-" * 50)
    result = engine.fission_energy()
    print(f"   Reaction: {result['reaction']}")
    print(f"   Total energy: {result['total_energy_MeV']} MeV")
    
    # Energy comparison
    print("\n📊 Nuclear vs Chemical Energy:")
    print("-" * 50)
    comp = engine.energy_comparison()
    print(f"   Fission: {comp['fission_U235_J_per_kg']:.1e} J/kg")
    print(f"   Fusion:  {comp['fusion_DT_J_per_kg']:.1e} J/kg")
    print(f"   TNT:     {comp['chemical_TNT_J_per_kg']:.1e} J/kg")
    print(f"   Fusion is {comp['ratio_fusion_to_chemical']:.0e}× more energy dense than TNT!")


if __name__ == "__main__":
    demo()
