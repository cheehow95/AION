"""
AION Particle Physics Engine
============================

Comprehensive particle physics implementation covering:
- Standard Model particles (quarks, leptons, bosons)
- Particle properties and classifications
- Conservation laws
- Cross-sections and decay rates
- Feynman diagram structures
- Hadrons and mesons

Based on the Standard Model of particle physics.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class ParticleConstants:
    """Particle physics constants."""
    c = 299792458           # Speed of light (m/s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    e = 1.602176634e-19     # Elementary charge (C)
    
    # Coupling constants
    alpha_EM = 1 / 137.036  # Fine structure constant (EM)
    alpha_s = 0.118         # Strong coupling at Z mass
    G_F = 1.1663787e-5      # Fermi constant (GeV^-2)
    
    # Masses (GeV/c²)
    m_e = 0.000511          # Electron
    m_mu = 0.10566          # Muon
    m_tau = 1.777           # Tau
    m_u = 0.00216           # Up quark
    m_d = 0.00467           # Down quark
    m_c = 1.27              # Charm quark
    m_s = 0.093             # Strange quark
    m_t = 172.76            # Top quark
    m_b = 4.18              # Bottom quark
    m_W = 80.379            # W boson
    m_Z = 91.1876           # Z boson
    m_H = 125.25            # Higgs boson
    
    # Conversion
    GeV_to_J = 1.602176634e-10  # GeV to Joules
    eV_to_J = 1.602176634e-19


# =============================================================================
# PARTICLE TYPES
# =============================================================================

class ParticleType(Enum):
    QUARK = "quark"
    LEPTON = "lepton"
    GAUGE_BOSON = "gauge_boson"
    SCALAR_BOSON = "scalar_boson"
    HADRON = "hadron"
    MESON = "meson"
    BARYON = "baryon"


class InteractionType(Enum):
    STRONG = "strong"
    ELECTROMAGNETIC = "electromagnetic"
    WEAK = "weak"
    GRAVITATIONAL = "gravitational"
    HIGGS = "higgs"


# =============================================================================
# FUNDAMENTAL PARTICLES
# =============================================================================

@dataclass
class Particle:
    """Fundamental or composite particle."""
    name: str
    symbol: str
    mass_GeV: float
    charge: float  # In units of e
    spin: float
    particle_type: ParticleType
    color_charge: Optional[str] = None  # For quarks: r, g, b
    generation: int = 1
    antiparticle: str = None
    interactions: List[InteractionType] = field(default_factory=list)
    lifetime_s: float = float('inf')  # Stable if infinite
    
    @property
    def mass_eV(self) -> float:
        return self.mass_GeV * 1e9
    
    @property
    def mass_kg(self) -> float:
        return self.mass_GeV * ParticleConstants.GeV_to_J / ParticleConstants.c ** 2
    
    @property
    def is_fermion(self) -> bool:
        return self.spin % 1 == 0.5
    
    @property
    def is_boson(self) -> bool:
        return self.spin % 1 == 0
    
    @property
    def is_stable(self) -> bool:
        return self.lifetime_s == float('inf')


# =============================================================================
# STANDARD MODEL DATABASE
# =============================================================================

class StandardModel:
    """
    The Standard Model of particle physics.
    """
    
    # QUARKS
    UP = Particle(
        name="up", symbol="u", mass_GeV=0.00216, charge=2/3, spin=0.5,
        particle_type=ParticleType.QUARK, color_charge="r,g,b",
        generation=1, antiparticle="ū",
        interactions=[InteractionType.STRONG, InteractionType.ELECTROMAGNETIC, InteractionType.WEAK]
    )
    DOWN = Particle(
        name="down", symbol="d", mass_GeV=0.00467, charge=-1/3, spin=0.5,
        particle_type=ParticleType.QUARK, color_charge="r,g,b",
        generation=1, antiparticle="d̄",
        interactions=[InteractionType.STRONG, InteractionType.ELECTROMAGNETIC, InteractionType.WEAK]
    )
    CHARM = Particle(
        name="charm", symbol="c", mass_GeV=1.27, charge=2/3, spin=0.5,
        particle_type=ParticleType.QUARK, color_charge="r,g,b",
        generation=2, antiparticle="c̄",
        interactions=[InteractionType.STRONG, InteractionType.ELECTROMAGNETIC, InteractionType.WEAK]
    )
    STRANGE = Particle(
        name="strange", symbol="s", mass_GeV=0.093, charge=-1/3, spin=0.5,
        particle_type=ParticleType.QUARK, color_charge="r,g,b",
        generation=2, antiparticle="s̄",
        interactions=[InteractionType.STRONG, InteractionType.ELECTROMAGNETIC, InteractionType.WEAK]
    )
    TOP = Particle(
        name="top", symbol="t", mass_GeV=172.76, charge=2/3, spin=0.5,
        particle_type=ParticleType.QUARK, color_charge="r,g,b",
        generation=3, antiparticle="t̄",
        interactions=[InteractionType.STRONG, InteractionType.ELECTROMAGNETIC, InteractionType.WEAK],
        lifetime_s=5e-25
    )
    BOTTOM = Particle(
        name="bottom", symbol="b", mass_GeV=4.18, charge=-1/3, spin=0.5,
        particle_type=ParticleType.QUARK, color_charge="r,g,b",
        generation=3, antiparticle="b̄",
        interactions=[InteractionType.STRONG, InteractionType.ELECTROMAGNETIC, InteractionType.WEAK],
        lifetime_s=1.3e-12
    )
    
    # LEPTONS
    ELECTRON = Particle(
        name="electron", symbol="e⁻", mass_GeV=0.000511, charge=-1, spin=0.5,
        particle_type=ParticleType.LEPTON, generation=1, antiparticle="e⁺",
        interactions=[InteractionType.ELECTROMAGNETIC, InteractionType.WEAK]
    )
    ELECTRON_NEUTRINO = Particle(
        name="electron neutrino", symbol="νₑ", mass_GeV=1e-9, charge=0, spin=0.5,
        particle_type=ParticleType.LEPTON, generation=1, antiparticle="ν̄ₑ",
        interactions=[InteractionType.WEAK]
    )
    MUON = Particle(
        name="muon", symbol="μ⁻", mass_GeV=0.10566, charge=-1, spin=0.5,
        particle_type=ParticleType.LEPTON, generation=2, antiparticle="μ⁺",
        interactions=[InteractionType.ELECTROMAGNETIC, InteractionType.WEAK],
        lifetime_s=2.197e-6
    )
    MUON_NEUTRINO = Particle(
        name="muon neutrino", symbol="νᵤ", mass_GeV=1e-9, charge=0, spin=0.5,
        particle_type=ParticleType.LEPTON, generation=2, antiparticle="ν̄ᵤ",
        interactions=[InteractionType.WEAK]
    )
    TAU = Particle(
        name="tau", symbol="τ⁻", mass_GeV=1.777, charge=-1, spin=0.5,
        particle_type=ParticleType.LEPTON, generation=3, antiparticle="τ⁺",
        interactions=[InteractionType.ELECTROMAGNETIC, InteractionType.WEAK],
        lifetime_s=2.9e-13
    )
    TAU_NEUTRINO = Particle(
        name="tau neutrino", symbol="ντ", mass_GeV=1e-9, charge=0, spin=0.5,
        particle_type=ParticleType.LEPTON, generation=3, antiparticle="ν̄τ",
        interactions=[InteractionType.WEAK]
    )
    
    # GAUGE BOSONS
    PHOTON = Particle(
        name="photon", symbol="γ", mass_GeV=0, charge=0, spin=1,
        particle_type=ParticleType.GAUGE_BOSON, antiparticle="γ",
        interactions=[InteractionType.ELECTROMAGNETIC]
    )
    GLUON = Particle(
        name="gluon", symbol="g", mass_GeV=0, charge=0, spin=1,
        particle_type=ParticleType.GAUGE_BOSON, color_charge="8 types",
        antiparticle="g",
        interactions=[InteractionType.STRONG]
    )
    W_PLUS = Particle(
        name="W+ boson", symbol="W⁺", mass_GeV=80.379, charge=1, spin=1,
        particle_type=ParticleType.GAUGE_BOSON, antiparticle="W⁻",
        interactions=[InteractionType.WEAK],
        lifetime_s=3e-25
    )
    W_MINUS = Particle(
        name="W- boson", symbol="W⁻", mass_GeV=80.379, charge=-1, spin=1,
        particle_type=ParticleType.GAUGE_BOSON, antiparticle="W⁺",
        interactions=[InteractionType.WEAK],
        lifetime_s=3e-25
    )
    Z_BOSON = Particle(
        name="Z boson", symbol="Z⁰", mass_GeV=91.1876, charge=0, spin=1,
        particle_type=ParticleType.GAUGE_BOSON, antiparticle="Z⁰",
        interactions=[InteractionType.WEAK],
        lifetime_s=3e-25
    )
    
    # SCALAR BOSON
    HIGGS = Particle(
        name="Higgs boson", symbol="H⁰", mass_GeV=125.25, charge=0, spin=0,
        particle_type=ParticleType.SCALAR_BOSON, antiparticle="H⁰",
        interactions=[InteractionType.HIGGS],
        lifetime_s=1.6e-22
    )
    
    @classmethod
    def all_quarks(cls) -> List[Particle]:
        return [cls.UP, cls.DOWN, cls.CHARM, cls.STRANGE, cls.TOP, cls.BOTTOM]
    
    @classmethod
    def all_leptons(cls) -> List[Particle]:
        return [cls.ELECTRON, cls.ELECTRON_NEUTRINO, cls.MUON, cls.MUON_NEUTRINO,
                cls.TAU, cls.TAU_NEUTRINO]
    
    @classmethod
    def all_gauge_bosons(cls) -> List[Particle]:
        return [cls.PHOTON, cls.GLUON, cls.W_PLUS, cls.W_MINUS, cls.Z_BOSON]
    
    @classmethod
    def all_particles(cls) -> List[Particle]:
        return cls.all_quarks() + cls.all_leptons() + cls.all_gauge_bosons() + [cls.HIGGS]


# =============================================================================
# HADRONS (COMPOSITE PARTICLES)
# =============================================================================

class Hadrons:
    """Common hadrons (quark composites)."""
    
    # BARYONS (qqq)
    PROTON = Particle(
        name="proton", symbol="p", mass_GeV=0.938272, charge=1, spin=0.5,
        particle_type=ParticleType.BARYON, antiparticle="p̄",
        interactions=[InteractionType.STRONG, InteractionType.ELECTROMAGNETIC, InteractionType.WEAK]
    )
    NEUTRON = Particle(
        name="neutron", symbol="n", mass_GeV=0.939565, charge=0, spin=0.5,
        particle_type=ParticleType.BARYON, antiparticle="n̄",
        interactions=[InteractionType.STRONG, InteractionType.WEAK],
        lifetime_s=879.4  # Free neutron
    )
    LAMBDA = Particle(
        name="lambda", symbol="Λ⁰", mass_GeV=1.116, charge=0, spin=0.5,
        particle_type=ParticleType.BARYON, lifetime_s=2.6e-10
    )
    SIGMA_PLUS = Particle(
        name="sigma+", symbol="Σ⁺", mass_GeV=1.189, charge=1, spin=0.5,
        particle_type=ParticleType.BARYON, lifetime_s=8e-11
    )
    OMEGA_MINUS = Particle(
        name="omega-", symbol="Ω⁻", mass_GeV=1.672, charge=-1, spin=1.5,
        particle_type=ParticleType.BARYON, lifetime_s=8.2e-11
    )
    
    # MESONS (q q̄)
    PION_PLUS = Particle(
        name="pion+", symbol="π⁺", mass_GeV=0.13957, charge=1, spin=0,
        particle_type=ParticleType.MESON, antiparticle="π⁻",
        interactions=[InteractionType.STRONG, InteractionType.ELECTROMAGNETIC, InteractionType.WEAK],
        lifetime_s=2.6e-8
    )
    PION_ZERO = Particle(
        name="pion0", symbol="π⁰", mass_GeV=0.13498, charge=0, spin=0,
        particle_type=ParticleType.MESON, antiparticle="π⁰",
        lifetime_s=8.5e-17
    )
    KAON_PLUS = Particle(
        name="kaon+", symbol="K⁺", mass_GeV=0.494, charge=1, spin=0,
        particle_type=ParticleType.MESON, antiparticle="K⁻",
        lifetime_s=1.24e-8
    )
    ETA = Particle(
        name="eta", symbol="η", mass_GeV=0.548, charge=0, spin=0,
        particle_type=ParticleType.MESON, lifetime_s=5e-19
    )
    JPSI = Particle(
        name="J/psi", symbol="J/ψ", mass_GeV=3.097, charge=0, spin=1,
        particle_type=ParticleType.MESON, lifetime_s=7e-21
    )
    UPSILON = Particle(
        name="upsilon", symbol="Υ", mass_GeV=9.46, charge=0, spin=1,
        particle_type=ParticleType.MESON, lifetime_s=1.2e-20
    )


# =============================================================================
# CONSERVATION LAWS
# =============================================================================

class ConservationLaws:
    """Check conservation laws in particle interactions."""
    
    @staticmethod
    def check_charge(initial: List[Particle], final: List[Particle]) -> bool:
        """Check charge conservation."""
        q_initial = sum(p.charge for p in initial)
        q_final = sum(p.charge for p in final)
        return abs(q_initial - q_final) < 0.01
    
    @staticmethod
    def check_baryon_number(initial: List[Particle], final: List[Particle]) -> bool:
        """Check baryon number conservation."""
        def baryon_number(p: Particle) -> int:
            if p.particle_type == ParticleType.QUARK:
                return 1/3
            elif p.particle_type == ParticleType.BARYON:
                return 1
            return 0
        
        B_initial = sum(baryon_number(p) for p in initial)
        B_final = sum(baryon_number(p) for p in final)
        return abs(B_initial - B_final) < 0.01
    
    @staticmethod
    def check_lepton_number(initial: List[Particle], final: List[Particle]) -> bool:
        """Check lepton number conservation."""
        def lepton_number(p: Particle) -> int:
            if p.particle_type == ParticleType.LEPTON:
                return 1 if p.charge <= 0 else -1  # Simplified
            return 0
        
        L_initial = sum(lepton_number(p) for p in initial)
        L_final = sum(lepton_number(p) for p in final)
        return L_initial == L_final


# =============================================================================
# PARTICLE PHYSICS CALCULATIONS
# =============================================================================

class ParticlePhysics:
    """Particle physics calculations."""
    
    @staticmethod
    def decay_width_to_lifetime(width_GeV: float) -> float:
        """
        Convert decay width Γ to lifetime τ.
        τ = ℏ/Γ
        """
        hbar = ParticleConstants.hbar
        width_J = width_GeV * ParticleConstants.GeV_to_J
        return hbar / width_J
    
    @staticmethod
    def lifetime_to_decay_width(lifetime_s: float) -> float:
        """Convert lifetime to decay width in GeV."""
        hbar = ParticleConstants.hbar
        width_J = hbar / lifetime_s
        return width_J / ParticleConstants.GeV_to_J
    
    @staticmethod
    def decay_length(lifetime_s: float, gamma: float) -> float:
        """
        Calculate decay length in lab frame.
        L = βγcτ
        """
        c = ParticleConstants.c
        beta = math.sqrt(1 - 1/gamma**2)
        return beta * gamma * c * lifetime_s
    
    @staticmethod
    def invariant_mass(E_total: float, p_total: float) -> float:
        """
        Calculate invariant mass.
        m²c⁴ = E² - (pc)²
        """
        c = ParticleConstants.c
        m2c4 = E_total**2 - (p_total * c)**2
        if m2c4 < 0:
            return 0
        return math.sqrt(m2c4) / c**2
    
    @staticmethod
    def threshold_energy(m_target: float, m_products: List[float]) -> float:
        """
        Calculate threshold energy for particle production.
        E_threshold = (Σm_products)²c² / (2 × m_target × c²) for fixed target
        """
        c = ParticleConstants.c
        sum_products = sum(m_products)
        return (sum_products ** 2 * c ** 2) / (2 * m_target * c ** 2)
    
    @staticmethod
    def compton_wavelength(mass_GeV: float) -> float:
        """
        Calculate Compton wavelength.
        λ_C = h/(mc)
        """
        h = ParticleConstants.hbar * 2 * math.pi
        c = ParticleConstants.c
        m = mass_GeV * ParticleConstants.GeV_to_J / c ** 2
        return h / (m * c)
    
    @staticmethod
    def de_broglie_wavelength(mass_GeV: float, kinetic_energy_GeV: float) -> float:
        """Calculate de Broglie wavelength."""
        h = ParticleConstants.hbar * 2 * math.pi
        E_k = kinetic_energy_GeV * ParticleConstants.GeV_to_J
        m = mass_GeV * ParticleConstants.GeV_to_J / ParticleConstants.c ** 2
        
        p = math.sqrt(2 * m * E_k)
        return h / p


# =============================================================================
# FEYNMAN DIAGRAM VERTICES
# =============================================================================

class FeynmanVertex:
    """Represent a Feynman diagram vertex."""
    
    QED_VERTICES = [
        ("e⁻ → e⁻ + γ", "Electron emits photon"),
        ("e⁺ + e⁻ → γ", "Electron-positron annihilation"),
        ("γ → e⁺ + e⁻", "Pair production"),
    ]
    
    QCD_VERTICES = [
        ("q → q + g", "Quark emits gluon"),
        ("g → g + g", "Triple gluon vertex"),
        ("g + g → g + g", "Four-gluon vertex"),
    ]
    
    WEAK_VERTICES = [
        ("e⁻ → νₑ + W⁻", "Electron-W vertex"),
        ("u → d + W⁺", "Quark flavor change"),
        ("Z → f + f̄", "Z decay to fermions"),
        ("H → W⁺ + W⁻", "Higgs-W coupling"),
    ]
    
    @classmethod
    def all_vertices(cls) -> Dict[str, List[Tuple[str, str]]]:
        return {
            'QED': cls.QED_VERTICES,
            'QCD': cls.QCD_VERTICES,
            'Weak': cls.WEAK_VERTICES
        }


# =============================================================================
# PARTICLE ENGINE - MAIN INTERFACE
# =============================================================================

class ParticleEngine:
    """
    AION Particle Physics Engine.
    """
    
    def __init__(self):
        self.sm = StandardModel
        self.hadrons = Hadrons
        self.physics = ParticlePhysics
    
    def get_particle(self, name: str) -> Optional[Particle]:
        """Get particle by name."""
        name_lower = name.lower()
        for p in self.sm.all_particles():
            if p.name.lower() == name_lower or p.symbol.lower() == name_lower:
                return p
        return None
    
    def particle_info(self, name: str) -> Dict:
        """Get complete information about a particle."""
        p = self.get_particle(name)
        if not p:
            return {'error': f'Particle {name} not found'}
        
        return {
            'name': p.name,
            'symbol': p.symbol,
            'type': p.particle_type.value,
            'mass_GeV': p.mass_GeV,
            'mass_eV': p.mass_eV,
            'charge': p.charge,
            'spin': p.spin,
            'is_fermion': p.is_fermion,
            'is_boson': p.is_boson,
            'is_stable': p.is_stable,
            'lifetime_s': p.lifetime_s if not p.is_stable else 'stable',
            'generation': p.generation,
            'antiparticle': p.antiparticle,
            'interactions': [i.value for i in p.interactions]
        }
    
    def list_particles(self, category: str = 'all') -> List[Dict]:
        """List particles by category."""
        if category == 'quarks':
            particles = self.sm.all_quarks()
        elif category == 'leptons':
            particles = self.sm.all_leptons()
        elif category == 'bosons':
            particles = self.sm.all_gauge_bosons() + [self.sm.HIGGS]
        else:
            particles = self.sm.all_particles()
        
        return [{'name': p.name, 'symbol': p.symbol, 'mass_GeV': p.mass_GeV, 
                 'charge': p.charge, 'spin': p.spin} for p in particles]
    
    def compare_masses(self) -> Dict:
        """Compare particle masses."""
        particles = self.sm.all_particles()
        sorted_particles = sorted(particles, key=lambda p: p.mass_GeV, reverse=True)
        
        return {
            'heaviest': sorted_particles[0].name,
            'lightest_massive': next((p.name for p in reversed(sorted_particles) if p.mass_GeV > 0), None),
            'mass_hierarchy': [(p.name, p.mass_GeV) for p in sorted_particles[:10]]
        }
    
    def decay_properties(self, lifetime_s: float, mass_GeV: float, gamma: float = 1.0) -> Dict:
        """Calculate decay properties."""
        width = self.physics.lifetime_to_decay_width(lifetime_s)
        decay_length = self.physics.decay_length(lifetime_s, gamma)
        
        return {
            'lifetime_s': lifetime_s,
            'decay_width_GeV': width,
            'decay_length_m': decay_length,
            'gamma_factor': gamma
        }
    
    def standard_model_summary(self) -> Dict:
        """Get Standard Model summary."""
        return {
            'quarks': {
                'count': 6,
                'generations': 3,
                'up_type': ['up', 'charm', 'top'],
                'down_type': ['down', 'strange', 'bottom'],
                'color_charge': True
            },
            'leptons': {
                'count': 6,
                'generations': 3,
                'charged': ['electron', 'muon', 'tau'],
                'neutrinos': ['νₑ', 'νᵤ', 'ντ'],
                'color_charge': False
            },
            'gauge_bosons': {
                'photon': 'electromagnetic force',
                'gluon': 'strong force (8 types)',
                'W±': 'weak force (charged)',
                'Z⁰': 'weak force (neutral)'
            },
            'higgs': {
                'mass_GeV': 125.25,
                'role': 'gives mass to particles'
            },
            'forces': {
                'strong': {'mediator': 'gluon', 'range': '~1 fm', 'relative_strength': 1},
                'electromagnetic': {'mediator': 'photon', 'range': 'infinite', 'relative_strength': 1/137},
                'weak': {'mediator': 'W,Z', 'range': '~0.001 fm', 'relative_strength': 1e-6},
                'gravity': {'mediator': 'graviton?', 'range': 'infinite', 'relative_strength': 1e-38}
            }
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Particle Physics Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          ⚛️ AION PARTICLE PHYSICS ENGINE ⚛️                               ║
║                                                                           ║
║     Standard Model, Quarks, Leptons, Bosons, Hadrons                     ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = ParticleEngine()
    
    # Quarks
    print("🔴 Quarks:")
    print("-" * 50)
    for p in engine.list_particles('quarks'):
        print(f"   {p['symbol']:4} mass = {p['mass_GeV']:.4f} GeV, Q = {p['charge']:+.2f}e")
    
    # Leptons
    print("\n🔵 Leptons:")
    print("-" * 50)
    for p in engine.list_particles('leptons'):
        print(f"   {p['symbol']:4} mass = {p['mass_GeV']:.6f} GeV, Q = {p['charge']:+.0f}e")
    
    # Bosons
    print("\n🟡 Bosons:")
    print("-" * 50)
    for p in engine.list_particles('bosons'):
        print(f"   {p['symbol']:4} mass = {p['mass_GeV']:.2f} GeV, spin = {p['spin']}")
    
    # Mass hierarchy
    print("\n📊 Mass Hierarchy (heaviest first):")
    print("-" * 50)
    result = engine.compare_masses()
    for name, mass in result['mass_hierarchy'][:6]:
        print(f"   {name:15} {mass:>10.3f} GeV")
    
    # Standard Model summary
    print("\n📋 Standard Model Summary:")
    print("-" * 50)
    summary = engine.standard_model_summary()
    print(f"   Quarks: {summary['quarks']['count']} (3 generations)")
    print(f"   Leptons: {summary['leptons']['count']} (3 generations)")
    print(f"   Force carriers: γ, g, W±, Z⁰")
    print(f"   Higgs boson: {summary['higgs']['mass_GeV']} GeV")


if __name__ == "__main__":
    demo()
