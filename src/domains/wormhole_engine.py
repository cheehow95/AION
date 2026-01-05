"""
AION Wormhole Physics Engine
============================

Theoretical wormhole physics simulation covering:
- Einstein-Rosen bridges
- Morris-Thorne traversable wormholes
- Exotic matter requirements
- Traversability conditions
- Causality and chronology protection

Based on general relativity solutions for wormholes.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class WormholeConstants:
    """Constants for wormhole physics."""
    G = 6.67430e-11         # Gravitational constant (m³/kg/s²)
    c = 299792458           # Speed of light (m/s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    
    # Derived
    c2 = c ** 2
    c4 = c ** 4
    
    # Planck units
    l_p = math.sqrt(hbar * G / c**3)   # Planck length ~1.6e-35 m
    t_p = math.sqrt(hbar * G / c**5)   # Planck time ~5.4e-44 s
    m_p = math.sqrt(hbar * c / G)      # Planck mass ~2.2e-8 kg
    
    # Energy density
    rho_p = m_p * c2 / l_p**3          # Planck energy density


# =============================================================================
# WORMHOLE TYPES
# =============================================================================

class WormholeType(Enum):
    """Types of wormholes."""
    EINSTEIN_ROSEN = "einstein_rosen"    # Non-traversable (Schwarzschild)
    MORRIS_THORNE = "morris_thorne"      # Traversable
    VISSER = "visser"                    # Thin-shell wormhole
    ELLIS = "ellis"                      # Drainhole (special case)


# =============================================================================
# MORRIS-THORNE WORMHOLE
# =============================================================================

@dataclass
class MorrisThorneWormhole:
    """
    Morris-Thorne traversable wormhole.
    
    Metric:
    ds² = -e^(2Φ(r)) c²dt² + dr²/(1 - b(r)/r) + r²dΩ²
    
    where:
    - Φ(r) is the redshift function
    - b(r) is the shape function
    - b(r₀) = r₀ at the throat
    """
    
    throat_radius: float      # r₀ (meters)
    redshift_param: float = 0.0  # Controls gravitational redshift
    
    def __post_init__(self):
        self.G = WormholeConstants.G
        self.c = WormholeConstants.c
        self.c2 = WormholeConstants.c2
    
    def shape_function(self, r: float) -> float:
        """
        Shape function b(r).
        
        Simple model: b(r) = r₀²/r
        
        Must satisfy:
        1. b(r₀) = r₀
        2. b(r) < r for r > r₀
        3. b'(r₀) < 1 (flare-out condition)
        """
        if r <= 0:
            return float('inf')
        return self.throat_radius ** 2 / r
    
    def redshift_function(self, r: float) -> float:
        """
        Redshift function Φ(r).
        
        For zero tidal forces: Φ(r) = constant
        """
        return self.redshift_param
    
    def proper_radial_distance(self, r: float) -> float:
        """
        Calculate proper distance from throat.
        
        l(r) = ∫ dr / √(1 - b(r)/r)
        
        For b(r) = r₀²/r:
        l(r) = √(r² - r₀²)
        """
        if r < self.throat_radius:
            return 0.0
        
        return math.sqrt(r ** 2 - self.throat_radius ** 2)
    
    def embedding_z(self, r: float) -> float:
        """
        Embedding function for visualization.
        
        dz/dr = ±√(b(r)/(r - b(r)))
        
        For b(r) = r₀²/r:
        z(r) = ±r₀ × arccosh(r/r₀)
        """
        if r <= self.throat_radius:
            return 0.0
        
        return self.throat_radius * math.acosh(r / self.throat_radius)
    
    def metric_coefficient_grr(self, r: float) -> float:
        """
        g_rr component of metric.
        
        g_rr = 1/(1 - b(r)/r)
        """
        b = self.shape_function(r)
        if r <= b:
            return float('inf')
        return 1 / (1 - b / r)
    
    def metric_coefficient_gtt(self, r: float) -> float:
        """
        g_tt component of metric.
        
        g_tt = -e^(2Φ(r)) c²
        """
        Phi = self.redshift_function(r)
        return -math.exp(2 * Phi) * self.c2
    
    def is_traversable(self) -> bool:
        """
        Check if wormhole is traversable.
        
        Requires:
        1. Φ(r) finite everywhere (no horizon)
        2. Proper radial distance finite
        3. Tidal forces survivable
        """
        # For our simple model with constant Φ
        return self.redshift_param > -float('inf')
    
    def flare_out_condition(self, r: float) -> bool:
        """
        Check flare-out condition at radius r.
        
        d²r/dz² > 0 at throat, which requires:
        (b - b'r) / (2b²) > 0 → b'(r₀) < 1
        
        For b(r) = r₀²/r: b'(r) = -r₀²/r² → b'(r₀) = -1 < 1 ✓
        """
        r0 = self.throat_radius
        b_prime = -r0 ** 2 / r ** 2
        
        return b_prime < 1
    
    def tidal_acceleration(self, r: float, radial_separation: float) -> float:
        """
        Estimate tidal acceleration for traveler.
        
        For zero-tidal-force wormhole: a_tidal ~ 0
        """
        # For our simple model with constant Φ
        return 0.0
    
    def travel_time(self, velocity: float) -> float:
        """
        Calculate proper time to traverse wormhole.
        
        For traveler moving at velocity v through the throat.
        """
        # Proper distance through throat ~ 2 × π × r₀ (approximate)
        distance = 2 * math.pi * self.throat_radius
        
        # Gamma factor
        gamma = 1 / math.sqrt(1 - velocity ** 2 / self.c2)
        
        # Proper time = distance / (γv)
        return distance / (gamma * velocity)


# =============================================================================
# EXOTIC MATTER REQUIREMENTS
# =============================================================================

class ExoticMatter:
    """
    Exotic matter calculations for wormholes.
    
    Traversable wormholes require matter that violates the
    Weak Energy Condition (WEC): T_μν U^μ U^ν < 0
    
    This means negative energy density from some observer's perspective.
    """
    
    @staticmethod
    def energy_density_at_throat(throat_radius: float) -> float:
        """
        Calculate required exotic matter energy density at throat.
        
        For Morris-Thorne with b(r) = r₀²/r:
        ρ = -c²/(8πGr₀²)
        
        Negative energy density!
        """
        G = WormholeConstants.G
        c = WormholeConstants.c
        
        return -c ** 2 / (8 * math.pi * G * throat_radius ** 2)
    
    @staticmethod
    def total_exotic_mass(throat_radius: float) -> float:
        """
        Estimate total exotic mass required.
        
        M_exotic ~ -c²r₀/(2G) = -r₀c²/(2G)
        
        This is NEGATIVE mass (exotic matter).
        """
        G = WormholeConstants.G
        c = WormholeConstants.c
        
        return -throat_radius * c ** 2 / (2 * G)
    
    @staticmethod
    def tension_at_throat(throat_radius: float) -> float:
        """
        Calculate radial tension required at throat.
        
        τ = c⁴/(8πGr₀²)
        
        Enormous tension (negative pressure)!
        """
        G = WormholeConstants.G
        c = WormholeConstants.c
        
        return c ** 4 / (8 * math.pi * G * throat_radius ** 2)
    
    @staticmethod
    def casimir_energy_density(plate_separation: float) -> float:
        """
        Calculate Casimir effect energy density (known source of negative energy).
        
        ρ = -π²ℏc/(240 d⁴)
        
        Only significant at nanometer scales.
        """
        hbar = WormholeConstants.hbar
        c = WormholeConstants.c
        
        return -math.pi ** 2 * hbar * c / (240 * plate_separation ** 4)
    
    @staticmethod
    def compare_to_casimir(throat_radius: float) -> Dict:
        """
        Compare required exotic energy density to Casimir effect.
        """
        required = abs(ExoticMatter.energy_density_at_throat(throat_radius))
        
        # Casimir at 1 nm separation
        casimir_1nm = abs(ExoticMatter.casimir_energy_density(1e-9))
        
        return {
            'required_energy_density_J_m3': required,
            'casimir_1nm_J_m3': casimir_1nm,
            'ratio': required / casimir_1nm,
            'feasibility': 'EXTREMELY DIFFICULT' if required > casimir_1nm * 1e30 else 'THEORETICALLY POSSIBLE'
        }


# =============================================================================
# CAUSALITY AND TIME TRAVEL
# =============================================================================

class CausalityAnalysis:
    """
    Analyze causality issues with wormholes.
    
    Wormholes connecting different times could create
    Closed Timelike Curves (CTCs) - paths through spacetime
    that loop back to their origin.
    """
    
    @staticmethod
    def can_create_time_machine(wormhole: MorrisThorneWormhole, 
                                 mouth_velocity: float,
                                 time_running: float) -> Dict:
        """
        Check if wormhole can be used for time travel.
        
        If one mouth is accelerated (twin paradox) or in stronger
        gravitational field, the mouths can become time-shifted.
        """
        c = WormholeConstants.c
        
        # Time dilation from velocity (twin paradox)
        if mouth_velocity >= c:
            return {'can_travel_back': False, 'error': 'Velocity >= c'}
        
        gamma = 1 / math.sqrt(1 - (mouth_velocity / c) ** 2)
        
        # Time difference accumulated
        time_difference = time_running * (1 - 1/gamma)
        
        # If wormhole proper length < c × time_difference, CTCs possible
        wormhole_proper_length = 2 * wormhole.proper_radial_distance(2 * wormhole.throat_radius)
        
        ctc_possible = wormhole_proper_length < c * time_difference
        
        return {
            'time_difference_seconds': time_difference,
            'gamma': gamma,
            'wormhole_proper_length': wormhole_proper_length,
            'ctc_possible': ctc_possible,
            'chronology_protection': 'Hawking\'s Chronology Protection Conjecture suggests quantum effects prevent CTCs'
        }
    
    @staticmethod
    def grandfather_paradox_analysis() -> str:
        """
        Theoretical analysis of grandfather paradox.
        """
        return """
        GRANDFATHER PARADOX RESOLUTIONS:
        
        1. NOVIKOV SELF-CONSISTENCY PRINCIPLE:
           Only self-consistent histories are allowed.
           You cannot change the past - any "changes" were
           always part of history.
        
        2. MANY-WORLDS INTERPRETATION:
           Traveling back creates a new branch of the multiverse.
           The original timeline is unaffected.
        
        3. CHRONOLOGY PROTECTION CONJECTURE (Hawking):
           Quantum effects (e.g., vacuum fluctuations) become
           infinite as CTCs form, destroying the wormhole.
        
        4. SELF-AVOIDING ORBITS:
           The universe has mechanisms to prevent paradoxes
           (physical laws prevent paradox-creating actions).
        """


# =============================================================================
# WORMHOLE ENGINE - MAIN INTERFACE
# =============================================================================

class WormholeEngine:
    """
    AION Wormhole Engine for theoretical calculations.
    """
    
    def __init__(self):
        self.c = WormholeConstants.c
    
    def create_wormhole(self, throat_radius_m: float, 
                        redshift_param: float = 0) -> MorrisThorneWormhole:
        """Create a Morris-Thorne traversable wormhole."""
        return MorrisThorneWormhole(throat_radius_m, redshift_param)
    
    def analyze(self, throat_radius_km: float) -> Dict:
        """Complete analysis of a wormhole."""
        r0 = throat_radius_km * 1000
        wh = self.create_wormhole(r0)
        
        exotic_mass = ExoticMatter.total_exotic_mass(r0)
        energy_density = ExoticMatter.energy_density_at_throat(r0)
        tension = ExoticMatter.tension_at_throat(r0)
        
        M_sun = 1.989e30
        
        return {
            'throat_radius_km': throat_radius_km,
            'is_traversable': wh.is_traversable(),
            'flare_out_satisfied': wh.flare_out_condition(r0),
            
            # Geometry
            'embedding_height_km': wh.embedding_z(2 * r0) / 1000,
            'proper_distance_2r0_km': wh.proper_radial_distance(2 * r0) / 1000,
            
            # Exotic matter (negative)
            'exotic_mass_solar': exotic_mass / M_sun,
            'exotic_mass_kg': exotic_mass,
            'energy_density_J_m3': energy_density,
            'tension_Pa': tension,
            
            # Travel
            'travel_time_at_0.1c_hours': wh.travel_time(0.1 * self.c) / 3600,
            'travel_time_at_0.5c_minutes': wh.travel_time(0.5 * self.c) / 60,
        }
    
    def exotic_matter_requirements(self, throat_radius_km: float) -> Dict:
        """Calculate exotic matter requirements."""
        r0 = throat_radius_km * 1000
        
        return {
            'throat_radius_km': throat_radius_km,
            'exotic_mass_kg': ExoticMatter.total_exotic_mass(r0),
            'energy_density_J_m3': ExoticMatter.energy_density_at_throat(r0),
            'tension_Pa': ExoticMatter.tension_at_throat(r0),
            'comparison': ExoticMatter.compare_to_casimir(r0)
        }
    
    def embedding_diagram(self, throat_radius_km: float, n_points: int = 100) -> List[Dict]:
        """
        Generate embedding diagram for visualization.
        
        Returns list of (r, z) coordinates for the wormhole surface.
        """
        r0 = throat_radius_km * 1000
        wh = self.create_wormhole(r0)
        
        points = []
        
        for i in range(n_points):
            r = r0 * (1 + 2 * i / n_points)
            z_plus = wh.embedding_z(r) / 1000
            z_minus = -z_plus
            
            points.append({
                'r_km': r / 1000,
                'z_upper_km': z_plus,
                'z_lower_km': z_minus
            })
        
        return points
    
    def time_travel_potential(self, throat_radius_km: float,
                               accelerated_velocity_c: float,
                               time_years: float) -> Dict:
        """Analyze time travel potential of wormhole."""
        r0 = throat_radius_km * 1000
        wh = self.create_wormhole(r0)
        
        v = accelerated_velocity_c * self.c
        t = time_years * 365.25 * 24 * 3600
        
        result = CausalityAnalysis.can_create_time_machine(wh, v, t)
        result['throat_radius_km'] = throat_radius_km
        result['velocity_c'] = accelerated_velocity_c
        result['time_years'] = time_years
        
        return result


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Wormhole Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🌀 AION WORMHOLE ENGINE 🌀                                       ║
║                                                                           ║
║     Einstein-Rosen Bridges, Traversable Wormholes, Exotic Matter         ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = WormholeEngine()
    
    # Human-sized wormhole
    print("🚶 Human-sized Wormhole (throat = 1 km):")
    print("-" * 50)
    result = engine.analyze(1.0)
    print(f"   Traversable: {result['is_traversable']}")
    print(f"   Flare-out condition: {result['flare_out_satisfied']}")
    print(f"   Exotic mass required: {result['exotic_mass_solar']:.2e} M☉")
    print(f"   (That's {abs(result['exotic_mass_kg']):.2e} kg of NEGATIVE mass!)")
    print(f"   Travel time at 0.1c: {result['travel_time_at_0.1c_hours']:.2f} hours")
    
    # Exotic matter
    print("\n⚗️ Exotic Matter Requirements:")
    print("-" * 50)
    result = engine.exotic_matter_requirements(1.0)
    print(f"   Energy density: {result['energy_density_J_m3']:.2e} J/m³")
    print(f"   (This is NEGATIVE - violates energy conditions)")
    print(f"   Tension: {result['tension_Pa']:.2e} Pa")
    print(f"   Comparison to Casimir: {result['comparison']['ratio']:.2e}× stronger needed")
    
    # Larger wormhole
    print("\n🚀 Spacecraft-sized Wormhole (throat = 100 km):")
    print("-" * 50)
    result = engine.analyze(100.0)
    print(f"   Exotic mass: {result['exotic_mass_solar']:.2e} M☉")
    print(f"   Travel time at 0.5c: {result['travel_time_at_0.5c_minutes']:.1f} minutes")
    
    # Time travel
    print("\n⏰ Time Travel Analysis:")
    print("-" * 50)
    result = engine.time_travel_potential(
        throat_radius_km=1.0,
        accelerated_velocity_c=0.9,
        time_years=10
    )
    print(f"   One mouth accelerated at 0.9c for 10 years")
    print(f"   Time difference accumulated: {result['time_difference_seconds']/(3600*24*365):.2f} years")
    print(f"   Closed timelike curves possible: {result['ctc_possible']}")
    print(f"   Note: {result['chronology_protection']}")
    
    # Theoretical analysis
    print("\n📚 Grandfather Paradox Resolutions:")
    print(CausalityAnalysis.grandfather_paradox_analysis())


if __name__ == "__main__":
    demo()
