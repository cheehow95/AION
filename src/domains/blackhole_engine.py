"""
AION Black Hole Physics Engine
==============================

Complete black hole physics simulation covering:
- Schwarzschild (non-rotating) black holes
- Kerr (rotating) black holes
- Hawking radiation and thermodynamics
- Particle orbits and geodesics
- Tidal forces (spaghettification)

Based on general relativity solutions.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class BlackHoleConstants:
    """Constants for black hole physics."""
    G = 6.67430e-11         # Gravitational constant (m³/kg/s²)
    c = 299792458           # Speed of light (m/s)
    h = 6.62607015e-34      # Planck constant (J·s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    k_B = 1.380649e-23      # Boltzmann constant (J/K)
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W/m²/K⁴)
    
    # Derived
    c2 = c ** 2
    c3 = c ** 3
    c4 = c ** 4
    
    # Planck units
    l_p = math.sqrt(hbar * G / c3)   # Planck length ~1.6e-35 m
    t_p = math.sqrt(hbar * G / c**5)  # Planck time ~5.4e-44 s
    m_p = math.sqrt(hbar * c / G)     # Planck mass ~2.2e-8 kg
    
    # Astronomical
    M_sun = 1.989e30        # Solar mass (kg)


# =============================================================================
# BLACK HOLE TYPES
# =============================================================================

class BlackHoleType(Enum):
    """Types of black holes based on properties."""
    SCHWARZSCHILD = "schwarzschild"  # Non-rotating, uncharged
    KERR = "kerr"                    # Rotating, uncharged
    REISSNER_NORDSTROM = "rn"        # Non-rotating, charged
    KERR_NEWMAN = "kerr_newman"      # Rotating, charged


@dataclass
class BlackHole:
    """
    Black hole with configurable parameters.
    """
    mass: float              # kg
    spin: float = 0.0        # Dimensionless spin parameter a* = J/(Mc) ∈ [0, 1]
    charge: float = 0.0      # Coulombs (usually 0, charged BHs neutralize quickly)
    
    def __post_init__(self):
        self.G = BlackHoleConstants.G
        self.c = BlackHoleConstants.c
        self.c2 = BlackHoleConstants.c2
        self.hbar = BlackHoleConstants.hbar
        self.k_B = BlackHoleConstants.k_B
    
    @property
    def type(self) -> BlackHoleType:
        """Determine black hole type."""
        if self.spin == 0 and self.charge == 0:
            return BlackHoleType.SCHWARZSCHILD
        elif self.spin != 0 and self.charge == 0:
            return BlackHoleType.KERR
        elif self.spin == 0 and self.charge != 0:
            return BlackHoleType.REISSNER_NORDSTROM
        else:
            return BlackHoleType.KERR_NEWMAN
    
    @property
    def schwarzschild_radius(self) -> float:
        """
        Schwarzschild radius: r_s = 2GM/c²
        
        This is the event horizon for non-rotating black holes.
        """
        return 2 * self.G * self.mass / self.c2
    
    @property
    def r_s(self) -> float:
        """Alias for schwarzschild_radius."""
        return self.schwarzschild_radius
    
    @property
    def event_horizon_radius(self) -> float:
        """
        Event horizon radius.
        
        Schwarzschild: r = r_s
        Kerr: r = (r_s/2) + √((r_s/2)² - a²)
        where a = J/(Mc) = spin × GM/c²
        """
        if self.spin == 0:
            return self.r_s
        
        # Kerr metric
        m = self.G * self.mass / self.c2  # GM/c² (geometric mass)
        a = self.spin * m  # Spin parameter in geometric units
        
        return m + math.sqrt(m ** 2 - a ** 2)
    
    @property
    def inner_horizon_radius(self) -> Optional[float]:
        """
        Inner (Cauchy) horizon for Kerr black holes.
        r_- = m - √(m² - a²)
        """
        if self.spin == 0:
            return None
        
        m = self.G * self.mass / self.c2
        a = self.spin * m
        
        return m - math.sqrt(m ** 2 - a ** 2)
    
    @property
    def ergosphere_radius(self) -> float:
        """
        Ergosphere radius at equator (Kerr black holes).
        r_ergo = m + √(m² - a²cos²θ)
        
        At equator (θ = π/2): r_ergo = 2m = r_s
        """
        if self.spin == 0:
            return self.r_s
        
        return self.r_s  # At equator
    
    @property
    def photon_sphere_radius(self) -> float:
        """
        Photon sphere radius where light can orbit.
        
        Schwarzschild: r_ph = 1.5 r_s
        """
        if self.spin == 0:
            return 1.5 * self.r_s
        
        # Kerr - depends on orbit direction, return prograde
        m = self.G * self.mass / self.c2
        a = self.spin * m
        return 2 * m * (1 + math.cos(2 * math.acos(-a / m) / 3))
    
    @property
    def isco_radius(self) -> float:
        """
        Innermost Stable Circular Orbit (ISCO).
        
        Schwarzschild: r_isco = 3 r_s
        Kerr: depends on spin (prograde orbit)
        """
        if self.spin == 0:
            return 3 * self.r_s
        
        # Kerr - prograde orbit
        a = self.spin
        z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
        z2 = math.sqrt(3 * a**2 + z1**2)
        
        return self.r_s * (3 + z2 - math.sqrt((3 - z1) * (3 + z1 + 2 * z2))) / 2
    
    @property
    def hawking_temperature(self) -> float:
        """
        Hawking temperature of the black hole.
        
        T = ℏc³ / (8πGMk_B)
        
        Smaller black holes are hotter!
        """
        return (self.hbar * self.c ** 3) / (8 * math.pi * self.G * self.mass * self.k_B)
    
    @property
    def hawking_luminosity(self) -> float:
        """
        Hawking radiation power.
        
        P = ℏc⁶ / (15360 π G² M²)
        
        Extremely small for stellar-mass black holes.
        """
        return (self.hbar * self.c ** 6) / (15360 * math.pi * self.G ** 2 * self.mass ** 2)
    
    @property
    def evaporation_time(self) -> float:
        """
        Time for complete evaporation via Hawking radiation.
        
        t = 5120 π G² M³ / (ℏc⁴)
        """
        return (5120 * math.pi * self.G ** 2 * self.mass ** 3) / (self.hbar * self.c ** 4)
    
    @property
    def bekenstein_entropy(self) -> float:
        """
        Bekenstein-Hawking entropy.
        
        S = k_B c³ A / (4 G ℏ) = k_B A / (4 l_p²)
        
        where A = 4π r_s² is the horizon area.
        """
        A = 4 * math.pi * self.schwarzschild_radius ** 2
        l_p = BlackHoleConstants.l_p
        return self.k_B * A / (4 * l_p ** 2)
    
    @property
    def surface_gravity(self) -> float:
        """
        Surface gravity at event horizon.
        
        κ = c⁴/(4GM)  for Schwarzschild
        """
        if self.spin == 0:
            return self.c ** 4 / (4 * self.G * self.mass)
        
        # Kerr
        r_plus = self.event_horizon_radius
        m = self.G * self.mass / self.c2
        a = self.spin * m
        return (r_plus - m) / (2 * r_plus ** 2 + 2 * a ** 2) * self.c2


# =============================================================================
# BLACK HOLE PHYSICS CALCULATIONS
# =============================================================================

class BlackHolePhysics:
    """
    Physics calculations for black holes.
    """
    
    @staticmethod
    def escape_velocity(bh: BlackHole, r: float) -> float:
        """
        Calculate escape velocity at radius r.
        
        v_esc = √(2GM/r) = c√(r_s/r)
        
        At event horizon, v_esc = c.
        """
        if r <= bh.r_s:
            return BlackHoleConstants.c  # At or inside horizon, nothing escapes
        
        return BlackHoleConstants.c * math.sqrt(bh.r_s / r)
    
    @staticmethod
    def gravitational_redshift(bh: BlackHole, r_emit: float) -> float:
        """
        Calculate gravitational redshift z.
        
        z = 1/√(1 - r_s/r) - 1
        
        Light emitted near the black hole is redshifted.
        """
        if r_emit <= bh.r_s:
            return float('inf')  # Infinite redshift at horizon
        
        return 1 / math.sqrt(1 - bh.r_s / r_emit) - 1
    
    @staticmethod
    def time_dilation_factor(bh: BlackHole, r: float) -> float:
        """
        Calculate gravitational time dilation factor.
        
        dτ/dt = √(1 - r_s/r)
        
        Time runs slower closer to the black hole.
        """
        if r <= bh.r_s:
            return 0.0
        
        return math.sqrt(1 - bh.r_s / r)
    
    @staticmethod
    def proper_distance_to_horizon(bh: BlackHole, r: float) -> float:
        """
        Calculate proper distance from r to event horizon.
        
        This integral diverges as r → r_s in coordinate time,
        but proper distance is finite.
        """
        if r <= bh.r_s:
            return 0.0
        
        r_s = bh.r_s
        
        # Approximate proper distance
        d = math.sqrt(r * (r - r_s)) + r_s * math.log((math.sqrt(r) + math.sqrt(r - r_s)) / math.sqrt(r_s))
        
        return d
    
    @staticmethod
    def tidal_force(bh: BlackHole, r: float, delta_r: float) -> float:
        """
        Calculate tidal acceleration (difference in gravitational pull
        across an extended object).
        
        a_tidal = 2GM Δr / r³
        
        This causes "spaghettification" near the black hole.
        """
        G = BlackHoleConstants.G
        return 2 * G * bh.mass * delta_r / r ** 3
    
    @staticmethod
    def spaghettification_radius(bh: BlackHole, body_length: float = 2.0, 
                                  max_acceleration: float = 10 * 9.8) -> float:
        """
        Calculate radius at which tidal forces become lethal.
        
        For a human (~2m tall), survivable tidal acceleration ~10g.
        
        r = (2GM × L / a_max)^(1/3)
        """
        G = BlackHoleConstants.G
        return (2 * G * bh.mass * body_length / max_acceleration) ** (1/3)
    
    @staticmethod
    def orbital_period(bh: BlackHole, r: float) -> float:
        """
        Calculate orbital period at radius r.
        
        T = 2π √(r³ / GM) × relativistic correction
        """
        G = BlackHoleConstants.G
        
        if r <= bh.isco_radius:
            return float('inf')  # No stable orbits inside ISCO
        
        # Newtonian + GR correction
        T = 2 * math.pi * math.sqrt(r ** 3 / (G * bh.mass))
        
        # GR correction factor
        factor = 1 / math.sqrt(1 - 1.5 * bh.r_s / r)
        
        return T * factor
    
    @staticmethod
    def orbital_velocity(bh: BlackHole, r: float) -> float:
        """
        Calculate orbital velocity at radius r.
        
        v = √(GM/r) × √(1/(1 - 1.5 r_s/r))
        """
        G = BlackHoleConstants.G
        c = BlackHoleConstants.c
        
        if r <= bh.isco_radius:
            return c  # Speed of light at ISCO limit
        
        v_newton = math.sqrt(G * bh.mass / r)
        
        # GR correction
        correction = 1 / math.sqrt(1 - 1.5 * bh.r_s / r)
        
        return min(v_newton * correction, c)
    
    @staticmethod
    def penrose_energy_extraction(bh: BlackHole) -> float:
        """
        Maximum energy extractable from rotating black hole via Penrose process.
        
        Efficiency η = 1 - √(1 - (J/(Mc))²)
        
        Up to ~29% of mass-energy for maximally spinning black hole.
        """
        if bh.spin == 0:
            return 0.0
        
        return bh.mass * bh.c2 * (1 - math.sqrt(1 + math.sqrt(1 - bh.spin ** 2)) / math.sqrt(2))


# =============================================================================
# BLACK HOLE SIMULATOR
# =============================================================================

class BlackHoleSimulator:
    """
    Simulate particle trajectories near black holes.
    """
    
    def __init__(self, bh: BlackHole):
        self.bh = bh
        self.G = BlackHoleConstants.G
        self.c = BlackHoleConstants.c
    
    def simulate_radial_infall(self, r0: float, n_steps: int = 1000) -> List[Dict]:
        """
        Simulate radial free-fall into black hole.
        
        Proper time is finite to reach the horizon!
        """
        trajectory = []
        
        r = r0
        v = 0.0  # Start at rest
        tau = 0.0  # Proper time
        t = 0.0    # Coordinate time
        
        dt = 0.001  # Time step
        
        for _ in range(n_steps):
            if r <= self.bh.r_s * 1.001:
                break
            
            # Acceleration
            a = self.G * self.bh.mass / r ** 2
            
            # Relativistic correction
            factor = 1 - self.bh.r_s / r
            
            # Update
            v += a * dt
            if v > self.c:
                v = self.c * 0.999
            
            r -= v * dt
            
            # Proper time
            dtau = dt * math.sqrt(abs(factor - v ** 2 / self.c ** 2))
            tau += dtau
            t += dt
            
            trajectory.append({
                'coordinate_time': t,
                'proper_time': tau,
                'radius': r,
                'velocity': v,
                'r_over_rs': r / self.bh.r_s
            })
        
        return trajectory
    
    def simulate_circular_orbit(self, r0: float, n_orbits: int = 1) -> List[Dict]:
        """
        Simulate stable circular orbit around black hole.
        """
        if r0 < self.bh.isco_radius:
            return []  # No stable circular orbit
        
        trajectory = []
        
        period = BlackHolePhysics.orbital_period(self.bh, r0)
        v = BlackHolePhysics.orbital_velocity(self.bh, r0)
        
        dt = period / 100
        t = 0.0
        theta = 0.0
        
        n_steps = int(n_orbits * period / dt)
        
        for _ in range(n_steps):
            x = r0 * math.cos(theta)
            y = r0 * math.sin(theta)
            
            # Proper time experienced by orbiting observer
            dtau = dt * BlackHolePhysics.time_dilation_factor(self.bh, r0)
            
            trajectory.append({
                'time': t,
                'x': x,
                'y': y,
                'theta': theta,
                'proper_time_elapsed': dtau * len(trajectory)
            })
            
            omega = v / r0
            theta += omega * dt
            t += dt
        
        return trajectory


# =============================================================================
# BLACK HOLE ENGINE - MAIN INTERFACE
# =============================================================================

class BlackHoleEngine:
    """
    AION Black Hole Engine for calculations and simulations.
    """
    
    def __init__(self):
        self.M_sun = BlackHoleConstants.M_sun
    
    def create_black_hole(self, mass_solar: float, spin: float = 0) -> BlackHole:
        """Create a black hole with given mass in solar masses."""
        return BlackHole(mass_solar * self.M_sun, spin)
    
    def analyze(self, mass_solar: float, spin: float = 0) -> Dict:
        """Complete analysis of a black hole."""
        bh = self.create_black_hole(mass_solar, spin)
        
        return {
            'type': bh.type.value,
            'mass_kg': bh.mass,
            'mass_solar': mass_solar,
            'spin': spin,
            
            # Geometry
            'schwarzschild_radius_km': bh.schwarzschild_radius / 1000,
            'event_horizon_km': bh.event_horizon_radius / 1000,
            'photon_sphere_km': bh.photon_sphere_radius / 1000,
            'isco_km': bh.isco_radius / 1000,
            
            # Thermodynamics
            'hawking_temperature_K': bh.hawking_temperature,
            'hawking_luminosity_W': bh.hawking_luminosity,
            'evaporation_time_years': bh.evaporation_time / (365.25 * 24 * 3600),
            'entropy_bits': bh.bekenstein_entropy / (BlackHoleConstants.k_B * math.log(2)),
            
            # Physics
            'surface_gravity_m_s2': bh.surface_gravity,
        }
    
    def tidal_effects(self, mass_solar: float, distance_km: float, body_size_m: float = 2.0) -> Dict:
        """Calculate tidal effects at given distance."""
        bh = self.create_black_hole(mass_solar)
        r = distance_km * 1000
        
        tidal_accel = BlackHolePhysics.tidal_force(bh, r, body_size_m)
        safe_radius = BlackHolePhysics.spaghettification_radius(bh, body_size_m)
        
        return {
            'distance_km': distance_km,
            'distance_rs': r / bh.r_s,
            'tidal_acceleration_g': tidal_accel / 9.8,
            'is_survivable': r > safe_radius,
            'spaghettification_radius_km': safe_radius / 1000,
            'spaghettification_inside_horizon': safe_radius < bh.r_s
        }
    
    def time_dilation(self, mass_solar: float, distance_km: float) -> Dict:
        """Calculate time dilation effects."""
        bh = self.create_black_hole(mass_solar)
        r = distance_km * 1000
        
        factor = BlackHolePhysics.time_dilation_factor(bh, r)
        redshift = BlackHolePhysics.gravitational_redshift(bh, r)
        
        return {
            'distance_km': distance_km,
            'distance_rs': r / bh.r_s,
            'time_dilation_factor': factor,
            'clock_rate_percent': factor * 100,
            'gravitational_redshift_z': redshift
        }
    
    def compare_sizes(self) -> Dict:
        """Compare black holes of different masses."""
        masses = [1, 10, 100, 1e6, 4e6, 1e9]  # Solar masses
        
        results = []
        for m in masses:
            bh = self.create_black_hole(m)
            results.append({
                'mass_solar': m,
                'rs_km': bh.r_s / 1000,
                'hawking_temp_K': bh.hawking_temperature,
                'evap_time_years': bh.evaporation_time / (365.25 * 24 * 3600)
            })
        
        return {'black_holes': results}


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Black Hole Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🕳️ AION BLACK HOLE ENGINE 🕳️                                    ║
║                                                                           ║
║     Event Horizons, Hawking Radiation, Tidal Forces                      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = BlackHoleEngine()
    
    # Stellar black hole
    print("⭐ Stellar Black Hole (10 M☉):")
    print("-" * 50)
    result = engine.analyze(10)
    print(f"   Schwarzschild radius: {result['schwarzschild_radius_km']:.1f} km")
    print(f"   Photon sphere: {result['photon_sphere_km']:.1f} km")
    print(f"   ISCO: {result['isco_km']:.1f} km")
    print(f"   Hawking temperature: {result['hawking_temperature_K']:.2e} K")
    print(f"   Evaporation time: {result['evaporation_time_years']:.2e} years")
    
    # Sagittarius A*
    print("\n🌌 Sagittarius A* (4 million M☉):")
    print("-" * 50)
    result = engine.analyze(4e6)
    print(f"   Event horizon: {result['event_horizon_km']/1e6:.2f} million km")
    print(f"   Event horizon: {result['event_horizon_km']/(1.496e8):.2f} AU")
    print(f"   Hawking temperature: {result['hawking_temperature_K']:.2e} K")
    
    # Rotating black hole
    print("\n🌀 Kerr Black Hole (10 M☉, a* = 0.9):")
    print("-" * 50)
    result = engine.analyze(10, spin=0.9)
    print(f"   Event horizon: {result['event_horizon_km']:.1f} km")
    print(f"   ISCO (prograde): {result['isco_km']:.1f} km")
    
    # Tidal forces
    print("\n💀 Tidal Effects (10 M☉, 2m human):")
    print("-" * 50)
    result = engine.tidal_effects(10, 100, 2.0)
    print(f"   At 100 km from center:")
    print(f"   Distance: {result['distance_rs']:.1f} × r_s")
    print(f"   Tidal acceleration: {result['tidal_acceleration_g']:.0f} g")
    print(f"   Survivable: {result['is_survivable']}")
    print(f"   Spaghettification radius: {result['spaghettification_radius_km']:.0f} km")
    print(f"   (Inside horizon: {result['spaghettification_inside_horizon']})")
    
    # Supermassive vs stellar
    print("\n🔍 Supermassive is safer to approach!")
    print("-" * 50)
    for mass in [10, 1e6]:
        result = engine.tidal_effects(mass, distance_km=30, body_size_m=2.0)
        bh = engine.create_black_hole(mass)
        print(f"   {mass} M☉: Tidal force at 30km = {result['tidal_acceleration_g']:.0e} g")
    
    # Time dilation
    print("\n⏰ Time Dilation near 10 M☉ black hole:")
    print("-" * 50)
    for dist in [100, 50, 35, 30]:
        result = engine.time_dilation(10, dist)
        print(f"   {dist} km ({result['distance_rs']:.1f} r_s): "
              f"clocks run at {result['clock_rate_percent']:.1f}% speed")


if __name__ == "__main__":
    demo()
