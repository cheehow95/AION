"""
AION Physics Simulation Domain Engine
======================================

A physics simulation engine for AION agents to explore and understand
physical phenomena through simulation.

Features:
- Classical mechanics (projectiles, orbits, pendulums)
- Electromagnetic fields and forces
- Wave mechanics
- Thermodynamics basics
- Quantum mechanics concepts
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class Constants:
    """Fundamental physical constants (SI units)."""
    
    # Mechanics
    G = 6.67430e-11       # Gravitational constant (m³/kg/s²)
    g = 9.80665           # Standard gravity (m/s²)
    c = 299792458         # Speed of light (m/s)
    
    # Electromagnetism
    k_e = 8.9875517923e9  # Coulomb's constant (N·m²/C²)
    epsilon_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
    mu_0 = 1.25663706212e-6       # Vacuum permeability (H/m)
    e = 1.602176634e-19   # Elementary charge (C)
    
    # Thermodynamics
    k_B = 1.380649e-23    # Boltzmann constant (J/K)
    R = 8.314462618       # Gas constant (J/mol/K)
    N_A = 6.02214076e23   # Avogadro's number (1/mol)
    
    # Quantum
    h = 6.62607015e-34    # Planck constant (J·s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    
    # Particles
    m_e = 9.1093837015e-31   # Electron mass (kg)
    m_p = 1.67262192369e-27  # Proton mass (kg)
    m_n = 1.67492749804e-27  # Neutron mass (kg)


# =============================================================================
# VECTOR OPERATIONS
# =============================================================================

@dataclass
class Vector3:
    """3D vector for physics calculations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self) -> 'Vector3':
        return Vector3(-self.x, -self.y, -self.z)
    
    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3') -> 'Vector3':
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3':
        mag = self.magnitude()
        if mag < 1e-10:
            return Vector3(0, 0, 0)
        return self / mag
    
    def __repr__(self):
        return f"({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


# =============================================================================
# PARTICLE DYNAMICS
# =============================================================================

@dataclass
class Particle:
    """A point mass with position, velocity, and properties."""
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    acceleration: Vector3 = field(default_factory=Vector3)
    mass: float = 1.0
    charge: float = 0.0
    name: str = "particle"
    
    def kinetic_energy(self) -> float:
        """Calculate kinetic energy: KE = ½mv²"""
        return 0.5 * self.mass * self.velocity.dot(self.velocity)
    
    def momentum(self) -> Vector3:
        """Calculate momentum: p = mv"""
        return self.velocity * self.mass
    
    def apply_force(self, force: Vector3, dt: float):
        """Apply force for time dt using F = ma."""
        self.acceleration = force / self.mass
        self.velocity = self.velocity + self.acceleration * dt
        self.position = self.position + self.velocity * dt


# =============================================================================
# CLASSICAL MECHANICS SIMULATIONS
# =============================================================================

class ProjectileMotion:
    """
    Simulate projectile motion under gravity.
    Includes optional air resistance.
    """
    
    def __init__(self, mass: float = 1.0, drag_coefficient: float = 0.0):
        self.mass = mass
        self.drag_coefficient = drag_coefficient
        self.g = Constants.g
    
    def simulate(self, v0: float, angle: float, dt: float = 0.01,
                 max_time: float = 100.0) -> List[Tuple[float, float, float]]:
        """
        Simulate projectile trajectory.
        
        Args:
            v0: Initial velocity (m/s)
            angle: Launch angle (degrees)
            dt: Time step (s)
            max_time: Maximum simulation time (s)
        
        Returns:
            List of (time, x, y) tuples
        """
        angle_rad = math.radians(angle)
        vx = v0 * math.cos(angle_rad)
        vy = v0 * math.sin(angle_rad)
        
        x, y = 0.0, 0.0
        t = 0.0
        trajectory = [(t, x, y)]
        
        while y >= 0 and t < max_time:
            # Calculate forces
            fx = -self.drag_coefficient * vx
            fy = -self.mass * self.g - self.drag_coefficient * vy
            
            # Update velocity
            vx += (fx / self.mass) * dt
            vy += (fy / self.mass) * dt
            
            # Update position
            x += vx * dt
            y += vy * dt
            t += dt
            
            trajectory.append((t, x, y))
        
        return trajectory
    
    def max_range(self, v0: float) -> float:
        """Calculate maximum range (at 45°, no drag)."""
        return (v0 ** 2) / self.g
    
    def max_height(self, v0: float, angle: float) -> float:
        """Calculate maximum height."""
        vy = v0 * math.sin(math.radians(angle))
        return (vy ** 2) / (2 * self.g)


class OrbitalMechanics:
    """
    Simulate orbital mechanics using Newton's law of gravitation.
    """
    
    def __init__(self, central_mass: float = 5.972e24):  # Earth mass by default
        self.M = central_mass
        self.G = Constants.G
    
    def orbital_velocity(self, radius: float) -> float:
        """Calculate circular orbital velocity: v = √(GM/r)"""
        return math.sqrt(self.G * self.M / radius)
    
    def orbital_period(self, semi_major_axis: float) -> float:
        """Calculate orbital period using Kepler's third law: T = 2π√(a³/GM)"""
        return 2 * math.pi * math.sqrt(semi_major_axis**3 / (self.G * self.M))
    
    def escape_velocity(self, radius: float) -> float:
        """Calculate escape velocity: v_esc = √(2GM/r)"""
        return math.sqrt(2 * self.G * self.M / radius)
    
    def simulate_orbit(self, r0: float, v0: float, dt: float = 100.0,
                       n_steps: int = 1000) -> List[Tuple[float, float]]:
        """
        Simulate orbit starting with radial position and tangential velocity.
        
        Returns list of (x, y) positions.
        """
        # Initial conditions (starting on x-axis, moving in +y direction)
        x, y = r0, 0.0
        vx, vy = 0.0, v0
        
        trajectory = [(x, y)]
        
        for _ in range(n_steps):
            # Distance from center
            r = math.sqrt(x*x + y*y)
            
            # Gravitational acceleration
            a = -self.G * self.M / (r**3)
            ax = a * x
            ay = a * y
            
            # Velocity Verlet integration
            x_new = x + vx * dt + 0.5 * ax * dt**2
            y_new = y + vy * dt + 0.5 * ay * dt**2
            
            r_new = math.sqrt(x_new*x_new + y_new*y_new)
            a_new = -self.G * self.M / (r_new**3)
            ax_new = a_new * x_new
            ay_new = a_new * y_new
            
            vx = vx + 0.5 * (ax + ax_new) * dt
            vy = vy + 0.5 * (ay + ay_new) * dt
            
            x, y = x_new, y_new
            trajectory.append((x, y))
        
        return trajectory


class SimplePendulum:
    """
    Simulate a simple pendulum.
    """
    
    def __init__(self, length: float = 1.0, g: float = None):
        self.length = length
        self.g = g or Constants.g
    
    def period(self) -> float:
        """Calculate period for small oscillations: T = 2π√(L/g)"""
        return 2 * math.pi * math.sqrt(self.length / self.g)
    
    def simulate(self, theta0: float, omega0: float = 0.0,
                 dt: float = 0.01, n_steps: int = 1000) -> List[Tuple[float, float]]:
        """
        Simulate pendulum motion.
        
        Args:
            theta0: Initial angle (radians)
            omega0: Initial angular velocity (rad/s)
            dt: Time step (s)
        
        Returns:
            List of (time, angle) tuples
        """
        theta = theta0
        omega = omega0
        t = 0.0
        
        trajectory = [(t, theta)]
        
        for _ in range(n_steps):
            # Angular acceleration: α = -(g/L)sin(θ)
            alpha = -(self.g / self.length) * math.sin(theta)
            
            # Update using Euler-Cromer method
            omega += alpha * dt
            theta += omega * dt
            t += dt
            
            trajectory.append((t, theta))
        
        return trajectory


# =============================================================================
# ELECTROMAGNETISM
# =============================================================================

class ElectricField:
    """
    Calculate electric fields from point charges.
    """
    
    def __init__(self):
        self.k = Constants.k_e
        self.charges: List[Tuple[Vector3, float]] = []  # (position, charge)
    
    def add_charge(self, position: Vector3, charge: float):
        """Add a point charge."""
        self.charges.append((position, charge))
    
    def field_at(self, point: Vector3) -> Vector3:
        """
        Calculate electric field at a point.
        E = kq/r² (radially outward for positive charge)
        """
        E = Vector3(0, 0, 0)
        
        for pos, q in self.charges:
            r_vec = point - pos
            r = r_vec.magnitude()
            
            if r < 1e-10:
                continue  # Skip if at charge location
            
            E_mag = self.k * q / (r**2)
            E = E + r_vec.normalize() * E_mag
        
        return E
    
    def potential_at(self, point: Vector3) -> float:
        """
        Calculate electric potential at a point.
        V = kq/r
        """
        V = 0.0
        
        for pos, q in self.charges:
            r = (point - pos).magnitude()
            if r > 1e-10:
                V += self.k * q / r
        
        return V
    
    def force_on_charge(self, position: Vector3, charge: float) -> Vector3:
        """Calculate force on a test charge: F = qE"""
        E = self.field_at(position)
        return E * charge


class MagneticField:
    """
    Calculate magnetic fields and Lorentz force.
    """
    
    def __init__(self, B: Vector3 = None):
        self.B = B or Vector3(0, 0, 0)  # Uniform field
    
    def lorentz_force(self, q: float, v: Vector3, E: Vector3 = None) -> Vector3:
        """
        Calculate Lorentz force: F = q(E + v × B)
        """
        E = E or Vector3(0, 0, 0)
        return (E + v.cross(self.B)) * q
    
    def cyclotron_frequency(self, q: float, m: float) -> float:
        """
        Calculate cyclotron frequency: ω = qB/m
        """
        return abs(q) * self.B.magnitude() / m
    
    def larmor_radius(self, q: float, m: float, v_perp: float) -> float:
        """
        Calculate Larmor (gyration) radius: r = mv_⊥/(qB)
        """
        B_mag = self.B.magnitude()
        if B_mag < 1e-10:
            return float('inf')
        return m * v_perp / (abs(q) * B_mag)


# =============================================================================
# WAVE MECHANICS
# =============================================================================

class Wave:
    """
    Model wave phenomena.
    """
    
    def __init__(self, amplitude: float = 1.0, wavelength: float = 1.0,
                 frequency: float = 1.0, phase: float = 0.0):
        self.A = amplitude
        self.wavelength = wavelength
        self.f = frequency
        self.phi = phase
    
    @property
    def k(self) -> float:
        """Wave number: k = 2π/λ"""
        return 2 * math.pi / self.wavelength
    
    @property
    def omega(self) -> float:
        """Angular frequency: ω = 2πf"""
        return 2 * math.pi * self.f
    
    @property
    def velocity(self) -> float:
        """Wave velocity: v = fλ"""
        return self.f * self.wavelength
    
    @property
    def period(self) -> float:
        """Period: T = 1/f"""
        return 1.0 / self.f
    
    def displacement(self, x: float, t: float) -> float:
        """
        Calculate wave displacement: y = A sin(kx - ωt + φ)
        """
        return self.A * math.sin(self.k * x - self.omega * t + self.phi)
    
    def energy_density(self) -> float:
        """
        Energy density for harmonic wave (proportional to A²)
        """
        return 0.5 * self.A**2  # Simplified, missing medium properties
    
    @staticmethod
    def superposition(waves: List['Wave'], x: float, t: float) -> float:
        """Calculate superposition of multiple waves."""
        return sum(w.displacement(x, t) for w in waves)
    
    @staticmethod
    def standing_wave(wave1: 'Wave', x: float, t: float) -> float:
        """
        Create standing wave from incident and reflected waves.
        Standing wave: y = 2A sin(kx) cos(ωt)
        """
        A = wave1.A
        k = wave1.k
        omega = wave1.omega
        return 2 * A * math.sin(k * x) * math.cos(omega * t)


# =============================================================================
# THERMODYNAMICS
# =============================================================================

class IdealGas:
    """
    Model ideal gas behavior.
    """
    
    def __init__(self, n_moles: float = 1.0):
        self.n = n_moles
        self.R = Constants.R
    
    def pressure(self, V: float, T: float) -> float:
        """Calculate pressure: PV = nRT → P = nRT/V"""
        return self.n * self.R * T / V
    
    def volume(self, P: float, T: float) -> float:
        """Calculate volume: V = nRT/P"""
        return self.n * self.R * T / P
    
    def temperature(self, P: float, V: float) -> float:
        """Calculate temperature: T = PV/(nR)"""
        return P * V / (self.n * self.R)
    
    def internal_energy(self, T: float, dof: int = 3) -> float:
        """
        Calculate internal energy for ideal gas.
        U = (dof/2) * nRT
        dof = 3 for monatomic, 5 for diatomic
        """
        return (dof / 2) * self.n * self.R * T
    
    def heat_capacity_v(self, dof: int = 3) -> float:
        """Heat capacity at constant volume: Cv = (dof/2)nR"""
        return (dof / 2) * self.n * self.R
    
    def heat_capacity_p(self, dof: int = 3) -> float:
        """Heat capacity at constant pressure: Cp = Cv + nR"""
        return self.heat_capacity_v(dof) + self.n * self.R


class Entropy:
    """Entropy calculations."""
    
    @staticmethod
    def boltzmann(W: int) -> float:
        """Boltzmann entropy: S = k_B ln(W)"""
        if W <= 0:
            return float('-inf')
        return Constants.k_B * math.log(W)
    
    @staticmethod
    def ideal_gas_change(n: float, T1: float, T2: float, V1: float, V2: float) -> float:
        """
        Entropy change for ideal gas:
        ΔS = nCv*ln(T2/T1) + nR*ln(V2/V1)
        """
        Cv = 1.5 * Constants.R  # Monatomic
        return n * Cv * math.log(T2/T1) + n * Constants.R * math.log(V2/V1)


# =============================================================================
# QUANTUM MECHANICS (CONCEPTUAL)
# =============================================================================

class QuantumMechanics:
    """
    Basic quantum mechanics calculations and concepts.
    """
    
    @staticmethod
    def de_broglie_wavelength(momentum: float) -> float:
        """de Broglie wavelength: λ = h/p"""
        return Constants.h / momentum
    
    @staticmethod
    def photon_energy(frequency: float) -> float:
        """Photon energy: E = hf"""
        return Constants.h * frequency
    
    @staticmethod
    def photon_frequency(energy: float) -> float:
        """Photon frequency from energy: f = E/h"""
        return energy / Constants.h
    
    @staticmethod
    def heisenberg_uncertainty(delta_x: float) -> float:
        """
        Heisenberg uncertainty principle.
        Given Δx, return minimum Δp: Δx·Δp ≥ ℏ/2
        """
        return Constants.hbar / (2 * delta_x)
    
    @staticmethod
    def hydrogen_energy(n: int) -> float:
        """
        Hydrogen atom energy levels: E_n = -13.6 eV / n²
        Returns energy in Joules.
        """
        E_1 = -13.6 * Constants.e  # Ground state in Joules
        return E_1 / (n**2)
    
    @staticmethod
    def particle_in_box_energy(n: int, L: float, m: float) -> float:
        """
        Energy levels for particle in 1D infinite well:
        E_n = n²h²/(8mL²)
        """
        return (n**2 * Constants.h**2) / (8 * m * L**2)


# =============================================================================
# PHYSICS ENGINE - Main Interface
# =============================================================================

class PhysicsEngine:
    """
    AION Physics Engine for simulations and calculations.
    """
    
    def __init__(self):
        self.history: List[Dict] = []
    
    # Mechanics
    def projectile(self, v0: float, angle: float, drag: float = 0.0) -> Dict:
        """Simulate projectile motion."""
        sim = ProjectileMotion(drag_coefficient=drag)
        trajectory = sim.simulate(v0, angle)
        
        max_range = trajectory[-1][1] if len(trajectory) > 1 else 0
        max_height = max(t[2] for t in trajectory)
        flight_time = trajectory[-1][0]
        
        return {
            "max_range": max_range,
            "max_height": max_height,
            "flight_time": flight_time,
            "trajectory": trajectory[:100]  # Limit points
        }
    
    def pendulum(self, length: float, theta0: float) -> Dict:
        """Simulate pendulum."""
        sim = SimplePendulum(length=length)
        trajectory = sim.simulate(math.radians(theta0))
        
        return {
            "period": sim.period(),
            "trajectory": trajectory[:200]
        }
    
    def orbital(self, radius: float, central_mass: float = 5.972e24) -> Dict:
        """Calculate orbital parameters."""
        sim = OrbitalMechanics(central_mass)
        
        return {
            "orbital_velocity": sim.orbital_velocity(radius),
            "orbital_period": sim.orbital_period(radius),
            "escape_velocity": sim.escape_velocity(radius)
        }
    
    # Electromagnetism
    def electric_field(self, charges: List[Tuple[Tuple, float]], point: Tuple) -> Dict:
        """Calculate electric field from point charges."""
        field = ElectricField()
        for pos, q in charges:
            field.add_charge(Vector3(*pos), q)
        
        test_point = Vector3(*point)
        E = field.field_at(test_point)
        V = field.potential_at(test_point)
        
        return {
            "field": (E.x, E.y, E.z),
            "field_magnitude": E.magnitude(),
            "potential": V
        }
    
    # Waves
    def wave_properties(self, amplitude: float, wavelength: float, frequency: float) -> Dict:
        """Calculate wave properties."""
        wave = Wave(amplitude, wavelength, frequency)
        
        return {
            "wave_number": wave.k,
            "angular_frequency": wave.omega,
            "velocity": wave.velocity,
            "period": wave.period
        }
    
    # Thermodynamics
    def ideal_gas_state(self, n_moles: float, P: float = None, V: float = None, T: float = None) -> Dict:
        """Calculate ideal gas state variables."""
        gas = IdealGas(n_moles)
        
        if T is None:
            T = gas.temperature(P, V)
        elif P is None:
            P = gas.pressure(V, T)
        elif V is None:
            V = gas.volume(P, T)
        
        return {
            "pressure": P,
            "volume": V,
            "temperature": T,
            "internal_energy": gas.internal_energy(T)
        }
    
    # Quantum
    def quantum_calculations(self, particle_mass: float = None, velocity: float = None,
                            photon_wavelength: float = None) -> Dict:
        """Various quantum mechanics calculations."""
        results = {}
        qm = QuantumMechanics
        
        if particle_mass and velocity:
            p = particle_mass * velocity
            results["de_broglie_wavelength"] = qm.de_broglie_wavelength(p)
            results["kinetic_energy"] = 0.5 * particle_mass * velocity**2
        
        if photon_wavelength:
            c = Constants.c
            f = c / photon_wavelength
            results["photon_frequency"] = f
            results["photon_energy"] = qm.photon_energy(f)
            results["photon_energy_eV"] = qm.photon_energy(f) / Constants.e
        
        return results


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Physics Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🔬 AION PHYSICS ENGINE 🔬                                        ║
║                                                                           ║
║     Classical mechanics, electromagnetism, waves, and quantum             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = PhysicsEngine()
    
    # Projectile motion
    print("🎯 Projectile Motion:")
    print("-" * 50)
    result = engine.projectile(v0=20, angle=45)
    print(f"   Initial velocity: 20 m/s at 45°")
    print(f"   Max range: {result['max_range']:.2f} m")
    print(f"   Max height: {result['max_height']:.2f} m")
    print(f"   Flight time: {result['flight_time']:.2f} s")
    
    # Pendulum
    print("\n⏰ Simple Pendulum:")
    print("-" * 50)
    result = engine.pendulum(length=1.0, theta0=10)
    print(f"   Length: 1.0 m, Initial angle: 10°")
    print(f"   Period: {result['period']:.4f} s")
    
    # Orbital mechanics
    print("\n🌍 Orbital Mechanics (Earth orbit):")
    print("-" * 50)
    LEO = 6.371e6 + 400e3  # Low Earth orbit (400 km altitude)
    result = engine.orbital(radius=LEO)
    print(f"   Altitude: 400 km")
    print(f"   Orbital velocity: {result['orbital_velocity']:.0f} m/s ({result['orbital_velocity']/1000:.2f} km/s)")
    print(f"   Orbital period: {result['orbital_period']/60:.1f} minutes")
    print(f"   Escape velocity: {result['escape_velocity']:.0f} m/s")
    
    # Electric field
    print("\n⚡ Electric Field:")
    print("-" * 50)
    charges = [((0, 0, 0), 1e-9), ((0.1, 0, 0), -1e-9)]  # Dipole
    point = (0.05, 0.05, 0)
    result = engine.electric_field(charges, point)
    print(f"   Dipole charges: +1nC at origin, -1nC at (0.1, 0, 0)")
    print(f"   Field at (0.05, 0.05, 0):")
    print(f"      |E| = {result['field_magnitude']:.2e} N/C")
    print(f"      V = {result['potential']:.2f} V")
    
    # Wave properties
    print("\n🌊 Wave Properties:")
    print("-" * 50)
    result = engine.wave_properties(amplitude=1.0, wavelength=2.0, frequency=5.0)
    print(f"   Wavelength: 2.0 m, Frequency: 5.0 Hz")
    print(f"   Wave velocity: {result['velocity']:.1f} m/s")
    print(f"   Period: {result['period']:.2f} s")
    
    # Ideal gas
    print("\n🌡️ Ideal Gas (1 mol):")
    print("-" * 50)
    result = engine.ideal_gas_state(n_moles=1.0, P=101325, T=300)
    print(f"   P = 101.3 kPa, T = 300 K")
    print(f"   V = {result['volume']*1000:.2f} L")
    print(f"   U = {result['internal_energy']:.0f} J")
    
    # Quantum
    print("\n⚛️ Quantum Mechanics:")
    print("-" * 50)
    result = engine.quantum_calculations(
        particle_mass=Constants.m_e,
        velocity=1e6,
        photon_wavelength=500e-9
    )
    print(f"   Electron at 1,000 km/s:")
    print(f"      de Broglie λ = {result['de_broglie_wavelength']*1e12:.3f} pm")
    print(f"   Green light (500 nm):")
    print(f"      Energy = {result['photon_energy_eV']:.2f} eV")
    
    # Physical constants
    print("\n📏 Physical Constants:")
    print("-" * 50)
    print(f"   Speed of light c = {Constants.c:,.0f} m/s")
    print(f"   Gravitational constant G = {Constants.G:.4e} m³/kg/s²")
    print(f"   Planck constant h = {Constants.h:.4e} J·s")
    print(f"   Boltzmann constant k_B = {Constants.k_B:.4e} J/K")


if __name__ == "__main__":
    demo()
