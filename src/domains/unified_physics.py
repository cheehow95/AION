"""
AION Unified Physics API
========================

High-level interface that integrates all AION physics domain engines:
- Classical mechanics (physics_engine)
- Optics (optics_engine)  
- Relativity (relativity_engine)
- Black holes (blackhole_engine)
- Wormholes (wormhole_engine)
- Multi-dimensional space (dimensions_engine)

Provides a single entry point for physics reasoning and calculations.
"""

from typing import Dict, List, Optional, Any
import math


# =============================================================================
# LAZY IMPORTS (to avoid circular dependencies)
# =============================================================================

_engines = {}

def _get_engine(name: str):
    """Lazy-load physics engines."""
    global _engines
    
    if name not in _engines:
        if name == 'physics':
            from src.domains.physics_engine import PhysicsEngine
            _engines[name] = PhysicsEngine()
        elif name == 'optics':
            from src.domains.optics_engine import OpticsEngine
            _engines[name] = OpticsEngine()
        elif name == 'relativity':
            from src.domains.relativity_engine import RelativityEngine
            _engines[name] = RelativityEngine()
        elif name == 'blackhole':
            from src.domains.blackhole_engine import BlackHoleEngine
            _engines[name] = BlackHoleEngine()
        elif name == 'wormhole':
            from src.domains.wormhole_engine import WormholeEngine
            _engines[name] = WormholeEngine()
        elif name == 'dimensions':
            from src.domains.dimensions_engine import DimensionsEngine
            _engines[name] = DimensionsEngine()
        elif name == 'quantum':
            from src.domains.quantum_engine import QuantumEngine
            _engines[name] = QuantumEngine()
        elif name == 'particle':
            from src.domains.particle_engine import ParticleEngine
            _engines[name] = ParticleEngine()
        elif name == 'nuclear':
            from src.domains.nuclear_engine import NuclearEngine
            _engines[name] = NuclearEngine()
        elif name == 'elements':
            from src.domains.elements_engine import ElementsEngine
            _engines[name] = ElementsEngine()
        elif name == 'qcompute':
            from src.domains.quantum_computing_engine import QuantumComputingEngine
            _engines[name] = QuantumComputingEngine()
    
    return _engines.get(name)


# =============================================================================
# UNIFIED PHYSICS ENGINE
# =============================================================================

class UnifiedPhysicsEngine:
    """
    AION Unified Physics Engine.
    
    Single interface to all physics domains for AI reasoning.
    """
    
    # Physical constants
    CONSTANTS = {
        'c': 299792458,           # Speed of light (m/s)
        'G': 6.67430e-11,         # Gravitational constant
        'h': 6.62607015e-34,      # Planck constant
        'hbar': 1.054571817e-34,  # Reduced Planck constant
        'k_B': 1.380649e-23,      # Boltzmann constant
        'e': 1.602176634e-19,     # Elementary charge
        'epsilon_0': 8.8541878128e-12,  # Vacuum permittivity
        'mu_0': 1.25663706212e-6,       # Vacuum permeability
        'm_e': 9.1093837015e-31,  # Electron mass
        'm_p': 1.67262192369e-27, # Proton mass
        'N_A': 6.02214076e23,     # Avogadro's number
        'R': 8.314462618,         # Gas constant
        'g': 9.80665,             # Standard gravity
        'M_sun': 1.989e30,        # Solar mass
        'M_earth': 5.972e24,      # Earth mass
        'R_earth': 6.371e6,       # Earth radius
    }
    
    def __init__(self):
        self.history: List[Dict] = []
    
    # =========================================================================
    # CLASSICAL MECHANICS
    # =========================================================================
    
    def projectile(self, velocity: float, angle: float, drag: float = 0) -> Dict:
        """Simulate projectile motion."""
        engine = _get_engine('physics')
        return engine.projectile(velocity, angle, drag)
    
    def pendulum(self, length: float, initial_angle: float) -> Dict:
        """Simulate simple pendulum."""
        engine = _get_engine('physics')
        return engine.pendulum(length, initial_angle)
    
    def orbital(self, radius: float, central_mass: float = None) -> Dict:
        """Calculate orbital parameters."""
        engine = _get_engine('physics')
        central_mass = central_mass or self.CONSTANTS['M_earth']
        return engine.orbital(radius, central_mass)
    
    # =========================================================================
    # OPTICS
    # =========================================================================
    
    def refraction(self, n1: float, n2: float, angle_degrees: float) -> Dict:
        """Calculate refraction at interface."""
        engine = _get_engine('optics')
        return engine.refraction(n1, n2, angle_degrees)
    
    def lens(self, focal_length: float, object_distance: float, 
             object_height: float = 1.0) -> Dict:
        """Analyze thin lens imaging."""
        engine = _get_engine('optics')
        return engine.lens_analysis(focal_length, object_distance, object_height)
    
    def interference(self, wavelength: float, slit_separation: float,
                     screen_distance: float) -> Dict:
        """Calculate double-slit interference pattern."""
        engine = _get_engine('optics')
        return engine.interference_pattern(wavelength, slit_separation, screen_distance)
    
    def laser_beam(self, wavelength: float, beam_waist: float, distance: float) -> Dict:
        """Calculate Gaussian laser beam properties."""
        engine = _get_engine('optics')
        return engine.laser_beam(wavelength, beam_waist, distance)
    
    # =========================================================================
    # RELATIVITY
    # =========================================================================
    
    def lorentz_factor(self, velocity: float) -> float:
        """Calculate Lorentz factor γ."""
        c = self.CONSTANTS['c']
        beta = velocity / c
        if beta >= 1:
            return float('inf')
        return 1 / math.sqrt(1 - beta**2)
    
    def time_dilation(self, proper_time: float, velocity: float) -> Dict:
        """Calculate time dilation effects."""
        engine = _get_engine('relativity')
        return engine.time_dilation(proper_time, velocity)
    
    def length_contraction(self, proper_length: float, velocity: float) -> Dict:
        """Calculate length contraction effects."""
        engine = _get_engine('relativity')
        return engine.length_contraction(proper_length, velocity)
    
    def twin_paradox(self, distance_ly: float, velocity_fraction_c: float) -> Dict:
        """Calculate twin paradox scenario."""
        engine = _get_engine('relativity')
        return engine.twin_paradox(distance_ly, velocity_fraction_c)
    
    def gravitational_effects(self, mass: float, radius: float) -> Dict:
        """Calculate GR effects near massive object."""
        engine = _get_engine('relativity')
        return engine.gravitational_effects(mass, radius)
    
    # =========================================================================
    # BLACK HOLES
    # =========================================================================
    
    def black_hole(self, mass_solar: float, spin: float = 0) -> Dict:
        """Analyze black hole properties."""
        engine = _get_engine('blackhole')
        return engine.analyze(mass_solar, spin)
    
    def schwarzschild_radius(self, mass: float) -> float:
        """Calculate Schwarzschild radius: r_s = 2GM/c²"""
        G = self.CONSTANTS['G']
        c = self.CONSTANTS['c']
        return 2 * G * mass / (c * c)
    
    def hawking_temperature(self, mass: float) -> float:
        """Calculate Hawking temperature."""
        hbar = self.CONSTANTS['hbar']
        c = self.CONSTANTS['c']
        G = self.CONSTANTS['G']
        k_B = self.CONSTANTS['k_B']
        return hbar * c**3 / (8 * math.pi * G * mass * k_B)
    
    def tidal_effects(self, mass_solar: float, distance_km: float, 
                      body_size: float = 2.0) -> Dict:
        """Calculate tidal forces near black hole."""
        engine = _get_engine('blackhole')
        return engine.tidal_effects(mass_solar, distance_km, body_size)
    
    # =========================================================================
    # WORMHOLES
    # =========================================================================
    
    def wormhole(self, throat_radius_km: float) -> Dict:
        """Analyze Morris-Thorne wormhole."""
        engine = _get_engine('wormhole')
        return engine.analyze(throat_radius_km)
    
    def exotic_matter(self, throat_radius_km: float) -> Dict:
        """Calculate exotic matter requirements."""
        engine = _get_engine('wormhole')
        return engine.exotic_matter_requirements(throat_radius_km)
    
    # =========================================================================
    # MULTI-DIMENSIONAL SPACE
    # =========================================================================
    
    def n_sphere_volume(self, radius: float, n_dimensions: int) -> float:
        """Calculate volume of n-dimensional sphere."""
        engine = _get_engine('dimensions')
        props = engine.n_sphere_properties(radius, n_dimensions)
        return props['volume']
    
    def tesseract(self, size: float = 1.0, rotation: float = 0) -> Dict:
        """Analyze 4D tesseract."""
        engine = _get_engine('dimensions')
        return engine.tesseract_analysis(size, rotation)
    
    def string_theory(self) -> Dict:
        """Get string theory overview."""
        engine = _get_engine('dimensions')
        return engine.string_theory_overview()
    
    def extra_dimensions(self, n_extra: int, M_star_TeV: float) -> Dict:
        """Test ADD extra dimension model."""
        engine = _get_engine('dimensions')
        return engine.extra_dimensions_test(n_extra, M_star_TeV)
    
    # =========================================================================
    # QUANTUM MECHANICS
    # =========================================================================
    
    def quantum_well(self, length_nm: float, n_levels: int = 5) -> Dict:
        """Analyze particle in infinite square well."""
        engine = _get_engine('quantum')
        return engine.infinite_well(length_nm, n_levels)
    
    def harmonic_oscillator(self, omega: float = 1e15, n_levels: int = 5) -> Dict:
        """Analyze quantum harmonic oscillator."""
        engine = _get_engine('quantum')
        return engine.harmonic_oscillator(omega, n_levels)
    
    def hydrogen_spectrum(self, series: str = 'balmer', n_max: int = 6) -> List[Dict]:
        """Calculate hydrogen spectral series."""
        engine = _get_engine('quantum')
        return engine.hydrogen_spectrum(series, n_max)
    
    def bell_states(self) -> Dict:
        """Get quantum Bell states (entanglement)."""
        engine = _get_engine('quantum')
        return engine.bell_states()
    
    def uncertainty(self, delta_x_nm: float = 1.0) -> Dict:
        """Analyze Heisenberg uncertainty consequences."""
        engine = _get_engine('quantum')
        return engine.uncertainty_analysis(delta_x_nm)
    
    # =========================================================================
    # PARTICLE PHYSICS
    # =========================================================================
    
    def particle(self, name: str) -> Dict:
        """Get particle information."""
        engine = _get_engine('particle')
        return engine.particle_info(name)
    
    def standard_model(self) -> Dict:
        """Get Standard Model summary."""
        engine = _get_engine('particle')
        return engine.standard_model_summary()
    
    def list_particles(self, category: str = 'all') -> List[Dict]:
        """List particles by category (quarks, leptons, bosons)."""
        engine = _get_engine('particle')
        return engine.list_particles(category)
    
    # =========================================================================
    # NUCLEAR PHYSICS
    # =========================================================================
    
    def nucleus(self, Z: int, A: int) -> Dict:
        """Analyze atomic nucleus."""
        engine = _get_engine('nuclear')
        return engine.nucleus(Z, A)
    
    def binding_energy_curve(self) -> List[Dict]:
        """Get binding energy per nucleon curve."""
        engine = _get_engine('nuclear')
        return engine.binding_energy_curve()
    
    def radioactive_decay(self, half_life: float, initial_atoms: float = 1e24) -> Dict:
        """Analyze radioactive decay."""
        engine = _get_engine('nuclear')
        return engine.radioactive_decay_analysis(half_life, initial_atoms)
    
    def fusion_energy(self, reaction: str = 'DT') -> Dict:
        """Get fusion reaction energy (DT, DD, or PP)."""
        engine = _get_engine('nuclear')
        return engine.fusion_energy(reaction)
    
    def fission_energy(self) -> Dict:
        """Get U-235 fission energy."""
        engine = _get_engine('nuclear')
        return engine.fission_energy()
    
    # =========================================================================
    # ELEMENTS
    # =========================================================================
    
    def element(self, identifier) -> Dict:
        """Get element by symbol or atomic number."""
        engine = _get_engine('elements')
        return engine.element(identifier)
    
    def search_elements(self, name: str) -> List[Dict]:
        """Search elements by name."""
        engine = _get_engine('elements')
        return engine.search(name)
    
    def compare_elements(self, symbols: List[str]) -> Dict:
        """Compare element properties."""
        engine = _get_engine('elements')
        return engine.compare(symbols)
    
    # =========================================================================
    # QUANTUM COMPUTING
    # =========================================================================
    
    def quantum_circuit(self, n_qubits: int):
        """Create a quantum circuit."""
        engine = _get_engine('qcompute')
        return engine.circuit(n_qubits)
    
    def quantum_bell_state(self, which: str = 'phi+') -> Dict:
        """Create Bell state."""
        engine = _get_engine('qcompute')
        return engine.bell_state(which)
    
    def quantum_ghz(self, n: int) -> Dict:
        """Create GHZ state."""
        engine = _get_engine('qcompute')
        return engine.ghz_state(n)
    
    def grover_search(self, n_qubits: int, target: int, shots: int = 100) -> Dict:
        """Run Grover's quantum search algorithm."""
        engine = _get_engine('qcompute')
        return engine.grover_search(n_qubits, target, shots)
    
    def quantum_fourier_transform(self, n: int, input_state: int = 0) -> Dict:
        """Perform Quantum Fourier Transform."""
        engine = _get_engine('qcompute')
        return engine.qft_demo(n, input_state)
    
    def bb84_simulation(self, n_bits: int = 100) -> Dict:
        """Simulate BB84 quantum key distribution."""
        engine = _get_engine('qcompute')
        return engine.bb84_simulation(n_bits)
    
    def quantum_gates(self) -> Dict:
        """List all quantum gates."""
        engine = _get_engine('qcompute')
        return engine.list_gates()
    
    def quantum_algorithms(self) -> Dict:
        """List quantum algorithms."""
        engine = _get_engine('qcompute')
        return engine.list_algorithms()
    
    # =========================================================================
    # HIGH-LEVEL REASONING
    # =========================================================================
    
    def explain(self, concept: str) -> str:
        """
        Explain a physics concept.
        """
        explanations = {
            'time_dilation': """
TIME DILATION (Special Relativity)
----------------------------------
Moving clocks run slow.

Formula: Δt = γΔt₀ where γ = 1/√(1 - v²/c²)

At 90% of light speed:
- γ ≈ 2.29
- 1 hour on spaceship = 2.29 hours on Earth
- Traveler ages slower

This is real and measured in:
- GPS satellites (corrected for)
- Particle accelerators (muon lifetime)
- Hafele-Keating experiment (1971)
""",
            'black_hole': """
BLACK HOLE
----------
A region of spacetime where gravity is so strong that nothing,
not even light, can escape from inside the event horizon.

Key radii:
- Event horizon: r_s = 2GM/c² (point of no return)
- Photon sphere: r = 1.5 r_s (light orbits)
- ISCO: r = 3 r_s (innermost stable orbit)

Properties:
- Hawking radiation: Black holes slowly evaporate
- Information paradox: What happens to information?
- Singularity: Infinite density at center (r = 0)
""",
            'wormhole': """
WORMHOLE (Einstein-Rosen Bridge)
--------------------------------
A theoretical tunnel through spacetime connecting distant regions.

Requirements for traversable wormhole:
1. Exotic matter (negative energy density)
2. Must avoid horizon formation
3. Tidal forces must be survivable

Problems:
- No known source of exotic matter
- May be unstable
- Chronology protection conjecture
""",
            'extra_dimensions': """
EXTRA DIMENSIONS
----------------
Beyond our 3 spatial + 1 time dimensions.

Theories requiring extra dimensions:
- String theory: 10 or 11 dimensions
- Kaluza-Klein: 5D unifies gravity + EM
- ADD model: Large extra dimensions

Why we don't see them:
- Compactified at tiny scales (Planck length)
- Or: gravity "leaks" into them (ADD model)

Experimental tests:
- Gravity deviations at sub-mm scales
- Missing energy at LHC
- Gravitational wave signatures
"""
        }
        
        key = concept.lower().replace(' ', '_').replace('-', '_')
        return explanations.get(key, f"No explanation available for '{concept}'")
    
    def calculate(self, equation: str, variables: Dict[str, float]) -> Dict:
        """
        Evaluate a physics equation with given variables.
        """
        # Common equations
        if equation == 'E=mc2':
            m = variables.get('m', 1)
            c = self.CONSTANTS['c']
            return {'E': m * c**2, 'units': 'Joules'}
        
        elif equation == 'F=ma':
            m = variables.get('m', 1)
            a = variables.get('a', 1)
            return {'F': m * a, 'units': 'Newtons'}
        
        elif equation == 'schwarzschild':
            M = variables.get('M', self.CONSTANTS['M_sun'])
            return {'r_s': self.schwarzschild_radius(M), 'units': 'meters'}
        
        elif equation == 'lorentz':
            v = variables.get('v', 0)
            return {'gamma': self.lorentz_factor(v)}
        
        elif equation == 'escape_velocity':
            M = variables.get('M', self.CONSTANTS['M_earth'])
            r = variables.get('r', self.CONSTANTS['R_earth'])
            G = self.CONSTANTS['G']
            return {'v_esc': math.sqrt(2 * G * M / r), 'units': 'm/s'}
        
        else:
            return {'error': f"Unknown equation: {equation}"}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def physics() -> UnifiedPhysicsEngine:
    """Get the unified physics engine instance."""
    return UnifiedPhysicsEngine()


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Unified Physics API."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🌌 AION UNIFIED PHYSICS API 🌌                                   ║
║                                                                           ║
║     Single interface to all physics domains                               ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    p = physics()
    
    # Classical mechanics
    print("🎯 Classical Mechanics:")
    print("-" * 50)
    result = p.projectile(20, 45)
    print(f"   Projectile (20 m/s at 45°): range = {result['max_range']:.1f} m")
    
    # Optics  
    print("\n🔦 Optics:")
    print("-" * 50)
    result = p.refraction(1.0, 1.52, 45)
    print(f"   Light entering glass at 45°: refracted to {result['refracted_angle']:.1f}°")
    
    # Relativity
    print("\n⚡ Relativity:")
    print("-" * 50)
    gamma = p.lorentz_factor(0.9 * p.CONSTANTS['c'])
    print(f"   At 0.9c: γ = {gamma:.2f}")
    result = p.twin_paradox(4.37, 0.9)
    print(f"   Alpha Centauri trip: traveler ages {result['traveler_time_years']:.1f} years")
    
    # Black holes
    print("\n🕳️ Black Holes:")
    print("-" * 50)
    result = p.black_hole(10)
    print(f"   10 M☉ black hole: r_s = {result['schwarzschild_radius_km']:.1f} km")
    print(f"   Hawking temp: {result['hawking_temperature_K']:.2e} K")
    
    # Wormholes
    print("\n🌀 Wormholes:")
    print("-" * 50)
    result = p.wormhole(1.0)
    print(f"   1 km throat: exotic mass = {result['exotic_mass_solar']:.2e} M☉")
    
    # Extra dimensions
    print("\n🌌 Extra Dimensions:")
    print("-" * 50)
    result = p.extra_dimensions(2, 1)
    print(f"   2 extra dims, M* = 1 TeV: R = {result['compactification_radius_m']:.2e} m")
    
    # Equations
    print("\n📐 Equations:")
    print("-" * 50)
    result = p.calculate('E=mc2', {'m': 1})
    print(f"   E = mc² for 1 kg: E = {result['E']:.2e} J")
    
    result = p.calculate('escape_velocity', {})
    print(f"   Earth escape velocity: {result['v_esc']/1000:.2f} km/s")


if __name__ == "__main__":
    demo()
