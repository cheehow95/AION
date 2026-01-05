"""
AION Quantum Mechanics Engine
=============================

Deep quantum mechanics implementation covering:
- Wave functions and probability
- Schrödinger equation solutions
- Quantum operators and observables
- Angular momentum and spin
- Quantum harmonic oscillator
- Hydrogen atom and orbitals
- Uncertainty relations
- Entanglement and Bell states

Based on rigorous quantum mechanical principles.
"""

import math
import cmath
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class QMConstants:
    """Quantum mechanical constants."""
    h = 6.62607015e-34       # Planck constant (J·s)
    hbar = 1.054571817e-34   # Reduced Planck constant (J·s)
    c = 299792458            # Speed of light (m/s)
    e = 1.602176634e-19      # Elementary charge (C)
    m_e = 9.1093837015e-31   # Electron mass (kg)
    m_p = 1.67262192369e-27  # Proton mass (kg)
    epsilon_0 = 8.8541878128e-12  # Vacuum permittivity
    k_e = 8.9875517923e9     # Coulomb's constant
    a_0 = 5.29177210903e-11  # Bohr radius (m)
    E_h = 4.3597447222071e-18  # Hartree energy (J)
    alpha = 7.2973525693e-3  # Fine structure constant
    mu_B = 9.2740100783e-24  # Bohr magneton (J/T)
    

# =============================================================================
# WAVE FUNCTIONS
# =============================================================================

class WaveFunction:
    """
    Quantum wave function ψ(x).
    
    |ψ(x)|² gives probability density.
    """
    
    def __init__(self, psi: Callable[[float], complex]):
        """
        Args:
            psi: Wave function ψ(x) returning complex amplitude
        """
        self.psi = psi
    
    def probability_density(self, x: float) -> float:
        """
        Probability density |ψ(x)|².
        """
        amplitude = self.psi(x)
        return abs(amplitude) ** 2
    
    def probability(self, x1: float, x2: float, dx: float = 0.01) -> float:
        """
        Probability of finding particle between x1 and x2.
        P = ∫|ψ(x)|² dx
        """
        total = 0.0
        x = x1
        while x < x2:
            total += self.probability_density(x) * dx
            x += dx
        return total
    
    def expectation_value_x(self, x_min: float, x_max: float, dx: float = 0.01) -> float:
        """
        Expectation value of position.
        <x> = ∫ ψ*(x) x ψ(x) dx
        """
        total = 0.0
        x = x_min
        while x < x_max:
            psi_x = self.psi(x)
            total += (psi_x.conjugate() * x * psi_x).real * dx
            x += dx
        return total
    
    def expectation_value_x2(self, x_min: float, x_max: float, dx: float = 0.01) -> float:
        """
        Expectation value of x².
        <x²> = ∫ ψ*(x) x² ψ(x) dx
        """
        total = 0.0
        x = x_min
        while x < x_max:
            psi_x = self.psi(x)
            total += (psi_x.conjugate() * x * x * psi_x).real * dx
            x += dx
        return total
    
    def uncertainty_x(self, x_min: float, x_max: float, dx: float = 0.01) -> float:
        """
        Position uncertainty Δx = √(<x²> - <x>²)
        """
        x_avg = self.expectation_value_x(x_min, x_max, dx)
        x2_avg = self.expectation_value_x2(x_min, x_max, dx)
        return math.sqrt(max(0, x2_avg - x_avg ** 2))


# =============================================================================
# QUANTUM STATES
# =============================================================================

@dataclass
class QuantumState:
    """
    Quantum state as superposition of basis states.
    |ψ⟩ = Σ cᵢ|i⟩
    """
    amplitudes: Dict[str, complex] = field(default_factory=dict)
    
    def normalize(self):
        """Normalize the state so Σ|cᵢ|² = 1."""
        norm = math.sqrt(sum(abs(c)**2 for c in self.amplitudes.values()))
        if norm > 0:
            self.amplitudes = {k: v/norm for k, v in self.amplitudes.items()}
    
    def probability(self, state: str) -> float:
        """Probability of measuring state |state⟩."""
        if state not in self.amplitudes:
            return 0.0
        return abs(self.amplitudes[state]) ** 2
    
    def measure(self) -> str:
        """Simulate measurement, returning collapsed state."""
        import random
        r = random.random()
        cumulative = 0.0
        for state, amp in self.amplitudes.items():
            cumulative += abs(amp) ** 2
            if r < cumulative:
                return state
        return list(self.amplitudes.keys())[-1]
    
    @staticmethod
    def superposition(states: List[str], equal: bool = True) -> 'QuantumState':
        """Create equal superposition of states."""
        n = len(states)
        amp = 1.0 / math.sqrt(n) if equal else 1.0
        return QuantumState({s: complex(amp, 0) for s in states})
    
    @staticmethod
    def bell_state(which: str = 'phi+') -> 'QuantumState':
        """
        Create a Bell state.
        
        |Φ+⟩ = (|00⟩ + |11⟩)/√2
        |Φ-⟩ = (|00⟩ - |11⟩)/√2
        |Ψ+⟩ = (|01⟩ + |10⟩)/√2
        |Ψ-⟩ = (|01⟩ - |10⟩)/√2
        """
        c = 1.0 / math.sqrt(2)
        if which == 'phi+':
            return QuantumState({'00': c, '11': c})
        elif which == 'phi-':
            return QuantumState({'00': c, '11': -c})
        elif which == 'psi+':
            return QuantumState({'01': c, '10': c})
        elif which == 'psi-':
            return QuantumState({'01': c, '10': -c})
        else:
            raise ValueError(f"Unknown Bell state: {which}")


class Qubit:
    """
    Two-level quantum system (qubit).
    |ψ⟩ = α|0⟩ + β|1⟩
    """
    
    def __init__(self, alpha: complex = 1.0, beta: complex = 0.0):
        self.alpha = alpha
        self.beta = beta
        self.normalize()
    
    def normalize(self):
        """Normalize so |α|² + |β|² = 1."""
        norm = math.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        if norm > 0:
            self.alpha /= norm
            self.beta /= norm
    
    @property
    def prob_0(self) -> float:
        """Probability of measuring |0⟩."""
        return abs(self.alpha) ** 2
    
    @property
    def prob_1(self) -> float:
        """Probability of measuring |1⟩."""
        return abs(self.beta) ** 2
    
    def bloch_coords(self) -> Tuple[float, float, float]:
        """
        Get Bloch sphere coordinates (x, y, z).
        
        |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
        x = sin(θ)cos(φ), y = sin(θ)sin(φ), z = cos(θ)
        """
        theta = 2 * math.acos(min(1, abs(self.alpha)))
        if abs(self.beta) > 1e-10:
            phi = cmath.phase(self.beta) - cmath.phase(self.alpha)
        else:
            phi = 0
        
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        
        return (x, y, z)
    
    def apply_gate(self, gate: str):
        """Apply quantum gate."""
        if gate == 'X':  # Pauli-X (NOT)
            self.alpha, self.beta = self.beta, self.alpha
        elif gate == 'Y':  # Pauli-Y
            self.alpha, self.beta = -1j * self.beta, 1j * self.alpha
        elif gate == 'Z':  # Pauli-Z
            self.beta = -self.beta
        elif gate == 'H':  # Hadamard
            c = 1 / math.sqrt(2)
            a, b = self.alpha, self.beta
            self.alpha = c * (a + b)
            self.beta = c * (a - b)
        elif gate == 'S':  # Phase gate
            self.beta *= 1j
        elif gate == 'T':  # π/8 gate
            self.beta *= cmath.exp(1j * math.pi / 4)
        self.normalize()


# =============================================================================
# SCHRÖDINGER EQUATION SOLUTIONS
# =============================================================================

class InfiniteSquareWell:
    """
    Particle in an infinite square well (1D box).
    
    V(x) = 0 for 0 < x < L
    V(x) = ∞ otherwise
    
    Energy: E_n = n²π²ℏ²/(2mL²)
    Wave function: ψ_n(x) = √(2/L) sin(nπx/L)
    """
    
    def __init__(self, length: float, mass: float = QMConstants.m_e):
        self.L = length
        self.m = mass
        self.hbar = QMConstants.hbar
    
    def energy(self, n: int) -> float:
        """Energy of n-th level (n = 1, 2, 3, ...)."""
        if n < 1:
            raise ValueError("n must be >= 1")
        return (n ** 2 * math.pi ** 2 * self.hbar ** 2) / (2 * self.m * self.L ** 2)
    
    def energy_eV(self, n: int) -> float:
        """Energy in electron volts."""
        return self.energy(n) / QMConstants.e
    
    def wavelength(self, n: int) -> float:
        """de Broglie wavelength for n-th state."""
        return 2 * self.L / n
    
    def wave_function(self, n: int, x: float) -> float:
        """Wave function ψ_n(x)."""
        if x < 0 or x > self.L:
            return 0.0
        return math.sqrt(2 / self.L) * math.sin(n * math.pi * x / self.L)
    
    def probability_density(self, n: int, x: float) -> float:
        """Probability density |ψ_n(x)|²."""
        return self.wave_function(n, x) ** 2


class QuantumHarmonicOscillator:
    """
    Quantum harmonic oscillator.
    
    V(x) = ½mω²x²
    
    Energy: E_n = (n + ½)ℏω
    """
    
    def __init__(self, mass: float = QMConstants.m_e, omega: float = 1e15):
        self.m = mass
        self.omega = omega
        self.hbar = QMConstants.hbar
    
    @property
    def zero_point_energy(self) -> float:
        """Ground state energy E_0 = ½ℏω."""
        return 0.5 * self.hbar * self.omega
    
    def energy(self, n: int) -> float:
        """Energy of n-th level (n = 0, 1, 2, ...)."""
        return (n + 0.5) * self.hbar * self.omega
    
    def energy_eV(self, n: int) -> float:
        """Energy in electron volts."""
        return self.energy(n) / QMConstants.e
    
    @property
    def characteristic_length(self) -> float:
        """Characteristic length scale √(ℏ/mω)."""
        return math.sqrt(self.hbar / (self.m * self.omega))
    
    def hermite(self, n: int, x: float) -> float:
        """Hermite polynomial H_n(x) using recursion."""
        if n == 0:
            return 1.0
        elif n == 1:
            return 2 * x
        else:
            h_prev2 = 1.0
            h_prev1 = 2 * x
            for k in range(2, n + 1):
                h = 2 * x * h_prev1 - 2 * (k - 1) * h_prev2
                h_prev2 = h_prev1
                h_prev1 = h
            return h_prev1
    
    def wave_function(self, n: int, x: float) -> float:
        """
        Wave function ψ_n(x).
        
        ψ_n(x) = (1/√(2^n n!)) × (mω/πℏ)^(1/4) × H_n(ξ) × e^(-ξ²/2)
        where ξ = x√(mω/ℏ)
        """
        xi = x * math.sqrt(self.m * self.omega / self.hbar)
        
        # Normalization
        norm = (self.m * self.omega / (math.pi * self.hbar)) ** 0.25
        norm /= math.sqrt(2 ** n * math.factorial(n))
        
        return norm * self.hermite(n, xi) * math.exp(-xi ** 2 / 2)


class HydrogenAtom:
    """
    Hydrogen atom quantum mechanics.
    
    Energy: E_n = -13.6 eV / n²
    Orbitals: ψ_nlm(r, θ, φ)
    """
    
    def __init__(self):
        self.a_0 = QMConstants.a_0  # Bohr radius
        self.E_1 = -13.6  # Ground state energy (eV)
    
    def energy_eV(self, n: int) -> float:
        """Energy of n-th level."""
        return self.E_1 / n ** 2
    
    def energy_J(self, n: int) -> float:
        """Energy in Joules."""
        return self.energy_eV(n) * QMConstants.e
    
    def transition_wavelength(self, n_upper: int, n_lower: int) -> float:
        """
        Wavelength of photon emitted in transition.
        """
        E_upper = self.energy_J(n_upper)
        E_lower = self.energy_J(n_lower)
        delta_E = abs(E_upper - E_lower)
        return QMConstants.h * QMConstants.c / delta_E
    
    def transition_frequency(self, n_upper: int, n_lower: int) -> float:
        """Frequency of emitted photon."""
        E_upper = self.energy_J(n_upper)
        E_lower = self.energy_J(n_lower)
        return abs(E_upper - E_lower) / QMConstants.h
    
    def orbital_radius(self, n: int) -> float:
        """Most probable radius for n-th shell."""
        return n ** 2 * self.a_0
    
    def orbital_velocity(self, n: int) -> float:
        """Classical orbital velocity v = αc/n."""
        return QMConstants.alpha * QMConstants.c / n
    
    def radial_wave_function_1s(self, r: float) -> float:
        """R_10(r) = 2(1/a_0)^(3/2) e^(-r/a_0)"""
        return 2 * (1/self.a_0) ** 1.5 * math.exp(-r / self.a_0)
    
    def radial_wave_function_2s(self, r: float) -> float:
        """R_20(r)"""
        rho = r / self.a_0
        norm = (1 / (2 * math.sqrt(2))) * (1/self.a_0) ** 1.5
        return norm * (2 - rho) * math.exp(-rho / 2)
    
    def radial_probability_density(self, n: int, l: int, r: float) -> float:
        """Radial probability density P(r) = r²|R_nl(r)|²."""
        if n == 1 and l == 0:
            R = self.radial_wave_function_1s(r)
        elif n == 2 and l == 0:
            R = self.radial_wave_function_2s(r)
        else:
            # Simplified for other states
            norm = 1 / (n ** 2 * self.a_0 ** 1.5)
            R = norm * math.exp(-r / (n * self.a_0))
        
        return r ** 2 * R ** 2
    
    def degeneracy(self, n: int) -> int:
        """Number of degenerate states for principal quantum number n."""
        return n ** 2  # Including spin: 2n² but here just orbital
    
    def quantum_numbers(self, n: int) -> List[Tuple[int, int, int]]:
        """All valid (n, l, m) combinations for given n."""
        states = []
        for l in range(n):
            for m in range(-l, l + 1):
                states.append((n, l, m))
        return states
    
    def series(self, name: str, n_upper: int) -> Dict:
        """
        Calculate spectral series.
        
        Lyman: n → 1 (UV)
        Balmer: n → 2 (visible)
        Paschen: n → 3 (IR)
        Brackett: n → 4 (IR)
        """
        series_config = {
            'lyman': 1,
            'balmer': 2,
            'paschen': 3,
            'brackett': 4
        }
        
        n_lower = series_config.get(name.lower(), 1)
        if n_upper <= n_lower:
            raise ValueError(f"n_upper must be > {n_lower}")
        
        wavelength = self.transition_wavelength(n_upper, n_lower)
        
        return {
            'series': name,
            'transition': f'{n_upper} → {n_lower}',
            'wavelength_nm': wavelength * 1e9,
            'energy_eV': abs(self.energy_eV(n_upper) - self.energy_eV(n_lower))
        }


# =============================================================================
# ANGULAR MOMENTUM AND SPIN
# =============================================================================

class AngularMomentum:
    """
    Quantum angular momentum.
    
    L² = l(l+1)ℏ²
    L_z = mℏ  where m = -l, -l+1, ..., l-1, l
    """
    
    hbar = QMConstants.hbar
    
    @staticmethod
    def L_squared(l: int) -> float:
        """Magnitude squared of angular momentum L² = l(l+1)ℏ²."""
        return l * (l + 1) * QMConstants.hbar ** 2
    
    @staticmethod
    def L_z(m: int) -> float:
        """z-component of angular momentum L_z = mℏ."""
        return m * QMConstants.hbar
    
    @staticmethod
    def L_magnitude(l: int) -> float:
        """Magnitude |L| = √(l(l+1))ℏ."""
        return math.sqrt(l * (l + 1)) * QMConstants.hbar
    
    @staticmethod
    def number_of_m_states(l: int) -> int:
        """Number of m states for given l: 2l + 1."""
        return 2 * l + 1
    
    @staticmethod
    def clebsch_gordan_simple(l1: int, l2: int) -> List[int]:
        """
        Possible values of total L when combining l1 and l2.
        L = |l1-l2|, |l1-l2|+1, ..., l1+l2
        """
        return list(range(abs(l1 - l2), l1 + l2 + 1))


class Spin:
    """
    Quantum spin.
    
    Electrons have spin s = 1/2.
    S² = s(s+1)ℏ² = (3/4)ℏ²
    S_z = m_s ℏ  where m_s = ±1/2
    """
    
    hbar = QMConstants.hbar
    
    @staticmethod
    def S_squared(s: float) -> float:
        """S² = s(s+1)ℏ²."""
        return s * (s + 1) * QMConstants.hbar ** 2
    
    @staticmethod
    def S_z(m_s: float) -> float:
        """S_z = m_s ℏ."""
        return m_s * QMConstants.hbar
    
    @staticmethod
    def electron_spin_states() -> List[Tuple[float, str]]:
        """Electron spin states."""
        return [
            (0.5, 'spin up |↑⟩'),
            (-0.5, 'spin down |↓⟩')
        ]
    
    @staticmethod
    def magnetic_moment(g: float, m_s: float) -> float:
        """
        Spin magnetic moment μ = -g × μ_B × m_s.
        g ≈ 2.002 for electron.
        """
        return -g * QMConstants.mu_B * m_s
    
    @staticmethod
    def spinor_up() -> Tuple[complex, complex]:
        """Spin-up spinor |↑⟩ = (1, 0)."""
        return (1 + 0j, 0 + 0j)
    
    @staticmethod
    def spinor_down() -> Tuple[complex, complex]:
        """Spin-down spinor |↓⟩ = (0, 1)."""
        return (0 + 0j, 1 + 0j)


# =============================================================================
# UNCERTAINTY PRINCIPLE
# =============================================================================

class UncertaintyPrinciple:
    """
    Heisenberg Uncertainty Principle.
    
    ΔxΔp ≥ ℏ/2
    ΔEΔt ≥ ℏ/2
    """
    
    hbar = QMConstants.hbar
    
    @staticmethod
    def minimum_momentum_uncertainty(delta_x: float) -> float:
        """Minimum momentum uncertainty from position uncertainty."""
        return QMConstants.hbar / (2 * delta_x)
    
    @staticmethod
    def minimum_position_uncertainty(delta_p: float) -> float:
        """Minimum position uncertainty from momentum uncertainty."""
        return QMConstants.hbar / (2 * delta_p)
    
    @staticmethod
    def minimum_energy_uncertainty(delta_t: float) -> float:
        """Minimum energy uncertainty from time uncertainty."""
        return QMConstants.hbar / (2 * delta_t)
    
    @staticmethod
    def minimum_time_uncertainty(delta_E: float) -> float:
        """Minimum time uncertainty from energy uncertainty."""
        return QMConstants.hbar / (2 * delta_E)
    
    @staticmethod
    def zero_point_energy(omega: float) -> float:
        """Zero-point energy from uncertainty: E_0 = ℏω/2."""
        return QMConstants.hbar * omega / 2
    
    @staticmethod
    def virtual_particle_lifetime(mass: float) -> float:
        """
        Maximum lifetime of virtual particle.
        Δt ≤ ℏ/(2mc²)
        """
        return QMConstants.hbar / (2 * mass * QMConstants.c ** 2)


# =============================================================================
# QUANTUM ENGINE - MAIN INTERFACE
# =============================================================================

class QuantumEngine:
    """
    AION Quantum Mechanics Engine for calculations and simulations.
    """
    
    def __init__(self):
        self.constants = QMConstants
        self.hydrogen = HydrogenAtom()
        self.uncertainty = UncertaintyPrinciple()
    
    def infinite_well(self, length_nm: float, n_levels: int = 5) -> Dict:
        """Analyze particle in infinite square well."""
        L = length_nm * 1e-9
        well = InfiniteSquareWell(L)
        
        levels = []
        for n in range(1, n_levels + 1):
            levels.append({
                'n': n,
                'energy_eV': well.energy_eV(n),
                'wavelength_nm': well.wavelength(n) * 1e9
            })
        
        return {
            'length_nm': length_nm,
            'energy_levels': levels,
            'ground_state_eV': well.energy_eV(1)
        }
    
    def harmonic_oscillator(self, omega_rad_s: float = 1e15, n_levels: int = 5) -> Dict:
        """Analyze quantum harmonic oscillator."""
        qho = QuantumHarmonicOscillator(omega=omega_rad_s)
        
        levels = []
        for n in range(n_levels):
            levels.append({
                'n': n,
                'energy_eV': qho.energy_eV(n)
            })
        
        return {
            'omega': omega_rad_s,
            'zero_point_energy_eV': qho.zero_point_energy / QMConstants.e,
            'characteristic_length_pm': qho.characteristic_length * 1e12,
            'energy_levels': levels
        }
    
    def hydrogen_spectrum(self, series: str = 'balmer', n_max: int = 6) -> List[Dict]:
        """Calculate hydrogen spectral series."""
        series_config = {'lyman': 1, 'balmer': 2, 'paschen': 3, 'brackett': 4}
        n_lower = series_config.get(series.lower(), 2)
        
        lines = []
        for n in range(n_lower + 1, n_max + 1):
            result = self.hydrogen.series(series, n)
            lines.append(result)
        
        return lines
    
    def qubit_state(self, theta: float, phi: float) -> Dict:
        """
        Create qubit state on Bloch sphere.
        |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
        """
        alpha = math.cos(theta / 2)
        beta = cmath.exp(1j * phi) * math.sin(theta / 2)
        
        qubit = Qubit(alpha, beta)
        
        return {
            'alpha': alpha,
            'beta': complex(beta),
            'prob_0': qubit.prob_0,
            'prob_1': qubit.prob_1,
            'bloch_coords': qubit.bloch_coords()
        }
    
    def bell_states(self) -> Dict[str, Dict]:
        """Get all four Bell states."""
        states = {}
        for name in ['phi+', 'phi-', 'psi+', 'psi-']:
            state = QuantumState.bell_state(name)
            states[name] = {
                'amplitudes': {k: complex(v) for k, v in state.amplitudes.items()},
                'entangled': True,
                'description': self._bell_description(name)
            }
        return states
    
    def _bell_description(self, name: str) -> str:
        descs = {
            'phi+': '|Φ+⟩ = (|00⟩ + |11⟩)/√2',
            'phi-': '|Φ-⟩ = (|00⟩ - |11⟩)/√2',
            'psi+': '|Ψ+⟩ = (|01⟩ + |10⟩)/√2',
            'psi-': '|Ψ-⟩ = (|01⟩ - |10⟩)/√2 (singlet)'
        }
        return descs.get(name, '')
    
    def uncertainty_analysis(self, delta_x_nm: float = 1.0) -> Dict:
        """Analyze uncertainty principle consequences."""
        delta_x = delta_x_nm * 1e-9
        delta_p = UncertaintyPrinciple.minimum_momentum_uncertainty(delta_x)
        
        # For electron
        m_e = QMConstants.m_e
        min_velocity = delta_p / m_e
        min_energy = delta_p ** 2 / (2 * m_e)
        
        return {
            'delta_x_nm': delta_x_nm,
            'delta_p_kg_m_s': delta_p,
            'min_velocity_m_s': min_velocity,
            'min_energy_eV': min_energy / QMConstants.e,
            'heisenberg_limit': QMConstants.hbar / 2
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Quantum Mechanics Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          ⚛️ AION QUANTUM MECHANICS ENGINE ⚛️                              ║
║                                                                           ║
║     Wave Functions, Schrödinger Equation, Spin, Entanglement             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = QuantumEngine()
    
    # Infinite well
    print("📦 Particle in 1nm Box:")
    print("-" * 50)
    result = engine.infinite_well(1.0, 3)
    for level in result['energy_levels']:
        print(f"   n={level['n']}: E = {level['energy_eV']:.3f} eV")
    
    # Harmonic oscillator
    print("\n🎵 Quantum Harmonic Oscillator:")
    print("-" * 50)
    result = engine.harmonic_oscillator(1e15, 4)
    print(f"   Zero-point energy: {result['zero_point_energy_eV']:.4f} eV")
    for level in result['energy_levels'][:3]:
        print(f"   n={level['n']}: E = {level['energy_eV']:.4f} eV")
    
    # Hydrogen
    print("\n🔴 Hydrogen Balmer Series (visible lines):")
    print("-" * 50)
    lines = engine.hydrogen_spectrum('balmer', 6)
    for line in lines:
        print(f"   {line['transition']}: λ = {line['wavelength_nm']:.1f} nm")
    
    # Bell states
    print("\n🔗 Bell States (Entanglement):")
    print("-" * 50)
    states = engine.bell_states()
    for name, info in states.items():
        print(f"   {info['description']}")
    
    # Uncertainty
    print("\n❓ Uncertainty Principle (1 nm localization):")
    print("-" * 50)
    result = engine.uncertainty_analysis(1.0)
    print(f"   Δx = {result['delta_x_nm']} nm")
    print(f"   Minimum Δp = {result['delta_p_kg_m_s']:.2e} kg·m/s")
    print(f"   Minimum velocity ≈ {result['min_velocity_m_s']/1000:.0f} km/s")


if __name__ == "__main__":
    demo()
