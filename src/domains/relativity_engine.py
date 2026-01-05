"""
AION Relativity Domain Engine
=============================

Complete relativity simulation engine covering:
- Special Relativity (Lorentz transformations, time dilation, length contraction)
- General Relativity (Schwarzschild metric, gravitational effects)
- Gravitational Waves

Physical principles implemented with real equations.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class RelativityConstants:
    """Relativistic constants."""
    c = 299792458           # Speed of light (m/s)
    G = 6.67430e-11         # Gravitational constant (m³/kg/s²)
    h = 6.62607015e-34      # Planck constant (J·s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    k_B = 1.380649e-23      # Boltzmann constant (J/K)
    
    # Derived constants
    c2 = c ** 2             # c²
    c4 = c ** 4             # c⁴
    
    # Solar system masses (kg)
    M_sun = 1.989e30
    M_earth = 5.972e24
    
    # Schwarzschild radii (m)
    r_s_sun = 2 * G * M_sun / c2      # ~2.95 km
    r_s_earth = 2 * G * M_earth / c2  # ~8.87 mm


# =============================================================================
# SPECIAL RELATIVITY
# =============================================================================

class SpecialRelativity:
    """
    Special Relativity calculations.
    
    Postulates:
    1. Laws of physics are the same in all inertial frames
    2. Speed of light is constant (c) in all inertial frames
    """
    
    c = RelativityConstants.c
    c2 = RelativityConstants.c2
    
    @staticmethod
    def lorentz_factor(v: float) -> float:
        """
        Calculate Lorentz factor γ.
        γ = 1/√(1 - v²/c²)
        
        Args:
            v: Velocity (m/s)
        """
        beta = v / SpecialRelativity.c
        if beta >= 1.0:
            return float('inf')
        return 1.0 / math.sqrt(1 - beta ** 2)
    
    @staticmethod
    def beta(v: float) -> float:
        """Calculate β = v/c."""
        return v / SpecialRelativity.c
    
    @staticmethod
    def time_dilation(proper_time: float, v: float) -> float:
        """
        Calculate dilated time observed by stationary observer.
        Δt = γ × Δt₀
        
        Moving clocks run slow.
        """
        gamma = SpecialRelativity.lorentz_factor(v)
        return gamma * proper_time
    
    @staticmethod
    def length_contraction(proper_length: float, v: float) -> float:
        """
        Calculate contracted length observed by stationary observer.
        L = L₀/γ
        
        Moving objects appear shorter in direction of motion.
        """
        gamma = SpecialRelativity.lorentz_factor(v)
        return proper_length / gamma
    
    @staticmethod
    def relativistic_mass(rest_mass: float, v: float) -> float:
        """
        Calculate relativistic mass.
        m = γm₀
        
        Note: Modern physics prefers invariant mass + relativistic momentum.
        """
        gamma = SpecialRelativity.lorentz_factor(v)
        return gamma * rest_mass
    
    @staticmethod
    def relativistic_momentum(rest_mass: float, v: float) -> float:
        """
        Calculate relativistic momentum.
        p = γm₀v
        """
        gamma = SpecialRelativity.lorentz_factor(v)
        return gamma * rest_mass * v
    
    @staticmethod
    def relativistic_kinetic_energy(rest_mass: float, v: float) -> float:
        """
        Calculate relativistic kinetic energy.
        KE = (γ - 1)m₀c²
        """
        gamma = SpecialRelativity.lorentz_factor(v)
        return (gamma - 1) * rest_mass * SpecialRelativity.c2
    
    @staticmethod
    def total_energy(rest_mass: float, v: float) -> float:
        """
        Calculate total relativistic energy.
        E = γm₀c²
        """
        gamma = SpecialRelativity.lorentz_factor(v)
        return gamma * rest_mass * SpecialRelativity.c2
    
    @staticmethod
    def rest_energy(rest_mass: float) -> float:
        """
        Calculate rest energy.
        E₀ = m₀c²
        """
        return rest_mass * SpecialRelativity.c2
    
    @staticmethod
    def energy_momentum_relation(rest_mass: float, momentum: float) -> float:
        """
        Energy-momentum relation.
        E² = (pc)² + (m₀c²)²
        """
        c = SpecialRelativity.c
        return math.sqrt((momentum * c) ** 2 + (rest_mass * c * c) ** 2)
    
    @staticmethod
    def velocity_addition(v1: float, v2: float) -> float:
        """
        Relativistic velocity addition.
        u = (v₁ + v₂)/(1 + v₁v₂/c²)
        
        Velocities don't simply add - result never exceeds c.
        """
        c2 = SpecialRelativity.c2
        return (v1 + v2) / (1 + v1 * v2 / c2)
    
    @staticmethod
    def doppler_shift(v: float, approaching: bool = True) -> float:
        """
        Relativistic Doppler shift factor.
        
        f_observed / f_source = √((1+β)/(1-β)) for approaching
                              = √((1-β)/(1+β)) for receding
        """
        beta = SpecialRelativity.beta(v)
        if approaching:
            return math.sqrt((1 + beta) / (1 - beta))
        else:
            return math.sqrt((1 - beta) / (1 + beta))
    
    @staticmethod
    def redshift(v: float) -> float:
        """
        Calculate cosmological redshift z.
        z = √((1+β)/(1-β)) - 1 (for recession)
        """
        beta = SpecialRelativity.beta(v)
        return math.sqrt((1 + beta) / (1 - beta)) - 1


class LorentzTransform:
    """
    Lorentz transformations between inertial frames.
    """
    
    c = RelativityConstants.c
    
    @staticmethod
    def transform_coordinates(x: float, t: float, v: float) -> Tuple[float, float]:
        """
        Transform (x, t) from frame S to frame S' moving at velocity v.
        
        x' = γ(x - vt)
        t' = γ(t - vx/c²)
        """
        gamma = SpecialRelativity.lorentz_factor(v)
        c2 = LorentzTransform.c ** 2
        
        x_prime = gamma * (x - v * t)
        t_prime = gamma * (t - v * x / c2)
        
        return (x_prime, t_prime)
    
    @staticmethod
    def inverse_transform(x_prime: float, t_prime: float, v: float) -> Tuple[float, float]:
        """
        Inverse Lorentz transformation (S' to S).
        
        x = γ(x' + vt')
        t = γ(t' + vx'/c²)
        """
        gamma = SpecialRelativity.lorentz_factor(v)
        c2 = LorentzTransform.c ** 2
        
        x = gamma * (x_prime + v * t_prime)
        t = gamma * (t_prime + v * x_prime / c2)
        
        return (x, t)
    
    @staticmethod
    def transform_4velocity(ux: float, uy: float, uz: float, v: float) -> Tuple[float, float, float]:
        """
        Transform velocity components from S to S'.
        """
        gamma_v = SpecialRelativity.lorentz_factor(v)
        c2 = LorentzTransform.c ** 2
        
        denominator = 1 - v * ux / c2
        
        ux_prime = (ux - v) / denominator
        uy_prime = uy / (gamma_v * denominator)
        uz_prime = uz / (gamma_v * denominator)
        
        return (ux_prime, uy_prime, uz_prime)


class TwinParadox:
    """
    Twin paradox calculations.
    
    One twin travels at high speed, the other stays on Earth.
    The traveling twin ages less due to time dilation.
    """
    
    @staticmethod
    def age_difference(earth_time: float, travel_velocity: float) -> Dict:
        """
        Calculate age difference between twins.
        
        Args:
            earth_time: Time elapsed on Earth (years)
            travel_velocity: Velocity of traveling twin (m/s)
        
        Returns:
            Dict with proper times for both twins
        """
        gamma = SpecialRelativity.lorentz_factor(travel_velocity)
        
        # Earth twin ages normally
        earth_age = earth_time
        
        # Traveling twin experiences time dilation
        traveler_age = earth_time / gamma
        
        return {
            'earth_twin_age': earth_age,
            'traveler_twin_age': traveler_age,
            'age_difference': earth_age - traveler_age,
            'gamma': gamma,
            'time_dilation_factor': gamma
        }
    
    @staticmethod
    def round_trip(distance: float, velocity: float) -> Dict:
        """
        Calculate times for round-trip journey.
        
        Args:
            distance: One-way distance (m)
            velocity: Travel velocity (m/s)
        """
        c = RelativityConstants.c
        gamma = SpecialRelativity.lorentz_factor(velocity)
        
        # Earth frame
        earth_travel_time = 2 * distance / velocity
        
        # Traveler frame (proper time)
        proper_time = earth_travel_time / gamma
        
        # Contracted distance seen by traveler
        contracted_distance = distance / gamma
        
        return {
            'earth_time': earth_travel_time,
            'proper_time': proper_time,
            'contracted_distance': contracted_distance,
            'age_difference': earth_travel_time - proper_time
        }


# =============================================================================
# GENERAL RELATIVITY
# =============================================================================

class GeneralRelativity:
    """
    General Relativity calculations.
    
    Core concept: Gravity is the curvature of spacetime caused by mass-energy.
    Einstein field equations: G_μν = (8πG/c⁴) T_μν
    """
    
    G = RelativityConstants.G
    c = RelativityConstants.c
    c2 = RelativityConstants.c2
    
    @staticmethod
    def schwarzschild_radius(mass: float) -> float:
        """
        Calculate Schwarzschild radius.
        r_s = 2GM/c²
        
        This is the event horizon radius for a non-rotating black hole.
        """
        return 2 * GeneralRelativity.G * mass / GeneralRelativity.c2
    
    @staticmethod
    def gravitational_time_dilation(mass: float, r: float) -> float:
        """
        Calculate gravitational time dilation factor.
        
        √(1 - r_s/r) = √(1 - 2GM/(rc²))
        
        Clocks run slower in stronger gravitational fields.
        """
        r_s = GeneralRelativity.schwarzschild_radius(mass)
        
        if r <= r_s:
            return 0.0  # Inside event horizon
        
        return math.sqrt(1 - r_s / r)
    
    @staticmethod
    def time_dilation_between(mass: float, r1: float, r2: float) -> float:
        """
        Calculate relative time dilation between two radii.
        
        Returns ratio: (proper time at r1) / (proper time at r2)
        """
        factor1 = GeneralRelativity.gravitational_time_dilation(mass, r1)
        factor2 = GeneralRelativity.gravitational_time_dilation(mass, r2)
        
        return factor1 / factor2
    
    @staticmethod
    def gravitational_redshift(mass: float, r_emit: float, r_obs: float) -> float:
        """
        Calculate gravitational redshift.
        
        z = √(1 - r_s/r_obs) / √(1 - r_s/r_emit) - 1
        
        Light climbing out of a gravitational well is redshifted.
        """
        factor_emit = GeneralRelativity.gravitational_time_dilation(mass, r_emit)
        factor_obs = GeneralRelativity.gravitational_time_dilation(mass, r_obs)
        
        return factor_obs / factor_emit - 1
    
    @staticmethod
    def light_bending_angle(mass: float, impact_parameter: float) -> float:
        """
        Calculate light deflection angle (weak field approximation).
        
        α = 4GM/(c²b)
        
        where b is the impact parameter (closest approach distance).
        """
        return 4 * GeneralRelativity.G * mass / (GeneralRelativity.c2 * impact_parameter)
    
    @staticmethod
    def shapiro_delay(mass: float, r1: float, r2: float, b: float) -> float:
        """
        Calculate Shapiro time delay for light passing near a massive object.
        
        Δt ≈ (4GM/c³) × ln((r1 + r2 + d)/(r1 + r2 - d))
        
        where d is the path length.
        """
        G = GeneralRelativity.G
        c = GeneralRelativity.c
        
        d = math.sqrt((r1 - r2) ** 2 + b ** 2)
        
        return (4 * G * mass / c ** 3) * math.log((r1 + r2 + d) / (r1 + r2 - d))
    
    @staticmethod
    def frame_dragging_velocity(mass: float, angular_momentum: float, r: float, theta: float = math.pi/2) -> float:
        """
        Calculate frame-dragging angular velocity (Lense-Thirring effect).
        
        ω = 2GJ/(c²r³) for equatorial plane
        
        Rotating masses drag spacetime around them.
        """
        G = GeneralRelativity.G
        c2 = GeneralRelativity.c2
        
        return 2 * G * angular_momentum / (c2 * r ** 3)
    
    @staticmethod
    def isco_radius(mass: float, spin: float = 0) -> float:
        """
        Calculate Innermost Stable Circular Orbit (ISCO).
        
        For Schwarzschild (spin=0): r_isco = 6GM/c² = 3r_s
        For Kerr: depends on spin parameter a = J/(Mc)
        """
        r_s = GeneralRelativity.schwarzschild_radius(mass)
        
        if spin == 0:
            return 3 * r_s  # Schwarzschild ISCO
        
        # Simplified Kerr ISCO (prograde orbit)
        # Full calculation requires solving cubic equation
        a = spin  # Dimensionless spin 0 ≤ a ≤ 1
        return r_s * (3 + 2 * math.sqrt(3 * (1 - a ** 2)) - a)
    
    @staticmethod
    def photon_sphere_radius(mass: float) -> float:
        """
        Calculate photon sphere radius.
        
        r_ph = 3GM/c² = 1.5 r_s
        
        Light can orbit the black hole at this radius (unstable).
        """
        return 1.5 * GeneralRelativity.schwarzschild_radius(mass)


# =============================================================================
# GRAVITATIONAL WAVES
# =============================================================================

class GravitationalWaves:
    """
    Gravitational wave physics.
    
    GWs are ripples in spacetime caused by accelerating masses.
    First directly detected in 2015 (LIGO).
    """
    
    G = RelativityConstants.G
    c = RelativityConstants.c
    
    @staticmethod
    def strain_amplitude(m1: float, m2: float, r: float, R: float) -> float:
        """
        Calculate gravitational wave strain amplitude.
        
        h ≈ (4G²M₁M₂)/(c⁴Rr)
        
        for circular binary orbit at separation r, observed at distance R.
        """
        G = GravitationalWaves.G
        c = GravitationalWaves.c
        
        return (4 * G ** 2 * m1 * m2) / (c ** 4 * R * r)
    
    @staticmethod
    def chirp_mass(m1: float, m2: float) -> float:
        """
        Calculate chirp mass.
        
        M_chirp = (m1 × m2)^(3/5) / (m1 + m2)^(1/5)
        
        This is the key mass parameter for GW observations.
        """
        return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    
    @staticmethod
    def orbital_frequency(m1: float, m2: float, separation: float) -> float:
        """
        Calculate orbital frequency of binary system.
        
        f = (1/2π) √(G(M1+M2)/r³)
        """
        G = GravitationalWaves.G
        M = m1 + m2
        
        return math.sqrt(G * M / separation ** 3) / (2 * math.pi)
    
    @staticmethod
    def gw_frequency(orbital_frequency: float) -> float:
        """
        GW frequency is twice the orbital frequency.
        f_GW = 2 × f_orbital
        """
        return 2 * orbital_frequency
    
    @staticmethod
    def energy_loss_rate(m1: float, m2: float, separation: float) -> float:
        """
        Calculate energy loss rate due to gravitational wave emission.
        
        dE/dt = -(32/5) × (G⁴/c⁵) × (M₁²M₂²(M₁+M₂))/r⁵
        """
        G = GravitationalWaves.G
        c = GravitationalWaves.c
        
        M = m1 + m2
        return -(32/5) * (G ** 4 / c ** 5) * (m1 ** 2 * m2 ** 2 * M) / separation ** 5
    
    @staticmethod
    def inspiral_time(m1: float, m2: float, initial_separation: float) -> float:
        """
        Estimate time to merger from initial separation.
        
        t ≈ (5/256) × (c⁵/G³) × r⁴/(M₁M₂(M₁+M₂))
        """
        G = GravitationalWaves.G
        c = GravitationalWaves.c
        
        r = initial_separation
        M = m1 + m2
        
        return (5/256) * (c ** 5 / G ** 3) * (r ** 4) / (m1 * m2 * M)
    
    @staticmethod
    def ligo_sensitivity_frequency() -> Tuple[float, float]:
        """
        LIGO frequency sensitivity range.
        Most sensitive: 100-300 Hz
        """
        return (10.0, 5000.0)  # Hz


# =============================================================================
# RELATIVITY ENGINE - MAIN INTERFACE
# =============================================================================

class RelativityEngine:
    """
    AION Relativity Engine for calculations and simulations.
    """
    
    def __init__(self):
        self.sr = SpecialRelativity()
        self.gr = GeneralRelativity()
        self.gw = GravitationalWaves()
        self.c = RelativityConstants.c
    
    def special_relativity_effects(self, velocity: float) -> Dict:
        """Calculate all special relativity effects at given velocity."""
        gamma = SpecialRelativity.lorentz_factor(velocity)
        beta = velocity / self.c
        
        return {
            'beta': beta,
            'gamma': gamma,
            'time_dilation_factor': gamma,
            'length_contraction_factor': 1 / gamma,
            'relativistic_mass_factor': gamma,
            'kinetic_energy_classical_ratio': (gamma - 1) / (0.5 * beta ** 2) if beta > 0 else 1
        }
    
    def time_dilation(self, proper_time: float, velocity: float) -> Dict:
        """Calculate time dilation effects."""
        gamma = SpecialRelativity.lorentz_factor(velocity)
        
        return {
            'proper_time': proper_time,
            'dilated_time': proper_time * gamma,
            'gamma': gamma,
            'time_lost': proper_time * (gamma - 1)
        }
    
    def length_contraction(self, proper_length: float, velocity: float) -> Dict:
        """Calculate length contraction effects."""
        gamma = SpecialRelativity.lorentz_factor(velocity)
        
        return {
            'proper_length': proper_length,
            'contracted_length': proper_length / gamma,
            'gamma': gamma,
            'contraction_percent': (1 - 1/gamma) * 100
        }
    
    def twin_paradox(self, distance_ly: float, velocity_fraction_c: float) -> Dict:
        """
        Calculate twin paradox scenario.
        
        Args:
            distance_ly: Distance to destination in light-years
            velocity_fraction_c: Velocity as fraction of c (e.g., 0.9 for 90% c)
        """
        c = self.c
        v = velocity_fraction_c * c
        
        # Convert light-years to meters
        ly_to_m = 9.461e15
        distance_m = distance_ly * ly_to_m
        
        result = TwinParadox.round_trip(distance_m, v)
        
        # Convert to years
        seconds_per_year = 365.25 * 24 * 3600
        
        return {
            'distance_ly': distance_ly,
            'velocity_c': velocity_fraction_c,
            'earth_time_years': result['earth_time'] / seconds_per_year,
            'traveler_time_years': result['proper_time'] / seconds_per_year,
            'age_difference_years': result['age_difference'] / seconds_per_year,
            'gamma': SpecialRelativity.lorentz_factor(v)
        }
    
    def gravitational_effects(self, mass: float, radius: float) -> Dict:
        """Calculate general relativity effects near massive object."""
        r_s = GeneralRelativity.schwarzschild_radius(mass)
        
        return {
            'schwarzschild_radius': r_s,
            'time_dilation_factor': GeneralRelativity.gravitational_time_dilation(mass, radius),
            'photon_sphere': GeneralRelativity.photon_sphere_radius(mass),
            'isco': GeneralRelativity.isco_radius(mass),
            'is_inside_event_horizon': radius <= r_s,
            'escape_velocity': math.sqrt(2 * RelativityConstants.G * mass / radius) if radius > r_s else self.c
        }
    
    def gw_binary(self, m1_solar: float, m2_solar: float, separation_km: float) -> Dict:
        """Calculate gravitational wave properties for binary system."""
        M_sun = RelativityConstants.M_sun
        m1 = m1_solar * M_sun
        m2 = m2_solar * M_sun
        r = separation_km * 1000
        
        f_orb = GravitationalWaves.orbital_frequency(m1, m2, r)
        
        return {
            'chirp_mass_solar': GravitationalWaves.chirp_mass(m1, m2) / M_sun,
            'orbital_frequency_hz': f_orb,
            'gw_frequency_hz': GravitationalWaves.gw_frequency(f_orb),
            'merger_time_years': GravitationalWaves.inspiral_time(m1, m2, r) / (365.25 * 24 * 3600),
            'energy_loss_rate_watts': abs(GravitationalWaves.energy_loss_rate(m1, m2, r))
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Relativity Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          ⚡ AION RELATIVITY ENGINE ⚡                                      ║
║                                                                           ║
║     Special Relativity, General Relativity, Gravitational Waves          ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = RelativityEngine()
    c = RelativityConstants.c
    
    # Special Relativity
    print("🚀 Special Relativity Effects at 0.9c:")
    print("-" * 50)
    v = 0.9 * c
    result = engine.special_relativity_effects(v)
    print(f"   Velocity: 0.9c ({v/1e6:.0f} km/s)")
    print(f"   Lorentz factor γ: {result['gamma']:.3f}")
    print(f"   Time dilation: clocks run {1/result['gamma']:.1%} as fast")
    print(f"   Length contraction: objects are {1/result['gamma']:.1%} of rest length")
    
    # Twin Paradox
    print("\n👫 Twin Paradox (Trip to Alpha Centauri):")
    print("-" * 50)
    result = engine.twin_paradox(distance_ly=4.37, velocity_fraction_c=0.9)
    print(f"   Distance: {result['distance_ly']:.2f} light-years")
    print(f"   Velocity: {result['velocity_c']*100:.0f}% of c")
    print(f"   Earth twin ages: {result['earth_time_years']:.2f} years")
    print(f"   Traveler ages: {result['traveler_time_years']:.2f} years")
    print(f"   Age difference: {result['age_difference_years']:.2f} years")
    
    # Gravitational Time Dilation
    print("\n🌍 Gravitational Effects (Earth surface vs GPS satellite):")
    print("-" * 50)
    M_earth = RelativityConstants.M_earth
    R_earth = 6.371e6
    R_gps = R_earth + 20200e3
    
    factor_surface = GeneralRelativity.gravitational_time_dilation(M_earth, R_earth)
    factor_gps = GeneralRelativity.gravitational_time_dilation(M_earth, R_gps)
    
    diff_per_day = (factor_gps / factor_surface - 1) * 86400 * 1e6
    print(f"   Time dilation factor (surface): {factor_surface:.12f}")
    print(f"   Time dilation factor (GPS): {factor_gps:.12f}")
    print(f"   GPS clocks gain: ~{diff_per_day:.1f} μs/day (GR only)")
    
    # Black Hole
    print("\n🕳️ Black Hole (10 solar masses):")
    print("-" * 50)
    result = engine.gravitational_effects(10 * RelativityConstants.M_sun, 100e3)
    print(f"   Schwarzschild radius: {result['schwarzschild_radius']/1000:.1f} km")
    print(f"   Photon sphere: {result['photon_sphere']/1000:.1f} km")
    print(f"   ISCO: {result['isco']/1000:.1f} km")
    print(f"   Time dilation at 100 km: {result['time_dilation_factor']:.4f}")
    
    # Gravitational Waves
    print("\n🌊 Gravitational Waves (Binary Neutron Stars):")
    print("-" * 50)
    result = engine.gw_binary(m1_solar=1.4, m2_solar=1.4, separation_km=100)
    print(f"   Chirp mass: {result['chirp_mass_solar']:.2f} M☉")
    print(f"   GW frequency: {result['gw_frequency_hz']:.1f} Hz")
    print(f"   Time to merger: {result['merger_time_years']:.2e} years")
    print(f"   Energy loss rate: {result['energy_loss_rate_watts']:.2e} W")


if __name__ == "__main__":
    demo()
