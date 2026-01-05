"""
AION Optics Domain Engine
=========================

Complete optics simulation engine covering:
- Geometric optics (ray tracing, reflection, refraction)
- Wave optics (interference, diffraction, polarization)
- Modern optics (lasers, fiber optics, holography)

Physical principles implemented with real equations.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class OpticsConstants:
    """Optical constants."""
    c = 299792458           # Speed of light in vacuum (m/s)
    n_vacuum = 1.0          # Refractive index of vacuum
    n_air = 1.000293        # Refractive index of air
    n_water = 1.333         # Refractive index of water
    n_glass = 1.52          # Refractive index of crown glass
    n_diamond = 2.417       # Refractive index of diamond
    h = 6.62607015e-34      # Planck constant (J·s)


# =============================================================================
# REFRACTIVE INDEX DATABASE
# =============================================================================

REFRACTIVE_INDICES = {
    'vacuum': 1.0,
    'air': 1.000293,
    'water': 1.333,
    'ice': 1.31,
    'glass': 1.52,
    'crown_glass': 1.52,
    'flint_glass': 1.66,
    'diamond': 2.417,
    'sapphire': 1.77,
    'quartz': 1.544,
    'acrylic': 1.49,
    'polycarbonate': 1.585,
    'silicon': 3.42,
    'germanium': 4.0,
}


# =============================================================================
# GEOMETRIC OPTICS
# =============================================================================

class SnellsLaw:
    """
    Snell's Law of Refraction.
    n₁ sin(θ₁) = n₂ sin(θ₂)
    """
    
    @staticmethod
    def refracted_angle(n1: float, n2: float, theta1: float) -> Optional[float]:
        """
        Calculate refracted angle.
        
        Args:
            n1: Refractive index of first medium
            n2: Refractive index of second medium
            theta1: Incident angle (radians)
            
        Returns:
            Refracted angle (radians) or None if total internal reflection
        """
        sin_theta2 = (n1 / n2) * math.sin(theta1)
        
        if abs(sin_theta2) > 1.0:
            return None  # Total internal reflection
        
        return math.asin(sin_theta2)
    
    @staticmethod
    def critical_angle(n1: float, n2: float) -> Optional[float]:
        """
        Calculate critical angle for total internal reflection.
        Only valid when n1 > n2.
        
        θc = arcsin(n₂/n₁)
        """
        if n1 <= n2:
            return None  # No critical angle (no TIR possible)
        
        return math.asin(n2 / n1)
    
    @staticmethod
    def brewster_angle(n1: float, n2: float) -> float:
        """
        Calculate Brewster's angle (polarization angle).
        θB = arctan(n₂/n₁)
        
        At this angle, reflected light is fully polarized.
        """
        return math.atan(n2 / n1)
    
    @staticmethod
    def fresnel_reflectance(n1: float, n2: float, theta1: float) -> Dict[str, float]:
        """
        Calculate Fresnel reflection coefficients.
        
        Returns R_s (s-polarized) and R_p (p-polarized).
        """
        theta2 = SnellsLaw.refracted_angle(n1, n2, theta1)
        
        if theta2 is None:
            return {'R_s': 1.0, 'R_p': 1.0, 'R_avg': 1.0}  # TIR
        
        cos1 = math.cos(theta1)
        cos2 = math.cos(theta2)
        
        # Fresnel equations
        r_s = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
        r_p = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
        
        R_s = r_s ** 2
        R_p = r_p ** 2
        
        return {
            'R_s': R_s,
            'R_p': R_p,
            'R_avg': (R_s + R_p) / 2,
            'T_avg': 1 - (R_s + R_p) / 2
        }


@dataclass
class ThinLens:
    """
    Thin lens optics.
    
    Lens equation: 1/f = 1/d_o + 1/d_i
    Magnification: M = -d_i/d_o = h_i/h_o
    """
    focal_length: float  # meters (positive for converging, negative for diverging)
    diameter: float = 0.05  # meters
    
    @property
    def is_converging(self) -> bool:
        return self.focal_length > 0
    
    @property
    def power(self) -> float:
        """Optical power in diopters (D = 1/f in meters)."""
        return 1.0 / self.focal_length
    
    @property
    def f_number(self) -> float:
        """f-number (f/D)."""
        return abs(self.focal_length) / self.diameter
    
    def image_distance(self, object_distance: float) -> float:
        """
        Calculate image distance using thin lens equation.
        1/f = 1/d_o + 1/d_i → d_i = (f * d_o) / (d_o - f)
        """
        if abs(object_distance - self.focal_length) < 1e-10:
            return float('inf')  # Object at focal point
        
        return (self.focal_length * object_distance) / (object_distance - self.focal_length)
    
    def magnification(self, object_distance: float) -> float:
        """Calculate linear magnification: M = -d_i/d_o"""
        d_i = self.image_distance(object_distance)
        if d_i == float('inf'):
            return float('inf')
        return -d_i / object_distance
    
    def image_properties(self, object_distance: float, object_height: float = 1.0) -> Dict:
        """
        Calculate complete image properties.
        """
        d_i = self.image_distance(object_distance)
        M = self.magnification(object_distance)
        
        is_real = d_i > 0
        is_inverted = M < 0
        
        return {
            'image_distance': d_i,
            'magnification': M,
            'image_height': abs(M) * object_height,
            'is_real': is_real,
            'is_inverted': is_inverted,
            'is_enlarged': abs(M) > 1
        }
    
    @staticmethod
    def lensmakers_equation(n: float, R1: float, R2: float) -> float:
        """
        Calculate focal length from lens geometry.
        1/f = (n-1) * (1/R1 - 1/R2)
        
        Sign convention: R > 0 for center on right side
        """
        return 1.0 / ((n - 1) * (1/R1 - 1/R2))


@dataclass
class SphericalMirror:
    """
    Spherical mirror (concave or convex).
    
    Mirror equation: 1/f = 1/d_o + 1/d_i
    f = R/2 (focal length is half the radius of curvature)
    """
    radius_of_curvature: float  # positive for concave, negative for convex
    diameter: float = 0.1
    
    @property
    def focal_length(self) -> float:
        return self.radius_of_curvature / 2
    
    @property
    def is_concave(self) -> bool:
        return self.radius_of_curvature > 0
    
    def image_distance(self, object_distance: float) -> float:
        f = self.focal_length
        if abs(object_distance - f) < 1e-10:
            return float('inf')
        return (f * object_distance) / (object_distance - f)
    
    def magnification(self, object_distance: float) -> float:
        d_i = self.image_distance(object_distance)
        if d_i == float('inf'):
            return float('inf')
        return -d_i / object_distance


class Prism:
    """
    Optical prism for dispersion.
    """
    
    def __init__(self, apex_angle: float, n_d: float = 1.52):
        """
        Args:
            apex_angle: Prism apex angle (radians)
            n_d: Refractive index at sodium D line (589 nm)
        """
        self.apex_angle = apex_angle
        self.n_d = n_d
    
    def minimum_deviation(self, n: float = None) -> float:
        """
        Calculate minimum deviation angle.
        δ_min = 2 * arcsin(n * sin(A/2)) - A
        """
        n = n or self.n_d
        return 2 * math.asin(n * math.sin(self.apex_angle / 2)) - self.apex_angle
    
    def deviation(self, incident_angle: float, n: float = None) -> Optional[float]:
        """
        Calculate deviation for arbitrary incident angle.
        """
        n = n or self.n_d
        
        # First refraction
        theta1_ref = SnellsLaw.refracted_angle(1.0, n, incident_angle)
        if theta1_ref is None:
            return None
        
        # Angle at second surface
        theta2 = self.apex_angle - theta1_ref
        
        # Second refraction
        theta2_ref = SnellsLaw.refracted_angle(n, 1.0, theta2)
        if theta2_ref is None:
            return None  # TIR at second surface
        
        return incident_angle + theta2_ref - self.apex_angle
    
    def dispersion(self, n_blue: float, n_red: float) -> float:
        """
        Calculate angular dispersion.
        """
        dev_blue = self.minimum_deviation(n_blue)
        dev_red = self.minimum_deviation(n_red)
        return dev_blue - dev_red


# =============================================================================
# WAVE OPTICS
# =============================================================================

class Interference:
    """
    Wave interference phenomena.
    """
    
    @staticmethod
    def double_slit_maxima(wavelength: float, slit_separation: float, 
                           screen_distance: float, m_max: int = 5) -> List[float]:
        """
        Calculate positions of bright fringes (maxima) for double-slit.
        
        Condition: d sin(θ) = mλ
        Position on screen: y = L tan(θ) ≈ mλL/d (small angle)
        """
        positions = []
        for m in range(-m_max, m_max + 1):
            sin_theta = m * wavelength / slit_separation
            if abs(sin_theta) <= 1:
                theta = math.asin(sin_theta)
                y = screen_distance * math.tan(theta)
                positions.append(y)
        return positions
    
    @staticmethod
    def fringe_spacing(wavelength: float, slit_separation: float, 
                       screen_distance: float) -> float:
        """
        Calculate fringe spacing (small angle approximation).
        Δy = λL/d
        """
        return wavelength * screen_distance / slit_separation
    
    @staticmethod
    def thin_film_condition(n_film: float, thickness: float, wavelength: float,
                            n_incident: float = 1.0, n_substrate: float = 1.5) -> Dict:
        """
        Analyze thin film interference.
        
        Optical path difference: 2nt (plus phase shifts at interfaces)
        """
        # Phase shift at first interface (n_incident < n_film → π shift)
        phase1 = math.pi if n_incident < n_film else 0
        
        # Phase shift at second interface (n_film < n_substrate → π shift)
        phase2 = math.pi if n_film < n_substrate else 0
        
        total_phase_shift = phase1 + phase2
        
        # Optical path difference
        opd = 2 * n_film * thickness
        
        # Condition for constructive interference
        wavelength_in_film = wavelength / n_film
        
        # Constructive if OPD + phase_shift = mλ
        m = (opd / wavelength + total_phase_shift / (2 * math.pi))
        
        is_constructive = abs(m - round(m)) < 0.1
        
        return {
            'optical_path_difference': opd,
            'phase_shift': total_phase_shift,
            'is_constructive': is_constructive,
            'order': round(m)
        }
    
    @staticmethod
    def michelson_interferometer(wavelength: float, mirror_displacement: float) -> int:
        """
        Calculate number of fringes shifted in Michelson interferometer.
        N = 2d/λ
        """
        return round(2 * mirror_displacement / wavelength)


class Diffraction:
    """
    Diffraction phenomena.
    """
    
    @staticmethod
    def single_slit_minima(wavelength: float, slit_width: float, m_max: int = 5) -> List[float]:
        """
        Calculate angles of dark fringes for single-slit diffraction.
        
        Condition: a sin(θ) = mλ (m = ±1, ±2, ...)
        """
        angles = []
        for m in range(1, m_max + 1):
            sin_theta = m * wavelength / slit_width
            if sin_theta <= 1:
                angles.append(math.asin(sin_theta))
                angles.append(-math.asin(sin_theta))
        return sorted(angles)
    
    @staticmethod
    def single_slit_central_width(wavelength: float, slit_width: float, 
                                   screen_distance: float) -> float:
        """
        Calculate width of central maximum.
        w = 2λL/a
        """
        return 2 * wavelength * screen_distance / slit_width
    
    @staticmethod
    def circular_aperture_airy_radius(wavelength: float, diameter: float, 
                                       focal_length: float) -> float:
        """
        Calculate radius of Airy disk (first dark ring).
        r = 1.22 λf/D
        """
        return 1.22 * wavelength * focal_length / diameter
    
    @staticmethod
    def rayleigh_criterion(wavelength: float, diameter: float) -> float:
        """
        Rayleigh resolution criterion.
        θ_min = 1.22 λ/D (radians)
        """
        return 1.22 * wavelength / diameter
    
    @staticmethod
    def grating_maxima(wavelength: float, grating_spacing: float, m_max: int = 5) -> List[float]:
        """
        Calculate diffraction grating maxima angles.
        
        Condition: d sin(θ) = mλ
        """
        angles = []
        for m in range(-m_max, m_max + 1):
            sin_theta = m * wavelength / grating_spacing
            if abs(sin_theta) <= 1:
                angles.append((m, math.asin(sin_theta)))
        return angles
    
    @staticmethod
    def grating_resolving_power(total_lines: int, order: int) -> float:
        """
        Calculate resolving power of diffraction grating.
        R = mN (order × number of lines)
        """
        return order * total_lines
    
    @staticmethod
    def bragg_diffraction(wavelength: float, lattice_spacing: float, 
                          n_max: int = 5) -> List[float]:
        """
        Calculate Bragg diffraction angles (X-ray crystallography).
        
        Condition: 2d sin(θ) = nλ
        """
        angles = []
        for n in range(1, n_max + 1):
            sin_theta = n * wavelength / (2 * lattice_spacing)
            if sin_theta <= 1:
                angles.append((n, math.asin(sin_theta)))
        return angles


class Polarization:
    """
    Light polarization phenomena.
    """
    
    @staticmethod
    def malus_law(I0: float, angle: float) -> float:
        """
        Malus's law for polarized light through analyzer.
        I = I₀ cos²(θ)
        
        Args:
            I0: Incident intensity
            angle: Angle between polarizer and analyzer (radians)
        """
        return I0 * math.cos(angle) ** 2
    
    @staticmethod
    def intensity_after_n_polarizers(I0: float, angles: List[float]) -> float:
        """
        Calculate intensity after passing through multiple polarizers.
        """
        I = I0 / 2  # First polarizer (unpolarized → polarized)
        
        for i in range(1, len(angles)):
            delta_angle = angles[i] - angles[i-1]
            I *= math.cos(delta_angle) ** 2
        
        return I
    
    @staticmethod
    def brewsters_angle(n1: float, n2: float) -> float:
        """
        Calculate Brewster's angle.
        θ_B = arctan(n₂/n₁)
        """
        return math.atan(n2 / n1)


# =============================================================================
# MODERN OPTICS
# =============================================================================

class Laser:
    """
    Laser physics and coherence.
    """
    
    @staticmethod
    def photon_energy(wavelength: float) -> float:
        """
        Calculate photon energy.
        E = hc/λ
        """
        return OpticsConstants.h * OpticsConstants.c / wavelength
    
    @staticmethod
    def coherence_length(wavelength: float, linewidth: float) -> float:
        """
        Calculate coherence length.
        L_c = λ²/Δλ
        """
        return wavelength ** 2 / linewidth
    
    @staticmethod
    def beam_divergence(wavelength: float, beam_waist: float) -> float:
        """
        Calculate Gaussian beam divergence (half-angle).
        θ = λ/(πw₀)
        """
        return wavelength / (math.pi * beam_waist)
    
    @staticmethod
    def rayleigh_range(wavelength: float, beam_waist: float) -> float:
        """
        Calculate Rayleigh range (depth of focus).
        z_R = πw₀²/λ
        """
        return math.pi * beam_waist ** 2 / wavelength
    
    @staticmethod
    def beam_radius_at_z(beam_waist: float, z: float, rayleigh_range: float) -> float:
        """
        Calculate beam radius at distance z.
        w(z) = w₀ √(1 + (z/z_R)²)
        """
        return beam_waist * math.sqrt(1 + (z / rayleigh_range) ** 2)


class FiberOptics:
    """
    Optical fiber physics.
    """
    
    def __init__(self, n_core: float, n_cladding: float, core_radius: float = 4e-6):
        self.n_core = n_core
        self.n_cladding = n_cladding
        self.core_radius = core_radius
    
    @property
    def numerical_aperture(self) -> float:
        """
        Calculate numerical aperture.
        NA = √(n_core² - n_cladding²)
        """
        return math.sqrt(self.n_core ** 2 - self.n_cladding ** 2)
    
    @property
    def acceptance_angle(self) -> float:
        """
        Calculate acceptance angle (maximum coupling angle).
        θ_max = arcsin(NA)
        """
        return math.asin(self.numerical_aperture)
    
    @property
    def critical_angle(self) -> float:
        """
        Calculate critical angle for total internal reflection.
        """
        return math.asin(self.n_cladding / self.n_core)
    
    def normalized_frequency(self, wavelength: float) -> float:
        """
        Calculate V-number (normalized frequency).
        V = (2πa/λ) × NA
        """
        return (2 * math.pi * self.core_radius / wavelength) * self.numerical_aperture
    
    def number_of_modes(self, wavelength: float) -> int:
        """
        Estimate number of guided modes (step-index fiber).
        N ≈ V²/2 for large V
        """
        V = self.normalized_frequency(wavelength)
        if V < 2.405:
            return 1  # Single-mode
        return max(1, int(V ** 2 / 2))
    
    def is_single_mode(self, wavelength: float) -> bool:
        """
        Check if fiber is single-mode at given wavelength.
        Single-mode when V < 2.405
        """
        return self.normalized_frequency(wavelength) < 2.405


# =============================================================================
# OPTICS ENGINE - MAIN INTERFACE
# =============================================================================

class OpticsEngine:
    """
    AION Optics Engine for calculations and simulations.
    """
    
    def __init__(self):
        self.snell = SnellsLaw()
        self.interference = Interference()
        self.diffraction = Diffraction()
        self.polarization = Polarization()
        self.laser = Laser()
    
    def refraction(self, n1: float, n2: float, angle: float) -> Dict:
        """Calculate refraction properties."""
        theta2 = SnellsLaw.refracted_angle(n1, n2, math.radians(angle))
        critical = SnellsLaw.critical_angle(n1, n2)
        brewster = SnellsLaw.brewster_angle(n1, n2)
        fresnel = SnellsLaw.fresnel_reflectance(n1, n2, math.radians(angle))
        
        return {
            'refracted_angle': math.degrees(theta2) if theta2 else None,
            'critical_angle': math.degrees(critical) if critical else None,
            'brewster_angle': math.degrees(brewster),
            'reflectance': fresnel['R_avg'],
            'transmittance': fresnel['T_avg'],
            'total_internal_reflection': theta2 is None
        }
    
    def lens_analysis(self, focal_length: float, object_distance: float, 
                      object_height: float = 1.0) -> Dict:
        """Analyze thin lens imaging."""
        lens = ThinLens(focal_length)
        return lens.image_properties(object_distance, object_height)
    
    def interference_pattern(self, wavelength: float, slit_separation: float,
                             screen_distance: float) -> Dict:
        """Calculate double-slit interference pattern."""
        return {
            'maxima_positions': Interference.double_slit_maxima(
                wavelength, slit_separation, screen_distance),
            'fringe_spacing': Interference.fringe_spacing(
                wavelength, slit_separation, screen_distance)
        }
    
    def diffraction_pattern(self, wavelength: float, aperture: float,
                            screen_distance: float) -> Dict:
        """Calculate single-slit diffraction pattern."""
        return {
            'minima_angles': Diffraction.single_slit_minima(wavelength, aperture),
            'central_width': Diffraction.single_slit_central_width(
                wavelength, aperture, screen_distance)
        }
    
    def laser_beam(self, wavelength: float, beam_waist: float, distance: float) -> Dict:
        """Calculate laser beam properties."""
        z_R = Laser.rayleigh_range(wavelength, beam_waist)
        
        return {
            'divergence': math.degrees(Laser.beam_divergence(wavelength, beam_waist)),
            'rayleigh_range': z_R,
            'beam_radius_at_distance': Laser.beam_radius_at_z(beam_waist, distance, z_R),
            'coherence_length': Laser.coherence_length(wavelength, 1e-12)  # Typical linewidth
        }
    
    def fiber_analysis(self, n_core: float, n_cladding: float, 
                       wavelength: float = 1550e-9) -> Dict:
        """Analyze optical fiber."""
        fiber = FiberOptics(n_core, n_cladding)
        
        return {
            'numerical_aperture': fiber.numerical_aperture,
            'acceptance_angle': math.degrees(fiber.acceptance_angle),
            'is_single_mode': fiber.is_single_mode(wavelength),
            'number_of_modes': fiber.number_of_modes(wavelength),
            'v_number': fiber.normalized_frequency(wavelength)
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Optics Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🔦 AION OPTICS ENGINE 🔦                                         ║
║                                                                           ║
║     Geometric, Wave, and Modern Optics                                    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = OpticsEngine()
    
    # Refraction
    print("🌊 Refraction (Air → Glass):")
    print("-" * 50)
    result = engine.refraction(n1=1.0, n2=1.52, angle=45)
    print(f"   Incident angle: 45°")
    print(f"   Refracted angle: {result['refracted_angle']:.2f}°")
    print(f"   Brewster angle: {result['brewster_angle']:.2f}°")
    print(f"   Reflectance: {result['reflectance']*100:.1f}%")
    
    # Lens
    print("\n🔍 Thin Lens (f = 10 cm):")
    print("-" * 50)
    result = engine.lens_analysis(focal_length=0.1, object_distance=0.3, object_height=0.02)
    print(f"   Object: 30 cm from lens, height 2 cm")
    print(f"   Image distance: {result['image_distance']*100:.1f} cm")
    print(f"   Magnification: {result['magnification']:.2f}×")
    print(f"   Image: {'Real' if result['is_real'] else 'Virtual'}, "
          f"{'Inverted' if result['is_inverted'] else 'Upright'}")
    
    # Double-slit interference
    print("\n🌈 Double-Slit Interference:")
    print("-" * 50)
    wavelength = 550e-9  # Green light
    result = engine.interference_pattern(wavelength, slit_separation=0.1e-3, screen_distance=1.0)
    print(f"   Wavelength: 550 nm, Slit separation: 0.1 mm")
    print(f"   Fringe spacing: {result['fringe_spacing']*1000:.2f} mm")
    
    # Laser beam
    print("\n🔴 Laser Beam (He-Ne, w₀ = 0.5 mm):")
    print("-" * 50)
    result = engine.laser_beam(wavelength=632.8e-9, beam_waist=0.5e-3, distance=10)
    print(f"   Divergence: {result['divergence']*1000:.3f} mrad")
    print(f"   Rayleigh range: {result['rayleigh_range']:.2f} m")
    print(f"   Beam radius at 10 m: {result['beam_radius_at_distance']*1000:.2f} mm")
    
    # Fiber optics
    print("\n📡 Optical Fiber (SMF-28):")
    print("-" * 50)
    result = engine.fiber_analysis(n_core=1.4676, n_cladding=1.4616)
    print(f"   Numerical Aperture: {result['numerical_aperture']:.3f}")
    print(f"   Acceptance angle: {result['acceptance_angle']:.1f}°")
    print(f"   Single-mode at 1550 nm: {result['is_single_mode']}")


if __name__ == "__main__":
    demo()
