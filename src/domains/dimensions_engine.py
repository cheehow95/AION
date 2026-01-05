"""
AION Multi-Dimensional Space Engine
====================================

Physics of higher dimensions covering:
- N-dimensional geometry (hyperspheres, hypervolumes)
- 4D polytopes (tesseract, 24-cell, etc.)
- Kaluza-Klein theory (unified EM + gravity via 5D)
- String theory basics (10/11 dimensions)
- Extra dimension models (ADD, Randall-Sundrum)

Based on mathematical physics of higher dimensions.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum


# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================

class DimensionalConstants:
    """Constants for dimensional physics."""
    c = 299792458           # Speed of light (m/s)
    G = 6.67430e-11         # Gravitational constant (m³/kg/s²)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    
    # Planck units
    l_p = math.sqrt(hbar * G / c**3)   # Planck length ~1.6e-35 m
    m_p = math.sqrt(hbar * c / G)      # Planck mass ~2.2e-8 kg
    E_p = m_p * c**2                   # Planck energy ~1.2e9 GeV


# =============================================================================
# N-DIMENSIONAL GEOMETRY
# =============================================================================

def gamma_function(n: float) -> float:
    """
    Gamma function Γ(n).
    Γ(n) = (n-1)! for positive integers
    """
    if n == int(n) and n > 0:
        result = 1.0
        for i in range(1, int(n)):
            result *= i
        return result
    
    # Use Stirling's approximation for non-integers
    if n > 0.5:
        return math.sqrt(2 * math.pi / n) * ((n / math.e) ** n)
    
    # Use reflection formula for n < 0.5
    return math.pi / (math.sin(math.pi * n) * gamma_function(1 - n))


class NDimensionalGeometry:
    """
    Geometry in N dimensions.
    """
    
    @staticmethod
    def volume_of_n_sphere(radius: float, n: int) -> float:
        """
        Volume of n-sphere (n-ball).
        
        V_n(r) = π^(n/2) × r^n / Γ(n/2 + 1)
        
        n=2: πr² (disk)
        n=3: (4/3)πr³ (ball)
        n=4: (π²/2)r⁴
        """
        return (math.pi ** (n / 2)) * (radius ** n) / gamma_function(n / 2 + 1)
    
    @staticmethod
    def surface_of_n_sphere(radius: float, n: int) -> float:
        """
        Surface area of n-sphere (n-1 dimensional boundary).
        
        S_n(r) = n × V_n(r) / r = n × π^(n/2) × r^(n-1) / Γ(n/2 + 1)
        
        n=2: 2πr (circle)
        n=3: 4πr² (sphere)
        n=4: 2π²r³
        """
        return n * NDimensionalGeometry.volume_of_n_sphere(radius, n) / radius
    
    @staticmethod
    def distance(coords: List[float]) -> float:
        """
        Euclidean distance in N dimensions.
        
        d = √(x₁² + x₂² + ... + xₙ²)
        """
        return math.sqrt(sum(x ** 2 for x in coords))
    
    @staticmethod
    def distance_between(p1: List[float], p2: List[float]) -> float:
        """
        Distance between two points in N dimensions.
        """
        if len(p1) != len(p2):
            raise ValueError("Points must have same dimension")
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    @staticmethod
    def n_cube_volume(side: float, n: int) -> float:
        """
        Volume of n-dimensional cube (hypercube).
        
        V = s^n
        """
        return side ** n
    
    @staticmethod
    def n_cube_surface(side: float, n: int) -> float:
        """
        Surface area of n-cube.
        
        S = 2n × s^(n-1)
        """
        return 2 * n * side ** (n - 1)
    
    @staticmethod
    def n_cube_diagonal(side: float, n: int) -> float:
        """
        Space diagonal of n-cube.
        
        d = s × √n
        """
        return side * math.sqrt(n)
    
    @staticmethod
    def solid_angle_n_sphere(n: int) -> float:
        """
        Total solid angle of n-sphere.
        
        Ω_n = 2π^(n/2) / Γ(n/2)
        
        n=2: 2π (full circle)
        n=3: 4π (full sphere)
        """
        return 2 * math.pi ** (n / 2) / gamma_function(n / 2)


# =============================================================================
# 4D POLYTOPES (POLYCHORA)
# =============================================================================

class Polytope4D:
    """
    4-dimensional polytopes (polychora).
    """
    
    @staticmethod
    def tesseract() -> Dict:
        """
        Properties of tesseract (8-cell, hypercube).
        
        The 4D analog of a cube.
        """
        return {
            'name': 'Tesseract (8-cell)',
            'vertices': 16,
            'edges': 32,
            'faces': 24,   # squares
            'cells': 8,    # cubes
            'schläfli_symbol': '{4,3,3}',
            'vertex_figure': 'tetrahedron',
            'dual': '16-cell'
        }
    
    @staticmethod
    def tesseract_vertices(size: float = 1.0) -> List[Tuple[float, float, float, float]]:
        """
        Generate all 16 vertices of a tesseract centered at origin.
        
        Vertices at (±s, ±s, ±s, ±s)
        """
        s = size / 2
        vertices = []
        for w in [-s, s]:
            for x in [-s, s]:
                for y in [-s, s]:
                    for z in [-s, s]:
                        vertices.append((w, x, y, z))
        return vertices
    
    @staticmethod
    def tesseract_edges(vertices: List[Tuple]) -> List[Tuple[int, int]]:
        """
        Get edges of tesseract (pairs of adjacent vertices).
        
        Two vertices are adjacent if they differ in exactly one coordinate.
        """
        edges = []
        for i, v1 in enumerate(vertices):
            for j, v2 in enumerate(vertices):
                if i < j:
                    # Count differing coordinates
                    diffs = sum(1 for a, b in zip(v1, v2) if a != b)
                    if diffs == 1:
                        edges.append((i, j))
        return edges
    
    @staticmethod
    def project_to_3d(vertices_4d: List[Tuple], angle: float = 0) -> List[Tuple[float, float, float]]:
        """
        Project 4D vertices to 3D using rotation + perspective.
        
        Rotates in the w-z plane, then projects by dropping w.
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        projected = []
        for w, x, y, z in vertices_4d:
            # Rotate in w-z plane
            w_rot = w * cos_a - z * sin_a
            z_rot = w * sin_a + z * cos_a
            
            # Perspective projection (distance 3)
            d = 3
            factor = d / (d - w_rot)
            
            projected.append((x * factor, y * factor, z_rot * factor))
        
        return projected
    
    @staticmethod
    def sixteen_cell() -> Dict:
        """
        Properties of 16-cell (cross-polytope).
        
        The 4D analog of an octahedron.
        """
        return {
            'name': '16-cell (orthoplex)',
            'vertices': 8,
            'edges': 24,
            'faces': 32,   # triangles
            'cells': 16,   # tetrahedra
            'schläfli_symbol': '{3,3,4}',
            'vertex_figure': 'octahedron',
            'dual': 'tesseract'
        }
    
    @staticmethod
    def twenty_four_cell() -> Dict:
        """
        Properties of 24-cell.
        
        A unique 4D polytope with no 3D analog.
        """
        return {
            'name': '24-cell',
            'vertices': 24,
            'edges': 96,
            'faces': 96,   # triangles
            'cells': 24,   # octahedra
            'schläfli_symbol': '{3,4,3}',
            'vertex_figure': 'cube',
            'dual': '24-cell',  # Self-dual!
            'note': 'Unique to 4D, self-dual, most symmetric'
        }
    
    @staticmethod
    def all_regular_polychora() -> List[Dict]:
        """
        All 6 regular convex 4-polytopes (analogs of Platonic solids).
        """
        return [
            Polytope4D.tesseract(),
            Polytope4D.sixteen_cell(),
            Polytope4D.twenty_four_cell(),
            {
                'name': '5-cell (pentachoron)',
                'vertices': 5, 'edges': 10, 'faces': 10, 'cells': 5,
                'schläfli_symbol': '{3,3,3}',
                'note': '4D simplex, analog of tetrahedron'
            },
            {
                'name': '120-cell',
                'vertices': 600, 'edges': 1200, 'faces': 720, 'cells': 120,
                'schläfli_symbol': '{5,3,3}',
                'note': 'Made of 120 dodecahedra'
            },
            {
                'name': '600-cell',
                'vertices': 120, 'edges': 720, 'faces': 1200, 'cells': 600,
                'schläfli_symbol': '{3,3,5}',
                'note': 'Made of 600 tetrahedra'
            }
        ]


# =============================================================================
# KALUZA-KLEIN THEORY
# =============================================================================

class KaluzaKlein:
    """
    Kaluza-Klein theory: unifying gravity and electromagnetism in 5D.
    
    The basic idea:
    - Spacetime has 4 + 1 dimensions
    - The 5th dimension is compactified (rolled up) at a tiny radius
    - 5D gravity → 4D gravity + electromagnetism
    """
    
    @staticmethod
    def compactification_radius() -> float:
        """
        The 5th dimension must be compactified at the Planck scale
        to be consistent with observations.
        
        R ~ l_p ~ 1.6 × 10⁻³⁵ m
        """
        return DimensionalConstants.l_p
    
    @staticmethod
    def effective_4d_gravity(G5: float, R: float) -> float:
        """
        4D gravitational constant from 5D theory.
        
        G₄ = G₅ / (2πR)
        """
        return G5 / (2 * math.pi * R)
    
    @staticmethod
    def electric_charge_from_momentum(p5: float, R: float) -> float:
        """
        Electric charge arises from momentum in the 5th dimension.
        
        Momentum in compact dimension is quantized:
        p₅ = n/R (n = integer)
        
        This appears as electric charge in 4D!
        """
        # Charge quantum related to p5
        return p5 * R * math.sqrt(16 * math.pi * DimensionalConstants.G)
    
    @staticmethod
    def kaluza_klein_tower_mass(n: int, R: float) -> float:
        """
        Mass of n-th Kaluza-Klein excitation.
        
        M_n = n × ℏc / R
        
        These are massive replicas of particles that would be
        observable if extra dimensions exist.
        """
        hbar = DimensionalConstants.hbar
        c = DimensionalConstants.c
        
        return n * hbar * c / R
    
    @staticmethod
    def first_kk_mode_mass(R_m: float) -> Dict:
        """
        Calculate mass of first KK mode for given compactification radius.
        """
        hbar = DimensionalConstants.hbar
        c = DimensionalConstants.c
        
        m = hbar * c / R_m
        m_GeV = m * c ** 2 / (1.602e-10)  # Convert to GeV
        
        return {
            'radius_m': R_m,
            'mass_kg': m,
            'mass_GeV': m_GeV,
            'within_lhc_reach': m_GeV < 14000  # LHC energy ~14 TeV
        }


# =============================================================================
# STRING THEORY BASICS
# =============================================================================

class StringTheory:
    """
    Basic concepts from string theory.
    
    String theory requires extra spatial dimensions:
    - Bosonic string theory: 26 dimensions
    - Superstring theory: 10 dimensions
    - M-theory: 11 dimensions
    """
    
    @staticmethod
    def critical_dimensions() -> Dict:
        """
        Critical dimensions for different string theories.
        """
        return {
            'bosonic_string': {
                'total_dimensions': 26,
                'spacetime': 25 + 1,
                'note': 'Has tachyons, superseded by superstrings'
            },
            'superstring': {
                'total_dimensions': 10,
                'spacetime': 9 + 1,
                'compact_dimensions': 6,
                'note': 'Five consistent superstring theories'
            },
            'm_theory': {
                'total_dimensions': 11,
                'spacetime': 10 + 1,
                'note': 'Unifies all five superstring theories'
            }
        }
    
    @staticmethod
    def string_length() -> float:
        """
        Fundamental string length scale.
        
        l_s ≈ l_p = √(ℏG/c³) ≈ 1.6 × 10⁻³⁵ m
        """
        return DimensionalConstants.l_p
    
    @staticmethod
    def string_tension() -> float:
        """
        String tension.
        
        T = c⁴ / (2πα'ℓs²) ≈ E_p / l_p
        
        Enormous: ~10³⁹ newtons!
        """
        l_s = StringTheory.string_length()
        return DimensionalConstants.E_p / l_s
    
    @staticmethod
    def calabi_yau_compactification() -> str:
        """
        Information about Calabi-Yau manifolds.
        """
        return """
        CALABI-YAU MANIFOLDS
        
        In superstring theory, the 6 extra dimensions are compactified
        on a Calabi-Yau manifold - a special type of 6-dimensional space.
        
        Properties:
        - Complex 3-dimensional (6 real dimensions)
        - Ricci-flat (solves vacuum Einstein equations)
        - Has SU(3) holonomy
        - Preserves N=1 supersymmetry in 4D
        
        The shape of the Calabi-Yau determines:
        - Number of particle generations (3 in Standard Model)
        - Coupling constants
        - Particle masses
        
        There are estimated to be ~10⁵⁰⁰ possible Calabi-Yau shapes,
        each giving different physics. This is the "string landscape."
        """


# =============================================================================
# EXTRA DIMENSION MODELS
# =============================================================================

class ExtraDimensionModels:
    """
    Phenomenological models with extra dimensions.
    """
    
    @staticmethod
    def add_model(n_extra: int, M_star_TeV: float) -> Dict:
        """
        Arkani-Hamed–Dimopoulos–Dvali (ADD) model.
        
        Large extra dimensions explain the weakness of gravity.
        Gravity spreads into n extra dimensions of size R.
        
        M_Planck² ≈ M_star^(2+n) × R^n
        """
        M_pl = 1.22e19  # Planck mass in GeV
        M_star = M_star_TeV * 1000  # Convert to GeV
        
        # Calculate required compactification radius
        # R ≈ (M_pl²/M_star^(2+n))^(1/n)
        R_eV_inv = (M_pl ** 2 / M_star ** (2 + n_extra)) ** (1 / n_extra)
        
        # Convert eV^-1 to meters (1 eV^-1 ≈ 2e-7 m)
        R_m = R_eV_inv * 2e-7
        
        return {
            'n_extra_dimensions': n_extra,
            'M_star_TeV': M_star_TeV,
            'compactification_radius_m': R_m,
            'experimentally_viable': R_m < 1e-3,  # Must be < mm
            'note': 'Gravity modified at distances < R'
        }
    
    @staticmethod
    def randall_sundrum(k: float, r_c: float) -> Dict:
        """
        Randall-Sundrum model with warped extra dimension.
        
        Two branes bounding a 5D anti-de Sitter space.
        The exponential warp factor explains hierarchy problem.
        
        Scale hierarchy: M_weak / M_Planck = e^(-k×r_c)
        """
        # Warp factor
        warp = math.exp(-k * r_c)
        
        M_pl = 1.22e19  # GeV
        M_weak = M_pl * warp
        
        return {
            'k_curvature': k,
            'r_c_distance': r_c,
            'warp_factor': warp,
            'effective_scale_GeV': M_weak,
            'solves_hierarchy': abs(math.log10(M_weak) - 3) < 1,  # ~TeV scale?
            'note': 'Only requires ONE extra dimension'
        }
    
    @staticmethod
    def gravity_deviation(n_extra: int, R: float, r: float) -> float:
        """
        Modified gravitational force law at distance r.
        
        For distances r < R:
        F ∝ 1/r^(2+n) instead of 1/r²
        
        Returns ratio F_extra / F_Newton
        """
        if r >= R:
            return 1.0  # No deviation at large distances
        
        return (R / r) ** n_extra


# =============================================================================
# DIMENSIONS ENGINE - MAIN INTERFACE
# =============================================================================

class DimensionsEngine:
    """
    AION Multi-Dimensional Space Engine.
    """
    
    def __init__(self):
        self.geometry = NDimensionalGeometry()
        self.kaluza_klein = KaluzaKlein()
        self.string = StringTheory()
    
    def n_sphere_properties(self, radius: float, n: int) -> Dict:
        """Calculate n-sphere properties."""
        return {
            'dimension': n,
            'radius': radius,
            'volume': NDimensionalGeometry.volume_of_n_sphere(radius, n),
            'surface_area': NDimensionalGeometry.surface_of_n_sphere(radius, n),
            'solid_angle': NDimensionalGeometry.solid_angle_n_sphere(n)
        }
    
    def tesseract_analysis(self, size: float = 1.0, rotation_angle: float = 0) -> Dict:
        """Analyze tesseract geometry."""
        info = Polytope4D.tesseract()
        vertices_4d = Polytope4D.tesseract_vertices(size)
        edges = Polytope4D.tesseract_edges(vertices_4d)
        vertices_3d = Polytope4D.project_to_3d(vertices_4d, rotation_angle)
        
        return {
            'info': info,
            'vertices_4d': vertices_4d,
            'vertices_3d_projection': vertices_3d,
            'edges': edges,
            'n_vertices': len(vertices_4d),
            'n_edges': len(edges)
        }
    
    def all_polychora(self) -> List[Dict]:
        """Get all 6 regular 4D polytopes."""
        return Polytope4D.all_regular_polychora()
    
    def kaluza_klein_analysis(self, compactification_radius_m: float) -> Dict:
        """Analyze Kaluza-Klein theory predictions."""
        return KaluzaKlein.first_kk_mode_mass(compactification_radius_m)
    
    def string_theory_overview(self) -> Dict:
        """Get string theory overview."""
        return {
            'critical_dimensions': StringTheory.critical_dimensions(),
            'string_length_m': StringTheory.string_length(),
            'string_tension_N': StringTheory.string_tension(),
            'calabi_yau_info': StringTheory.calabi_yau_compactification()
        }
    
    def extra_dimensions_test(self, n_extra: int, M_star_TeV: float) -> Dict:
        """Test predictions of ADD model."""
        return ExtraDimensionModels.add_model(n_extra, M_star_TeV)
    
    def dimension_comparison(self) -> List[Dict]:
        """Compare geometry across dimensions."""
        results = []
        for n in range(1, 11):
            sphere_vol = NDimensionalGeometry.volume_of_n_sphere(1.0, n)
            cube_vol = NDimensionalGeometry.n_cube_volume(1.0, n)
            diagonal = NDimensionalGeometry.n_cube_diagonal(1.0, n)
            
            results.append({
                'dimension': n,
                'unit_sphere_volume': sphere_vol,
                'unit_cube_volume': cube_vol,
                'sphere_to_cube_ratio': sphere_vol / cube_vol if cube_vol > 0 else 0,
                'unit_cube_diagonal': diagonal
            })
        
        return results


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Dimensions Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🌌 AION MULTI-DIMENSIONAL SPACE ENGINE 🌌                        ║
║                                                                           ║
║     N-dimensional Geometry, 4D Polytopes, String Theory                  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = DimensionsEngine()
    
    # N-sphere volumes
    print("🔮 Unit Sphere Volume by Dimension:")
    print("-" * 50)
    for n in range(1, 8):
        vol = NDimensionalGeometry.volume_of_n_sphere(1.0, n)
        print(f"   {n}D sphere: {vol:.6f}")
    print("   → Volume DECREASES for n > 5!")
    
    # Tesseract
    print("\n📦 Tesseract (4D Hypercube):")
    print("-" * 50)
    result = engine.tesseract_analysis()
    info = result['info']
    print(f"   Vertices: {info['vertices']}")
    print(f"   Edges: {info['edges']}")
    print(f"   Faces: {info['faces']} (squares)")
    print(f"   Cells: {info['cells']} (cubes)")
    print(f"   Schläfli symbol: {info['schläfli_symbol']}")
    
    # All polychora
    print("\n🎲 All 6 Regular 4D Polytopes:")
    print("-" * 50)
    for p in engine.all_polychora():
        print(f"   {p['name']}: {p['vertices']}v, {p['edges']}e, {p['cells']} cells")
    
    # Kaluza-Klein
    print("\n🔬 Kaluza-Klein Theory:")
    print("-" * 50)
    kk = engine.kaluza_klein_analysis(1e-35)
    print(f"   Compactification at Planck scale: {kk['radius_m']:.1e} m")
    print(f"   First KK mode mass: {kk['mass_GeV']:.1e} GeV")
    print(f"   Within LHC reach: {kk['within_lhc_reach']}")
    
    # Extra dimensions
    print("\n🔭 ADD Model (Large Extra Dimensions):")
    print("-" * 50)
    for n in [2, 4, 6]:
        result = ExtraDimensionModels.add_model(n, 1)  # 1 TeV
        print(f"   {n} extra dims, M* = 1 TeV: R = {result['compactification_radius_m']:.2e} m")
    
    # String theory
    print("\n🎸 String Theory:")
    print("-" * 50)
    st = engine.string_theory_overview()
    dims = st['critical_dimensions']
    print(f"   Superstring: {dims['superstring']['total_dimensions']} dimensions")
    print(f"   M-theory: {dims['m_theory']['total_dimensions']} dimensions")
    print(f"   String length: {st['string_length_m']:.2e} m (Planck scale)")
    print(f"   String tension: {st['string_tension_N']:.2e} N (enormous!)")


if __name__ == "__main__":
    demo()
