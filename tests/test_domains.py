"""
AION Domain Engines Test Suite
==============================

Comprehensive tests for all physics and science domain engines.
"""

import unittest
import math
import sys
sys.path.insert(0, '.')


class TestPhysicsEngine(unittest.TestCase):
    """Test classical physics engine."""
    
    def setUp(self):
        from src.domains.physics_engine import PhysicsEngine
        self.engine = PhysicsEngine()
    
    def test_projectile_motion(self):
        """Test projectile calculation."""
        result = self.engine.projectile(20, 45)
        self.assertIn('max_range', result)
        self.assertIn('max_height', result)
        self.assertIn('flight_time', result)
        self.assertGreater(result['max_range'], 0)
    
    def test_orbital_mechanics(self):
        """Test orbital period calculation."""
        result = self.engine.orbital(400000 + 6.371e6)  # ISS altitude
        self.assertIn('orbital_period', result)
        self.assertIn('orbital_velocity', result)
    
    def test_pendulum(self):
        """Test simple pendulum."""
        result = self.engine.pendulum(1.0, 10)
        self.assertIn('period', result)
        self.assertAlmostEqual(result['period'], 2.0, delta=0.1)


class TestOpticsEngine(unittest.TestCase):
    """Test optics engine."""
    
    def setUp(self):
        from src.domains.optics_engine import OpticsEngine
        self.engine = OpticsEngine()
    
    def test_snell_law(self):
        """Test Snell's law refraction."""
        result = self.engine.refraction(1.0, 1.5, 45)
        self.assertIn('refracted_angle', result)
        self.assertLess(result['refracted_angle'], 45)
    
    def test_total_internal_reflection(self):
        """Test TIR condition."""
        # Critical angle is returned in refraction result
        result = self.engine.refraction(1.5, 1.0, 30)
        self.assertIn('critical_angle', result)
        self.assertIsNotNone(result['critical_angle'])
        self.assertGreater(result['critical_angle'], 0)


class TestRelativityEngine(unittest.TestCase):
    """Test relativity engine."""
    
    def setUp(self):
        from src.domains.relativity_engine import RelativityEngine
        self.engine = RelativityEngine()
    
    def test_special_relativity_effects(self):
        """Test SR effects calculation."""
        # At 0.9c
        result = self.engine.special_relativity_effects(0.9 * 3e8)
        self.assertIn('gamma', result)
        self.assertGreater(result['gamma'], 2)
    
    def test_time_dilation(self):
        """Test time dilation."""
        result = self.engine.time_dilation(1.0, 0.9 * 3e8)
        self.assertIn('dilated_time', result)
        self.assertGreater(result['dilated_time'], 1.0)
    
    def test_twin_paradox(self):
        """Test twin paradox calculation."""
        result = self.engine.twin_paradox(4.37, 0.9)
        self.assertIn('traveler_time_years', result)
        self.assertIn('earth_time_years', result)


class TestQuantumEngine(unittest.TestCase):
    """Test quantum mechanics engine."""
    
    def setUp(self):
        from src.domains.quantum_engine import QuantumEngine
        self.engine = QuantumEngine()
    
    def test_infinite_well(self):
        """Test particle in box."""
        result = self.engine.infinite_well(1.0, 5)
        self.assertIn('energy_levels', result)
        self.assertEqual(len(result['energy_levels']), 5)
        # Energy increases with n^2
        E1 = result['energy_levels'][0]['energy_eV']
        E2 = result['energy_levels'][1]['energy_eV']
        self.assertAlmostEqual(E2/E1, 4, delta=0.1)
    
    def test_harmonic_oscillator(self):
        """Test QHO energy levels."""
        result = self.engine.harmonic_oscillator(1e15, 3)
        self.assertIn('zero_point_energy_eV', result)
        self.assertGreater(result['zero_point_energy_eV'], 0)
    
    def test_hydrogen_spectrum(self):
        """Test hydrogen spectral lines."""
        lines = self.engine.hydrogen_spectrum('balmer', 5)
        self.assertGreater(len(lines), 0)
        # H-alpha should be around 656 nm
        self.assertAlmostEqual(lines[0]['wavelength_nm'], 656, delta=5)


class TestParticleEngine(unittest.TestCase):
    """Test particle physics engine."""
    
    def setUp(self):
        from src.domains.particle_engine import ParticleEngine
        self.engine = ParticleEngine()
    
    def test_particle_info(self):
        """Test particle data retrieval."""
        electron = self.engine.particle_info('electron')
        self.assertEqual(electron['charge'], -1)
        self.assertAlmostEqual(electron['mass_GeV'], 0.000511, delta=0.0001)
    
    def test_quarks(self):
        """Test quark listing."""
        quarks = self.engine.list_particles('quarks')
        self.assertEqual(len(quarks), 6)
    
    def test_standard_model(self):
        """Test SM summary."""
        sm = self.engine.standard_model_summary()
        self.assertEqual(sm['quarks']['count'], 6)
        self.assertEqual(sm['leptons']['count'], 6)


class TestNuclearEngine(unittest.TestCase):
    """Test nuclear physics engine."""
    
    def setUp(self):
        from src.domains.nuclear_engine import NuclearEngine
        self.engine = NuclearEngine()
    
    def test_binding_energy(self):
        """Test binding energy calculation."""
        result = self.engine.nucleus(26, 56, 'Fe', 'Iron')
        self.assertIn('binding_energy_MeV', result)
        # Iron-56 has ~8.8 MeV/nucleon
        self.assertAlmostEqual(result['binding_per_nucleon_MeV'], 8.8, delta=0.5)
    
    def test_radioactive_decay(self):
        """Test decay calculations."""
        half_life = 5730 * 365.25 * 24 * 3600  # C-14
        result = self.engine.radioactive_decay_analysis(half_life)
        self.assertIn('fraction_remaining', result)
    
    def test_fusion_energy(self):
        """Test fusion energy."""
        result = self.engine.fusion_energy('DT')
        self.assertEqual(result['energy_MeV'], 17.6)


class TestElementsEngine(unittest.TestCase):
    """Test elements database."""
    
    def setUp(self):
        from src.domains.elements_engine import ElementsEngine
        self.engine = ElementsEngine()
    
    def test_element_by_symbol(self):
        """Test element lookup by symbol."""
        gold = self.engine.element('Au')
        self.assertEqual(gold['Z'], 79)
        self.assertEqual(gold['name'], 'Gold')
    
    def test_element_by_number(self):
        """Test element lookup by Z."""
        uranium = self.engine.element(92)
        self.assertEqual(uranium['symbol'], 'U')
    
    def test_all_elements(self):
        """Test we have all 118 elements."""
        from src.domains.elements_engine import PeriodicTable
        PeriodicTable._init_elements()
        self.assertEqual(len(PeriodicTable._elements_by_Z), 118)


class TestQuantumComputingEngine(unittest.TestCase):
    """Test quantum computing engine."""
    
    def setUp(self):
        from src.domains.quantum_computing_engine import QuantumComputingEngine
        self.engine = QuantumComputingEngine()
    
    def test_bell_state(self):
        """Test Bell state creation."""
        result = self.engine.bell_state('phi+')
        self.assertTrue(result['entangled'])
        self.assertEqual(len(result['state_vector']), 4)
    
    def test_grover_search(self):
        """Test Grover's algorithm."""
        result = self.engine.grover_search(2, 3, shots=50)
        self.assertIn('success_rate', result)
        # Should find target with high probability
        self.assertGreater(result['success_rate'], 0.5)
    
    def test_bb84(self):
        """Test BB84 simulation."""
        result = self.engine.bb84_simulation(100)
        self.assertIn('key_length', result)
        # Should have roughly 50% key rate
        self.assertGreater(result['efficiency'], 0.3)


class TestBlackholeEngine(unittest.TestCase):
    """Test black hole physics."""
    
    def setUp(self):
        from src.domains.blackhole_engine import BlackHoleEngine
        self.engine = BlackHoleEngine()
    
    def test_schwarzschild_radius(self):
        """Test event horizon calculation."""
        result = self.engine.analyze(10, 0)  # 10 solar masses
        self.assertIn('schwarzschild_radius_km', result)
        # 10 M_sun should be ~30 km
        self.assertAlmostEqual(result['schwarzschild_radius_km'], 30, delta=5)
    
    def test_hawking_temperature(self):
        """Test Hawking radiation."""
        result = self.engine.analyze(10, 0)
        self.assertIn('hawking_temperature_K', result)
        self.assertGreater(result['hawking_temperature_K'], 0)


class TestChemistryEngine(unittest.TestCase):
    """Test chemistry engine."""
    
    def setUp(self):
        from src.domains.chemistry_engine import ChemistryEngine
        self.engine = ChemistryEngine()
    
    def test_element_properties(self):
        """Test element property retrieval."""
        result = self.engine.get_element('H')
        self.assertIsNotNone(result)
    
    def test_molecular_weight(self):
        """Test molecular weight calculation."""
        result = self.engine.molecular_weight('H2O')
        self.assertAlmostEqual(result, 18.015, delta=0.1)


class TestMathEngine(unittest.TestCase):
    """Test math engine."""
    
    def setUp(self):
        from src.domains.math_engine import MathEngine
        self.engine = MathEngine()
    
    def test_differentiate(self):
        """Test symbolic differentiation."""
        result = self.engine.differentiate('x^2')
        # Accept common equivalent representations
        self.assertIn(result.replace(' ', '').replace('(', '').replace(')', ''), ['2*x', '(2*x)'])
    
    def test_evaluate(self):
        """Test expression evaluation."""
        result = self.engine.evaluate('2+3*4')
        self.assertEqual(result, 14)


class TestUnifiedPhysics(unittest.TestCase):
    """Test unified physics API."""
    
    def setUp(self):
        from src.domains.unified_physics import physics
        self.p = physics()
    
    def test_constants(self):
        """Test physical constants."""
        self.assertAlmostEqual(self.p.CONSTANTS['c'], 3e8, delta=1e6)
    
    def test_projectile(self):
        """Test projectile via unified API."""
        result = self.p.projectile(20, 45)
        self.assertIn('max_range', result)
    
    def test_element(self):
        """Test element via unified API."""
        result = self.p.element('Fe')
        self.assertEqual(result['Z'], 26)
    
    def test_quantum_gates(self):
        """Test quantum gates via unified API."""
        gates = self.p.quantum_gates()
        self.assertIn('single_qubit', gates)


if __name__ == '__main__':
    unittest.main(verbosity=2)
