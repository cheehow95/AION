"""
AION Domains Package
====================

Scientific domain engines for physics, chemistry, biology, and more.
All imports are optional - missing modules won't break the package.
"""

__all__ = []

# Core Physics
try:
    from .physics_engine import PhysicsEngine
    __all__.append('PhysicsEngine')
except ImportError:
    pass

try:
    from .optics_engine import OpticsEngine
    __all__.append('OpticsEngine')
except ImportError:
    pass

try:
    from .relativity_engine import RelativityEngine
    __all__.append('RelativityEngine')
except ImportError:
    pass

try:
    from .quantum_engine import QuantumEngine
    __all__.append('QuantumEngine')
except ImportError:
    pass

try:
    from .quantum_computing_engine import QuantumComputingEngine
    __all__.append('QuantumComputingEngine')
except ImportError:
    pass

try:
    from .particle_engine import ParticleEngine
    __all__.append('ParticleEngine')
except ImportError:
    pass

try:
    from .nuclear_engine import NuclearEngine
    __all__.append('NuclearEngine')
except ImportError:
    pass

try:
    from .blackhole_engine import BlackHoleEngine
    __all__.append('BlackHoleEngine')
except ImportError:
    pass

try:
    from .wormhole_engine import WormholeEngine
    __all__.append('WormholeEngine')
except ImportError:
    pass

try:
    from .dimensions_engine import DimensionsEngine
    __all__.append('DimensionsEngine')
except ImportError:
    pass

try:
    from .elements_engine import ElementsEngine
    __all__.append('ElementsEngine')
except ImportError:
    pass

try:
    from .unified_physics import UnifiedPhysicsEngine
    __all__.append('UnifiedPhysicsEngine')
except ImportError:
    pass

# Chemistry & Math
try:
    from .chemistry_engine import ChemistryEngine
    __all__.append('ChemistryEngine')
except ImportError:
    pass

try:
    from .math_engine import MathEngine
    __all__.append('MathEngine')
except ImportError:
    pass

# Protein/Life Sciences - Core
try:
    from .protein_folding import ProteinFolder, ProteinStructure, analyze_sequence
    __all__.extend(['ProteinFolder', 'ProteinStructure', 'analyze_sequence'])
except ImportError:
    pass

try:
    from .protein_physics import ProteinStructurePredictor
    __all__.append('ProteinStructurePredictor')
except ImportError:
    pass

try:
    from .protein_dynamics import FoldingSimulator, simulate_folding
    __all__.extend(['FoldingSimulator', 'simulate_folding'])
except ImportError:
    pass

try:
    from .protein_unified import ProteinFoldingEngine, fold_protein, predict_structure
    __all__.extend(['ProteinFoldingEngine', 'fold_protein', 'predict_structure'])
except ImportError:
    pass

# Optional modules
try:
    from .drug_binding import PocketDetector, SimpleDocking
    __all__.extend(['PocketDetector', 'SimpleDocking'])
except ImportError:
    pass

try:
    from .cellular import CellularEngine
    __all__.append('CellularEngine')
except ImportError:
    pass

try:
    from .alphafold_db import AlphaFoldDB
    __all__.append('AlphaFoldDB')
except ImportError:
    pass

try:
    from .structure_api import StructureAPI
    __all__.append('StructureAPI')
except ImportError:
    pass
