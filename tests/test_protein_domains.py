"""
Tests for AION Protein Domain Modules

Tests for:
- protein_folding.py (HP lattice model, Monte Carlo)
- protein_physics.py (Structure predictor, secondary structure)
- alphafold_db.py (AlphaFold API client)
"""

import pytest
import asyncio
import sys
import numpy as np
sys.path.insert(0, '.')

# ============================================================================
# Tests for protein_folding.py
# ============================================================================

from src.domains.protein_folding import (
    AminoAcid, AminoAcidProperties, AA_PROPERTIES,
    ProteinStructure, ProteinFolder, analyze_sequence
)


class TestAminoAcidEnum:
    """Test AminoAcid enumeration."""
    
    def test_all_20_amino_acids_exist(self):
        """Test all standard amino acids are defined."""
        expected = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                    'THR', 'TRP', 'TYR', 'VAL']
        actual = [aa.name for aa in AminoAcid]
        assert sorted(actual) == sorted(expected)
    
    def test_amino_acid_single_letter_codes(self):
        """Test single letter codes are correct."""
        assert AminoAcid.ALA.value == "A"
        assert AminoAcid.GLY.value == "G"
        assert AminoAcid.TRP.value == "W"


class TestAminoAcidProperties:
    """Test amino acid property database."""
    
    def test_properties_database_complete(self):
        """Test all 20 amino acids have properties."""
        assert len(AA_PROPERTIES) == 20
    
    def test_hydrophobicity_values(self):
        """Test hydrophobicity scale is reasonable."""
        # Isoleucine is most hydrophobic
        assert AA_PROPERTIES["I"].hydrophobicity > 4.0
        # Arginine is most hydrophilic
        assert AA_PROPERTIES["R"].hydrophobicity < -4.0
    
    def test_charge_values(self):
        """Test charge assignments."""
        # Positive: R, K, H
        assert AA_PROPERTIES["R"].charge > 0
        assert AA_PROPERTIES["K"].charge > 0
        # Negative: D, E
        assert AA_PROPERTIES["D"].charge < 0
        assert AA_PROPERTIES["E"].charge < 0
        # Neutral: A
        assert AA_PROPERTIES["A"].charge == 0


class TestProteinStructure:
    """Test ProteinStructure class."""
    
    def test_structure_creation(self):
        """Test creating a protein structure."""
        coords = [(0, 0), (1, 0), (2, 0)]
        structure = ProteinStructure("AAA", coords)
        
        assert structure.sequence == "AAA"
        assert len(structure.coordinates) == 3
        assert structure.energy == 0.0
    
    def test_energy_calculation_hydrophobic(self):
        """Test energy calculation for hydrophobic contacts."""
        # Create structure where hydrophobic residues are neighbors
        # I and L are hydrophobic
        coords = [(0, 0), (1, 0), (2, 0), (2, 1)]  # Last two non-adjacent in seq, but adjacent on lattice
        structure = ProteinStructure("AAIL", coords)
        
        energy = structure.calculate_energy()
        # Should have some energy contribution
        assert isinstance(energy, float)
    
    def test_energy_electrostatic_attraction(self):
        """Test opposite charges attract."""
        # K is positive (index 0), D is negative (index 3)
        # They must be at least 2 positions apart in sequence, but adjacent on lattice
        # Sequence: KAAD - K at (0,0), A at (1,0), A at (1,1), D at (0,1)
        # K and D are 3 apart in sequence and adjacent on lattice (Manhattan dist = 1)
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        structure = ProteinStructure("KAAD", coords)
        
        energy = structure.calculate_energy()
        # Opposite charges should give negative (favorable) energy
        assert energy < 0


class TestProteinFolder:
    """Test ProteinFolder Monte Carlo simulation."""
    
    def test_folder_initialization(self):
        """Test folder initialization."""
        folder = ProteinFolder("ACDEFGH")
        
        assert folder.sequence == "ACDEFGH"
        assert folder.length == 7
        assert folder.best_structure is None
    
    def test_random_conformation_generation(self):
        """Test generating random conformations."""
        folder = ProteinFolder("AAAA")
        structure = folder.generate_random_conformation()
        
        assert len(structure.coordinates) == 4
        assert len(set(structure.coordinates)) == 4  # All unique (self-avoiding)
    
    def test_folding_simulation(self):
        """Test Monte Carlo folding simulation."""
        folder = ProteinFolder("HPHHPPHH")  # H=hydrophobic-like, P=polar-like
        result = folder.fold(iterations=100)
        
        assert result is not None
        assert len(result.coordinates) == 8
        assert result.energy is not None
    
    def test_visualization(self):
        """Test ASCII visualization."""
        folder = ProteinFolder("AAA")
        structure = folder.generate_random_conformation()
        viz = folder.visualize_structure(structure)
        
        assert isinstance(viz, str)
        assert len(viz) > 0


class TestAnalyzeSequence:
    """Test sequence analysis function."""
    
    def test_basic_analysis(self):
        """Test basic sequence analysis."""
        result = analyze_sequence("AAKDE")
        
        assert result['length'] == 5
        assert 'hydrophobic_count' in result
        assert 'charged_count' in result
        assert 'net_charge' in result
    
    def test_hydrophobic_counting(self):
        """Test hydrophobic residue counting."""
        # I, L, V, F are hydrophobic (hydrophobicity > 1.0)
        result = analyze_sequence("ILVF")
        assert result['hydrophobic_count'] == 4
    
    def test_net_charge_calculation(self):
        """Test net charge calculation."""
        # K+, R+ = +2, D-, E- = -2, net = 0
        result = analyze_sequence("KRDE")
        assert result['net_charge'] == 0


# ============================================================================
# Tests for protein_physics.py
# ============================================================================

from src.domains.protein_physics import (
    SecondaryStructure, BackboneAngles, AminoAcidChemistry,
    AMINO_ACID_CHEMISTRY, Atom3D, Residue3D, ProteinStructurePredictor
)


class TestSecondaryStructure:
    """Test SecondaryStructure enum."""
    
    def test_structure_types(self):
        """Test all secondary structure types exist."""
        assert SecondaryStructure.COIL.value == "coil"
        assert SecondaryStructure.HELIX.value == "helix"
        assert SecondaryStructure.SHEET.value == "sheet"
        assert SecondaryStructure.TURN.value == "turn"


class TestBackboneAngles:
    """Test BackboneAngles class."""
    
    def test_alpha_helix_angles(self):
        """Test alpha helix ideal angles."""
        angles = BackboneAngles.alpha_helix()
        assert angles.phi == -60
        assert angles.psi == -45
    
    def test_beta_sheet_angles(self):
        """Test beta sheet ideal angles."""
        angles = BackboneAngles.beta_sheet()
        assert angles.phi == -120
        assert angles.psi == 135
    
    def test_random_coil_angles(self):
        """Test random coil generates valid angles."""
        angles = BackboneAngles.random_coil()
        assert -180 <= angles.phi <= 0
        assert -60 <= angles.psi <= 180


class TestAminoAcidChemistry:
    """Test amino acid chemistry database."""
    
    def test_chemistry_database_complete(self):
        """Test all amino acids have chemistry data."""
        assert len(AMINO_ACID_CHEMISTRY) == 20
    
    def test_special_properties(self):
        """Test special amino acid flags."""
        assert AMINO_ACID_CHEMISTRY['P'].is_proline is True
        assert AMINO_ACID_CHEMISTRY['G'].is_glycine is True
        assert AMINO_ACID_CHEMISTRY['C'].is_cysteine is True
    
    def test_propensities(self):
        """Test secondary structure propensities."""
        # Alanine has high helix propensity
        assert AMINO_ACID_CHEMISTRY['A'].helix_propensity > 1.0
        # Valine has high sheet propensity
        assert AMINO_ACID_CHEMISTRY['V'].sheet_propensity > 1.5


class TestAtom3D:
    """Test Atom3D class."""
    
    def test_atom_creation(self):
        """Test creating an atom."""
        atom = Atom3D(1.0, 2.0, 3.0, 'C', 'CA')
        
        assert atom.x == 1.0
        assert atom.y == 2.0
        assert atom.z == 3.0
        assert atom.element == 'C'
    
    def test_to_array(self):
        """Test converting to numpy array."""
        atom = Atom3D(1.0, 2.0, 3.0)
        arr = atom.to_array()
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        assert list(arr) == [1.0, 2.0, 3.0]
    
    def test_distance_calculation(self):
        """Test distance calculation between atoms."""
        atom1 = Atom3D(0, 0, 0)
        atom2 = Atom3D(3, 4, 0)
        
        dist = atom1.distance_to(atom2)
        assert abs(dist - 5.0) < 0.001  # 3-4-5 triangle


class TestResidue3D:
    """Test Residue3D class."""
    
    def test_residue_creation(self):
        """Test creating a residue."""
        chem = AMINO_ACID_CHEMISTRY['A']
        residue = Residue3D(0, 'A', chem)
        
        assert residue.index == 0
        assert residue.aa_code == 'A'
        assert residue.secondary == SecondaryStructure.COIL
        assert residue.confidence == 70.0


class TestProteinStructurePredictor:
    """Test ProteinStructurePredictor class."""
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = ProteinStructurePredictor("ACDEFGH")
        
        assert predictor.sequence == "ACDEFGH"
        assert len(predictor.residues) == 0  # Before prediction
    
    def test_structure_prediction(self):
        """Test full structure prediction."""
        predictor = ProteinStructurePredictor("AEAAAKEAAAK")  # Helix-forming
        residues = predictor.predict()
        
        assert len(residues) == 11
        assert all(r.ca is not None for r in residues)
    
    def test_secondary_structure_prediction(self):
        """Test secondary structure is assigned."""
        predictor = ProteinStructurePredictor("AEAAAKEAAAK")
        predictor.predict()
        
        # Check secondary structure is assigned
        for res in predictor.residues:
            assert res.secondary in [SecondaryStructure.HELIX, 
                                     SecondaryStructure.SHEET,
                                     SecondaryStructure.COIL,
                                     SecondaryStructure.TURN]
    
    def test_get_summary(self):
        """Test structure summary generation."""
        predictor = ProteinStructurePredictor("ACDEFGH")
        predictor.predict()
        summary = predictor.get_summary()
        
        assert 'length' in summary
        assert 'helix_percent' in summary
        assert 'sheet_percent' in summary
        assert 'radius_of_gyration' in summary
        assert 'average_confidence' in summary


# ============================================================================
# Tests for alphafold_db.py
# ============================================================================

from src.domains.alphafold_db import (
    AlphaFoldStructure, AlphaFoldDB, AlphaFoldAgent, PROTEOMES
)


class TestProteomes:
    """Test proteome database."""
    
    def test_proteomes_exist(self):
        """Test proteome database is populated."""
        assert len(PROTEOMES) >= 8
    
    def test_human_proteome(self):
        """Test human proteome entry."""
        assert 'human' in PROTEOMES
        assert 'id' in PROTEOMES['human']
        assert 'size_mb' in PROTEOMES['human']


class TestAlphaFoldStructure:
    """Test AlphaFoldStructure dataclass."""
    
    def test_structure_creation(self):
        """Test creating a structure."""
        structure = AlphaFoldStructure(
            uniprot_id="P00533",
            gene_name="EGFR",
            organism="Homo sapiens",
            sequence_length=1210,
            pdb_url="https://example.com/AF-P00533-F1.pdb",
            cif_url="https://example.com/AF-P00533-F1.cif",
            pae_url="https://example.com/AF-P00533-F1_pae.json",
            confidence_score=0.92
        )
        
        assert structure.uniprot_id == "P00533"
        assert structure.gene_name == "EGFR"
        assert structure.experimental_resolved is False
    
    def test_get_download_urls(self):
        """Test getting download URLs."""
        structure = AlphaFoldStructure(
            uniprot_id="P00533",
            gene_name="EGFR",
            organism="Homo sapiens",
            sequence_length=1210,
            pdb_url="https://example.com/test.pdb",
            cif_url="https://example.com/test.cif",
            pae_url="https://example.com/test_pae.json",
            confidence_score=0.92
        )
        
        urls = structure.get_download_urls()
        assert 'pdb' in urls
        assert 'cif' in urls
        assert 'pae' in urls


class TestAlphaFoldDB:
    """Test AlphaFoldDB API client."""
    
    def test_db_initialization(self):
        """Test database initialization."""
        db = AlphaFoldDB()
        
        assert db.cache == {}
        assert db.stats['queries'] == 0
    
    def test_get_structure_urls(self):
        """Test URL generation."""
        db = AlphaFoldDB()
        urls = db.get_structure_urls("P00533")
        
        assert 'pdb' in urls
        assert 'cif' in urls
        assert 'P00533' in urls['pdb']
        assert 'alphafold' in urls['pdb'].lower()
    
    def test_get_example_proteins(self):
        """Test getting example proteins."""
        db = AlphaFoldDB()
        examples = db.get_example_proteins()
        
        assert len(examples) >= 5
        assert all('id' in p for p in examples)
        assert all('name' in p for p in examples)
    
    def test_get_proteome_download_url(self):
        """Test proteome download URL."""
        db = AlphaFoldDB()
        url = db.get_proteome_download_url("human")
        
        assert url is not None
        assert "alphafold" in url.lower()
        assert "tar" in url
    
    def test_proteome_url_invalid_organism(self):
        """Test proteome URL for invalid organism."""
        db = AlphaFoldDB()
        url = db.get_proteome_download_url("martian")
        
        assert url is None
    
    @pytest.mark.asyncio
    async def test_fetch_structure(self):
        """Test fetching structure metadata."""
        db = AlphaFoldDB()
        structure = await db.fetch_structure("P00533")
        
        assert structure is not None
        assert structure.uniprot_id == "P00533"
        assert db.stats['queries'] == 1
    
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache functionality."""
        db = AlphaFoldDB()
        
        # First fetch
        await db.fetch_structure("P00533")
        assert db.stats['cache_hits'] == 0
        
        # Second fetch - should be cached
        await db.fetch_structure("P00533")
        assert db.stats['cache_hits'] == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestDomainIntegration:
    """Integration tests across domain modules."""
    
    def test_folding_to_physics_workflow(self):
        """Test workflow from folding to physics analysis."""
        # Generate sequence
        sequence = "AEAAAKEAAAKEAAAK"  # Helix-forming
        
        # Analyze with basic folding
        analysis = analyze_sequence(sequence)
        assert analysis['length'] == 16
        
        # Predict with physics
        predictor = ProteinStructurePredictor(sequence)
        residues = predictor.predict()
        
        assert len(residues) == 16
        # Should have some helix content for this sequence
        summary = predictor.get_summary()
        assert summary['helix_percent'] > 0 or summary['sheet_percent'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
