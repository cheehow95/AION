"""
AION Drug Binding and Docking Module
=====================================

Provides:
- Binding pocket detection (Fpocket-like)
- Druggability scoring
- Simple docking simulation
- Binding affinity estimation
- Pharmacophore features
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class InteractionType(Enum):
    """Types of protein-ligand interactions."""
    HYDROPHOBIC = "hydrophobic"
    HYDROGEN_BOND = "hydrogen_bond"
    SALT_BRIDGE = "salt_bridge"
    PI_STACKING = "pi_stacking"
    CATION_PI = "cation_pi"
    HALOGEN_BOND = "halogen_bond"


@dataclass
class Atom3D:
    """Simple 3D atom representation."""
    x: float
    y: float
    z: float
    element: str = "C"
    residue: str = ""
    residue_idx: int = 0
    
    def distance_to(self, other: 'Atom3D') -> float:
        return math.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        )


@dataclass
class BindingPocket:
    """A detected binding pocket."""
    id: int
    center: Tuple[float, float, float]
    volume: float  # Å³
    surface_area: float  # Å²
    depth: float  # Å
    residues: List[str]  # Residue codes lining the pocket
    druggability_score: float  # 0-1
    hydrophobicity: float  # -1 to 1
    
    @property
    def is_druggable(self) -> bool:
        return self.druggability_score > 0.5 and self.volume > 200
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "center": self.center,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "depth": self.depth,
            "residues": self.residues,
            "druggability": self.druggability_score,
            "hydrophobicity": self.hydrophobicity,
            "is_druggable": self.is_druggable
        }


@dataclass
class Ligand:
    """Simple ligand representation."""
    name: str
    atoms: List[Atom3D]
    molecular_weight: float
    logP: float  # Lipophilicity
    h_bond_donors: int
    h_bond_acceptors: int
    rotatable_bonds: int
    
    @property
    def lipinski_compliant(self) -> bool:
        """Check Lipinski's Rule of Five."""
        return (
            self.molecular_weight <= 500 and
            self.logP <= 5 and
            self.h_bond_donors <= 5 and
            self.h_bond_acceptors <= 10
        )
    
    def get_center(self) -> Tuple[float, float, float]:
        if not self.atoms:
            return (0, 0, 0)
        x = sum(a.x for a in self.atoms) / len(self.atoms)
        y = sum(a.y for a in self.atoms) / len(self.atoms)
        z = sum(a.z for a in self.atoms) / len(self.atoms)
        return (x, y, z)


@dataclass
class DockingPose:
    """A docking pose for a ligand."""
    ligand_position: Tuple[float, float, float]
    orientation: Tuple[float, float, float]  # Euler angles
    binding_energy: float  # kcal/mol
    interactions: List[Dict[str, Any]]
    rmsd: float = 0.0  # From reference if available
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.ligand_position,
            "orientation": self.orientation,
            "binding_energy": self.binding_energy,
            "interactions": self.interactions,
            "rmsd": self.rmsd
        }


@dataclass
class BindingResult:
    """Complete binding analysis result."""
    pocket: BindingPocket
    ligand: Ligand
    poses: List[DockingPose]
    best_pose: DockingPose
    estimated_kd: float  # Dissociation constant (nM)
    le: float  # Ligand efficiency
    pharmacophore: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pocket": self.pocket.to_dict(),
            "ligand_name": self.ligand.name,
            "best_pose": self.best_pose.to_dict(),
            "binding_energy": self.best_pose.binding_energy,
            "estimated_kd_nM": self.estimated_kd,
            "ligand_efficiency": self.le,
            "pharmacophore": self.pharmacophore
        }


class PocketDetector:
    """
    Detects binding pockets in protein structures.
    Uses simplified alpha-sphere approach (Fpocket-like).
    """
    
    # Residue properties for pocket analysis
    RESIDUE_HYDROPHOBICITY = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
    }
    
    def __init__(self, backbone_coords: List[Tuple[float, float, float]], 
                 sequence: str):
        self.coords = backbone_coords
        self.sequence = sequence
    
    def detect_pockets(self, min_volume: float = 100) -> List[BindingPocket]:
        """Detect all binding pockets."""
        pockets = []
        n = len(self.coords)
        
        if n < 10:
            return pockets
        
        # Find cavity centers using distance clustering
        pocket_centers = self._find_cavity_centers()
        
        for idx, center in enumerate(pocket_centers):
            pocket = self._analyze_pocket(idx, center)
            if pocket.volume >= min_volume:
                pockets.append(pocket)
        
        # Sort by druggability
        pockets.sort(key=lambda p: p.druggability_score, reverse=True)
        
        return pockets
    
    def _find_cavity_centers(self) -> List[Tuple[float, float, float]]:
        """Find potential cavity centers."""
        centers = []
        n = len(self.coords)
        
        # Look for regions with high local curvature
        for i in range(2, n - 2):
            # Check if this could be a pocket center
            local_coords = [self.coords[j] for j in range(max(0, i-5), min(n, i+6))]
            
            # Calculate local center
            cx = sum(c[0] for c in local_coords) / len(local_coords)
            cy = sum(c[1] for c in local_coords) / len(local_coords)
            cz = sum(c[2] for c in local_coords) / len(local_coords)
            
            # Check if there's a "cavity" (low density region nearby)
            # Simplified: look for concave regions
            if i > 5 and i < n - 5:
                vec1 = (
                    self.coords[i][0] - self.coords[i-3][0],
                    self.coords[i][1] - self.coords[i-3][1],
                    self.coords[i][2] - self.coords[i-3][2]
                )
                vec2 = (
                    self.coords[i+3][0] - self.coords[i][0],
                    self.coords[i+3][1] - self.coords[i][1],
                    self.coords[i+3][2] - self.coords[i][2]
                )
                
                # Dot product check for concavity
                dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
                
                if dot < 0:  # Concave region
                    centers.append((cx, cy + 5, cz))  # Offset from backbone
        
        # Cluster nearby centers
        return self._cluster_centers(centers)
    
    def _cluster_centers(self, centers: List[Tuple[float, float, float]], 
                        threshold: float = 8.0) -> List[Tuple[float, float, float]]:
        """Cluster nearby pocket centers."""
        if not centers:
            return centers
        
        clustered = []
        used = set()
        
        for i, c1 in enumerate(centers):
            if i in used:
                continue
            
            cluster = [c1]
            for j, c2 in enumerate(centers[i+1:], i+1):
                dist = math.sqrt(sum((a-b)**2 for a, b in zip(c1, c2)))
                if dist < threshold:
                    cluster.append(c2)
                    used.add(j)
            
            # Average cluster
            avg = (
                sum(c[0] for c in cluster) / len(cluster),
                sum(c[1] for c in cluster) / len(cluster),
                sum(c[2] for c in cluster) / len(cluster)
            )
            clustered.append(avg)
        
        return clustered[:5]  # Max 5 pockets
    
    def _analyze_pocket(self, idx: int, center: Tuple[float, float, float]) -> BindingPocket:
        """Analyze a pocket at given center."""
        # Find residues near pocket
        nearby_residues = []
        
        for i, coord in enumerate(self.coords):
            dist = math.sqrt(sum((a-b)**2 for a, b in zip(coord, center)))
            if dist < 10:  # 10 Å radius
                if i < len(self.sequence):
                    nearby_residues.append((i, self.sequence[i]))
        
        # Calculate pocket properties
        residue_codes = [r[1] for r in nearby_residues]
        
        # Volume estimation (simplified)
        volume = len(nearby_residues) * 50  # Rough estimate
        
        # Surface area
        surface_area = volume ** (2/3) * 4.84  # Sphere approximation
        
        # Depth (distance from center to nearest backbone)
        depth = min(
            math.sqrt(sum((a-b)**2 for a, b in zip(coord, center)))
            for coord in self.coords
        ) if self.coords else 5.0
        
        # Hydrophobicity
        hydrophobicity = sum(
            self.RESIDUE_HYDROPHOBICITY.get(r, 0) 
            for r in residue_codes
        ) / max(1, len(residue_codes))
        
        # Druggability score
        druggability = self._calculate_druggability(
            volume, depth, hydrophobicity, len(residue_codes)
        )
        
        return BindingPocket(
            id=idx,
            center=center,
            volume=volume,
            surface_area=surface_area,
            depth=depth,
            residues=residue_codes,
            druggability_score=druggability,
            hydrophobicity=hydrophobicity
        )
    
    def _calculate_druggability(self, volume: float, depth: float, 
                               hydrophobicity: float, num_residues: int) -> float:
        """Calculate druggability score (0-1)."""
        score = 0.0
        
        # Volume contribution (250-1000 Å³ optimal)
        if 200 < volume < 1200:
            score += 0.3 * (1 - abs(volume - 500) / 700)
        
        # Depth (5-12 Å optimal)
        if 4 < depth < 15:
            score += 0.2 * (1 - abs(depth - 8) / 7)
        
        # Hydrophobicity (slightly hydrophobic optimal)
        if -0.5 < hydrophobicity < 1.0:
            score += 0.25
        
        # Enclosure (more residues = better defined pocket)
        if num_residues >= 8:
            score += 0.25
        elif num_residues >= 5:
            score += 0.15
        
        return min(1.0, score)


class SimpleDocking:
    """
    Simple docking simulation.
    Uses scoring function-based approach.
    """
    
    def __init__(self, pocket: BindingPocket, backbone_coords: List[Tuple[float, float, float]]):
        self.pocket = pocket
        self.coords = backbone_coords
    
    def dock_ligand(self, ligand: Ligand, num_poses: int = 10) -> List[DockingPose]:
        """Generate and score docking poses."""
        poses = []
        
        for _ in range(num_poses):
            # Generate random pose around pocket center
            position = (
                self.pocket.center[0] + random.gauss(0, 2),
                self.pocket.center[1] + random.gauss(0, 2),
                self.pocket.center[2] + random.gauss(0, 2)
            )
            
            orientation = (
                random.uniform(0, 2*math.pi),
                random.uniform(0, 2*math.pi),
                random.uniform(0, 2*math.pi)
            )
            
            # Score pose
            energy, interactions = self._score_pose(ligand, position, orientation)
            
            poses.append(DockingPose(
                ligand_position=position,
                orientation=orientation,
                binding_energy=energy,
                interactions=interactions
            ))
        
        # Sort by energy
        poses.sort(key=lambda p: p.binding_energy)
        
        return poses
    
    def _score_pose(self, ligand: Ligand, position: Tuple[float, float, float],
                   orientation: Tuple[float, float, float]) -> Tuple[float, List[Dict]]:
        """Score a docking pose."""
        energy = 0.0
        interactions = []
        
        # Distance from pocket center (should be inside)
        dist_to_center = math.sqrt(sum((a-b)**2 for a, b in zip(position, self.pocket.center)))
        if dist_to_center > 5:
            energy += 2.0 * (dist_to_center - 5)  # Penalty for leaving pocket
        
        # Van der Waals / steric
        for coord in self.coords:
            dist = math.sqrt(sum((a-b)**2 for a, b in zip(position, coord)))
            if dist < 2.5:
                energy += 10.0  # Clash
            elif dist < 6:
                # Attractive VdW
                energy -= 0.5 * math.exp(-(dist - 4)**2 / 2)
                
                if dist < 4:
                    interactions.append({
                        "type": InteractionType.HYDROPHOBIC.value,
                        "distance": dist
                    })
        
        # Hydrophobic contribution
        energy -= 0.5 * self.pocket.hydrophobicity * ligand.logP
        
        # H-bond contribution
        energy -= 0.7 * min(ligand.h_bond_donors, 3)
        energy -= 0.5 * min(ligand.h_bond_acceptors, 5)
        
        if ligand.h_bond_donors > 0:
            interactions.append({
                "type": InteractionType.HYDROGEN_BOND.value,
                "count": ligand.h_bond_donors
            })
        
        # Entropy penalty for rotatable bonds
        energy += 0.3 * ligand.rotatable_bonds
        
        return energy, interactions


class BindingAnalyzer:
    """
    Complete binding analysis pipeline.
    """
    
    def __init__(self, backbone_coords: List[Tuple[float, float, float]], 
                 sequence: str):
        self.coords = backbone_coords
        self.sequence = sequence
        self.pocket_detector = PocketDetector(backbone_coords, sequence)
    
    def find_binding_sites(self) -> List[BindingPocket]:
        """Find all potential binding sites."""
        return self.pocket_detector.detect_pockets()
    
    def dock_to_best_site(self, ligand: Ligand) -> Optional[BindingResult]:
        """Dock ligand to the best binding site."""
        pockets = self.find_binding_sites()
        
        if not pockets:
            return None
        
        best_pocket = pockets[0]  # Most druggable
        
        # Dock
        docker = SimpleDocking(best_pocket, self.coords)
        poses = docker.dock_ligand(ligand, num_poses=20)
        
        if not poses:
            return None
        
        best_pose = poses[0]
        
        # Calculate Kd from binding energy
        # ΔG = RT ln(Kd) => Kd = exp(ΔG/RT)
        R = 0.001987  # kcal/(mol·K)
        T = 300  # K
        kd_M = math.exp(best_pose.binding_energy / (R * T))
        kd_nM = kd_M * 1e9
        
        # Ligand efficiency
        heavy_atoms = ligand.molecular_weight / 12  # Rough estimate
        le = -best_pose.binding_energy / heavy_atoms if heavy_atoms > 0 else 0
        
        # Pharmacophore
        pharmacophore = self._extract_pharmacophore(best_pose, best_pocket)
        
        return BindingResult(
            pocket=best_pocket,
            ligand=ligand,
            poses=poses,
            best_pose=best_pose,
            estimated_kd=kd_nM,
            le=le,
            pharmacophore=pharmacophore
        )
    
    def _extract_pharmacophore(self, pose: DockingPose, 
                              pocket: BindingPocket) -> Dict[str, Any]:
        """Extract pharmacophore features from binding pose."""
        features = {
            "hydrophobic_centers": [],
            "h_bond_donors": [],
            "h_bond_acceptors": [],
            "positive_ionizable": [],
            "negative_ionizable": [],
            "aromatic_rings": []
        }
        
        # Based on pocket residues
        for res in pocket.residues:
            if res in 'VILMFYW':
                features["hydrophobic_centers"].append(res)
            if res in 'ST':
                features["h_bond_donors"].append(res)
            if res in 'STNDQ':
                features["h_bond_acceptors"].append(res)
            if res in 'KRH':
                features["positive_ionizable"].append(res)
            if res in 'DE':
                features["negative_ionizable"].append(res)
            if res in 'FYW':
                features["aromatic_rings"].append(res)
        
        return features


def analyze_drug_binding(sequence: str, 
                        backbone_coords: List[Tuple[float, float, float]],
                        ligand_name: str = "test_ligand") -> Dict[str, Any]:
    """
    Complete drug binding analysis.
    
    Args:
        sequence: Protein sequence
        backbone_coords: Cα coordinates
        ligand_name: Name of test ligand
    
    Returns:
        Comprehensive binding analysis
    """
    print(f"💊 Analyzing drug binding for {len(sequence)} residue protein...")
    
    # Create test ligand (drug-like properties)
    test_ligand = Ligand(
        name=ligand_name,
        atoms=[Atom3D(0, 0, 0, "C")],  # Simplified
        molecular_weight=350,
        logP=2.5,
        h_bond_donors=2,
        h_bond_acceptors=5,
        rotatable_bonds=4
    )
    
    # Analyze
    analyzer = BindingAnalyzer(backbone_coords, sequence)
    
    # Find pockets
    pockets = analyzer.find_binding_sites()
    print(f"   Found {len(pockets)} potential binding sites")
    
    # Dock to best site
    result = analyzer.dock_to_best_site(test_ligand)
    
    if result:
        print(f"   Best binding energy: {result.best_pose.binding_energy:.2f} kcal/mol")
        print(f"   Estimated Kd: {result.estimated_kd:.1f} nM")
    
    return {
        "sequence_length": len(sequence),
        "pockets": [p.to_dict() for p in pockets],
        "num_druggable_pockets": sum(1 for p in pockets if p.is_druggable),
        "binding_result": result.to_dict() if result else None,
        "ligand_lipinski": test_ligand.lipinski_compliant
    }
