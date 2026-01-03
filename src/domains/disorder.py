"""
AION Intrinsically Disordered Protein Module
=============================================

Provides:
- Disorder prediction (IUPred-like)
- Fuzzy complex detection
- Liquid-liquid phase separation (LLPS) propensity
- Post-translational modification site prediction
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class DisorderType(Enum):
    """Types of protein disorder."""
    ORDERED = "ordered"
    WEAKLY_DISORDERED = "weakly_disordered"
    DISORDERED = "disordered"
    STRONGLY_DISORDERED = "strongly_disordered"


class PTMType(Enum):
    """Post-translational modification types."""
    PHOSPHORYLATION = "phosphorylation"
    ACETYLATION = "acetylation"
    METHYLATION = "methylation"
    UBIQUITINATION = "ubiquitination"
    SUMOYLATION = "sumoylation"
    GLYCOSYLATION = "glycosylation"


@dataclass
class DisorderRegion:
    """A disordered region in a protein."""
    start: int
    end: int
    sequence: str
    avg_score: float
    disorder_type: DisorderType
    functional_annotation: str = ""
    
    @property
    def length(self) -> int:
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "sequence": self.sequence,
            "score": self.avg_score,
            "type": self.disorder_type.value,
            "annotation": self.functional_annotation
        }


@dataclass
class PTMSite:
    """A predicted PTM site."""
    position: int
    residue: str
    ptm_type: PTMType
    score: float
    context: str  # Surrounding sequence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "residue": self.residue,
            "modification": self.ptm_type.value,
            "score": self.score,
            "context": self.context
        }


@dataclass
class LLPSPrediction:
    """Liquid-liquid phase separation prediction."""
    propensity_score: float  # 0-1
    phase_separating: bool
    driving_regions: List[Tuple[int, int]]
    sticker_residues: List[int]  # Aromatic/charged residues
    spacer_regions: List[Tuple[int, int]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "propensity": self.propensity_score,
            "phase_separating": self.phase_separating,
            "driving_regions": self.driving_regions,
            "num_stickers": len(self.sticker_residues),
            "spacer_regions": self.spacer_regions
        }


class DisorderPredictor:
    """
    Predicts intrinsically disordered regions (IDRs).
    Based on IUPred-like energy estimation approach.
    """
    
    # Energy potentials for disorder prediction
    # Lower energy = more likely ordered
    ENERGY_MATRIX = {
        'A': {'contact': 0.06, 'disorder': 0.12},
        'R': {'contact': -0.18, 'disorder': 0.25},
        'N': {'contact': -0.15, 'disorder': 0.22},
        'D': {'contact': -0.20, 'disorder': 0.28},
        'C': {'contact': 0.10, 'disorder': 0.08},
        'Q': {'contact': -0.12, 'disorder': 0.20},
        'E': {'contact': -0.18, 'disorder': 0.26},
        'G': {'contact': -0.08, 'disorder': 0.35},
        'H': {'contact': 0.02, 'disorder': 0.15},
        'I': {'contact': 0.22, 'disorder': -0.10},
        'L': {'contact': 0.18, 'disorder': -0.05},
        'K': {'contact': -0.15, 'disorder': 0.24},
        'M': {'contact': 0.12, 'disorder': 0.02},
        'F': {'contact': 0.20, 'disorder': -0.08},
        'P': {'contact': -0.25, 'disorder': 0.38},
        'S': {'contact': -0.10, 'disorder': 0.18},
        'T': {'contact': -0.05, 'disorder': 0.14},
        'W': {'contact': 0.15, 'disorder': -0.05},
        'Y': {'contact': 0.08, 'disorder': 0.05},
        'V': {'contact': 0.18, 'disorder': -0.08}
    }
    
    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
        self.n = len(sequence)
    
    def predict(self, window: int = 21) -> List[float]:
        """
        Predict disorder score for each residue.
        
        Returns scores where > 0.5 = disordered.
        """
        scores = []
        half_window = window // 2
        
        for i in range(self.n):
            # Get window
            start = max(0, i - half_window)
            end = min(self.n, i + half_window + 1)
            segment = self.sequence[start:end]
            
            # Calculate disorder energy
            disorder_energy = sum(
                self.ENERGY_MATRIX.get(aa, {'disorder': 0.1})['disorder']
                for aa in segment
            ) / len(segment)
            
            # Calculate contact energy (globular propensity)
            contact_energy = sum(
                self.ENERGY_MATRIX.get(aa, {'contact': 0})['contact']
                for aa in segment
            ) / len(segment)
            
            # Disorder score = sigmoid of (disorder - contact)
            score = 1 / (1 + math.exp(-(disorder_energy - contact_energy) * 5))
            scores.append(score)
        
        return self._smooth_scores(scores)
    
    def _smooth_scores(self, scores: List[float], window: int = 5) -> List[float]:
        """Smooth disorder scores."""
        smoothed = []
        half = window // 2
        
        for i in range(len(scores)):
            start = max(0, i - half)
            end = min(len(scores), i + half + 1)
            smoothed.append(sum(scores[start:end]) / (end - start))
        
        return smoothed
    
    def find_disordered_regions(self, threshold: float = 0.5, 
                               min_length: int = 5) -> List[DisorderRegion]:
        """Find contiguous disordered regions."""
        scores = self.predict()
        regions = []
        
        in_region = False
        region_start = 0
        
        for i, score in enumerate(scores):
            if score >= threshold:
                if not in_region:
                    in_region = True
                    region_start = i
            else:
                if in_region:
                    if i - region_start >= min_length:
                        avg_score = sum(scores[region_start:i]) / (i - region_start)
                        regions.append(self._create_region(region_start, i, avg_score))
                    in_region = False
        
        # Handle region at end
        if in_region and self.n - region_start >= min_length:
            avg_score = sum(scores[region_start:]) / (self.n - region_start)
            regions.append(self._create_region(region_start, self.n, avg_score))
        
        return regions
    
    def _create_region(self, start: int, end: int, avg_score: float) -> DisorderRegion:
        """Create a DisorderRegion object."""
        # Classify disorder type
        if avg_score > 0.8:
            disorder_type = DisorderType.STRONGLY_DISORDERED
        elif avg_score > 0.6:
            disorder_type = DisorderType.DISORDERED
        else:
            disorder_type = DisorderType.WEAKLY_DISORDERED
        
        # Functional annotation based on composition
        segment = self.sequence[start:end]
        annotation = self._annotate_region(segment)
        
        return DisorderRegion(
            start=start,
            end=end,
            sequence=segment,
            avg_score=avg_score,
            disorder_type=disorder_type,
            functional_annotation=annotation
        )
    
    def _annotate_region(self, segment: str) -> str:
        """Annotate disordered region function."""
        # Check for linear motifs
        if segment.count('S') + segment.count('T') > len(segment) * 0.3:
            return "Potential phosphorylation enriched"
        if segment.count('P') > len(segment) * 0.2:
            return "Proline-rich (potential PPII helix)"
        if segment.count('Q') + segment.count('N') > len(segment) * 0.3:
            return "Q/N-rich (potential prion-like)"
        if segment.count('E') + segment.count('D') > len(segment) * 0.3:
            return "Acidic (potential transactivation)"
        if segment.count('K') + segment.count('R') > len(segment) * 0.3:
            return "Basic (potential nucleic acid binding)"
        
        return "General IDR"


class PTMPredictor:
    """Predicts post-translational modification sites."""
    
    # PTM consensus motifs (simplified)
    PTM_MOTIFS = {
        PTMType.PHOSPHORYLATION: {
            'S': ['...SP..', '...S..K', 'R..S...'],  # Kinase motifs
            'T': ['...TP..', '...T..K'],
            'Y': ['...Y...']  # Tyrosine
        },
        PTMType.ACETYLATION: {
            'K': ['...K...']  # Lysine acetylation
        },
        PTMType.METHYLATION: {
            'K': ['...K...'],  # Lysine methylation
            'R': ['...RG..', '...R...']  # Arginine methylation
        },
        PTMType.UBIQUITINATION: {
            'K': ['...K...']  # Lysine ubiquitination
        },
        PTMType.GLYCOSYLATION: {
            'N': ['N.S', 'N.T'],  # N-linked
            'S': ['...S...'],  # O-linked
            'T': ['...T...']
        }
    }
    
    def __init__(self, sequence: str, disorder_scores: List[float] = None):
        self.sequence = sequence.upper()
        self.disorder_scores = disorder_scores or [0.5] * len(sequence)
    
    def predict_sites(self) -> List[PTMSite]:
        """Predict all potential PTM sites."""
        sites = []
        
        for ptm_type, residue_motifs in self.PTM_MOTIFS.items():
            for target_residue, motifs in residue_motifs.items():
                sites.extend(self._find_sites(ptm_type, target_residue))
        
        # Sort by score
        sites.sort(key=lambda s: s.score, reverse=True)
        
        return sites
    
    def _find_sites(self, ptm_type: PTMType, target: str) -> List[PTMSite]:
        """Find sites for a specific PTM type."""
        sites = []
        
        for i, aa in enumerate(self.sequence):
            if aa != target:
                continue
            
            # Calculate score based on context
            score = self._calculate_ptm_score(i, ptm_type)
            
            # PTMs often occur in disordered regions
            if i < len(self.disorder_scores):
                score *= (1 + 0.5 * self.disorder_scores[i])
            
            if score > 0.3:  # Threshold
                context = self.sequence[max(0, i-3):min(len(self.sequence), i+4)]
                sites.append(PTMSite(
                    position=i,
                    residue=aa,
                    ptm_type=ptm_type,
                    score=min(1.0, score),
                    context=context
                ))
        
        return sites
    
    def _calculate_ptm_score(self, pos: int, ptm_type: PTMType) -> float:
        """Calculate PTM likelihood score."""
        score = 0.4  # Base score
        
        # Get context
        start = max(0, pos - 5)
        end = min(len(self.sequence), pos + 6)
        context = self.sequence[start:end]
        
        # Phosphorylation: check for kinase motifs
        if ptm_type == PTMType.PHOSPHORYLATION:
            if 'P' in context[context.find(self.sequence[pos])+1:]:
                score += 0.3  # Proline-directed
            if 'K' in context or 'R' in context:
                score += 0.2  # Basic residue nearby
        
        # Acetylation: prefers certain contexts
        elif ptm_type == PTMType.ACETYLATION:
            if self.disorder_scores[pos] > 0.5:
                score += 0.2
        
        return score


class LLPSPredictor:
    """
    Predicts liquid-liquid phase separation propensity.
    Based on sticker-spacer model.
    """
    
    # Sticker residues (promote phase separation)
    STICKERS = {'F', 'Y', 'W', 'R', 'K'}
    
    # π-cation and charge interactions
    PI_RESIDUES = {'F', 'Y', 'W'}
    CATION_RESIDUES = {'R', 'K'}
    
    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
    
    def predict(self) -> LLPSPrediction:
        """Predict LLPS propensity."""
        # Find sticker residues
        stickers = [i for i, aa in enumerate(self.sequence) if aa in self.STICKERS]
        
        # Find driving regions (aromatic + charged clusters)
        driving_regions = self._find_driving_regions()
        
        # Find spacer regions (between stickers)
        spacer_regions = self._find_spacer_regions(stickers)
        
        # Calculate propensity score
        propensity = self._calculate_propensity(stickers, driving_regions)
        
        return LLPSPrediction(
            propensity_score=propensity,
            phase_separating=propensity > 0.5,
            driving_regions=driving_regions,
            sticker_residues=stickers,
            spacer_regions=spacer_regions
        )
    
    def _find_driving_regions(self, window: int = 20) -> List[Tuple[int, int]]:
        """Find regions that drive phase separation."""
        regions = []
        n = len(self.sequence)
        
        for i in range(0, n - window + 1, window // 2):
            segment = self.sequence[i:i+window]
            
            # Count stickers
            sticker_count = sum(1 for aa in segment if aa in self.STICKERS)
            
            # Check for π-cation motifs
            pi_count = sum(1 for aa in segment if aa in self.PI_RESIDUES)
            cation_count = sum(1 for aa in segment if aa in self.CATION_RESIDUES)
            
            if sticker_count >= 4 or (pi_count >= 2 and cation_count >= 2):
                regions.append((i, min(i + window, n)))
        
        return self._merge_regions(regions)
    
    def _find_spacer_regions(self, stickers: List[int]) -> List[Tuple[int, int]]:
        """Find spacer regions between stickers."""
        spacers = []
        
        for i in range(len(stickers) - 1):
            gap = stickers[i+1] - stickers[i]
            if gap > 5:  # Significant spacer
                spacers.append((stickers[i] + 1, stickers[i+1]))
        
        return spacers
    
    def _merge_regions(self, regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping regions."""
        if not regions:
            return []
        
        merged = [regions[0]]
        for start, end in regions[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        return merged
    
    def _calculate_propensity(self, stickers: List[int], 
                             driving_regions: List[Tuple[int, int]]) -> float:
        """Calculate overall LLPS propensity."""
        n = len(self.sequence)
        
        # Sticker density
        sticker_density = len(stickers) / n if n > 0 else 0
        
        # Driving region coverage
        driving_coverage = sum(e - s for s, e in driving_regions) / n if n > 0 else 0
        
        # Aromatic content
        aromatic_count = sum(1 for aa in self.sequence if aa in 'FYW')
        aromatic_ratio = aromatic_count / n if n > 0 else 0
        
        # Combined score
        propensity = (
            0.4 * min(sticker_density * 10, 1) +
            0.3 * driving_coverage +
            0.3 * min(aromatic_ratio * 10, 1)
        )
        
        return min(1.0, propensity)


def analyze_disorder(sequence: str) -> Dict[str, Any]:
    """
    Complete disorder analysis.
    
    Returns comprehensive disorder, PTM, and LLPS predictions.
    """
    print(f"🌊 Analyzing disorder for {len(sequence)} residue protein...")
    
    # Disorder prediction
    disorder_pred = DisorderPredictor(sequence)
    disorder_scores = disorder_pred.predict()
    regions = disorder_pred.find_disordered_regions()
    
    print(f"   Found {len(regions)} disordered regions")
    
    # PTM prediction
    ptm_pred = PTMPredictor(sequence, disorder_scores)
    ptm_sites = ptm_pred.predict_sites()
    
    print(f"   Found {len(ptm_sites)} potential PTM sites")
    
    # LLPS prediction
    llps_pred = LLPSPredictor(sequence)
    llps = llps_pred.predict()
    
    print(f"   LLPS propensity: {llps.propensity_score:.2f}")
    
    # Calculate statistics
    disordered_fraction = sum(1 for s in disorder_scores if s > 0.5) / len(disorder_scores)
    
    return {
        "sequence_length": len(sequence),
        "disorder_scores": disorder_scores,
        "disordered_fraction": disordered_fraction,
        "disordered_regions": [r.to_dict() for r in regions],
        "ptm_sites": [s.to_dict() for s in ptm_sites[:20]],  # Top 20
        "llps": llps.to_dict(),
        "classification": "IDP" if disordered_fraction > 0.5 else "Ordered"
    }
