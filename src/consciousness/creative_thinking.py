"""
AION Creative Thinking Engine
=============================

Human-like creative cognition capabilities:
- Divergent thinking (generate many ideas)
- Convergent thinking (refine to best ideas)
- Analogical reasoning (connect distant concepts)
- Conceptual blending (merge concepts creatively)
- Imagination and mental simulation
- Insight and "aha" moments
- Dream synthesis (unconscious creativity)
- Intuition modeling

Enables truly creative, human-like thought.
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from enum import Enum
from collections import defaultdict
import json


# =============================================================================
# CREATIVITY TYPES
# =============================================================================

class CreativityMode(Enum):
    """Modes of creative thinking."""
    DIVERGENT = "divergent"        # Generate many possibilities
    CONVERGENT = "convergent"      # Refine to best solutions
    LATERAL = "lateral"            # Unexpected connections
    ANALOGICAL = "analogical"      # Cross-domain transfer
    ASSOCIATIVE = "associative"    # Free association
    COMBINATORIAL = "combinatorial" # Combine existing ideas
    TRANSFORMATIVE = "transformative" # Radically new concepts


class EmotionalTone(Enum):
    """Emotional coloring for creative output."""
    CURIOUS = "curious"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    WHIMSICAL = "whimsical"
    MELANCHOLIC = "melancholic"
    HOPEFUL = "hopeful"
    MYSTERIOUS = "mysterious"
    PASSIONATE = "passionate"


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Concept:
    """A conceptual unit for creative thinking."""
    name: str
    domain: str
    properties: Dict[str, Any] = field(default_factory=dict)
    associations: List[str] = field(default_factory=list)
    abstraction_level: float = 0.5  # 0=concrete, 1=abstract
    emotional_valence: float = 0.0  # -1=negative, +1=positive
    novelty: float = 0.5  # How novel/unusual


@dataclass
class Idea:
    """A creative idea or insight."""
    content: str
    concepts: List[str]
    novelty_score: float
    usefulness_score: float
    surprise_score: float
    origin: str  # How it was generated
    elaborations: List[str] = field(default_factory=list)
    
    @property
    def creativity_score(self) -> float:
        """Combined creativity metric (novelty × usefulness)."""
        return self.novelty_score * self.usefulness_score


@dataclass
class MentalImage:
    """Mental imagery for imagination."""
    description: str
    sensory_modality: str  # visual, auditory, etc.
    vividness: float
    emotional_tone: EmotionalTone
    associated_concepts: List[str]


@dataclass
class Analogy:
    """A cross-domain analogy."""
    source_domain: str
    target_domain: str
    source_concept: str
    target_concept: str
    mapping: Dict[str, str]
    strength: float
    insight: str


@dataclass
class ConceptBlend:
    """A conceptual blend of two ideas."""
    input1: Concept
    input2: Concept
    emergent_structure: Dict[str, Any]
    novel_properties: List[str]
    name: str
    description: str


# =============================================================================
# DIVERGENT THINKING
# =============================================================================

class DivergentThinker:
    """
    Generate many diverse ideas.
    Models fluency, flexibility, originality, and elaboration.
    """
    
    def __init__(self):
        self.idea_cache = []
        self.association_network = defaultdict(set)
    
    def brainstorm(self, seed: str, n: int = 10, 
                   mode: CreativityMode = CreativityMode.DIVERGENT) -> List[Idea]:
        """Generate n diverse ideas from a seed concept."""
        ideas = []
        
        # Different strategies based on mode
        if mode == CreativityMode.DIVERGENT:
            ideas = self._fluent_generation(seed, n)
        elif mode == CreativityMode.LATERAL:
            ideas = self._lateral_generation(seed, n)
        elif mode == CreativityMode.ASSOCIATIVE:
            ideas = self._associative_generation(seed, n)
        elif mode == CreativityMode.COMBINATORIAL:
            ideas = self._combinatorial_generation(seed, n)
        
        # Score for novelty
        for idea in ideas:
            idea.novelty_score = self._assess_novelty(idea)
            idea.surprise_score = self._assess_surprise(idea)
        
        return ideas
    
    def _fluent_generation(self, seed: str, n: int) -> List[Idea]:
        """Generate many ideas quickly (fluency)."""
        ideas = []
        
        # Technique 1: What if questions
        what_ifs = [
            f"What if {seed} could fly?",
            f"What if {seed} was invisible?",
            f"What if {seed} could talk?",
            f"What if we reversed {seed}?",
            f"What if {seed} was 100x bigger?",
            f"What if {seed} was microscopic?",
            f"What if {seed} existed underwater?",
            f"What if {seed} was alive?",
        ]
        
        for wif in what_ifs[:n//2]:
            ideas.append(Idea(
                content=wif,
                concepts=[seed],
                novelty_score=0.0,
                usefulness_score=0.5,
                surprise_score=0.0,
                origin="what_if"
            ))
        
        # Technique 2: Random modifiers
        modifiers = ["quantum", "musical", "edible", "ancient", "futuristic",
                     "emotional", "fractal", "living", "crystalline", "ethereal"]
        
        for mod in random.sample(modifiers, min(n//2, len(modifiers))):
            ideas.append(Idea(
                content=f"{mod} {seed}",
                concepts=[seed, mod],
                novelty_score=0.0,
                usefulness_score=0.5,
                surprise_score=0.0,
                origin="modifier"
            ))
        
        return ideas[:n]
    
    def _lateral_generation(self, seed: str, n: int) -> List[Idea]:
        """Generate ideas by unexpected connections."""
        ideas = []
        
        # Random domain jumps
        domains = ["music", "cooking", "sports", "nature", "technology",
                   "art", "medicine", "astronomy", "dance", "architecture"]
        
        for domain in random.sample(domains, min(n, len(domains))):
            connection = self._find_unexpected_connection(seed, domain)
            ideas.append(Idea(
                content=connection,
                concepts=[seed, domain],
                novelty_score=0.0,
                usefulness_score=0.5,
                surprise_score=0.0,
                origin="lateral"
            ))
        
        return ideas
    
    def _find_unexpected_connection(self, concept: str, domain: str) -> str:
        """Find unexpected connection between concept and domain."""
        connections = {
            "music": f"{concept} as a symphony of interactions",
            "cooking": f"Recipe for the perfect {concept}",
            "sports": f"The championship of {concept}",
            "nature": f"{concept} as an ecosystem",
            "technology": f"The algorithm of {concept}",
            "art": f"{concept} as a masterpiece",
            "medicine": f"Diagnosing problems in {concept}",
            "astronomy": f"{concept} as a constellation of ideas",
            "dance": f"The choreography of {concept}",
            "architecture": f"Building the structure of {concept}",
        }
        return connections.get(domain, f"{concept} meets {domain}")
    
    def _associative_generation(self, seed: str, n: int) -> List[Idea]:
        """Free association chain."""
        ideas = []
        current = seed
        chain = [seed]
        
        for _ in range(n):
            # Get free association
            next_concept = self._free_associate(current)
            chain.append(next_concept)
            
            ideas.append(Idea(
                content=f"From {seed} to {next_concept}: {' → '.join(chain[-3:])}",
                concepts=chain[-3:],
                novelty_score=0.0,
                usefulness_score=0.5,
                surprise_score=0.0,
                origin="association"
            ))
            current = next_concept
        
        return ideas
    
    def _free_associate(self, concept: str) -> str:
        """Generate free association."""
        # Simplified - in real system would use embeddings
        associations = {
            "default": ["dream", "memory", "future", "connection", "pattern",
                       "flow", "energy", "space", "time", "color", "sound",
                       "texture", "movement", "stillness", "growth", "change"]
        }
        return random.choice(associations.get(concept, associations["default"]))
    
    def _combinatorial_generation(self, seed: str, n: int) -> List[Idea]:
        """Combine with other concepts."""
        ideas = []
        combinators = ["and", "with", "through", "via", "despite", 
                       "because of", "transforming into", "dancing with"]
        
        concepts = ["light", "water", "time", "music", "nature", "code",
                   "emotion", "memory", "dream", "space"]
        
        for _ in range(n):
            combinator = random.choice(combinators)
            other = random.choice(concepts)
            ideas.append(Idea(
                content=f"{seed} {combinator} {other}",
                concepts=[seed, other],
                novelty_score=0.0,
                usefulness_score=0.5,
                surprise_score=0.0,
                origin="combinatorial"
            ))
        
        return ideas
    
    def _assess_novelty(self, idea: Idea) -> float:
        """Assess how novel an idea is."""
        # Check against cached ideas
        if idea.content in [i.content for i in self.idea_cache]:
            return 0.1
        
        # Novel combinations get higher scores
        if len(idea.concepts) > 1:
            return 0.7 + random.random() * 0.3
        
        return 0.4 + random.random() * 0.3
    
    def _assess_surprise(self, idea: Idea) -> float:
        """Assess how surprising an idea is."""
        # Lateral and associative ideas tend to be more surprising
        if idea.origin in ["lateral", "association"]:
            return 0.7 + random.random() * 0.3
        return 0.3 + random.random() * 0.4


# =============================================================================
# ANALOGICAL REASONING
# =============================================================================

class AnalogicalReasoner:
    """
    Find and apply analogies across domains.
    Based on Structure-Mapping Theory.
    """
    
    def __init__(self):
        self.domain_knowledge = {}
        self.analogy_cache = []
    
    def find_analogy(self, source: str, source_domain: str,
                     target_domain: str) -> Analogy:
        """Find analogy from source domain to target domain."""
        
        # Find structural correspondences
        mapping = self._find_structural_mapping(source, source_domain, target_domain)
        
        # Generate insight
        insight = self._generate_insight(source, mapping, target_domain)
        
        analogy = Analogy(
            source_domain=source_domain,
            target_domain=target_domain,
            source_concept=source,
            target_concept=mapping.get('target_concept', ''),
            mapping=mapping,
            strength=self._assess_analogy_strength(mapping),
            insight=insight
        )
        
        self.analogy_cache.append(analogy)
        return analogy
    
    def _find_structural_mapping(self, source: str, source_domain: str,
                                  target_domain: str) -> Dict[str, str]:
        """Find structural correspondences."""
        # Simplified mapping - real system would use deeper analysis
        domain_mappings = {
            ("physics", "economics"): {
                "energy": "money",
                "force": "incentive",
                "momentum": "market trend",
                "equilibrium": "market equilibrium",
                "entropy": "market uncertainty"
            },
            ("biology", "organization"): {
                "cell": "team",
                "organism": "company",
                "ecosystem": "market",
                "evolution": "innovation",
                "DNA": "culture",
                "metabolism": "operations"
            },
            ("music", "writing"): {
                "melody": "narrative",
                "harmony": "themes",
                "rhythm": "pacing",
                "composition": "story structure",
                "improvisation": "creative writing"
            }
        }
        
        key = (source_domain, target_domain)
        reverse_key = (target_domain, source_domain)
        
        if key in domain_mappings:
            mapping = domain_mappings[key]
        elif reverse_key in domain_mappings:
            mapping = {v: k for k, v in domain_mappings[reverse_key].items()}
        else:
            # Generate generic mapping
            mapping = {
                "structure": "structure",
                "process": "process",
                "relationship": "relationship"
            }
        
        mapping["target_concept"] = f"{source} in {target_domain}"
        return mapping
    
    def _generate_insight(self, source: str, mapping: Dict, 
                          target_domain: str) -> str:
        """Generate insight from analogy."""
        return f"Just as {source} works in its domain, " \
               f"we can apply similar principles to {target_domain}"
    
    def _assess_analogy_strength(self, mapping: Dict) -> float:
        """Assess how strong/useful an analogy is."""
        # More mappings = stronger analogy
        return min(1.0, len(mapping) * 0.15)
    
    def explain_through_analogy(self, concept: str, 
                                familiar_domain: str) -> str:
        """Explain a concept using familiar domain analogy."""
        analogy = self.find_analogy(concept, "abstract", familiar_domain)
        return f"{concept} is like {analogy.mapping.get('target_concept', 'something similar')} in {familiar_domain}. {analogy.insight}"


# =============================================================================
# CONCEPTUAL BLENDING
# =============================================================================

class ConceptualBlender:
    """
    Blend concepts to create novel emergent structures.
    Based on Fauconnier & Turner's Conceptual Blending Theory.
    """
    
    def blend(self, concept1: Concept, concept2: Concept) -> ConceptBlend:
        """Create a conceptual blend of two concepts."""
        
        # Find shared structure (generic space)
        shared = self._find_shared_structure(concept1, concept2)
        
        # Project elements from both inputs
        projected1 = self._project_elements(concept1, shared)
        projected2 = self._project_elements(concept2, shared)
        
        # Create emergent structure
        emergent = self._create_emergent_structure(projected1, projected2)
        
        # Find novel properties
        novel = self._find_novel_properties(concept1, concept2, emergent)
        
        # Generate blend name and description
        name = self._generate_blend_name(concept1, concept2)
        description = self._generate_blend_description(concept1, concept2, emergent)
        
        return ConceptBlend(
            input1=concept1,
            input2=concept2,
            emergent_structure=emergent,
            novel_properties=novel,
            name=name,
            description=description
        )
    
    def _find_shared_structure(self, c1: Concept, c2: Concept) -> Dict:
        """Find generic space shared by both concepts."""
        shared = {}
        
        # Check for shared properties
        for prop in c1.properties:
            if prop in c2.properties:
                shared[prop] = "shared"
        
        # Check for shared associations
        shared_assoc = set(c1.associations) & set(c2.associations)
        if shared_assoc:
            shared["associations"] = list(shared_assoc)
        
        return shared
    
    def _project_elements(self, concept: Concept, shared: Dict) -> Dict:
        """Project concept elements to blend space."""
        projection = {
            "name": concept.name,
            "unique_properties": {k: v for k, v in concept.properties.items() 
                                  if k not in shared},
            "domain": concept.domain
        }
        return projection
    
    def _create_emergent_structure(self, proj1: Dict, proj2: Dict) -> Dict:
        """Create emergent structure from projections."""
        emergent = {
            "combined_name": f"{proj1['name']}-{proj2['name']}",
            "properties": {**proj1.get('unique_properties', {}), 
                          **proj2.get('unique_properties', {})},
            "domains": [proj1.get('domain'), proj2.get('domain')],
            "new_relations": self._generate_new_relations(proj1, proj2)
        }
        return emergent
    
    def _generate_new_relations(self, proj1: Dict, proj2: Dict) -> List[str]:
        """Generate new relations that emerge from the blend."""
        return [
            f"{proj1['name']} enables {proj2['name']}",
            f"{proj2['name']} transforms {proj1['name']}",
            f"fusion of {proj1['name']} and {proj2['name']} creates synergy"
        ]
    
    def _find_novel_properties(self, c1: Concept, c2: Concept, 
                               emergent: Dict) -> List[str]:
        """Find properties that are novel to the blend."""
        original_props = set(c1.properties.keys()) | set(c2.properties.keys())
        emergent_props = set(emergent.get('properties', {}).keys())
        
        # Also add conceptual novelties
        novel = [
            f"Neither {c1.name} nor {c2.name} alone can achieve this",
            f"A new category combining {c1.domain} and {c2.domain}"
        ]
        
        return novel
    
    def _generate_blend_name(self, c1: Concept, c2: Concept) -> str:
        """Generate creative name for blend."""
        # Try portmanteau
        n1, n2 = c1.name, c2.name
        if len(n1) >= 3 and len(n2) >= 3:
            return n1[:len(n1)//2] + n2[len(n2)//2:]
        return f"{n1}-{n2}"
    
    def _generate_blend_description(self, c1: Concept, c2: Concept, 
                                    emergent: Dict) -> str:
        """Generate description of the blend."""
        return f"A novel concept combining the {c1.domain} nature of {c1.name} " \
               f"with the {c2.domain} qualities of {c2.name}, creating something " \
               f"that transcends both original concepts."


# =============================================================================
# IMAGINATION ENGINE
# =============================================================================

class ImaginationEngine:
    """
    Generate mental imagery and explore hypothetical scenarios.
    """
    
    def __init__(self):
        self.mental_images = []
        self.scenarios = []
    
    def imagine(self, prompt: str, 
                modality: str = "visual",
                tone: EmotionalTone = EmotionalTone.CURIOUS) -> MentalImage:
        """Generate a mental image."""
        
        # Generate vivid description
        description = self._generate_vivid_description(prompt, modality, tone)
        
        # Extract associated concepts
        concepts = self._extract_concepts(prompt)
        
        image = MentalImage(
            description=description,
            sensory_modality=modality,
            vividness=0.7 + random.random() * 0.3,
            emotional_tone=tone,
            associated_concepts=concepts
        )
        
        self.mental_images.append(image)
        return image
    
    def _generate_vivid_description(self, prompt: str, modality: str,
                                     tone: EmotionalTone) -> str:
        """Generate vivid sensory description."""
        tone_adjectives = {
            EmotionalTone.CURIOUS: ["intriguing", "fascinating", "mysterious"],
            EmotionalTone.PLAYFUL: ["whimsical", "dancing", "bubbling"],
            EmotionalTone.SERIOUS: ["profound", "weighty", "solemn"],
            EmotionalTone.WHIMSICAL: ["fantastical", "dreamy", "ethereal"],
            EmotionalTone.MELANCHOLIC: ["fading", "echoing", "wistful"],
            EmotionalTone.HOPEFUL: ["glowing", "rising", "blooming"],
            EmotionalTone.MYSTERIOUS: ["shrouded", "glimmering", "ancient"],
            EmotionalTone.PASSIONATE: ["burning", "intense", "vivid"]
        }
        
        modality_templates = {
            "visual": "Imagine seeing {adj} {prompt}, with {details}",
            "auditory": "Hear the {adj} sounds of {prompt}, {details}",
            "tactile": "Feel the {adj} texture of {prompt}, {details}",
            "olfactory": "Sense the {adj} aroma of {prompt}, {details}",
            "kinesthetic": "Experience the {adj} movement of {prompt}, {details}"
        }
        
        adj = random.choice(tone_adjectives.get(tone, ["vivid"]))
        details = self._generate_sensory_details(prompt, modality)
        
        template = modality_templates.get(modality, modality_templates["visual"])
        return template.format(adj=adj, prompt=prompt, details=details)
    
    def _generate_sensory_details(self, prompt: str, modality: str) -> str:
        """Generate specific sensory details."""
        details = {
            "visual": ["colors shifting like aurora", "light dancing on surfaces",
                      "shadows forming intricate patterns", "depth stretching to infinity"],
            "auditory": ["resonating with harmonic overtones", "echoing in vast spaces",
                        "rhythms pulsing like a heartbeat", "silence pregnant with meaning"],
            "tactile": ["smooth as polished stone", "electric with possibility",
                       "warm like sunlight", "textured like ancient bark"],
            "olfactory": ["hints of rain on warm earth", "traces of distant flowers",
                         "crisp like morning air", "rich with history"],
            "kinesthetic": ["flowing like water", "spiraling upward", 
                           "expanding outward", "settling into stillness"]
        }
        return random.choice(details.get(modality, details["visual"]))
    
    def _extract_concepts(self, prompt: str) -> List[str]:
        """Extract key concepts from prompt."""
        # Simplified - would use NLP in real system
        words = prompt.lower().split()
        return [w for w in words if len(w) > 3][:5]
    
    def explore_scenario(self, what_if: str) -> Dict:
        """Explore a hypothetical scenario."""
        
        # Generate consequences
        immediate = self._generate_immediate_consequences(what_if)
        downstream = self._generate_downstream_effects(what_if)
        
        # Generate alternative paths
        alternatives = self._generate_alternative_paths(what_if)
        
        scenario = {
            "premise": what_if,
            "immediate_consequences": immediate,
            "downstream_effects": downstream,
            "alternative_paths": alternatives,
            "key_insight": self._extract_key_insight(what_if, immediate, downstream)
        }
        
        self.scenarios.append(scenario)
        return scenario
    
    def _generate_immediate_consequences(self, what_if: str) -> List[str]:
        """Generate immediate consequences of scenario."""
        return [
            f"First, {what_if.lower()} would immediately change how we perceive...",
            f"Initial reactions would include surprise, adaptation, and exploration",
            f"Systems and structures would begin to reorganize"
        ]
    
    def _generate_downstream_effects(self, what_if: str) -> List[str]:
        """Generate long-term downstream effects."""
        return [
            "New patterns would emerge from the chaos",
            "Unexpected benefits and challenges would arise",
            "Society/systems would find a new equilibrium"
        ]
    
    def _generate_alternative_paths(self, what_if: str) -> List[str]:
        """Generate alternative paths the scenario could take."""
        return [
            "Path 1: Gradual adoption leading to peaceful transformation",
            "Path 2: Resistance and conflict before eventual acceptance",
            "Path 3: Partial implementation creating hybrid systems"
        ]
    
    def _extract_key_insight(self, what_if: str, immediate: List, 
                            downstream: List) -> str:
        """Extract the key insight from scenario exploration."""
        return f"Exploring '{what_if}' reveals that change creates both " \
               f"disruption and opportunity, and adaptation is key to thriving."


# =============================================================================
# INSIGHT GENERATOR
# =============================================================================

class InsightGenerator:
    """
    Generate "aha" moments and insights.
    Models incubation and sudden illumination.
    """
    
    def __init__(self):
        self.incubating = []
        self.insights = []
    
    def incubate(self, problem: str, background_concepts: List[str]):
        """Put a problem into incubation."""
        self.incubating.append({
            "problem": problem,
            "concepts": background_concepts,
            "start_time": 0,
            "connections_found": []
        })
    
    def check_for_insight(self) -> Optional[Dict]:
        """Check if any incubating problem has reached insight."""
        for item in self.incubating:
            # Simulate incubation (random chance of insight)
            if random.random() < 0.3:  # 30% chance
                insight = self._generate_insight(item)
                self.insights.append(insight)
                self.incubating.remove(item)
                return insight
        return None
    
    def _generate_insight(self, incubating: Dict) -> Dict:
        """Generate an insight."""
        problem = incubating["problem"]
        concepts = incubating["concepts"]
        
        # Find unexpected connection
        if len(concepts) >= 2:
            c1, c2 = random.sample(concepts, 2)
            connection = f"The key is seeing how {c1} relates to {c2}"
        else:
            connection = "A new perspective reveals the solution"
        
        return {
            "problem": problem,
            "insight": connection,
            "aha_moment": f"💡 Aha! {connection}!",
            "explanation": f"By stepping back from {problem}, we can see that {connection}. "
                          f"This reframe opens new possibilities.",
            "next_steps": [
                f"Explore the {c1}-{c2} connection further" if len(concepts) >= 2 else "Explore further",
                "Test this new perspective",
                "Look for similar patterns elsewhere"
            ]
        }
    
    def force_insight(self, problem: str, concepts: List[str]) -> Dict:
        """Force an insight through deliberate techniques."""
        techniques = [
            self._reframe_problem,
            self._reverse_assumptions,
            self._find_hidden_constraint,
            self._change_granularity
        ]
        
        technique = random.choice(techniques)
        return technique(problem, concepts)
    
    def _reframe_problem(self, problem: str, concepts: List[str]) -> Dict:
        """Insight through reframing."""
        return {
            "problem": problem,
            "technique": "reframing",
            "insight": f"Instead of solving '{problem}', what if we asked a different question?",
            "reframed_problem": f"How might we transform {problem} into an opportunity?"
        }
    
    def _reverse_assumptions(self, problem: str, concepts: List[str]) -> Dict:
        """Insight through assumption reversal."""
        return {
            "problem": problem,
            "technique": "assumption_reversal",
            "insight": f"What if the opposite of our assumption about {problem} were true?",
            "reversed_view": "The constraint might actually be the key to the solution."
        }
    
    def _find_hidden_constraint(self, problem: str, concepts: List[str]) -> Dict:
        """Insight through finding hidden constraints."""
        return {
            "problem": problem,
            "technique": "hidden_constraint",
            "insight": f"We've been assuming something about {problem} that isn't necessarily true.",
            "hidden_assumption": "The real constraint is not what we thought."
        }
    
    def _change_granularity(self, problem: str, concepts: List[str]) -> Dict:
        """Insight through changing scale/granularity."""
        return {
            "problem": problem,
            "technique": "granularity_shift",
            "insight": f"Looking at {problem} from a different scale reveals new patterns.",
            "zoomed_out": "At a higher level, the pattern becomes clear.",
            "zoomed_in": "At a detailed level, the mechanism becomes clear."
        }


# =============================================================================
# INTUITION MODELER
# =============================================================================

class IntuitionModeler:
    """
    Model intuitive, fast thinking (System 1).
    Pattern recognition without explicit reasoning.
    """
    
    def __init__(self):
        self.pattern_memory = []
        self.gut_feelings = []
    
    def get_intuition(self, situation: str) -> Dict:
        """Get intuitive response to a situation."""
        
        # Quick pattern match
        pattern = self._pattern_match(situation)
        
        # Generate gut feeling
        feeling = self._generate_gut_feeling(situation, pattern)
        
        # Confidence in intuition
        confidence = self._assess_intuition_confidence(pattern)
        
        intuition = {
            "situation": situation,
            "gut_feeling": feeling,
            "pattern_matched": pattern,
            "confidence": confidence,
            "should_override": confidence < 0.5,  # When intuition might be wrong
            "warning": "Intuition may be biased" if confidence < 0.3 else None
        }
        
        self.gut_feelings.append(intuition)
        return intuition
    
    def _pattern_match(self, situation: str) -> Optional[str]:
        """Quick pattern matching."""
        # Check against known patterns
        patterns = {
            "danger": ["risk", "threat", "warning", "problem", "issue"],
            "opportunity": ["chance", "possibility", "potential", "opening"],
            "familiarity": ["similar", "like", "reminds", "same"],
            "novelty": ["new", "different", "unusual", "strange"],
            "complexity": ["complex", "complicated", "intricate", "many"]
        }
        
        situation_lower = situation.lower()
        for pattern_type, keywords in patterns.items():
            if any(kw in situation_lower for kw in keywords):
                return pattern_type
        
        return "neutral"
    
    def _generate_gut_feeling(self, situation: str, pattern: str) -> str:
        """Generate intuitive gut feeling."""
        feelings = {
            "danger": "Something feels off here. Proceed with caution.",
            "opportunity": "This feels promising. Worth exploring further.",
            "familiarity": "I've seen something like this before.",
            "novelty": "This is new territory. Stay curious but alert.",
            "complexity": "There's more here than meets the eye.",
            "neutral": "Uncertain. Need more information."
        }
        return feelings.get(pattern, feelings["neutral"])
    
    def _assess_intuition_confidence(self, pattern: str) -> float:
        """Assess how much to trust intuition."""
        # Strong patterns get higher confidence
        if pattern in ["danger", "opportunity"]:
            return 0.7 + random.random() * 0.2
        elif pattern == "novelty":
            return 0.3 + random.random() * 0.2  # Less reliable for novel situations
        else:
            return 0.4 + random.random() * 0.2


# =============================================================================
# CREATIVE THINKING ENGINE
# =============================================================================

class CreativeThinkingEngine:
    """
    Main engine for human-like creative thinking.
    Integrates all creative cognition components.
    """
    
    def __init__(self):
        self.divergent = DivergentThinker()
        self.analogical = AnalogicalReasoner()
        self.blender = ConceptualBlender()
        self.imagination = ImaginationEngine()
        self.insights = InsightGenerator()
        self.intuition = IntuitionModeler()
        
        self.creative_history = []
    
    def think_creatively(self, prompt: str, 
                         mode: CreativityMode = CreativityMode.DIVERGENT) -> Dict:
        """Main entry point for creative thinking."""
        
        result = {
            "prompt": prompt,
            "mode": mode.value,
            "ideas": [],
            "analogies": [],
            "blends": [],
            "images": [],
            "insights": [],
            "intuition": None
        }
        
        # Get intuition first
        result["intuition"] = self.intuition.get_intuition(prompt)
        
        # Generate ideas
        result["ideas"] = self.divergent.brainstorm(prompt, n=5, mode=mode)
        
        # Find analogies
        domains = ["nature", "music", "technology"]
        for domain in domains:
            analogy = self.analogical.find_analogy(prompt, "abstract", domain)
            result["analogies"].append(analogy)
        
        # Create conceptual blend
        concept1 = Concept(name=prompt, domain="input", properties={"original": True})
        concept2 = Concept(name="creativity", domain="meta", properties={"generative": True})
        result["blends"].append(self.blender.blend(concept1, concept2))
        
        # Generate mental image
        result["images"].append(
            self.imagination.imagine(prompt, tone=EmotionalTone.CURIOUS)
        )
        
        # Check for insights
        self.insights.incubate(prompt, [prompt, "solution", "pattern"])
        insight = self.insights.check_for_insight()
        if insight:
            result["insights"].append(insight)
        
        # Store in history
        self.creative_history.append(result)
        
        return result
    
    def generate_creative_solution(self, problem: str) -> Dict:
        """Generate creative solution to a problem."""
        
        # Phase 1: Divergent (open up)
        ideas = self.divergent.brainstorm(problem, n=10, mode=CreativityMode.DIVERGENT)
        lateral = self.divergent.brainstorm(problem, n=5, mode=CreativityMode.LATERAL)
        all_ideas = ideas + lateral
        
        # Phase 2: Explore through analogy
        analogies = []
        for domain in ["nature", "technology", "art"]:
            analogies.append(self.analogical.find_analogy(problem, "problem", domain))
        
        # Phase 3: Force insight
        insight = self.insights.force_insight(problem, [problem, "solution", "user"])
        
        # Phase 4: Convergent (narrow down)
        best_ideas = sorted(all_ideas, key=lambda x: x.creativity_score, reverse=True)[:3]
        
        # Phase 5: Synthesize
        solution = self._synthesize_solution(problem, best_ideas, analogies, insight)
        
        return {
            "problem": problem,
            "exploration": {
                "total_ideas_generated": len(all_ideas),
                "analogies_explored": len(analogies),
                "insight_found": insight
            },
            "best_ideas": [{"content": i.content, "score": i.creativity_score} for i in best_ideas],
            "synthesized_solution": solution
        }
    
    def _synthesize_solution(self, problem: str, ideas: List[Idea],
                            analogies: List[Analogy], insight: Dict) -> str:
        """Synthesize final creative solution."""
        solution_parts = []
        
        # Extract key elements
        idea_essence = ideas[0].content if ideas else "novel approach"
        analogy_essence = analogies[0].insight if analogies else "new perspective"
        insight_essence = insight.get("insight", "unexpected connection")
        
        solution = f"""
Creative Solution for: {problem}

Core Insight: {insight_essence}

Approach: {idea_essence}

Analogy: {analogy_essence}

This solution combines divergent exploration, cross-domain 
transfer, and insight to address {problem} in a novel way.
"""
        return solution
    
    def dream_synthesis(self, themes: List[str]) -> Dict:
        """
        Dream-like synthesis of themes.
        Mimics unconscious creative processing.
        """
        
        # Random associations
        associations = []
        for theme in themes:
            assoc = self.divergent._free_associate(theme)
            associations.append((theme, assoc))
        
        # Create surreal blends
        blends = []
        if len(themes) >= 2:
            for i in range(0, len(themes) - 1, 2):
                c1 = Concept(name=themes[i], domain="dream")
                c2 = Concept(name=themes[i+1], domain="dream")
                blends.append(self.blender.blend(c1, c2))
        
        # Generate dreamlike imagery
        images = []
        for theme in themes[:3]:
            images.append(self.imagination.imagine(
                theme, 
                tone=random.choice(list(EmotionalTone))
            ))
        
        # Find hidden meanings
        hidden_meaning = f"In the dream-logic of {' and '.join(themes)}, " \
                        f"we find {random.choice(['transformation', 'unity', 'revelation', 'mystery'])}"
        
        return {
            "themes": themes,
            "associations": associations,
            "blends": [b.description for b in blends],
            "dream_images": [i.description for i in images],
            "hidden_meaning": hidden_meaning,
            "upon_waking": "New connections emerge from the dream state..."
        }
    
    def stats(self) -> Dict:
        """Get creative thinking statistics."""
        return {
            "sessions": len(self.creative_history),
            "total_ideas": sum(len(s.get("ideas", [])) for s in self.creative_history),
            "analogies_made": len(self.analogical.analogy_cache),
            "insights_gained": len(self.insights.insights),
            "intuitions_recorded": len(self.intuition.gut_feelings)
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Creative Thinking Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          🎨 AION CREATIVE THINKING ENGINE 🎨                              ║
║                                                                           ║
║     Human-Like Creativity: Imagination, Insight, Intuition               ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = CreativeThinkingEngine()
    
    # Brainstorming
    print("💡 Brainstorming: 'sustainable cities'")
    print("-" * 50)
    ideas = engine.divergent.brainstorm("sustainable cities", n=5)
    for idea in ideas[:3]:
        print(f"   • {idea.content}")
    
    # Analogical reasoning
    print("\n🔗 Analogy: Cities as Ecosystems")
    print("-" * 50)
    analogy = engine.analogical.find_analogy("city", "urban", "biology")
    print(f"   {analogy.insight}")
    
    # Conceptual blending
    print("\n🌀 Conceptual Blend: Nature + Technology")
    print("-" * 50)
    c1 = Concept(name="forest", domain="nature", properties={"living": True})
    c2 = Concept(name="computer", domain="technology", properties={"processing": True})
    blend = engine.blender.blend(c1, c2)
    print(f"   Name: {blend.name}")
    print(f"   {blend.description}")
    
    # Imagination
    print("\n🌈 Mental Imagery:")
    print("-" * 50)
    image = engine.imagination.imagine("a city that grows like a tree", 
                                        tone=EmotionalTone.HOPEFUL)
    print(f"   {image.description}")
    
    # Insight generation
    print("\n⚡ Forcing Insight:")
    print("-" * 50)
    insight = engine.insights.force_insight(
        "How to reduce urban pollution",
        ["nature", "technology", "community"]
    )
    print(f"   Technique: {insight.get('technique')}")
    print(f"   Insight: {insight.get('insight')}")
    
    # Intuition
    print("\n🎯 Intuitive Response:")
    print("-" * 50)
    intuition = engine.intuition.get_intuition(
        "A new opportunity for green infrastructure"
    )
    print(f"   Gut feeling: {intuition['gut_feeling']}")
    print(f"   Confidence: {intuition['confidence']:.0%}")
    
    print("\n" + "=" * 60)
    print("AION can now think creatively like a human! 🧠✨")


if __name__ == "__main__":
    demo()
