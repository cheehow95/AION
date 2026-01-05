"""
AION Knowledge Module Tests
===========================

Tests for knowledge graph and formal logic engines.
"""

import unittest
import sys
sys.path.insert(0, '.')


class TestKnowledgeGraph(unittest.TestCase):
    """Test knowledge graph engine."""
    
    def setUp(self):
        from src.knowledge.knowledge_graph import KnowledgeGraphEngine
        self.engine = KnowledgeGraphEngine()
    
    def test_add_triple(self):
        """Test adding a triple."""
        result = self.engine.add("Dog", "is_a", "Animal")
        self.assertTrue(result['success'])
        self.assertEqual(result['subject']['name'], 'Dog')
    
    def test_query(self):
        """Test querying triples."""
        self.engine.add("Dog", "is_a", "Mammal")
        self.engine.add("Cat", "is_a", "Mammal")
        
        results = self.engine.query(predicate="is_a", obj="Mammal")
        self.assertEqual(len(results), 2)
    
    def test_transitive_inference(self):
        """Test transitive inference."""
        self.engine.add("Dog", "is_a", "Mammal")
        self.engine.add("Mammal", "is_a", "Animal")
        
        inferred = self.engine.infer("is_a")
        # Should infer: Dog is_a Animal
        subjects = [t['subject'] for t in inferred]
        self.assertIn('Dog', subjects)
    
    def test_find_path(self):
        """Test path finding."""
        self.engine.add("A", "related_to", "B")
        self.engine.add("B", "related_to", "C")
        
        path = self.engine.find_path("A", "C")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "C")
    
    def test_stats(self):
        """Test graph statistics."""
        self.engine.add("X", "has", "Y")
        stats = self.engine.stats()
        self.assertEqual(stats['entities'], 2)
        self.assertEqual(stats['relations'], 1)


class TestFormalLogic(unittest.TestCase):
    """Test formal logic engine."""
    
    def setUp(self):
        from src.knowledge.formal_logic import FormalLogicEngine
        self.engine = FormalLogicEngine()
    
    def test_parse_proposition(self):
        """Test parsing simple proposition."""
        expr = self.engine.parse("P")
        self.assertEqual(str(expr), "P")
    
    def test_parse_negation(self):
        """Test parsing negation."""
        expr = self.engine.parse("¬P")
        self.assertEqual(str(expr), "¬P")
    
    def test_truth_table_tautology(self):
        """Test tautology detection."""
        result = self.engine.check_validity("P ∨ ¬P")
        self.assertTrue(result['tautology'])
    
    def test_truth_table_contradiction(self):
        """Test contradiction detection."""
        result = self.engine.check_validity("P ∧ ¬P")
        self.assertTrue(result['contradiction'])
    
    def test_modus_ponens(self):
        """Test modus ponens inference."""
        result = self.engine.modus_ponens("P", "P→Q")
        self.assertEqual(result, "Q")
    
    def test_theorem_proving(self):
        """Test theorem prover."""
        result = self.engine.prove(["P→Q", "Q→R"], "P→R")
        self.assertTrue(result['proved'])
    
    def test_equivalences(self):
        """Test logical equivalences list."""
        equivs = self.engine.equivalences()
        self.assertIn('De Morgan (AND)', equivs)
        self.assertIn('Double Negation', equivs)


class TestPropositionalLogic(unittest.TestCase):
    """Test propositional logic structures."""
    
    def test_and(self):
        """Test conjunction."""
        from src.knowledge.formal_logic import Proposition, And
        P = Proposition('P')
        Q = Proposition('Q')
        expr = And(P, Q)
        
        self.assertTrue(expr.evaluate({'P': True, 'Q': True}))
        self.assertFalse(expr.evaluate({'P': True, 'Q': False}))
    
    def test_or(self):
        """Test disjunction."""
        from src.knowledge.formal_logic import Proposition, Or
        P = Proposition('P')
        Q = Proposition('Q')
        expr = Or(P, Q)
        
        self.assertTrue(expr.evaluate({'P': True, 'Q': False}))
        self.assertFalse(expr.evaluate({'P': False, 'Q': False}))
    
    def test_implies(self):
        """Test implication."""
        from src.knowledge.formal_logic import Proposition, Implies
        P = Proposition('P')
        Q = Proposition('Q')
        expr = Implies(P, Q)
        
        # P→Q is false only when P is true and Q is false
        self.assertFalse(expr.evaluate({'P': True, 'Q': False}))
        self.assertTrue(expr.evaluate({'P': False, 'Q': False}))
        self.assertTrue(expr.evaluate({'P': True, 'Q': True}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
