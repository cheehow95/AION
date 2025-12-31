"""Tests for AION Lexer"""

import pytest
import sys
sys.path.insert(0, 'd:/Time/aion')

from src.lexer import Lexer, Token, TokenType, tokenize, LexerError


class TestLexer:
    """Test cases for the AION lexer."""
    
    def test_empty_source(self):
        """Test tokenizing empty source."""
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_keywords(self):
        """Test keyword recognition."""
        source = "agent model tool policy memory goal"
        tokens = tokenize(source)
        
        expected = [
            TokenType.AGENT, TokenType.MODEL, TokenType.TOOL,
            TokenType.POLICY, TokenType.MEMORY, TokenType.GOAL,
            TokenType.EOF
        ]
        
        assert [t.type for t in tokens] == expected
    
    def test_reasoning_keywords(self):
        """Test reasoning keyword recognition."""
        source = "think analyze reflect decide"
        tokens = tokenize(source)
        
        expected = [
            TokenType.THINK, TokenType.ANALYZE,
            TokenType.REFLECT, TokenType.DECIDE,
            TokenType.EOF
        ]
        
        assert [t.type for t in tokens] == expected
    
    def test_memory_types(self):
        """Test memory type keywords."""
        source = "working episodic long_term semantic"
        tokens = tokenize(source)
        
        expected = [
            TokenType.WORKING, TokenType.EPISODIC,
            TokenType.LONG_TERM, TokenType.SEMANTIC,
            TokenType.EOF
        ]
        
        assert [t.type for t in tokens] == expected
    
    def test_string_literal(self):
        """Test string literal parsing."""
        source = '"Hello, World!"'
        tokens = tokenize(source)
        
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "Hello, World!"
    
    def test_string_escape(self):
        """Test string escape sequences."""
        source = '"Line1\\nLine2\\tTab"'
        tokens = tokenize(source)
        
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "Line1\nLine2\tTab"
    
    def test_number_integer(self):
        """Test integer literal parsing."""
        source = "42 0 -17"
        tokens = tokenize(source)
        
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 42
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == 0
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == -17
    
    def test_number_float(self):
        """Test float literal parsing."""
        source = "3.14 0.5 -2.718"
        tokens = tokenize(source)
        
        assert tokens[0].value == 3.14
        assert tokens[1].value == 0.5
        assert tokens[2].value == -2.718
    
    def test_boolean_literals(self):
        """Test boolean literals."""
        source = "true false"
        tokens = tokenize(source)
        
        assert tokens[0].type == TokenType.TRUE
        assert tokens[0].value == True
        assert tokens[1].type == TokenType.FALSE
        assert tokens[1].value == False
    
    def test_null_literal(self):
        """Test null literal."""
        source = "null"
        tokens = tokenize(source)
        
        assert tokens[0].type == TokenType.NULL
        assert tokens[0].value == None
    
    def test_operators(self):
        """Test operator tokenization."""
        source = "+ - * / = == != < > <= >="
        tokens = tokenize(source)
        
        expected = [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.EQ, TokenType.EQEQ, TokenType.NEQ,
            TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE,
            TokenType.EOF
        ]
        
        assert [t.type for t in tokens] == expected
    
    def test_delimiters(self):
        """Test delimiter tokenization."""
        source = "( ) { } [ ] , : ."
        tokens = tokenize(source)
        
        expected = [
            TokenType.LPAREN, TokenType.RPAREN,
            TokenType.LBRACE, TokenType.RBRACE,
            TokenType.LBRACKET, TokenType.RBRACKET,
            TokenType.COMMA, TokenType.COLON, TokenType.DOT,
            TokenType.EOF
        ]
        
        assert [t.type for t in tokens] == expected
    
    def test_identifier(self):
        """Test identifier tokenization."""
        source = "myVar _private CamelCase snake_case"
        tokens = tokenize(source)
        
        assert all(t.type == TokenType.IDENTIFIER for t in tokens[:-1])
        assert tokens[0].value == "myVar"
        assert tokens[1].value == "_private"
    
    def test_comments(self):
        """Test comment handling."""
        source = """# This is a comment
agent Test"""
        tokens = tokenize(source)
        
        # Comments should be skipped
        assert tokens[0].type == TokenType.NEWLINE
        assert tokens[1].type == TokenType.AGENT
    
    def test_newlines_and_indentation(self):
        """Test newline and indentation handling."""
        source = """agent Test {
  goal "Test"
}"""
        tokens = tokenize(source)
        
        # Should have INDENT after colon-newline
        types = [t.type for t in tokens]
        assert TokenType.NEWLINE in types
    
    def test_agent_declaration(self):
        """Test tokenizing a complete agent declaration."""
        source = '''agent Assistant {
  goal "Help users"
}'''
        tokens = tokenize(source)
        
        # First few tokens
        assert tokens[0].type == TokenType.AGENT
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "Assistant"
        assert tokens[2].type == TokenType.LBRACE
    
    def test_line_column_tracking(self):
        """Test that line and column are tracked correctly."""
        source = "agent\nmodel"
        tokens = tokenize(source)
        
        assert tokens[0].line == 1
        assert tokens[2].line == 2  # After newline


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
