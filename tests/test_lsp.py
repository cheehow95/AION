"""Tests for AION Language Server Protocol"""

import pytest
import sys
sys.path.insert(0, 'd:/Time/aion')

from src.lsp.server import (
    AIONLanguageServer, Position, Range, Diagnostic,
    CompletionItem, CompletionItemKind, MessageType
)


class TestPosition:
    """Test Position dataclass."""
    
    def test_position_creation(self):
        """Test creating a Position."""
        pos = Position(line=5, character=10)
        
        assert pos.line == 5
        assert pos.character == 10


class TestRange:
    """Test Range dataclass."""
    
    def test_range_creation(self):
        """Test creating a Range."""
        start = Position(1, 0)
        end = Position(1, 10)
        r = Range(start=start, end=end)
        
        assert r.start.line == 1
        assert r.end.character == 10


class TestDiagnostic:
    """Test Diagnostic dataclass."""
    
    def test_diagnostic_creation(self):
        """Test creating a Diagnostic."""
        r = Range(Position(0, 0), Position(0, 5))
        diag = Diagnostic(
            range=r,
            message="Syntax error",
            severity=MessageType.ERROR
        )
        
        assert diag.message == "Syntax error"
        assert diag.severity == MessageType.ERROR
        assert diag.source == "aion"


class TestCompletionItem:
    """Test CompletionItem dataclass."""
    
    def test_completion_item_creation(self):
        """Test creating a CompletionItem."""
        item = CompletionItem(
            label="agent",
            kind=CompletionItemKind.KEYWORD,
            detail="Declare an agent"
        )
        
        assert item.label == "agent"
        assert item.kind == CompletionItemKind.KEYWORD


class TestAIONLanguageServer:
    """Test AIONLanguageServer class."""
    
    def test_init(self):
        """Test server initialization."""
        server = AIONLanguageServer()
        
        assert server.documents == {}
        assert server.diagnostics == {}
        assert len(server.keywords) > 0
    
    def test_did_open(self):
        """Test document open handler."""
        server = AIONLanguageServer()
        
        server.did_open("file:///test.aion", "agent Test {}")
        
        assert "file:///test.aion" in server.documents
        assert server.documents["file:///test.aion"] == "agent Test {}"
    
    def test_did_change(self):
        """Test document change handler."""
        server = AIONLanguageServer()
        server.did_open("file:///test.aion", "agent Test {}")
        
        server.did_change("file:///test.aion", "agent Updated {}")
        
        assert server.documents["file:///test.aion"] == "agent Updated {}"
    
    def test_did_close(self):
        """Test document close handler."""
        server = AIONLanguageServer()
        server.did_open("file:///test.aion", "agent Test {}")
        
        server.did_close("file:///test.aion")
        
        assert "file:///test.aion" not in server.documents
    
    def test_get_completions_keywords(self):
        """Test keyword completions."""
        server = AIONLanguageServer()
        server.did_open("file:///test.aion", "")
        
        completions = server.get_completions(
            "file:///test.aion",
            Position(0, 0)
        )
        
        labels = [c.label for c in completions]
        assert "agent" in labels
        assert "model" in labels
    
    def test_get_completions_with_prefix(self):
        """Test completions with prefix filter."""
        server = AIONLanguageServer()
        server.did_open("file:///test.aion", "ag")
        
        completions = server.get_completions(
            "file:///test.aion",
            Position(0, 2)
        )
        
        labels = [c.label for c in completions]
        assert "agent" in labels
    
    def test_get_completions_in_agent(self):
        """Test completions inside agent body."""
        server = AIONLanguageServer()
        source = "agent Test {\n  \n}"
        server.did_open("file:///test.aion", source)
        
        completions = server.get_completions(
            "file:///test.aion",
            Position(1, 2)
        )
        
        labels = [c.label for c in completions]
        assert "goal" in labels or "memory" in labels
    
    def test_get_hover_keyword(self):
        """Test hover documentation for keywords."""
        server = AIONLanguageServer()
        server.did_open("file:///test.aion", "agent Test {}")
        
        hover = server.get_hover(
            "file:///test.aion",
            Position(0, 0)
        )
        
        assert hover is not None
        assert "agent" in hover.lower()
    
    def test_get_hover_no_match(self):
        """Test hover with no matching keyword."""
        server = AIONLanguageServer()
        server.did_open("file:///test.aion", "xyz123")
        
        hover = server.get_hover(
            "file:///test.aion",
            Position(0, 0)
        )
        
        assert hover is None
    
    def test_get_hover_out_of_range(self):
        """Test hover with out of range position."""
        server = AIONLanguageServer()
        server.did_open("file:///test.aion", "agent")
        
        hover = server.get_hover(
            "file:///test.aion",
            Position(100, 0)
        )
        
        assert hover is None
    
    def test_get_diagnostics_valid_code(self):
        """Test diagnostics for valid code."""
        server = AIONLanguageServer()
        server.did_open("file:///test.aion", "agent Test { goal \"Test\" }")
        
        diagnostics = server.get_diagnostics("file:///test.aion")
        
        # Should have no errors for valid code
        assert len(diagnostics) == 0
    
    def test_get_diagnostics_empty(self):
        """Test diagnostics for non-existent document."""
        server = AIONLanguageServer()
        
        diagnostics = server.get_diagnostics("file:///nonexistent.aion")
        
        assert diagnostics == []
    
    def test_keywords_have_snippets(self):
        """Test that keywords have completion snippets."""
        server = AIONLanguageServer()
        
        for keyword, detail, snippet in server.keywords:
            assert isinstance(keyword, str)
            assert len(keyword) > 0
            assert isinstance(snippet, str)
    
    def test_all_reasoning_keywords_covered(self):
        """Test all reasoning keywords have completions."""
        server = AIONLanguageServer()
        
        keywords = [k[0] for k in server.keywords]
        
        assert "think" in keywords
        assert "analyze" in keywords
        assert "reflect" in keywords
        assert "decide" in keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
