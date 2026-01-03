"""Tests for AION Parser"""

import pytest
import sys
sys.path.insert(0, 'd:/Time/aion')

from src.parser import (
    parse, Parser, ParserError,
    Program, AgentDecl, GoalStmt, MemoryDecl, ModelDecl, ModelRef,
    ToolDecl, ToolRef, PolicyDecl, EventHandler,
    ThinkStmt, AnalyzeStmt, ReflectStmt, DecideStmt,
    IfStmt, WhenStmt, RepeatStmt,
    UseStmt, RespondStmt, StoreStmt, RecallStmt, AssignStmt,
    BinaryExpr, UnaryExpr, Literal, Identifier, MemberAccess, ListLiteral
)
from src.lexer import tokenize


class TestParser:
    """Test cases for the AION parser."""
    
    def test_empty_program(self):
        """Test parsing empty program."""
        program = parse("")
        assert isinstance(program, Program)
        assert len(program.declarations) == 0
    
    def test_agent_declaration(self):
        """Test parsing agent declaration."""
        source = '''agent Assistant {
  goal "Help users"
}'''
        program = parse(source)
        
        assert len(program.declarations) == 1
        agent = program.declarations[0]
        assert isinstance(agent, AgentDecl)
        assert agent.name == "Assistant"
    
    def test_agent_with_goal(self):
        """Test parsing agent with goal statement."""
        source = '''agent Test {
  goal "Test goal"
}'''
        program = parse(source)
        agent = program.declarations[0]
        
        goal = agent.body[0]
        assert isinstance(goal, GoalStmt)
        assert goal.goal == "Test goal"
    
    def test_agent_with_memory(self):
        """Test parsing agent with memory declarations."""
        source = '''agent Test {
  memory working
  memory long_term
}'''
        program = parse(source)
        agent = program.declarations[0]
        
        assert len(agent.body) == 2
        assert isinstance(agent.body[0], MemoryDecl)
        assert agent.body[0].memory_type == "working"
        assert agent.body[1].memory_type == "long_term"
    
    def test_model_declaration(self):
        """Test parsing model declaration."""
        source = '''model LLM {
  provider = "openai"
  name = "gpt-4"
}'''
        program = parse(source)
        
        model = program.declarations[0]
        assert isinstance(model, ModelDecl)
        assert model.name == "LLM"
        assert model.config['provider'] == "openai"
        assert model.config['name'] == "gpt-4"
    
    def test_tool_declaration(self):
        """Test parsing tool declaration."""
        source = '''tool calculator {
  trust = "high"
  cost = "low"
}'''
        program = parse(source)
        
        tool = program.declarations[0]
        assert isinstance(tool, ToolDecl)
        assert tool.name == "calculator"
        assert tool.config['trust'] == "high"
    
    def test_policy_declaration(self):
        """Test parsing policy declaration."""
        source = '''policy {
  max_tokens = 2048
  allow_web = true
}'''
        program = parse(source)
        
        policy = program.declarations[0]
        assert isinstance(policy, PolicyDecl)
        assert policy.config['max_tokens'] == 2048
        assert policy.config['allow_web'] == True
    
    def test_event_handler(self):
        """Test parsing event handler."""
        source = '''agent Test {
  on input(msg):
    respond
}'''
        program = parse(source)
        agent = program.declarations[0]
        
        handler = agent.body[0]
        assert isinstance(handler, EventHandler)
        assert handler.event_type == "input"
        assert handler.params == ["msg"]
    
    def test_reasoning_statements(self):
        """Test parsing reasoning statements."""
        source = '''agent Test {
  on input(x):
    think
    analyze x
    reflect
    decide answer
}'''
        program = parse(source)
        agent = program.declarations[0]
        handler = agent.body[0]
        
        assert isinstance(handler.body[0], ThinkStmt)
        assert isinstance(handler.body[1], AnalyzeStmt)
        assert isinstance(handler.body[2], ReflectStmt)
        assert isinstance(handler.body[3], DecideStmt)
    
    def test_if_statement(self):
        """Test parsing if statement."""
        source = '''agent Test {
  on input(x):
    if x > 0:
      respond
}'''
        program = parse(source)
        agent = program.declarations[0]
        handler = agent.body[0]
        
        if_stmt = handler.body[0]
        assert isinstance(if_stmt, IfStmt)
        assert isinstance(if_stmt.condition, BinaryExpr)
    
    def test_when_statement(self):
        """Test parsing when statement."""
        source = '''agent Test {
  on input(x):
    when ready:
      respond
}'''
        program = parse(source)
        agent = program.declarations[0]
        handler = agent.body[0]
        
        when_stmt = handler.body[0]
        assert isinstance(when_stmt, WhenStmt)
    
    def test_repeat_statement(self):
        """Test parsing repeat statement."""
        source = '''agent Test {
  on input(x):
    repeat 3 times:
      think
}'''
        program = parse(source)
        agent = program.declarations[0]
        handler = agent.body[0]
        
        repeat_stmt = handler.body[0]
        assert isinstance(repeat_stmt, RepeatStmt)
        assert isinstance(repeat_stmt.times, Literal)
        assert repeat_stmt.times.value == 3
    
    def test_use_statement(self):
        """Test parsing use statement."""
        source = '''agent Test {
  on input(x):
    use calculator(x)
}'''
        program = parse(source)
        agent = program.declarations[0]
        handler = agent.body[0]
        
        use_stmt = handler.body[0]
        assert isinstance(use_stmt, UseStmt)
        assert use_stmt.tool_name == "calculator"
    
    def test_store_recall_statements(self):
        """Test parsing store and recall statements."""
        source = '''agent Test {
  on input(x):
    store x in my_memory
    recall from my_memory
}'''
        program = parse(source)
        agent = program.declarations[0]
        handler = agent.body[0]
        
        store_stmt = handler.body[0]
        assert isinstance(store_stmt, StoreStmt)
        
        recall_stmt = handler.body[1]
        assert isinstance(recall_stmt, RecallStmt)
    
    def test_binary_expression(self):
        """Test parsing binary expressions."""
        source = '''agent Test {
  on input(x):
    if x + 1 > 5:
      respond
}'''
        program = parse(source)
        agent = program.declarations[0]
        handler = agent.body[0]
        
        if_stmt = handler.body[0]
        # x + 1 > 5 is (x + 1) > 5
        assert isinstance(if_stmt.condition, BinaryExpr)
        assert if_stmt.condition.operator == ">"
    
    def test_unary_expression(self):
        """Test parsing unary expressions."""
        source = '''agent Test {
  on input(x):
    if not done:
      respond
}'''
        program = parse(source)
        agent = program.declarations[0]
        handler = agent.body[0]
        
        if_stmt = handler.body[0]
        assert isinstance(if_stmt.condition, UnaryExpr)
        assert if_stmt.condition.operator == "not"
    
    def test_member_access(self):
        """Test parsing member access expressions."""
        source = '''agent Test {
  on input(x):
    analyze x.data.value
}'''
        program = parse(source)
        agent = program.declarations[0]
        handler = agent.body[0]
        
        analyze_stmt = handler.body[0]
        target = analyze_stmt.target
        assert isinstance(target, MemberAccess)
    
    def test_list_literal(self):
        """Test parsing list literals."""
        source = '''policy {
  allowed = [1, 2, 3]
}'''
        program = parse(source)
        policy = program.declarations[0]
        
        assert policy.config['allowed'] == [1, 2, 3]
    
    def test_complete_agent(self):
        """Test parsing a complete agent."""
        source = '''agent Assistant {
  goal "Help users"
  memory working
  memory long_term
  model LLM
  tool calculator
  
  on input(question):
    think
    analyze question
    decide answer
    respond answer
}'''
        program = parse(source)
        
        assert len(program.declarations) == 1
        agent = program.declarations[0]
        assert agent.name == "Assistant"
        assert len(agent.body) == 6  # goal, 2 memory, model, tool, event handler


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
