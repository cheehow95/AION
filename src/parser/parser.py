"""
AION Parser
Recursive descent parser that generates AST from tokens.
"""

from typing import Optional, Any
from ..lexer import Token, TokenType, Lexer
from .ast_nodes import (
    Program, AgentDecl, GoalStmt, MemoryDecl, ModelDecl, ModelRef,
    ToolDecl, ToolRef, PolicyDecl, EventHandler,
    ThinkStmt, AnalyzeStmt, ReflectStmt, DecideStmt,
    IfStmt, WhenStmt, RepeatStmt,
    UseStmt, RespondStmt, EmitStmt, StoreStmt, RecallStmt, AssignStmt,
    Expression, BinaryExpr, UnaryExpr, Literal, Identifier, MemberAccess, ListLiteral,
    ASTNode
)


class ParserError(Exception):
    """Raised when the parser encounters a syntax error."""
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"Parser error at line {token.line}, column {token.column}: {message}")


class Parser:
    """
    Recursive descent parser for the AION programming language.
    """
    
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
    
    @property
    def current(self) -> Token:
        """Returns the current token."""
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[self.pos]
    
    def peek(self, offset: int = 1) -> Token:
        """Peek at a token ahead."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[pos]
    
    def advance(self) -> Token:
        """Consume and return the current token."""
        token = self.current
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token
    
    def check(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        return self.current.type in types
    
    def match(self, *types: TokenType) -> Optional[Token]:
        """Match and consume if current token is in types."""
        if self.check(*types):
            return self.advance()
        return None
    
    def expect(self, token_type: TokenType, message: str) -> Token:
        """Expect a specific token type or raise error."""
        if self.check(token_type):
            return self.advance()
        raise ParserError(f"Expected {message}, got {self.current.type.name}", self.current)
    
    def skip_newlines(self) -> None:
        """Skip newline, indent, and dedent tokens (inside brace blocks)."""
        while self.check(TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT):
            self.advance()
    
    # ============ Top Level Parsing ============
    
    def parse(self) -> Program:
        """Parse the entire program."""
        declarations = []
        self.skip_newlines()
        
        while not self.check(TokenType.EOF):
            decl = self.parse_declaration()
            if decl:
                declarations.append(decl)
            self.skip_newlines()
        
        return Program(declarations)
    
    def parse_declaration(self) -> Optional[ASTNode]:
        """Parse a top-level declaration."""
        if self.check(TokenType.AGENT):
            return self.parse_agent_decl()
        elif self.check(TokenType.MODEL):
            return self.parse_model_decl()
        elif self.check(TokenType.TOOL):
            return self.parse_tool_decl()
        elif self.check(TokenType.POLICY):
            return self.parse_policy_decl()
        else:
            raise ParserError(f"Unexpected token: {self.current.type.name}", self.current)
    
    # ============ Agent Parsing ============
    
    def parse_agent_decl(self) -> AgentDecl:
        """Parse an agent declaration."""
        self.expect(TokenType.AGENT, "'agent'")
        name_token = self.expect(TokenType.IDENTIFIER, "agent name")
        self.expect(TokenType.LBRACE, "'{'")
        self.skip_newlines()
        
        body = []
        while not self.check(TokenType.RBRACE, TokenType.EOF):
            member = self.parse_agent_member()
            if member:
                body.append(member)
            self.skip_newlines()
        
        self.expect(TokenType.RBRACE, "'}'")
        return AgentDecl(name_token.value, body)
    
    def parse_agent_member(self) -> Optional[ASTNode]:
        """Parse an agent body member."""
        if self.check(TokenType.GOAL):
            return self.parse_goal_stmt()
        elif self.check(TokenType.MEMORY):
            return self.parse_memory_decl()
        elif self.check(TokenType.MODEL):
            return self.parse_model_ref()
        elif self.check(TokenType.TOOL):
            return self.parse_tool_ref()
        elif self.check(TokenType.POLICY):
            return self.parse_policy_block()
        elif self.check(TokenType.ON):
            return self.parse_event_handler()
        else:
            raise ParserError(f"Unexpected agent member: {self.current.type.name}", self.current)
    
    def parse_goal_stmt(self) -> GoalStmt:
        """Parse a goal statement."""
        self.expect(TokenType.GOAL, "'goal'")
        goal_token = self.expect(TokenType.STRING, "goal string")
        return GoalStmt(goal_token.value)
    
    def parse_memory_decl(self) -> MemoryDecl:
        """Parse a memory declaration."""
        self.expect(TokenType.MEMORY, "'memory'")
        
        memory_types = {
            TokenType.WORKING: 'working',
            TokenType.EPISODIC: 'episodic',
            TokenType.LONG_TERM: 'long_term',
            TokenType.SEMANTIC: 'semantic',
        }
        
        # Check for memory type keyword or identifier
        memory_type = None
        for token_type, type_name in memory_types.items():
            if self.check(token_type):
                self.advance()
                memory_type = type_name
                break
        
        if memory_type is None:
            if self.check(TokenType.IDENTIFIER):
                memory_type = self.advance().value
            else:
                raise ParserError("Expected memory type", self.current)
        
        config = {}
        if self.match(TokenType.LBRACE):
            config = self.parse_config_pairs()
            self.expect(TokenType.RBRACE, "'}'")
        
        return MemoryDecl(memory_type, config)
    
    def parse_model_decl(self) -> ModelDecl:
        """Parse a top-level model declaration."""
        self.expect(TokenType.MODEL, "'model'")
        name_token = self.expect(TokenType.IDENTIFIER, "model name")
        
        config = {}
        if self.match(TokenType.LBRACE):
            self.skip_newlines()
            config = self.parse_config_pairs()
            self.skip_newlines()
            self.expect(TokenType.RBRACE, "'}'")
        
        return ModelDecl(name_token.value, config)
    
    def parse_model_ref(self) -> ModelRef:
        """Parse a model reference within an agent."""
        self.expect(TokenType.MODEL, "'model'")
        name_token = self.expect(TokenType.IDENTIFIER, "model name")
        return ModelRef(name_token.value)
    
    def parse_tool_decl(self) -> ToolDecl:
        """Parse a top-level tool declaration."""
        self.expect(TokenType.TOOL, "'tool'")
        name_token = self.expect(TokenType.IDENTIFIER, "tool name")
        
        config = {}
        if self.match(TokenType.LBRACE):
            self.skip_newlines()
            config = self.parse_config_pairs()
            self.skip_newlines()
            self.expect(TokenType.RBRACE, "'}'")
        
        return ToolDecl(name_token.value, config)
    
    def parse_tool_ref(self) -> ToolRef:
        """Parse a tool reference within an agent."""
        self.expect(TokenType.TOOL, "'tool'")
        name_token = self.expect(TokenType.IDENTIFIER, "tool name")
        return ToolRef(name_token.value)
    
    def parse_policy_decl(self) -> PolicyDecl:
        """Parse a top-level policy declaration."""
        self.expect(TokenType.POLICY, "'policy'")
        
        name = None
        if self.check(TokenType.IDENTIFIER):
            name = self.advance().value
        
        self.expect(TokenType.LBRACE, "'{'")
        self.skip_newlines()
        config = self.parse_config_pairs()
        self.skip_newlines()
        self.expect(TokenType.RBRACE, "'}'")
        
        return PolicyDecl(name, config)
    
    def parse_policy_block(self) -> PolicyDecl:
        """Parse a policy block within an agent."""
        return self.parse_policy_decl()
    
    def parse_config_pairs(self) -> dict[str, Any]:
        """Parse configuration key-value pairs."""
        config = {}
        
        while self.check(TokenType.IDENTIFIER):
            key = self.advance().value
            self.expect(TokenType.EQ, "'='")
            value = self.parse_value()
            config[key] = value
            self.skip_newlines()
        
        return config
    
    def parse_value(self) -> Any:
        """Parse a configuration value."""
        if self.check(TokenType.STRING):
            return self.advance().value
        elif self.check(TokenType.NUMBER):
            return self.advance().value
        elif self.check(TokenType.TRUE, TokenType.FALSE):
            return self.advance().value
        elif self.check(TokenType.NULL):
            self.advance()
            return None
        elif self.check(TokenType.LBRACKET):
            return self.parse_list_value()
        else:
            raise ParserError(f"Expected value, got {self.current.type.name}", self.current)
    
    def parse_list_value(self) -> list:
        """Parse a list literal value."""
        self.expect(TokenType.LBRACKET, "'['")
        values = []
        
        if not self.check(TokenType.RBRACKET):
            values.append(self.parse_value())
            while self.match(TokenType.COMMA):
                values.append(self.parse_value())
        
        self.expect(TokenType.RBRACKET, "']'")
        return values
    
    # ============ Event Handler Parsing ============
    
    def parse_event_handler(self) -> EventHandler:
        """Parse an event handler."""
        self.expect(TokenType.ON, "'on'")
        
        event_types = {
            TokenType.INPUT: 'input',
            TokenType.ERROR: 'error',
            TokenType.TIMEOUT: 'timeout',
            TokenType.COMPLETE: 'complete',
        }
        
        event_type = None
        for token_type, type_name in event_types.items():
            if self.check(token_type):
                self.advance()
                event_type = type_name
                break
        
        if event_type is None:
            raise ParserError("Expected event type", self.current)
        
        # Parse parameters
        params = []
        self.expect(TokenType.LPAREN, "'('")
        if self.check(TokenType.IDENTIFIER):
            params.append(self.advance().value)
            while self.match(TokenType.COMMA):
                params.append(self.expect(TokenType.IDENTIFIER, "parameter name").value)
        self.expect(TokenType.RPAREN, "')'")
        
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        # Parse body (expect indent)
        body = self.parse_statement_block()
        
        return EventHandler(event_type, params, body)
    
    # ============ Statement Parsing ============
    
    def parse_statement_block(self) -> list[ASTNode]:
        """Parse a block of statements (indented)."""
        statements = []
        
        if self.match(TokenType.INDENT):
            while not self.check(TokenType.DEDENT, TokenType.EOF):
                stmt = self.parse_statement()
                if stmt:
                    statements.append(stmt)
                self.skip_newlines()
            self.match(TokenType.DEDENT)
        else:
            # Single statement on same line
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        
        return statements
    
    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement."""
        # Reasoning statements
        if self.check(TokenType.THINK):
            return self.parse_think_stmt()
        elif self.check(TokenType.ANALYZE):
            return self.parse_analyze_stmt()
        elif self.check(TokenType.REFLECT):
            return self.parse_reflect_stmt()
        elif self.check(TokenType.DECIDE):
            return self.parse_decide_stmt()
        
        # Control flow
        elif self.check(TokenType.IF):
            return self.parse_if_stmt()
        elif self.check(TokenType.WHEN):
            return self.parse_when_stmt()
        elif self.check(TokenType.REPEAT):
            return self.parse_repeat_stmt()
        
        # Actions
        elif self.check(TokenType.USE):
            return self.parse_use_stmt()
        elif self.check(TokenType.RESPOND):
            return self.parse_respond_stmt()
        elif self.check(TokenType.EMIT):
            return self.parse_emit_stmt()
        elif self.check(TokenType.STORE):
            return self.parse_store_stmt()
        elif self.check(TokenType.RECALL):
            return self.parse_recall_stmt()
        
        # Assignment or expression
        elif self.check(TokenType.IDENTIFIER):
            # Look ahead to check for assignment
            if self.peek().type == TokenType.EQ:
                return self.parse_assign_stmt()
            # Otherwise it's an expression statement (which we'll skip for now)
        
        return None
    
    # ============ Reasoning Statements ============
    
    def parse_think_stmt(self) -> ThinkStmt:
        """Parse a think statement."""
        self.expect(TokenType.THINK, "'think'")
        prompt = None
        if self.check(TokenType.STRING):
            prompt = self.advance().value
        return ThinkStmt(prompt)
    
    def parse_analyze_stmt(self) -> AnalyzeStmt:
        """Parse an analyze statement."""
        self.expect(TokenType.ANALYZE, "'analyze'")
        target = self.parse_expression()
        return AnalyzeStmt(target)
    
    def parse_reflect_stmt(self) -> ReflectStmt:
        """Parse a reflect statement."""
        self.expect(TokenType.REFLECT, "'reflect'")
        target = None
        if self.match(TokenType.ON):
            target = self.parse_expression()
        return ReflectStmt(target)
    
    def parse_decide_stmt(self) -> DecideStmt:
        """Parse a decide statement."""
        self.expect(TokenType.DECIDE, "'decide'")
        target = self.parse_expression()
        return DecideStmt(target)
    
    # ============ Control Flow ============
    
    def parse_if_stmt(self) -> IfStmt:
        """Parse an if/else statement."""
        self.expect(TokenType.IF, "'if'")
        condition = self.parse_expression()
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        then_body = self.parse_statement_block()
        
        else_body = []
        self.skip_newlines()
        if self.match(TokenType.ELSE):
            self.expect(TokenType.COLON, "':'")
            self.skip_newlines()
            else_body = self.parse_statement_block()
        
        return IfStmt(condition, then_body, else_body)
    
    def parse_when_stmt(self) -> WhenStmt:
        """Parse a when statement."""
        self.expect(TokenType.WHEN, "'when'")
        condition = self.parse_expression()
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        body = self.parse_statement_block()
        return WhenStmt(condition, body)
    
    def parse_repeat_stmt(self) -> RepeatStmt:
        """Parse a repeat statement."""
        self.expect(TokenType.REPEAT, "'repeat'")
        
        times = None
        if not self.check(TokenType.COLON):
            times = self.parse_expression()
            self.match(TokenType.TIMES)
        
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        body = self.parse_statement_block()
        return RepeatStmt(times, body)
    
    # ============ Action Statements ============
    
    def parse_use_stmt(self) -> UseStmt:
        """Parse a use statement."""
        self.expect(TokenType.USE, "'use'")
        name_token = self.expect(TokenType.IDENTIFIER, "tool name")
        
        args = []
        if self.match(TokenType.LPAREN):
            if not self.check(TokenType.RPAREN):
                args.append(self.parse_expression())
                while self.match(TokenType.COMMA):
                    args.append(self.parse_expression())
            self.expect(TokenType.RPAREN, "')'")
        
        return UseStmt(name_token.value, args)
    
    def parse_respond_stmt(self) -> RespondStmt:
        """Parse a respond statement."""
        self.expect(TokenType.RESPOND, "'respond'")
        value = None
        if not self.check(TokenType.NEWLINE, TokenType.EOF, TokenType.DEDENT):
            value = self.parse_expression()
        return RespondStmt(value)
    
    def parse_emit_stmt(self) -> EmitStmt:
        """Parse an emit statement."""
        self.expect(TokenType.EMIT, "'emit'")
        name_token = self.expect(TokenType.IDENTIFIER, "event name")
        
        args = []
        if self.match(TokenType.LPAREN):
            if not self.check(TokenType.RPAREN):
                args.append(self.parse_expression())
                while self.match(TokenType.COMMA):
                    args.append(self.parse_expression())
            self.expect(TokenType.RPAREN, "')'")
        
        return EmitStmt(name_token.value, args)
    
    def parse_store_stmt(self) -> StoreStmt:
        """Parse a store statement."""
        self.expect(TokenType.STORE, "'store'")
        value = self.parse_expression()
        
        memory_name = None
        if self.match(TokenType.IN):
            memory_name = self.expect(TokenType.IDENTIFIER, "memory name").value
        
        return StoreStmt(value, memory_name)
    
    def parse_recall_stmt(self) -> RecallStmt:
        """Parse a recall statement."""
        self.expect(TokenType.RECALL, "'recall'")
        
        memory_name = None
        condition = None
        
        if self.match(TokenType.FROM):
            memory_name = self.expect(TokenType.IDENTIFIER, "memory name").value
        
        if self.match(TokenType.WHERE):
            condition = self.parse_expression()
        
        return RecallStmt(memory_name, condition)
    
    def parse_assign_stmt(self) -> AssignStmt:
        """Parse an assignment statement."""
        name = self.expect(TokenType.IDENTIFIER, "variable name").value
        self.expect(TokenType.EQ, "'='")
        value = self.parse_expression()
        return AssignStmt(name, value)
    
    # ============ Expression Parsing ============
    
    def parse_expression(self) -> Expression:
        """Parse an expression."""
        return self.parse_logic_or()
    
    def parse_logic_or(self) -> Expression:
        """Parse OR expressions."""
        left = self.parse_logic_and()
        
        while self.match(TokenType.OR):
            right = self.parse_logic_and()
            left = BinaryExpr(left, 'or', right)
        
        return left
    
    def parse_logic_and(self) -> Expression:
        """Parse AND expressions."""
        left = self.parse_comparison()
        
        while self.match(TokenType.AND):
            right = self.parse_comparison()
            left = BinaryExpr(left, 'and', right)
        
        return left
    
    def parse_comparison(self) -> Expression:
        """Parse comparison expressions."""
        left = self.parse_term()
        
        comparison_ops = {
            TokenType.EQEQ: '==',
            TokenType.NEQ: '!=',
            TokenType.LT: '<',
            TokenType.GT: '>',
            TokenType.LTE: '<=',
            TokenType.GTE: '>=',
        }
        
        while self.current.type in comparison_ops:
            op_token = self.advance()
            right = self.parse_term()
            left = BinaryExpr(left, comparison_ops[op_token.type], right)
        
        return left
    
    def parse_term(self) -> Expression:
        """Parse addition/subtraction expressions."""
        left = self.parse_factor()
        
        while self.check(TokenType.PLUS, TokenType.MINUS):
            op = '+' if self.current.type == TokenType.PLUS else '-'
            self.advance()
            right = self.parse_factor()
            left = BinaryExpr(left, op, right)
        
        return left
    
    def parse_factor(self) -> Expression:
        """Parse multiplication/division expressions."""
        left = self.parse_unary()
        
        while self.check(TokenType.STAR, TokenType.SLASH):
            op = '*' if self.current.type == TokenType.STAR else '/'
            self.advance()
            right = self.parse_unary()
            left = BinaryExpr(left, op, right)
        
        return left
    
    def parse_unary(self) -> Expression:
        """Parse unary expressions."""
        if self.match(TokenType.NOT):
            operand = self.parse_unary()
            return UnaryExpr('not', operand)
        elif self.match(TokenType.MINUS):
            operand = self.parse_unary()
            return UnaryExpr('-', operand)
        
        return self.parse_primary()
    
    def parse_primary(self) -> Expression:
        """Parse primary expressions."""
        # Literals
        if self.check(TokenType.STRING):
            return Literal(self.advance().value)
        elif self.check(TokenType.NUMBER):
            return Literal(self.advance().value)
        elif self.check(TokenType.TRUE, TokenType.FALSE):
            return Literal(self.advance().value)
        elif self.check(TokenType.NULL):
            self.advance()
            return Literal(None)
        
        # List literal
        elif self.check(TokenType.LBRACKET):
            return self.parse_list_literal()
        
        # Parenthesized expression
        elif self.match(TokenType.LPAREN):
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN, "')'")
            return expr
        
        # Identifier with potential member access
        elif self.check(TokenType.IDENTIFIER):
            expr = Identifier(self.advance().value)
            while self.match(TokenType.DOT):
                member = self.expect(TokenType.IDENTIFIER, "member name").value
                expr = MemberAccess(expr, member)
            return expr
        
        raise ParserError(f"Expected expression, got {self.current.type.name}", self.current)
    
    def parse_list_literal(self) -> ListLiteral:
        """Parse a list literal expression."""
        self.expect(TokenType.LBRACKET, "'['")
        elements = []
        
        if not self.check(TokenType.RBRACKET):
            elements.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                elements.append(self.parse_expression())
        
        self.expect(TokenType.RBRACKET, "']'")
        return ListLiteral(elements)


def parse(source: str) -> Program:
    """
    Convenience function to parse source code and return an AST.
    
    Args:
        source: AION source code
        
    Returns:
        Program AST node
    """
    from ..lexer import tokenize
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse()
