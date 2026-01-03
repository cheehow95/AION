"""
AION Parser v2.0
Recursive descent parser that generates AST from tokens.
Supports all v2.0 language features.
"""

from typing import Optional, Any, List
from ..lexer import Token, TokenType, Lexer
from .ast_nodes import (
    # Core
    Program, ASTNode,
    # Declarations
    AgentDecl, GoalStmt, MemoryDecl, ModelDecl, ModelRef,
    ToolDecl, ToolRef, PolicyDecl, EventHandler,
    # Reasoning
    ThinkStmt, AnalyzeStmt, ReflectStmt, DecideStmt,
    # Control Flow v1
    IfStmt, WhenStmt, RepeatStmt,
    # Actions
    UseStmt, RespondStmt, EmitStmt, StoreStmt, RecallStmt, AssignStmt,
    # Expressions v1
    Expression, BinaryExpr, UnaryExpr, Literal, Identifier, MemberAccess, ListLiteral,
    # v2.0: Imports/Exports
    ImportStmt, ExportDecl,
    # v2.0: Type System
    TypeDef, TypeAnnotation,
    # v2.0: Async/Await
    AsyncEventHandler, AwaitExpr,
    # v2.0: Pattern Matching
    MatchStmt, CaseClause, PatternExpr,
    # v2.0: Error Handling
    TryStmt, RaiseStmt,
    # v2.0: Function Definitions
    FunctionDef, ParamDef, ReturnStmt, BreakStmt, ContinueStmt,
    # v2.0: Loops
    ForStmt,
    # v2.0: Concurrency
    ParallelBlock, SpawnExpr, JoinExpr,
    # v2.0: Pipeline & Expressions
    PipelineExpr, CallExpr, MapLiteral, WithExpr, IndexExpr, StringInterpolation,
    # v2.0: Decorators
    DecoratorExpr, DecoratedDecl
)


class ParserError(Exception):
    """Raised when the parser encounters a syntax error."""
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"Parser error at line {token.line}, column {token.column}: {message}")


class Parser:
    """
    Recursive descent parser for the AION programming language v2.0.
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
        """Skip newline tokens (preserves indent structure for statement blocks)."""
        while self.check(TokenType.NEWLINE):
            self.advance()
    
    def skip_in_braces(self) -> None:
        """Skip newline, indent, and dedent tokens inside brace blocks."""
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
        # v2.0: Handle decorators
        if self.check(TokenType.AT):
            return self.parse_decorated_decl()
        
        # v2.0: Imports
        if self.check(TokenType.IMPORT):
            return self.parse_import_stmt()
        
        # v2.0: Exports
        if self.check(TokenType.EXPORT):
            return self.parse_export_decl()
        
        # v2.0: Type definitions
        if self.check(TokenType.TYPE):
            return self.parse_type_def()
        
        # v1.0 declarations
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
    
    # ============ v2.0: Imports/Exports ============
    
    def parse_import_stmt(self) -> ImportStmt:
        """Parse import statement: import path.to.module as alias"""
        self.expect(TokenType.IMPORT, "'import'")
        
        # Parse module path (dotted identifier)
        path_parts = [self.expect(TokenType.IDENTIFIER, "module name").value]
        while self.match(TokenType.DOT):
            path_parts.append(self.expect(TokenType.IDENTIFIER, "module name").value)
        
        module_path = ".".join(path_parts)
        
        # Optional alias
        alias = None
        if self.match(TokenType.AS):
            alias = self.expect(TokenType.IDENTIFIER, "alias name").value
        
        return ImportStmt(module_path, alias)
    
    def parse_export_decl(self) -> ExportDecl:
        """Parse export declaration: export agent/tool/model Name"""
        self.expect(TokenType.EXPORT, "'export'")
        
        # Determine export type
        if self.check(TokenType.AGENT):
            self.advance()
            export_type = "agent"
        elif self.check(TokenType.TOOL):
            self.advance()
            export_type = "tool"
        elif self.check(TokenType.MODEL):
            self.advance()
            export_type = "model"
        else:
            raise ParserError("Expected 'agent', 'tool', or 'model' after 'export'", self.current)
        
        target_name = self.expect(TokenType.IDENTIFIER, "name to export").value
        return ExportDecl(export_type, target_name)
    
    # ============ v2.0: Type System ============
    
    def parse_type_def(self) -> TypeDef:
        """Parse type definition: type Name = { field: Type, ... }"""
        self.expect(TokenType.TYPE, "'type'")
        name = self.expect(TokenType.IDENTIFIER, "type name").value
        self.expect(TokenType.EQ, "'='")
        
        self.expect(TokenType.LBRACE, "'{'")
        self.skip_newlines()
        
        fields = {}
        while not self.check(TokenType.RBRACE, TokenType.EOF):
            field_name = self.expect(TokenType.IDENTIFIER, "field name").value
            self.expect(TokenType.COLON, "':'")
            field_type = self.parse_type_annotation_inline()
            fields[field_name] = field_type
            
            self.match(TokenType.COMMA)
            self.skip_newlines()
        
        self.expect(TokenType.RBRACE, "'}'")
        return TypeDef(name, fields)
    
    def parse_type_annotation(self) -> TypeAnnotation:
        """Parse :: Type or :: Type[T]"""
        self.expect(TokenType.DOUBLECOLON, "'::'")
        return self.parse_type_annotation_inline()
    
    def parse_type_annotation_inline(self) -> TypeAnnotation:
        """Parse Type or Type[T] without leading ::"""
        type_name = self.expect(TokenType.IDENTIFIER, "type name").value
        
        # Check for type parameters [T, U, ...]
        type_params = []
        if self.match(TokenType.LBRACKET):
            type_params.append(self.expect(TokenType.IDENTIFIER, "type parameter").value)
            while self.match(TokenType.COMMA):
                type_params.append(self.expect(TokenType.IDENTIFIER, "type parameter").value)
            self.expect(TokenType.RBRACKET, "']'")
        
        return TypeAnnotation(type_name, type_params)
    
    # ============ v2.0: Decorators ============
    
    def parse_decorated_decl(self) -> DecoratedDecl:
        """Parse decorated declaration: @decorator ... agent/tool/model Name"""
        decorators = []
        
        while self.check(TokenType.AT):
            self.advance()
            name = self.expect(TokenType.IDENTIFIER, "decorator name").value
            
            args = []
            if self.match(TokenType.LPAREN):
                if not self.check(TokenType.RPAREN):
                    args.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        args.append(self.parse_expression())
                self.expect(TokenType.RPAREN, "')'")
            
            decorators.append(DecoratorExpr(name, args))
            self.skip_newlines()
        
        # Parse the actual declaration
        declaration = self.parse_declaration()
        return DecoratedDecl(decorators, declaration)
    
    # ============ Agent Parsing ============
    
    def parse_agent_decl(self) -> AgentDecl:
        """Parse an agent declaration."""
        self.expect(TokenType.AGENT, "'agent'")
        name_token = self.expect(TokenType.IDENTIFIER, "agent name")
        self.expect(TokenType.LBRACE, "'{'")
        self.skip_in_braces()
        
        body = []
        while not self.check(TokenType.RBRACE, TokenType.EOF):
            member = self.parse_agent_member()
            if member:
                body.append(member)
            self.skip_in_braces()
        
        self.expect(TokenType.RBRACE, "'}'")
        return AgentDecl(name_token.value, body)
    
    def parse_agent_member(self) -> Optional[ASTNode]:
        """Parse an agent body member."""
        if self.check(TokenType.GOAL):
            return self.parse_goal_stmt()
        elif self.check(TokenType.MEMORY):
            return self.parse_memory_decl()
        elif self.check(TokenType.MODEL):
            return self.parse_model_ref_with_config()
        elif self.check(TokenType.TOOL):
            return self.parse_tool_ref()
        elif self.check(TokenType.POLICY):
            return self.parse_policy_block()
        # v2.0: async event handlers
        elif self.check(TokenType.ASYNC):
            return self.parse_async_event_handler()
        elif self.check(TokenType.ON):
            return self.parse_event_handler()
        # v2.0: function definitions inside agent
        elif self.check(TokenType.FUNCTION) if hasattr(TokenType, 'FUNCTION') else False:
            return self.parse_function_def()
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
        
        # v2.0: Optional type annotation
        type_annotation = None
        if self.check(TokenType.DOUBLECOLON):
            type_annotation = self.parse_type_annotation()
        
        config = {}
        if self.match(TokenType.LBRACE):
            config = self.parse_config_pairs()
            self.expect(TokenType.RBRACE, "'}'")
        
        if type_annotation:
            config['_type_annotation'] = type_annotation
        
        return MemoryDecl(memory_type, config)
    
    def parse_model_decl(self) -> ModelDecl:
        """Parse a top-level model declaration."""
        self.expect(TokenType.MODEL, "'model'")
        name_token = self.expect(TokenType.IDENTIFIER, "model name")
        
        config = {}
        if self.match(TokenType.LBRACE):
            self.skip_in_braces()
            config = self.parse_config_pairs()
            self.skip_in_braces()
            self.expect(TokenType.RBRACE, "'}'")
        
        return ModelDecl(name_token.value, config)
    
    def parse_model_ref(self) -> ModelRef:
        """Parse a model reference within an agent (simple)."""
        self.expect(TokenType.MODEL, "'model'")
        name_token = self.expect(TokenType.IDENTIFIER, "model name")
        return ModelRef(name_token.value)
    
    def parse_model_ref_with_config(self) -> ASTNode:
        """Parse a model reference with optional config block."""
        self.expect(TokenType.MODEL, "'model'")
        name_token = self.expect(TokenType.IDENTIFIER, "model name")
        
        # Check if there's a config block
        if self.match(TokenType.LBRACE):
            self.skip_in_braces()
            config = self.parse_config_pairs()
            self.skip_in_braces()
            self.expect(TokenType.RBRACE, "'}'")
            return ModelDecl(name_token.value, config)
        
        return ModelRef(name_token.value)
    
    def parse_tool_decl(self) -> ToolDecl:
        """Parse a top-level tool declaration."""
        self.expect(TokenType.TOOL, "'tool'")
        name_token = self.expect(TokenType.IDENTIFIER, "tool name")
        
        config = {}
        if self.match(TokenType.LBRACE):
            self.skip_in_braces()
            config = self.parse_config_pairs()
            self.skip_in_braces()
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
        self.skip_in_braces()
        config = self.parse_config_pairs()
        self.skip_in_braces()
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
        
        # Parse parameters with optional type annotations
        params = []
        self.expect(TokenType.LPAREN, "'('")
        if self.check(TokenType.IDENTIFIER):
            params.append(self.advance().value)
            # v2.0: Skip type annotation for now in simple handler
            if self.check(TokenType.DOUBLECOLON):
                self.parse_type_annotation()
            while self.match(TokenType.COMMA):
                params.append(self.expect(TokenType.IDENTIFIER, "parameter name").value)
                if self.check(TokenType.DOUBLECOLON):
                    self.parse_type_annotation()
        self.expect(TokenType.RPAREN, "')'")
        
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        # Parse body (expect indent)
        body = self.parse_statement_block()
        
        return EventHandler(event_type, params, body)
    
    def parse_async_event_handler(self) -> AsyncEventHandler:
        """Parse async event handler: async on event(params):"""
        self.expect(TokenType.ASYNC, "'async'")
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
        
        # Parse parameters with optional type annotations
        params = []
        param_types = []
        self.expect(TokenType.LPAREN, "'('")
        if self.check(TokenType.IDENTIFIER):
            params.append(self.advance().value)
            if self.check(TokenType.DOUBLECOLON):
                param_types.append(self.parse_type_annotation())
            else:
                param_types.append(None)
            while self.match(TokenType.COMMA):
                params.append(self.expect(TokenType.IDENTIFIER, "parameter name").value)
                if self.check(TokenType.DOUBLECOLON):
                    param_types.append(self.parse_type_annotation())
                else:
                    param_types.append(None)
        self.expect(TokenType.RPAREN, "')'")
        
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        body = self.parse_statement_block()
        
        return AsyncEventHandler(event_type, params, param_types, body)
    
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
        
        # v1.0 Control flow
        elif self.check(TokenType.IF):
            return self.parse_if_stmt()
        elif self.check(TokenType.WHEN):
            return self.parse_when_stmt()
        elif self.check(TokenType.REPEAT):
            return self.parse_repeat_stmt()
        
        # v2.0 Control flow
        elif self.check(TokenType.MATCH):
            return self.parse_match_stmt()
        elif self.check(TokenType.TRY):
            return self.parse_try_stmt()
        elif self.check(TokenType.FOR):
            return self.parse_for_stmt()
        elif self.check(TokenType.RETURN):
            return self.parse_return_stmt()
        elif self.check(TokenType.BREAK):
            return self.parse_break_stmt()
        elif self.check(TokenType.CONTINUE):
            return self.parse_continue_stmt()
        elif self.check(TokenType.RAISE) if hasattr(TokenType, 'RAISE') else False:
            return self.parse_raise_stmt()
        
        # v2.0 Concurrency
        elif self.check(TokenType.PARALLEL):
            return self.parse_parallel_block()
        elif self.check(TokenType.AWAIT):
            return self.parse_await_as_stmt()
        
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
            # Otherwise parse as expression statement
            expr = self.parse_expression()
            return expr  # Expression statements are valid
        
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
    
    # ============ v2.0: Pattern Matching ============
    
    def parse_match_stmt(self) -> MatchStmt:
        """Parse match statement: match expr: case patterns..."""
        self.expect(TokenType.MATCH, "'match'")
        target = self.parse_expression()
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        cases = []
        self.expect(TokenType.INDENT, "indented case clauses")
        
        while not self.check(TokenType.DEDENT, TokenType.EOF):
            case_clause = self.parse_case_clause()
            if case_clause:
                cases.append(case_clause)
            self.skip_newlines()
        
        self.match(TokenType.DEDENT)
        
        return MatchStmt(target, cases)
    
    def parse_case_clause(self) -> CaseClause:
        """Parse a case clause."""
        is_default = False
        patterns = []
        guard = None
        binding = None
        
        if self.match(TokenType.DEFAULT):
            is_default = True
        elif self.match(TokenType.CASE):
            # Parse pattern(s) with | for alternatives
            patterns.append(self.parse_pattern())
            while self.match(TokenType.PIPE) if hasattr(TokenType, 'PIPE') and self.check(TokenType.PIPE) else False:
                # Handle | for OR patterns - but PIPE is |> not |
                # For now, use simple string matching in patterns
                pass
            
            # Check for 'as' binding
            if self.match(TokenType.AS):
                if self.match(TokenType.LPAREN):
                    # Multiple bindings: as (a, b)
                    bindings = [self.expect(TokenType.IDENTIFIER, "binding").value]
                    while self.match(TokenType.COMMA):
                        bindings.append(self.expect(TokenType.IDENTIFIER, "binding").value)
                    self.expect(TokenType.RPAREN, "')'")
                    binding = tuple(bindings)
                else:
                    binding = self.expect(TokenType.IDENTIFIER, "binding").value
            
            # Check for 'where' guard
            if self.match(TokenType.WHERE):
                guard = self.parse_expression()
        else:
            raise ParserError("Expected 'case' or 'default'", self.current)
        
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        body = self.parse_statement_block()
        
        return CaseClause(patterns, guard, binding, body, is_default)
    
    def parse_pattern(self) -> Expression:
        """Parse a pattern expression."""
        # For now, patterns are just expressions
        # Could be string literals, regex patterns, wildcards, etc.
        return self.parse_expression()
    
    # ============ v2.0: Error Handling ============
    
    def parse_try_stmt(self) -> TryStmt:
        """Parse try/catch/finally statement."""
        self.expect(TokenType.TRY, "'try'")
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        try_body = self.parse_statement_block()
        
        catch_param = None
        catch_body = []
        finally_body = []
        
        self.skip_newlines()
        
        if self.match(TokenType.CATCH):
            if self.check(TokenType.IDENTIFIER):
                catch_param = self.advance().value
            self.expect(TokenType.COLON, "':'")
            self.skip_newlines()
            catch_body = self.parse_statement_block()
        
        self.skip_newlines()
        
        if self.match(TokenType.FINALLY):
            self.expect(TokenType.COLON, "':'")
            self.skip_newlines()
            finally_body = self.parse_statement_block()
        
        return TryStmt(try_body, catch_param, catch_body, finally_body)
    
    def parse_raise_stmt(self) -> RaiseStmt:
        """Parse raise statement."""
        self.advance()  # consume 'raise'
        expr = self.parse_expression()
        return RaiseStmt(expr)
    
    # ============ v2.0: Loops ============
    
    def parse_for_stmt(self) -> ForStmt:
        """Parse for each loop: for each item in collection:"""
        self.expect(TokenType.FOR, "'for'")
        self.match(TokenType.EACH)  # 'each' is optional
        
        variable = self.expect(TokenType.IDENTIFIER, "loop variable").value
        self.expect(TokenType.IN, "'in'")
        iterable = self.parse_expression()
        
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        body = self.parse_statement_block()
        
        return ForStmt(variable, iterable, body)
    
    def parse_return_stmt(self) -> ReturnStmt:
        """Parse return statement."""
        self.expect(TokenType.RETURN, "'return'")
        value = None
        if not self.check(TokenType.NEWLINE, TokenType.EOF, TokenType.DEDENT):
            value = self.parse_expression()
        return ReturnStmt(value)
    
    def parse_break_stmt(self) -> BreakStmt:
        """Parse break statement."""
        self.expect(TokenType.BREAK, "'break'")
        return BreakStmt()
    
    def parse_continue_stmt(self) -> ContinueStmt:
        """Parse continue statement."""
        self.expect(TokenType.CONTINUE, "'continue'")
        return ContinueStmt()
    
    # ============ v2.0: Concurrency ============
    
    def parse_parallel_block(self) -> ParallelBlock:
        """Parse parallel block: parallel: spawn tasks..."""
        self.expect(TokenType.PARALLEL, "'parallel'")
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        statements = self.parse_statement_block()
        
        return ParallelBlock(statements)
    
    def parse_await_as_stmt(self) -> AssignStmt:
        """Parse await as a statement: result = await expr"""
        # This is handled in assignment typically
        expr = self.parse_expression()
        return expr
    
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
    
    # ============ v2.0: Function Definitions ============
    
    def parse_function_def(self) -> FunctionDef:
        """Parse function definition: [async] function name(params) -> Type:"""
        is_async = False
        if self.match(TokenType.ASYNC):
            is_async = True
        
        # Note: FUNCTION token may not exist, check identifier "function"
        if self.check(TokenType.IDENTIFIER) and self.current.value == "function":
            self.advance()
        
        name = self.expect(TokenType.IDENTIFIER, "function name").value
        
        # Parse parameters
        self.expect(TokenType.LPAREN, "'('")
        params = []
        if not self.check(TokenType.RPAREN):
            params.append(self.parse_param_def())
            while self.match(TokenType.COMMA):
                params.append(self.parse_param_def())
        self.expect(TokenType.RPAREN, "')'")
        
        # Optional return type
        return_type = None
        if self.match(TokenType.ARROW):
            return_type = self.parse_type_annotation_inline()
        
        self.expect(TokenType.COLON, "':'")
        self.skip_newlines()
        
        body = self.parse_statement_block()
        
        return FunctionDef(name, params, return_type, body, is_async)
    
    def parse_param_def(self) -> ParamDef:
        """Parse parameter definition: name :: Type = default"""
        name = self.expect(TokenType.IDENTIFIER, "parameter name").value
        
        type_annotation = None
        if self.check(TokenType.DOUBLECOLON):
            type_annotation = self.parse_type_annotation()
        
        default_value = None
        if self.match(TokenType.EQ):
            default_value = self.parse_expression()
        
        return ParamDef(name, type_annotation, default_value)
    
    # ============ Expression Parsing ============
    
    def parse_expression(self) -> Expression:
        """Parse an expression."""
        return self.parse_pipeline()
    
    def parse_pipeline(self) -> Expression:
        """Parse pipeline expressions: expr |> func1 |> func2"""
        expr = self.parse_logic_or()
        
        if self.check(TokenType.PIPE):
            stages = []
            while self.match(TokenType.PIPE):
                stage = self.parse_logic_or()
                stages.append(stage)
            return PipelineExpr(expr, stages)
        
        return expr
    
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
        
        # v2.0: await expression
        elif self.match(TokenType.AWAIT):
            operand = self.parse_unary()
            return AwaitExpr(operand)
        
        # v2.0: spawn expression
        elif self.match(TokenType.SPAWN):
            operand = self.parse_unary()
            return SpawnExpr(operand)
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> Expression:
        """Parse postfix expressions (calls, member access, indexing)."""
        expr = self.parse_primary()
        
        while True:
            if self.match(TokenType.LPAREN):
                # Function call
                args = []
                if not self.check(TokenType.RPAREN):
                    args.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        args.append(self.parse_expression())
                self.expect(TokenType.RPAREN, "')'")
                expr = CallExpr(expr, args)
            
            elif self.match(TokenType.DOT):
                # Member access
                member = self.expect(TokenType.IDENTIFIER, "member name").value
                expr = MemberAccess(expr, member)
            
            elif self.match(TokenType.LBRACKET):
                # Index access
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET, "']'")
                expr = IndexExpr(expr, index)
            
            else:
                break
        
        return expr
    
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
        
        # v2.0: Map literal or block
        elif self.check(TokenType.LBRACE):
            return self.parse_map_literal()
        
        # v2.0: join() expression
        elif self.check(TokenType.JOIN):
            return self.parse_join_expr()
        
        # Parenthesized expression
        elif self.match(TokenType.LPAREN):
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN, "')'")
            return expr
        
        # Identifier
        elif self.check(TokenType.IDENTIFIER):
            return Identifier(self.advance().value)
        
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
    
    def parse_map_literal(self) -> MapLiteral:
        """Parse map literal: { key: value, ... }"""
        self.expect(TokenType.LBRACE, "'{'")
        self.skip_newlines()
        
        entries = []
        if not self.check(TokenType.RBRACE):
            key = self.expect(TokenType.IDENTIFIER, "key").value
            self.expect(TokenType.COLON, "':'")
            value = self.parse_expression()
            entries.append((key, value))
            
            while self.match(TokenType.COMMA):
                self.skip_newlines()
                if self.check(TokenType.RBRACE):
                    break  # Trailing comma
                key = self.expect(TokenType.IDENTIFIER, "key").value
                self.expect(TokenType.COLON, "':'")
                value = self.parse_expression()
                entries.append((key, value))
            self.skip_newlines()
        
        self.expect(TokenType.RBRACE, "'}'")
        return MapLiteral(entries)
    
    def parse_join_expr(self) -> JoinExpr:
        """Parse join(task1, task2, ...) expression."""
        self.expect(TokenType.JOIN, "'join'")
        self.expect(TokenType.LPAREN, "'('")
        
        tasks = []
        if not self.check(TokenType.RPAREN):
            tasks.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                tasks.append(self.parse_expression())
        
        self.expect(TokenType.RPAREN, "')'")
        return JoinExpr(tasks)


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
