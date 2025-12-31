"""
AION Lexer
Tokenizes AION source code into a stream of tokens.
"""

from typing import Generator, Optional
from .token_types import Token, TokenType, KEYWORDS


class LexerError(Exception):
    """Raised when the lexer encounters an invalid token."""
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Lexer error at line {line}, column {column}: {message}")


class Lexer:
    """
    Tokenizer for the AION programming language.
    Converts source code into a stream of tokens.
    """
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.indent_stack = [0]
        
    @property
    def current_char(self) -> Optional[str]:
        """Returns the current character or None if at end."""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek(self, offset: int = 1) -> Optional[str]:
        """Peek at a character ahead without consuming it."""
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self) -> Optional[str]:
        """Consume and return the current character."""
        char = self.current_char
        if char is not None:
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        return char
    
    def skip_whitespace(self) -> None:
        """Skip spaces and tabs (not newlines)."""
        while self.current_char in (' ', '\t'):
            self.advance()
    
    def skip_comment(self) -> None:
        """Skip a comment line starting with #."""
        while self.current_char is not None and self.current_char != '\n':
            self.advance()
    
    def read_string(self) -> Token:
        """Read a string literal enclosed in double quotes."""
        start_line = self.line
        start_col = self.column
        self.advance()  # consume opening "
        
        value = []
        while self.current_char is not None and self.current_char != '"':
            if self.current_char == '\\':
                self.advance()
                escape_char = self.current_char
                if escape_char == 'n':
                    value.append('\n')
                elif escape_char == 't':
                    value.append('\t')
                elif escape_char == '\\':
                    value.append('\\')
                elif escape_char == '"':
                    value.append('"')
                else:
                    value.append(escape_char or '')
                self.advance()
            else:
                value.append(self.current_char)
                self.advance()
        
        if self.current_char != '"':
            raise LexerError("Unterminated string literal", start_line, start_col)
        
        self.advance()  # consume closing "
        return Token(TokenType.STRING, ''.join(value), start_line, start_col)
    
    def read_number(self) -> Token:
        """Read a numeric literal (integer or float)."""
        start_line = self.line
        start_col = self.column
        value = []
        
        # Handle negative numbers
        if self.current_char == '-':
            value.append('-')
            self.advance()
        
        # Integer part
        while self.current_char is not None and self.current_char.isdigit():
            value.append(self.current_char)
            self.advance()
        
        # Decimal part
        if self.current_char == '.' and self.peek() and self.peek().isdigit():
            value.append('.')
            self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                value.append(self.current_char)
                self.advance()
        
        num_str = ''.join(value)
        num_value = float(num_str) if '.' in num_str else int(num_str)
        return Token(TokenType.NUMBER, num_value, start_line, start_col)
    
    def read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start_line = self.line
        start_col = self.column
        value = []
        
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            value.append(self.current_char)
            self.advance()
        
        identifier = ''.join(value)
        
        # Check if it's a keyword
        if identifier in KEYWORDS:
            token_type = KEYWORDS[identifier]
            # Handle boolean literals
            if identifier == 'true':
                return Token(token_type, True, start_line, start_col)
            elif identifier == 'false':
                return Token(token_type, False, start_line, start_col)
            elif identifier == 'null':
                return Token(token_type, None, start_line, start_col)
            return Token(token_type, identifier, start_line, start_col)
        
        return Token(TokenType.IDENTIFIER, identifier, start_line, start_col)
    
    def handle_newline(self) -> Generator[Token, None, None]:
        """Handle newlines and indentation changes."""
        start_line = self.line
        start_col = self.column
        
        yield Token(TokenType.NEWLINE, '\n', start_line, start_col)
        self.advance()  # consume newline
        
        # Skip blank lines
        while self.current_char == '\n':
            self.advance()
        
        # Count indentation
        indent = 0
        while self.current_char in (' ', '\t'):
            if self.current_char == ' ':
                indent += 1
            else:  # tab
                indent += 4
            self.advance()
        
        # Skip if it's a comment line or empty line
        if self.current_char == '#' or self.current_char == '\n' or self.current_char is None:
            return
        
        current_indent = self.indent_stack[-1]
        
        if indent > current_indent:
            self.indent_stack.append(indent)
            yield Token(TokenType.INDENT, indent, self.line, 1)
        elif indent < current_indent:
            while self.indent_stack and self.indent_stack[-1] > indent:
                self.indent_stack.pop()
                yield Token(TokenType.DEDENT, indent, self.line, 1)
    
    def tokenize(self) -> Generator[Token, None, None]:
        """Generate tokens from the source code."""
        while self.current_char is not None:
            start_line = self.line
            start_col = self.column
            
            # Skip whitespace (not newlines)
            if self.current_char in (' ', '\t'):
                self.skip_whitespace()
                continue
            
            # Comments
            if self.current_char == '#':
                self.skip_comment()
                continue
            
            # Newlines (with indent handling)
            if self.current_char == '\n':
                yield from self.handle_newline()
                continue
            
            # String literals
            if self.current_char == '"':
                yield self.read_string()
                continue
            
            # Number literals
            if self.current_char.isdigit() or (self.current_char == '-' and self.peek() and self.peek().isdigit()):
                yield self.read_number()
                continue
            
            # Identifiers and keywords
            if self.current_char.isalpha() or self.current_char == '_':
                yield self.read_identifier()
                continue
            
            # Two-character operators
            if self.current_char == '=' and self.peek() == '=':
                self.advance()
                self.advance()
                yield Token(TokenType.EQEQ, '==', start_line, start_col)
                continue
            
            if self.current_char == '!' and self.peek() == '=':
                self.advance()
                self.advance()
                yield Token(TokenType.NEQ, '!=', start_line, start_col)
                continue
            
            if self.current_char == '<' and self.peek() == '=':
                self.advance()
                self.advance()
                yield Token(TokenType.LTE, '<=', start_line, start_col)
                continue
            
            if self.current_char == '>' and self.peek() == '=':
                self.advance()
                self.advance()
                yield Token(TokenType.GTE, '>=', start_line, start_col)
                continue
            
            # Single-character tokens
            single_char_tokens = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.STAR,
                '/': TokenType.SLASH,
                '=': TokenType.EQ,
                '<': TokenType.LT,
                '>': TokenType.GT,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                ',': TokenType.COMMA,
                ':': TokenType.COLON,
                '.': TokenType.DOT,
            }
            
            if self.current_char in single_char_tokens:
                char = self.current_char
                self.advance()
                yield Token(single_char_tokens[char], char, start_line, start_col)
                continue
            
            # Unknown character
            raise LexerError(f"Unexpected character: {self.current_char!r}", start_line, start_col)
        
        # Emit remaining DEDENTs
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            yield Token(TokenType.DEDENT, 0, self.line, self.column)
        
        yield Token(TokenType.EOF, None, self.line, self.column)


def tokenize(source: str) -> list[Token]:
    """
    Convenience function to tokenize source code and return a list of tokens.
    
    Args:
        source: AION source code
        
    Returns:
        List of tokens
    """
    lexer = Lexer(source)
    return list(lexer.tokenize())
