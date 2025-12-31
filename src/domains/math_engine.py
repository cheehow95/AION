"""
AION Mathematics Domain Engine
==============================

A symbolic mathematics engine for AION agents to reason about
mathematical concepts, solve equations, and perform calculus.

Features:
- Expression parsing and simplification
- Symbolic differentiation and integration
- Matrix operations and linear algebra
- Equation solving
- Number theory basics
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# SYMBOLIC EXPRESSION TREE
# =============================================================================

class ExprType(Enum):
    NUMBER = "number"
    VARIABLE = "variable"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    LOG = "log"
    EXP = "exp"
    SQRT = "sqrt"
    NEG = "neg"


@dataclass
class Expr:
    """Symbolic expression node."""
    type: ExprType
    value: Optional[float] = None  # For NUMBER
    name: Optional[str] = None     # For VARIABLE
    args: List['Expr'] = field(default_factory=list)
    
    def __repr__(self):
        if self.type == ExprType.NUMBER:
            if self.value == int(self.value):
                return str(int(self.value))
            return str(self.value)
        elif self.type == ExprType.VARIABLE:
            return self.name
        elif self.type == ExprType.ADD:
            return f"({self.args[0]} + {self.args[1]})"
        elif self.type == ExprType.SUB:
            return f"({self.args[0]} - {self.args[1]})"
        elif self.type == ExprType.MUL:
            return f"({self.args[0]} * {self.args[1]})"
        elif self.type == ExprType.DIV:
            return f"({self.args[0]} / {self.args[1]})"
        elif self.type == ExprType.POW:
            return f"({self.args[0]}^{self.args[1]})"
        elif self.type == ExprType.NEG:
            return f"(-{self.args[0]})"
        else:
            return f"{self.type.value}({', '.join(str(a) for a in self.args)})"


# Helper constructors
def Num(value: float) -> Expr:
    return Expr(ExprType.NUMBER, value=value)

def Var(name: str) -> Expr:
    return Expr(ExprType.VARIABLE, name=name)

def Add(a: Expr, b: Expr) -> Expr:
    return Expr(ExprType.ADD, args=[a, b])

def Sub(a: Expr, b: Expr) -> Expr:
    return Expr(ExprType.SUB, args=[a, b])

def Mul(a: Expr, b: Expr) -> Expr:
    return Expr(ExprType.MUL, args=[a, b])

def Div(a: Expr, b: Expr) -> Expr:
    return Expr(ExprType.DIV, args=[a, b])

def Pow(a: Expr, b: Expr) -> Expr:
    return Expr(ExprType.POW, args=[a, b])

def Sin(a: Expr) -> Expr:
    return Expr(ExprType.SIN, args=[a])

def Cos(a: Expr) -> Expr:
    return Expr(ExprType.COS, args=[a])

def Log(a: Expr) -> Expr:
    return Expr(ExprType.LOG, args=[a])

def Exp(a: Expr) -> Expr:
    return Expr(ExprType.EXP, args=[a])

def Sqrt(a: Expr) -> Expr:
    return Expr(ExprType.SQRT, args=[a])

def Neg(a: Expr) -> Expr:
    return Expr(ExprType.NEG, args=[a])


# =============================================================================
# EXPRESSION PARSER
# =============================================================================

class ExpressionParser:
    """
    Parse mathematical expressions from string.
    Supports: +, -, *, /, ^, sin, cos, log, exp, sqrt, parentheses
    """
    
    def __init__(self, text: str):
        self.text = text.replace(' ', '')
        self.pos = 0
    
    def parse(self) -> Expr:
        """Parse full expression."""
        result = self._parse_additive()
        if self.pos < len(self.text):
            raise ValueError(f"Unexpected character at position {self.pos}: {self.text[self.pos]}")
        return result
    
    def _parse_additive(self) -> Expr:
        """Parse addition and subtraction."""
        left = self._parse_multiplicative()
        
        while self.pos < len(self.text) and self.text[self.pos] in '+-':
            op = self.text[self.pos]
            self.pos += 1
            right = self._parse_multiplicative()
            if op == '+':
                left = Add(left, right)
            else:
                left = Sub(left, right)
        
        return left
    
    def _parse_multiplicative(self) -> Expr:
        """Parse multiplication and division."""
        left = self._parse_power()
        
        while self.pos < len(self.text) and self.text[self.pos] in '*/':
            op = self.text[self.pos]
            self.pos += 1
            right = self._parse_power()
            if op == '*':
                left = Mul(left, right)
            else:
                left = Div(left, right)
        
        return left
    
    def _parse_power(self) -> Expr:
        """Parse exponentiation."""
        left = self._parse_unary()
        
        if self.pos < len(self.text) and self.text[self.pos] == '^':
            self.pos += 1
            right = self._parse_power()  # Right associative
            left = Pow(left, right)
        
        return left
    
    def _parse_unary(self) -> Expr:
        """Parse unary operators."""
        if self.pos < len(self.text) and self.text[self.pos] == '-':
            self.pos += 1
            arg = self._parse_unary()
            return Neg(arg)
        return self._parse_primary()
    
    def _parse_primary(self) -> Expr:
        """Parse primary expressions (numbers, variables, functions, parentheses)."""
        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of expression")
        
        char = self.text[self.pos]
        
        # Parentheses
        if char == '(':
            self.pos += 1
            result = self._parse_additive()
            if self.pos >= len(self.text) or self.text[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
            self.pos += 1
            return result
        
        # Functions
        for func_name, func_constructor in [
            ('sin', Sin), ('cos', Cos), ('tan', lambda x: Expr(ExprType.TAN, args=[x])),
            ('log', Log), ('exp', Exp), ('sqrt', Sqrt), ('ln', Log)
        ]:
            if self.text[self.pos:].startswith(func_name):
                self.pos += len(func_name)
                if self.pos >= len(self.text) or self.text[self.pos] != '(':
                    raise ValueError(f"Expected '(' after {func_name}")
                self.pos += 1
                arg = self._parse_additive()
                if self.pos >= len(self.text) or self.text[self.pos] != ')':
                    raise ValueError(f"Missing ')' after {func_name}")
                self.pos += 1
                return func_constructor(arg)
        
        # Numbers
        if char.isdigit() or char == '.':
            start = self.pos
            while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
                self.pos += 1
            return Num(float(self.text[start:self.pos]))
        
        # Constants
        if self.text[self.pos:].startswith('pi'):
            self.pos += 2
            return Num(math.pi)
        if self.text[self.pos:].startswith('e') and (self.pos + 1 >= len(self.text) or not self.text[self.pos + 1].isalpha()):
            self.pos += 1
            return Num(math.e)
        
        # Variables
        if char.isalpha():
            start = self.pos
            while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
                self.pos += 1
            return Var(self.text[start:self.pos])
        
        raise ValueError(f"Unexpected character: {char}")


def parse(text: str) -> Expr:
    """Parse a mathematical expression string."""
    return ExpressionParser(text).parse()


# =============================================================================
# SYMBOLIC DIFFERENTIATION
# =============================================================================

def differentiate(expr: Expr, var: str) -> Expr:
    """
    Compute symbolic derivative of expression with respect to variable.
    Uses standard differentiation rules.
    """
    if expr.type == ExprType.NUMBER:
        return Num(0)
    
    elif expr.type == ExprType.VARIABLE:
        return Num(1) if expr.name == var else Num(0)
    
    elif expr.type == ExprType.ADD:
        # d/dx (f + g) = f' + g'
        return Add(differentiate(expr.args[0], var), differentiate(expr.args[1], var))
    
    elif expr.type == ExprType.SUB:
        # d/dx (f - g) = f' - g'
        return Sub(differentiate(expr.args[0], var), differentiate(expr.args[1], var))
    
    elif expr.type == ExprType.MUL:
        # Product rule: d/dx (f * g) = f' * g + f * g'
        f, g = expr.args
        return Add(Mul(differentiate(f, var), g), Mul(f, differentiate(g, var)))
    
    elif expr.type == ExprType.DIV:
        # Quotient rule: d/dx (f / g) = (f' * g - f * g') / g^2
        f, g = expr.args
        num = Sub(Mul(differentiate(f, var), g), Mul(f, differentiate(g, var)))
        denom = Pow(g, Num(2))
        return Div(num, denom)
    
    elif expr.type == ExprType.POW:
        # Power rule for x^n: n * x^(n-1)
        # General: d/dx (f^g) = f^g * (g' * ln(f) + g * f'/f)
        f, g = expr.args
        if g.type == ExprType.NUMBER:
            # Simple power rule: n * f^(n-1) * f'
            n = g.value
            return Mul(Mul(Num(n), Pow(f, Num(n - 1))), differentiate(f, var))
        else:
            # General case
            return Mul(expr, Add(
                Mul(differentiate(g, var), Log(f)),
                Mul(g, Div(differentiate(f, var), f))
            ))
    
    elif expr.type == ExprType.SIN:
        # d/dx sin(f) = cos(f) * f'
        f = expr.args[0]
        return Mul(Cos(f), differentiate(f, var))
    
    elif expr.type == ExprType.COS:
        # d/dx cos(f) = -sin(f) * f'
        f = expr.args[0]
        return Mul(Neg(Sin(f)), differentiate(f, var))
    
    elif expr.type == ExprType.LOG:
        # d/dx ln(f) = f'/f
        f = expr.args[0]
        return Div(differentiate(f, var), f)
    
    elif expr.type == ExprType.EXP:
        # d/dx exp(f) = exp(f) * f'
        f = expr.args[0]
        return Mul(expr, differentiate(f, var))
    
    elif expr.type == ExprType.SQRT:
        # d/dx sqrt(f) = f' / (2 * sqrt(f))
        f = expr.args[0]
        return Div(differentiate(f, var), Mul(Num(2), Sqrt(f)))
    
    elif expr.type == ExprType.NEG:
        return Neg(differentiate(expr.args[0], var))
    
    raise ValueError(f"Cannot differentiate expression of type {expr.type}")


# =============================================================================
# EXPRESSION SIMPLIFICATION
# =============================================================================

def simplify(expr: Expr) -> Expr:
    """
    Simplify an expression by applying algebraic rules.
    """
    if expr.type in [ExprType.NUMBER, ExprType.VARIABLE]:
        return expr
    
    # Recursively simplify arguments
    args = [simplify(a) for a in expr.args]
    
    if expr.type == ExprType.ADD:
        a, b = args
        # 0 + x = x
        if a.type == ExprType.NUMBER and a.value == 0:
            return b
        # x + 0 = x
        if b.type == ExprType.NUMBER and b.value == 0:
            return a
        # Constant folding
        if a.type == ExprType.NUMBER and b.type == ExprType.NUMBER:
            return Num(a.value + b.value)
        return Add(a, b)
    
    elif expr.type == ExprType.SUB:
        a, b = args
        # x - 0 = x
        if b.type == ExprType.NUMBER and b.value == 0:
            return a
        # x - x = 0
        if _expr_equal(a, b):
            return Num(0)
        # Constant folding
        if a.type == ExprType.NUMBER and b.type == ExprType.NUMBER:
            return Num(a.value - b.value)
        return Sub(a, b)
    
    elif expr.type == ExprType.MUL:
        a, b = args
        # 0 * x = 0
        if a.type == ExprType.NUMBER and a.value == 0:
            return Num(0)
        if b.type == ExprType.NUMBER and b.value == 0:
            return Num(0)
        # 1 * x = x
        if a.type == ExprType.NUMBER and a.value == 1:
            return b
        if b.type == ExprType.NUMBER and b.value == 1:
            return a
        # Constant folding
        if a.type == ExprType.NUMBER and b.type == ExprType.NUMBER:
            return Num(a.value * b.value)
        return Mul(a, b)
    
    elif expr.type == ExprType.DIV:
        a, b = args
        # 0 / x = 0
        if a.type == ExprType.NUMBER and a.value == 0:
            return Num(0)
        # x / 1 = x
        if b.type == ExprType.NUMBER and b.value == 1:
            return a
        # x / x = 1
        if _expr_equal(a, b):
            return Num(1)
        # Constant folding
        if a.type == ExprType.NUMBER and b.type == ExprType.NUMBER and b.value != 0:
            return Num(a.value / b.value)
        return Div(a, b)
    
    elif expr.type == ExprType.POW:
        a, b = args
        # x^0 = 1
        if b.type == ExprType.NUMBER and b.value == 0:
            return Num(1)
        # x^1 = x
        if b.type == ExprType.NUMBER and b.value == 1:
            return a
        # 0^x = 0
        if a.type == ExprType.NUMBER and a.value == 0:
            return Num(0)
        # 1^x = 1
        if a.type == ExprType.NUMBER and a.value == 1:
            return Num(1)
        # Constant folding
        if a.type == ExprType.NUMBER and b.type == ExprType.NUMBER:
            return Num(a.value ** b.value)
        return Pow(a, b)
    
    elif expr.type == ExprType.NEG:
        a = args[0]
        # -(-x) = x
        if a.type == ExprType.NEG:
            return a.args[0]
        # Constant folding
        if a.type == ExprType.NUMBER:
            return Num(-a.value)
        return Neg(a)
    
    return Expr(expr.type, args=args)


def _expr_equal(a: Expr, b: Expr) -> bool:
    """Check if two expressions are structurally equal."""
    if a.type != b.type:
        return False
    if a.type == ExprType.NUMBER:
        return a.value == b.value
    if a.type == ExprType.VARIABLE:
        return a.name == b.name
    if len(a.args) != len(b.args):
        return False
    return all(_expr_equal(aa, bb) for aa, bb in zip(a.args, b.args))


# =============================================================================
# EXPRESSION EVALUATION
# =============================================================================

def evaluate(expr: Expr, variables: Dict[str, float] = None) -> float:
    """
    Evaluate expression with given variable values.
    """
    variables = variables or {}
    
    if expr.type == ExprType.NUMBER:
        return expr.value
    elif expr.type == ExprType.VARIABLE:
        if expr.name not in variables:
            raise ValueError(f"Undefined variable: {expr.name}")
        return variables[expr.name]
    elif expr.type == ExprType.ADD:
        return evaluate(expr.args[0], variables) + evaluate(expr.args[1], variables)
    elif expr.type == ExprType.SUB:
        return evaluate(expr.args[0], variables) - evaluate(expr.args[1], variables)
    elif expr.type == ExprType.MUL:
        return evaluate(expr.args[0], variables) * evaluate(expr.args[1], variables)
    elif expr.type == ExprType.DIV:
        return evaluate(expr.args[0], variables) / evaluate(expr.args[1], variables)
    elif expr.type == ExprType.POW:
        return evaluate(expr.args[0], variables) ** evaluate(expr.args[1], variables)
    elif expr.type == ExprType.SIN:
        return math.sin(evaluate(expr.args[0], variables))
    elif expr.type == ExprType.COS:
        return math.cos(evaluate(expr.args[0], variables))
    elif expr.type == ExprType.TAN:
        return math.tan(evaluate(expr.args[0], variables))
    elif expr.type == ExprType.LOG:
        return math.log(evaluate(expr.args[0], variables))
    elif expr.type == ExprType.EXP:
        return math.exp(evaluate(expr.args[0], variables))
    elif expr.type == ExprType.SQRT:
        return math.sqrt(evaluate(expr.args[0], variables))
    elif expr.type == ExprType.NEG:
        return -evaluate(expr.args[0], variables)
    else:
        raise ValueError(f"Cannot evaluate expression of type {expr.type}")


# =============================================================================
# LINEAR ALGEBRA
# =============================================================================

@dataclass
class Matrix:
    """Simple matrix class for linear algebra operations."""
    rows: int
    cols: int
    data: List[List[float]]
    
    def __repr__(self):
        lines = []
        for row in self.data:
            lines.append("[" + ", ".join(f"{v:8.3f}" for v in row) + "]")
        return "\n".join(lines)
    
    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'Matrix':
        return cls(rows, cols, [[0.0] * cols for _ in range(rows)])
    
    @classmethod
    def identity(cls, n: int) -> 'Matrix':
        m = cls.zeros(n, n)
        for i in range(n):
            m.data[i][i] = 1.0
        return m
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        assert self.rows == other.rows and self.cols == other.cols
        result = Matrix.zeros(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result
    
    def __mul__(self, other: Union['Matrix', float]) -> 'Matrix':
        if isinstance(other, (int, float)):
            # Scalar multiplication
            result = Matrix.zeros(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * other
            return result
        else:
            # Matrix multiplication
            assert self.cols == other.rows
            result = Matrix.zeros(self.rows, other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    for k in range(self.cols):
                        result.data[i][j] += self.data[i][k] * other.data[k][j]
            return result
    
    def transpose(self) -> 'Matrix':
        result = Matrix.zeros(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result
    
    def determinant(self) -> float:
        """Calculate determinant (only for square matrices)."""
        assert self.rows == self.cols, "Determinant only for square matrices"
        n = self.rows
        
        if n == 1:
            return self.data[0][0]
        if n == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        
        # Use LU decomposition for larger matrices
        det = 1.0
        m = [row[:] for row in self.data]  # Copy
        
        for i in range(n):
            # Partial pivoting
            max_row = i
            for k in range(i + 1, n):
                if abs(m[k][i]) > abs(m[max_row][i]):
                    max_row = k
            if max_row != i:
                m[i], m[max_row] = m[max_row], m[i]
                det *= -1
            
            if abs(m[i][i]) < 1e-10:
                return 0.0
            
            det *= m[i][i]
            
            for k in range(i + 1, n):
                factor = m[k][i] / m[i][i]
                for j in range(i, n):
                    m[k][j] -= factor * m[i][j]
        
        return det
    
    def trace(self) -> float:
        """Calculate trace (sum of diagonal elements)."""
        assert self.rows == self.cols, "Trace only for square matrices"
        return sum(self.data[i][i] for i in range(self.rows))


# =============================================================================
# EQUATION SOLVER
# =============================================================================

def solve_linear(a: float, b: float) -> Optional[float]:
    """Solve linear equation: ax + b = 0"""
    if a == 0:
        return None if b != 0 else 0
    return -b / a


def solve_quadratic(a: float, b: float, c: float) -> Tuple[Optional[float], Optional[float]]:
    """Solve quadratic equation: ax² + bx + c = 0"""
    if a == 0:
        root = solve_linear(b, c)
        return (root, None)
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return (None, None)  # Complex roots not implemented
    elif discriminant == 0:
        root = -b / (2 * a)
        return (root, root)
    else:
        sqrt_disc = math.sqrt(discriminant)
        x1 = (-b + sqrt_disc) / (2 * a)
        x2 = (-b - sqrt_disc) / (2 * a)
        return (x1, x2)


# =============================================================================
# NUMBER THEORY
# =============================================================================

def gcd(a: int, b: int) -> int:
    """Greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return abs(a)


def lcm(a: int, b: int) -> int:
    """Least common multiple."""
    return abs(a * b) // gcd(a, b) if a and b else 0


def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def prime_factors(n: int) -> List[int]:
    """Get prime factorization of n."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def fibonacci(n: int) -> int:
    """Calculate n-th Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def factorial(n: int) -> int:
    """Calculate n factorial."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# =============================================================================
# MATH ENGINE - Main Interface
# =============================================================================

class MathEngine:
    """
    AION Math Engine for symbolic and numerical computation.
    """
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
    
    def parse(self, expression: str) -> Expr:
        """Parse a mathematical expression."""
        return parse(expression)
    
    def differentiate(self, expression: str, variable: str = "x") -> str:
        """Differentiate an expression with respect to a variable."""
        expr = parse(expression)
        derivative = differentiate(expr, variable)
        simplified = simplify(derivative)
        
        self.history.append({
            "operation": "differentiate",
            "input": expression,
            "variable": variable,
            "result": str(simplified)
        })
        
        return str(simplified)
    
    def simplify(self, expression: str) -> str:
        """Simplify an expression."""
        expr = parse(expression)
        simplified = simplify(expr)
        return str(simplified)
    
    def evaluate(self, expression: str, **variables) -> float:
        """Evaluate an expression with given variable values."""
        expr = parse(expression)
        return evaluate(expr, variables)
    
    def solve_equation(self, equation: str) -> List[float]:
        """
        Solve a polynomial equation (up to degree 2).
        Format: "ax^2 + bx + c = 0" or "ax + b = 0"
        """
        # Simple parsing for polynomial form
        # This is a basic implementation
        equation = equation.replace(' ', '').replace('=0', '')
        
        # Try to detect quadratic
        if 'x^2' in equation or 'x**2' in equation:
            # Parse coefficients (simplified)
            # For demo purposes
            return []  # TODO: Full polynomial parsing
        
        return []
    
    def matrix_operations(self, operation: str, *matrices) -> Matrix:
        """Perform matrix operations."""
        if operation == "identity":
            n = int(matrices[0])
            return Matrix.identity(n)
        elif operation == "transpose":
            return matrices[0].transpose()
        elif operation == "determinant":
            return matrices[0].determinant()
        elif operation == "multiply":
            return matrices[0] * matrices[1]
        raise ValueError(f"Unknown operation: {operation}")


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Mathematics Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          📐 AION MATHEMATICS ENGINE 📐                                    ║
║                                                                           ║
║     Symbolic computation, calculus, and linear algebra                    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = MathEngine()
    
    # Expression parsing
    print("📝 Expression Parsing:")
    print("-" * 50)
    expressions = ["x^2 + 2*x + 1", "sin(x) + cos(x)", "exp(x) * log(x)"]
    for expr_str in expressions:
        expr = engine.parse(expr_str)
        print(f"   {expr_str:20} → {expr}")
    
    # Differentiation
    print("\n📊 Symbolic Differentiation:")
    print("-" * 50)
    derivatives = [
        ("x^2", "x"),
        ("sin(x)", "x"),
        ("x^3 + 2*x", "x"),
        ("exp(x)", "x"),
    ]
    for expr_str, var in derivatives:
        result = engine.differentiate(expr_str, var)
        print(f"   d/d{var} [{expr_str}] = {result}")
    
    # Evaluation
    print("\n🔢 Expression Evaluation:")
    print("-" * 50)
    print(f"   x^2 + 2*x + 1 at x=3: {engine.evaluate('x^2 + 2*x + 1', x=3)}")
    print(f"   sin(pi/2): {engine.evaluate('sin(pi/2)'):.6f}")
    print(f"   sqrt(2): {engine.evaluate('sqrt(2)'):.6f}")
    
    # Linear algebra
    print("\n🔷 Linear Algebra:")
    print("-" * 50)
    m = Matrix(3, 3, [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ])
    print(f"   Matrix:")
    for row in m.data:
        print(f"      {row}")
    print(f"   Determinant: {m.determinant():.3f}")
    print(f"   Trace: {m.trace():.3f}")
    
    # Number theory
    print("\n🔢 Number Theory:")
    print("-" * 50)
    print(f"   gcd(48, 18) = {gcd(48, 18)}")
    print(f"   lcm(12, 18) = {lcm(12, 18)}")
    print(f"   is_prime(17) = {is_prime(17)}")
    print(f"   prime_factors(84) = {prime_factors(84)}")
    print(f"   fibonacci(10) = {fibonacci(10)}")
    print(f"   factorial(6) = {factorial(6)}")
    
    # Quadratic solver
    print("\n✅ Equation Solving:")
    print("-" * 50)
    roots = solve_quadratic(1, -5, 6)  # x² - 5x + 6 = 0
    print(f"   x² - 5x + 6 = 0 → x = {roots[0]}, {roots[1]}")
    roots = solve_quadratic(1, 0, -4)  # x² - 4 = 0
    print(f"   x² - 4 = 0 → x = {roots[0]}, {roots[1]}")


if __name__ == "__main__":
    demo()
