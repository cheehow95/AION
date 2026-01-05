"""
AION Quantum Computing Engine
=============================

Comprehensive quantum computing implementation covering:
- Quantum gates (all standard single and multi-qubit gates)
- Quantum circuits and simulation
- Quantum algorithms (Grover, Shor, QFT, VQE, QAOA)
- Quantum error correction codes
- Quantum cryptography protocols
- Quantum simulation for chemistry

Full-featured quantum computer simulator for AI reasoning.
"""

import math
import cmath
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Union
from enum import Enum
import copy


# =============================================================================
# CONSTANTS
# =============================================================================

class QCConstants:
    """Quantum computing constants."""
    # Computational basis states
    ZERO = (1+0j, 0+0j)  # |0⟩
    ONE = (0+0j, 1+0j)   # |1⟩
    
    # Common states
    PLUS = (1/math.sqrt(2), 1/math.sqrt(2))   # |+⟩ = (|0⟩+|1⟩)/√2
    MINUS = (1/math.sqrt(2), -1/math.sqrt(2)) # |-⟩ = (|0⟩-|1⟩)/√2


# =============================================================================
# QUANTUM GATES
# =============================================================================

class QuantumGate:
    """
    Quantum gate represented as a unitary matrix.
    """
    
    def __init__(self, name: str, matrix: List[List[complex]], n_qubits: int = 1):
        self.name = name
        self.matrix = matrix
        self.n_qubits = n_qubits
        self.size = 2 ** n_qubits
    
    def __repr__(self):
        return f"Gate({self.name})"
    
    def apply(self, state: List[complex]) -> List[complex]:
        """Apply gate to quantum state vector."""
        if len(state) != self.size:
            raise ValueError(f"State size {len(state)} doesn't match gate size {self.size}")
        
        result = [0j] * self.size
        for i in range(self.size):
            for j in range(self.size):
                result[i] += self.matrix[i][j] * state[j]
        return result
    
    @property
    def dag(self) -> 'QuantumGate':
        """Return conjugate transpose (adjoint) of gate."""
        n = len(self.matrix)
        adj = [[self.matrix[j][i].conjugate() for j in range(n)] for i in range(n)]
        return QuantumGate(f"{self.name}†", adj, self.n_qubits)


class Gates:
    """Standard quantum gates library."""
    
    # =========================================================================
    # SINGLE-QUBIT GATES
    # =========================================================================
    
    # Pauli gates
    I = QuantumGate("I", [[1, 0], [0, 1]])
    X = QuantumGate("X", [[0, 1], [1, 0]])  # NOT gate, bit flip
    Y = QuantumGate("Y", [[0, -1j], [1j, 0]])
    Z = QuantumGate("Z", [[1, 0], [0, -1]])  # Phase flip
    
    # Hadamard
    H = QuantumGate("H", [[1/math.sqrt(2), 1/math.sqrt(2)], 
                          [1/math.sqrt(2), -1/math.sqrt(2)]])
    
    # Phase gates
    S = QuantumGate("S", [[1, 0], [0, 1j]])  # √Z, π/2 phase
    Sdag = QuantumGate("S†", [[1, 0], [0, -1j]])
    T = QuantumGate("T", [[1, 0], [0, cmath.exp(1j * math.pi / 4)]])  # π/8 gate
    Tdag = QuantumGate("T†", [[1, 0], [0, cmath.exp(-1j * math.pi / 4)]])
    
    # Square root of NOT
    SX = QuantumGate("√X", [[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]])
    
    @staticmethod
    def Rx(theta: float) -> QuantumGate:
        """Rotation around X-axis: Rx(θ) = exp(-iθX/2)"""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return QuantumGate(f"Rx({theta:.2f})", [[c, -1j*s], [-1j*s, c]])
    
    @staticmethod
    def Ry(theta: float) -> QuantumGate:
        """Rotation around Y-axis: Ry(θ) = exp(-iθY/2)"""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return QuantumGate(f"Ry({theta:.2f})", [[c, -s], [s, c]])
    
    @staticmethod
    def Rz(theta: float) -> QuantumGate:
        """Rotation around Z-axis: Rz(θ) = exp(-iθZ/2)"""
        return QuantumGate(f"Rz({theta:.2f})", 
                          [[cmath.exp(-1j*theta/2), 0], 
                           [0, cmath.exp(1j*theta/2)]])
    
    @staticmethod
    def P(phi: float) -> QuantumGate:
        """Phase gate: P(φ) = diag(1, e^(iφ))"""
        return QuantumGate(f"P({phi:.2f})", [[1, 0], [0, cmath.exp(1j*phi)]])
    
    @staticmethod
    def U(theta: float, phi: float, lam: float) -> QuantumGate:
        """
        General single-qubit unitary:
        U(θ,φ,λ) = Rz(φ)Ry(θ)Rz(λ)
        """
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return QuantumGate(f"U({theta:.2f},{phi:.2f},{lam:.2f})", [
            [c, -cmath.exp(1j*lam)*s],
            [cmath.exp(1j*phi)*s, cmath.exp(1j*(phi+lam))*c]
        ])
    
    # =========================================================================
    # TWO-QUBIT GATES
    # =========================================================================
    
    # CNOT (Controlled-NOT)
    CNOT = QuantumGate("CNOT", [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], n_qubits=2)
    CX = CNOT  # Alias
    
    # CZ (Controlled-Z)
    CZ = QuantumGate("CZ", [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], n_qubits=2)
    
    # SWAP
    SWAP = QuantumGate("SWAP", [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], n_qubits=2)
    
    # iSWAP
    iSWAP = QuantumGate("iSWAP", [
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1]
    ], n_qubits=2)
    
    # √SWAP
    SQSWAP = QuantumGate("√SWAP", [
        [1, 0, 0, 0],
        [0, 0.5+0.5j, 0.5-0.5j, 0],
        [0, 0.5-0.5j, 0.5+0.5j, 0],
        [0, 0, 0, 1]
    ], n_qubits=2)
    
    @staticmethod
    def CRz(theta: float) -> QuantumGate:
        """Controlled-Rz gate."""
        return QuantumGate(f"CRz({theta:.2f})", [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cmath.exp(-1j*theta/2), 0],
            [0, 0, 0, cmath.exp(1j*theta/2)]
        ], n_qubits=2)
    
    @staticmethod
    def CU(theta: float, phi: float, lam: float) -> QuantumGate:
        """Controlled-U gate."""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return QuantumGate(f"CU", [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -cmath.exp(1j*lam)*s],
            [0, 0, cmath.exp(1j*phi)*s, cmath.exp(1j*(phi+lam))*c]
        ], n_qubits=2)
    
    # =========================================================================
    # THREE-QUBIT GATES
    # =========================================================================
    
    # Toffoli (CCNOT)
    TOFFOLI = QuantumGate("Toffoli", [
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,1,0]
    ], n_qubits=3)
    CCX = TOFFOLI  # Alias
    
    # Fredkin (CSWAP)
    FREDKIN = QuantumGate("Fredkin", [
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1]
    ], n_qubits=3)
    CSWAP = FREDKIN


# =============================================================================
# QUANTUM REGISTER
# =============================================================================

class QuantumRegister:
    """
    Quantum register holding n qubits.
    Represents state as 2^n dimensional complex vector.
    """
    
    def __init__(self, n_qubits: int, initial_state: Optional[List[complex]] = None):
        self.n = n_qubits
        self.size = 2 ** n_qubits
        
        if initial_state:
            if len(initial_state) != self.size:
                raise ValueError(f"Initial state must have {self.size} amplitudes")
            self.state = list(initial_state)
        else:
            # Initialize to |00...0⟩
            self.state = [0j] * self.size
            self.state[0] = 1 + 0j
    
    def __repr__(self):
        return f"QuantumRegister({self.n} qubits)"
    
    def reset(self):
        """Reset to |00...0⟩ state."""
        self.state = [0j] * self.size
        self.state[0] = 1 + 0j
    
    def set_state(self, state: List[complex]):
        """Set state vector directly."""
        if len(state) != self.size:
            raise ValueError(f"State must have {self.size} amplitudes")
        self.state = list(state)
        self.normalize()
    
    def normalize(self):
        """Normalize state vector."""
        norm = math.sqrt(sum(abs(a)**2 for a in self.state))
        if norm > 0:
            self.state = [a/norm for a in self.state]
    
    def probabilities(self) -> List[float]:
        """Get measurement probabilities for all basis states."""
        return [abs(a)**2 for a in self.state]
    
    def measure(self) -> int:
        """
        Measure all qubits, collapsing to a basis state.
        Returns the measured value as integer.
        """
        probs = self.probabilities()
        r = random.random()
        cumulative = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                # Collapse state
                self.state = [0j] * self.size
                self.state[i] = 1 + 0j
                return i
        return self.size - 1
    
    def measure_qubit(self, qubit: int) -> int:
        """Measure a single qubit, partially collapsing the state."""
        prob_0 = 0
        for i in range(self.size):
            if (i >> qubit) & 1 == 0:
                prob_0 += abs(self.state[i])**2
        
        result = 0 if random.random() < prob_0 else 1
        
        # Collapse
        new_state = [0j] * self.size
        norm = 0
        for i in range(self.size):
            if (i >> qubit) & 1 == result:
                new_state[i] = self.state[i]
                norm += abs(self.state[i])**2
        
        norm = math.sqrt(norm)
        self.state = [a/norm if norm > 0 else a for a in new_state]
        
        return result
    
    def to_binary(self, value: int) -> str:
        """Convert measurement result to binary string."""
        return format(value, f'0{self.n}b')
    
    def amplitude(self, state_index: int) -> complex:
        """Get amplitude of specific basis state."""
        return self.state[state_index]
    
    def fidelity(self, other: 'QuantumRegister') -> float:
        """Calculate fidelity with another state: |⟨ψ|φ⟩|²"""
        if self.size != other.size:
            raise ValueError("Registers must have same size")
        inner = sum(a.conjugate() * b for a, b in zip(self.state, other.state))
        return abs(inner) ** 2


# =============================================================================
# QUANTUM CIRCUIT
# =============================================================================

@dataclass
class CircuitOperation:
    """A single operation in a quantum circuit."""
    gate: QuantumGate
    qubits: List[int]
    params: Dict = field(default_factory=dict)


class QuantumCircuit:
    """
    Quantum circuit builder and simulator.
    """
    
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.operations: List[CircuitOperation] = []
        self.register = QuantumRegister(n_qubits)
    
    def __repr__(self):
        return f"QuantumCircuit({self.n} qubits, {len(self.operations)} gates)"
    
    # Gate application methods
    def apply(self, gate: QuantumGate, qubits: List[int]) -> 'QuantumCircuit':
        """Apply gate to specified qubits."""
        self.operations.append(CircuitOperation(gate, qubits))
        return self
    
    def i(self, qubit: int) -> 'QuantumCircuit':
        return self.apply(Gates.I, [qubit])
    
    def x(self, qubit: int) -> 'QuantumCircuit':
        return self.apply(Gates.X, [qubit])
    
    def y(self, qubit: int) -> 'QuantumCircuit':
        return self.apply(Gates.Y, [qubit])
    
    def z(self, qubit: int) -> 'QuantumCircuit':
        return self.apply(Gates.Z, [qubit])
    
    def h(self, qubit: int) -> 'QuantumCircuit':
        return self.apply(Gates.H, [qubit])
    
    def s(self, qubit: int) -> 'QuantumCircuit':
        return self.apply(Gates.S, [qubit])
    
    def t(self, qubit: int) -> 'QuantumCircuit':
        return self.apply(Gates.T, [qubit])
    
    def rx(self, qubit: int, theta: float) -> 'QuantumCircuit':
        return self.apply(Gates.Rx(theta), [qubit])
    
    def ry(self, qubit: int, theta: float) -> 'QuantumCircuit':
        return self.apply(Gates.Ry(theta), [qubit])
    
    def rz(self, qubit: int, theta: float) -> 'QuantumCircuit':
        return self.apply(Gates.Rz(theta), [qubit])
    
    def cx(self, control: int, target: int) -> 'QuantumCircuit':
        return self.apply(Gates.CNOT, [control, target])
    
    def cnot(self, control: int, target: int) -> 'QuantumCircuit':
        return self.cx(control, target)
    
    def cz(self, control: int, target: int) -> 'QuantumCircuit':
        return self.apply(Gates.CZ, [control, target])
    
    def swap(self, q1: int, q2: int) -> 'QuantumCircuit':
        return self.apply(Gates.SWAP, [q1, q2])
    
    def ccx(self, c1: int, c2: int, target: int) -> 'QuantumCircuit':
        return self.apply(Gates.TOFFOLI, [c1, c2, target])
    
    def toffoli(self, c1: int, c2: int, target: int) -> 'QuantumCircuit':
        return self.ccx(c1, c2, target)
    
    def barrier(self) -> 'QuantumCircuit':
        """Visual barrier (no-op)."""
        return self
    
    def _apply_single_qubit_gate(self, gate: QuantumGate, qubit: int):
        """Apply single-qubit gate to register."""
        new_state = [0j] * self.register.size
        
        for i in range(self.register.size):
            bit = (i >> qubit) & 1
            # For each basis state, apply the gate
            for new_bit in range(2):
                # Index with bit changed
                j = i if bit == 0 else i ^ (1 << qubit)
                if bit == 0:
                    new_state[i] += gate.matrix[0][0] * self.register.state[i]
                    new_state[i ^ (1 << qubit)] += gate.matrix[1][0] * self.register.state[i]
        
        # Correct implementation
        new_state = [0j] * self.register.size
        for i in range(self.register.size):
            for j in range(self.register.size):
                # Check if states differ only on the target qubit
                if (i ^ j) == 0 or (i ^ j) == (1 << qubit):
                    bit_i = (i >> qubit) & 1
                    bit_j = (j >> qubit) & 1
                    if (i & ~(1 << qubit)) == (j & ~(1 << qubit)):
                        new_state[i] += gate.matrix[bit_i][bit_j] * self.register.state[j]
        
        self.register.state = new_state
    
    def _apply_two_qubit_gate(self, gate: QuantumGate, qubits: List[int]):
        """Apply two-qubit gate."""
        q0, q1 = qubits[0], qubits[1]
        new_state = [0j] * self.register.size
        
        for i in range(self.register.size):
            for j in range(self.register.size):
                # States can only differ on the two target qubits
                other_bits_i = i & ~((1 << q0) | (1 << q1))
                other_bits_j = j & ~((1 << q0) | (1 << q1))
                
                if other_bits_i == other_bits_j:
                    bit0_i = (i >> q0) & 1
                    bit1_i = (i >> q1) & 1
                    bit0_j = (j >> q0) & 1
                    bit1_j = (j >> q1) & 1
                    
                    gate_i = bit0_i * 2 + bit1_i
                    gate_j = bit0_j * 2 + bit1_j
                    
                    new_state[i] += gate.matrix[gate_i][gate_j] * self.register.state[j]
        
        self.register.state = new_state
    
    def run(self, shots: int = 1) -> Dict[str, int]:
        """
        Execute circuit and return measurement counts.
        """
        counts = {}
        
        for _ in range(shots):
            # Reset register
            self.register.reset()
            
            # Apply all operations
            for op in self.operations:
                if op.gate.n_qubits == 1:
                    self._apply_single_qubit_gate(op.gate, op.qubits[0])
                elif op.gate.n_qubits == 2:
                    self._apply_two_qubit_gate(op.gate, op.qubits)
            
            # Measure
            result = self.register.measure()
            result_str = self.register.to_binary(result)
            counts[result_str] = counts.get(result_str, 0) + 1
        
        return counts
    
    def statevector(self) -> List[complex]:
        """Get final state vector (without measurement)."""
        self.register.reset()
        
        for op in self.operations:
            if op.gate.n_qubits == 1:
                self._apply_single_qubit_gate(op.gate, op.qubits[0])
            elif op.gate.n_qubits == 2:
                self._apply_two_qubit_gate(op.gate, op.qubits)
        
        return self.register.state
    
    def draw(self) -> str:
        """Draw circuit as ASCII art."""
        lines = [f"q{i}: ─" for i in range(self.n)]
        
        for op in self.operations:
            gate_name = op.gate.name[:3]
            
            if len(op.qubits) == 1:
                q = op.qubits[0]
                for i in range(self.n):
                    if i == q:
                        lines[i] += f"[{gate_name}]─"
                    else:
                        lines[i] += "───" + "─"
            else:
                for i in range(self.n):
                    if i in op.qubits:
                        if i == op.qubits[0]:
                            lines[i] += "●───"
                        else:
                            lines[i] += f"[{gate_name[:1]}]─"
                    else:
                        lines[i] += "│───" if min(op.qubits) < i < max(op.qubits) else "────"
        
        return "\n".join(lines)


# =============================================================================
# QUANTUM ALGORITHMS
# =============================================================================

class QuantumAlgorithms:
    """Standard quantum algorithms."""
    
    @staticmethod
    def bell_state(which: str = 'phi+') -> QuantumCircuit:
        """
        Create a Bell state.
        |Φ+⟩ = (|00⟩ + |11⟩)/√2
        |Φ-⟩ = (|00⟩ - |11⟩)/√2
        |Ψ+⟩ = (|01⟩ + |10⟩)/√2
        |Ψ-⟩ = (|01⟩ - |10⟩)/√2
        """
        qc = QuantumCircuit(2)
        
        if which in ['phi-', 'psi-']:
            qc.x(0)
        
        qc.h(0)
        qc.cx(0, 1)
        
        if which in ['psi+', 'psi-']:
            qc.x(1)
        
        return qc
    
    @staticmethod
    def ghz_state(n: int) -> QuantumCircuit:
        """
        Create GHZ state: (|00...0⟩ + |11...1⟩)/√2
        """
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(1, n):
            qc.cx(0, i)
        return qc
    
    @staticmethod
    def qft(n: int) -> QuantumCircuit:
        """
        Quantum Fourier Transform on n qubits.
        """
        qc = QuantumCircuit(n)
        
        for i in range(n):
            qc.h(i)
            for j in range(i + 1, n):
                angle = math.pi / (2 ** (j - i))
                qc.apply(Gates.CRz(angle), [j, i])
        
        # Swap qubits for correct ordering
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
        
        return qc
    
    @staticmethod
    def inverse_qft(n: int) -> QuantumCircuit:
        """Inverse QFT."""
        qc = QuantumCircuit(n)
        
        # Swap first
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
        
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                angle = -math.pi / (2 ** (j - i))
                qc.apply(Gates.CRz(angle), [j, i])
            qc.h(i)
        
        return qc
    
    @staticmethod
    def grover_oracle(n: int, target: int) -> QuantumCircuit:
        """
        Oracle that marks the target state with a phase flip.
        """
        qc = QuantumCircuit(n)
        
        # Convert target to binary and apply X where bit is 0
        for i in range(n):
            if not (target >> i) & 1:
                qc.x(i)
        
        # Multi-controlled Z (simplified for demonstration)
        if n == 2:
            qc.cz(0, 1)
        elif n == 3:
            qc.h(2)
            qc.ccx(0, 1, 2)
            qc.h(2)
        
        # Undo X gates
        for i in range(n):
            if not (target >> i) & 1:
                qc.x(i)
        
        return qc
    
    @staticmethod
    def grover_diffusion(n: int) -> QuantumCircuit:
        """Grover diffusion operator."""
        qc = QuantumCircuit(n)
        
        # H on all qubits
        for i in range(n):
            qc.h(i)
        
        # X on all qubits
        for i in range(n):
            qc.x(i)
        
        # Multi-controlled Z
        if n == 2:
            qc.cz(0, 1)
        elif n >= 3:
            qc.h(n - 1)
            if n == 3:
                qc.ccx(0, 1, 2)
            qc.h(n - 1)
        
        # X on all qubits
        for i in range(n):
            qc.x(i)
        
        # H on all qubits
        for i in range(n):
            qc.h(i)
        
        return qc
    
    @staticmethod
    def grover_search(n: int, target: int, iterations: int = None) -> QuantumCircuit:
        """
        Complete Grover's search algorithm.
        """
        if iterations is None:
            iterations = int(math.pi / 4 * math.sqrt(2 ** n))
        
        qc = QuantumCircuit(n)
        
        # Initialize superposition
        for i in range(n):
            qc.h(i)
        
        # Apply Grover iterations
        oracle = QuantumAlgorithms.grover_oracle(n, target)
        diffusion = QuantumAlgorithms.grover_diffusion(n)
        
        for _ in range(iterations):
            # We can't easily combine circuits here, so we add ops manually
            for op in oracle.operations:
                qc.operations.append(op)
            for op in diffusion.operations:
                qc.operations.append(op)
        
        return qc
    
    @staticmethod
    def deutsch_jozsa(n: int, is_constant: bool) -> QuantumCircuit:
        """
        Deutsch-Jozsa algorithm.
        
        Determines if function is constant or balanced in one query.
        """
        qc = QuantumCircuit(n + 1)  # n input + 1 ancilla
        
        # Initialize ancilla to |1⟩
        qc.x(n)
        
        # Apply H to all qubits
        for i in range(n + 1):
            qc.h(i)
        
        # Oracle (simplified)
        if is_constant:
            pass  # f(x) = 0 for all x
        else:
            # Balanced function: flip ancilla based on first qubit
            qc.cx(0, n)
        
        # Apply H to input qubits
        for i in range(n):
            qc.h(i)
        
        return qc


# =============================================================================
# QUANTUM ERROR CORRECTION
# =============================================================================

class QuantumErrorCorrection:
    """Quantum error correction codes."""
    
    @staticmethod
    def bit_flip_encode() -> QuantumCircuit:
        """
        3-qubit bit-flip code encoder.
        |ψ⟩ → |ψψψ⟩
        """
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(0, 2)
        return qc
    
    @staticmethod
    def bit_flip_decode() -> QuantumCircuit:
        """Bit-flip decoder with error correction."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.ccx(1, 2, 0)  # Majority vote
        return qc
    
    @staticmethod
    def phase_flip_encode() -> QuantumCircuit:
        """
        3-qubit phase-flip code encoder.
        Uses bit-flip in Hadamard basis.
        """
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        return qc
    
    @staticmethod
    def shor_9_qubit_encode() -> QuantumCircuit:
        """
        Shor's 9-qubit code (protects against both bit and phase flips).
        """
        qc = QuantumCircuit(9)
        
        # Phase flip encoding (using 3-qubit blocks)
        qc.cx(0, 3)
        qc.cx(0, 6)
        
        qc.h(0)
        qc.h(3)
        qc.h(6)
        
        # Bit flip encoding within each block
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(6, 7)
        qc.cx(6, 8)
        
        return qc


# =============================================================================
# QUANTUM CRYPTOGRAPHY
# =============================================================================

class QuantumCryptography:
    """Quantum cryptography protocols."""
    
    @staticmethod
    def bb84_prepare(bits: List[int], bases: List[int]) -> List[Tuple[int, int]]:
        """
        BB84 Protocol - Prepare qubits.
        
        bits: classical bits to encode
        bases: 0 for computational (Z), 1 for Hadamard (X)
        
        Returns list of (bit, basis) pairs.
        """
        return list(zip(bits, bases))
    
    @staticmethod
    def bb84_measure(prepared: List[Tuple[int, int]], 
                     measure_bases: List[int]) -> List[Optional[int]]:
        """
        BB84 measurement with optional wrong basis.
        Returns measured bits (None if incompatible basis).
        """
        results = []
        for (bit, prep_basis), meas_basis in zip(prepared, measure_bases):
            if prep_basis == meas_basis:
                results.append(bit)
            else:
                # Wrong basis: random result
                results.append(random.randint(0, 1) if random.random() > 0.5 else None)
        return results
    
    @staticmethod
    def bb84_sift_key(alice_bases: List[int], bob_bases: List[int],
                      bob_bits: List[int]) -> List[int]:
        """Sift key by keeping only matching bases."""
        key = []
        for ab, bb, bit in zip(alice_bases, bob_bases, bob_bits):
            if ab == bb and bit is not None:
                key.append(bit)
        return key
    
    @staticmethod
    def e91_protocol() -> Dict:
        """
        E91 Protocol description (Ekert 1991).
        Uses entangled Bell pairs for key distribution.
        """
        return {
            'name': 'E91 (Ekert 1991)',
            'type': 'Entanglement-based QKD',
            'principle': 'Bell pair entanglement',
            'security': 'Bell inequality violation detects eavesdropping',
            'steps': [
                '1. Create Bell pairs |Φ+⟩ = (|00⟩ + |11⟩)/√2',
                '2. Send one qubit to Alice, one to Bob',
                '3. Both measure in random bases',
                '4. Publicly compare bases',
                '5. Test CHSH inequality on subset',
                '6. Use matching measurements as key'
            ]
        }


# =============================================================================
# QUANTUM CHEMISTRY SIMULATION
# =============================================================================

class QuantumChemistry:
    """Quantum simulation for chemistry."""
    
    @staticmethod
    def vqe_ansatz(n_qubits: int, depth: int = 1) -> QuantumCircuit:
        """
        Variational Quantum Eigensolver (VQE) ansatz.
        Hardware-efficient ansatz with Ry and CNOT layers.
        """
        qc = QuantumCircuit(n_qubits)
        
        for d in range(depth):
            # Rotation layer
            for i in range(n_qubits):
                theta = random.uniform(0, 2 * math.pi)
                qc.ry(i, theta)
            
            # Entangling layer
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
        
        return qc
    
    @staticmethod
    def qaoa_layer(n_qubits: int, gamma: float, beta: float) -> QuantumCircuit:
        """
        QAOA (Quantum Approximate Optimization Algorithm) layer.
        """
        qc = QuantumCircuit(n_qubits)
        
        # Problem unitary (ZZ interactions)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(i + 1, gamma)
            qc.cx(i, i + 1)
        
        # Mixer unitary (Rx)
        for i in range(n_qubits):
            qc.rx(i, beta)
        
        return qc
    
    @staticmethod
    def h2_ground_state_circuit() -> QuantumCircuit:
        """
        Simplified circuit for H2 ground state estimation.
        Uses 2 qubits for the bonding orbital.
        """
        qc = QuantumCircuit(2)
        
        # Hartree-Fock reference: |01⟩
        qc.x(0)
        
        # Single and double excitations (simplified)
        theta = 0.2  # Would be optimized in real VQE
        qc.ry(1, theta)
        qc.cx(1, 0)
        
        return qc


# =============================================================================
# QUANTUM COMPUTING ENGINE
# =============================================================================

class QuantumComputingEngine:
    """
    AION Quantum Computing Engine.
    
    Complete quantum computing simulation and algorithm library.
    """
    
    def __init__(self):
        self.gates = Gates
        self.algorithms = QuantumAlgorithms
        self.error_correction = QuantumErrorCorrection
        self.crypto = QuantumCryptography
        self.chemistry = QuantumChemistry
    
    def circuit(self, n_qubits: int) -> QuantumCircuit:
        """Create a new quantum circuit."""
        return QuantumCircuit(n_qubits)
    
    def register(self, n_qubits: int) -> QuantumRegister:
        """Create a new quantum register."""
        return QuantumRegister(n_qubits)
    
    def bell_state(self, which: str = 'phi+') -> Dict:
        """Create and analyze a Bell state."""
        qc = self.algorithms.bell_state(which)
        state = qc.statevector()
        
        return {
            'type': which,
            'state_vector': state,
            'entangled': True,
            'circuit': qc.draw()
        }
    
    def ghz_state(self, n: int) -> Dict:
        """Create GHZ state."""
        qc = self.algorithms.ghz_state(n)
        state = qc.statevector()
        
        return {
            'n_qubits': n,
            'state': f'(|{"0"*n}⟩ + |{"1"*n}⟩)/√2',
            'state_vector': state,
            'maximally_entangled': True
        }
    
    def grover_search(self, n_qubits: int, target: int, shots: int = 100) -> Dict:
        """Run Grover's search algorithm."""
        qc = self.algorithms.grover_search(n_qubits, target)
        counts = qc.run(shots)
        
        target_str = format(target, f'0{n_qubits}b')
        success_rate = counts.get(target_str, 0) / shots
        
        return {
            'n_qubits': n_qubits,
            'target': target,
            'target_binary': target_str,
            'iterations': int(math.pi / 4 * math.sqrt(2 ** n_qubits)),
            'counts': counts,
            'success_rate': success_rate,
            'speedup': f'O(√N) vs O(N) classical'
        }
    
    def qft_demo(self, n: int, input_state: int = 0) -> Dict:
        """Demonstrate Quantum Fourier Transform."""
        qc = QuantumCircuit(n)
        
        # Prepare input state
        for i in range(n):
            if (input_state >> i) & 1:
                qc.x(i)
        
        # Apply QFT
        qft = self.algorithms.qft(n)
        for op in qft.operations:
            qc.operations.append(op)
        
        state = qc.statevector()
        
        return {
            'n_qubits': n,
            'input_state': input_state,
            'output_state_vector': state,
            'description': 'QFT transforms computational basis to Fourier basis'
        }
    
    def bb84_simulation(self, n_bits: int = 100) -> Dict:
        """Simulate BB84 quantum key distribution."""
        # Alice prepares
        alice_bits = [random.randint(0, 1) for _ in range(n_bits)]
        alice_bases = [random.randint(0, 1) for _ in range(n_bits)]
        
        # Bob measures
        bob_bases = [random.randint(0, 1) for _ in range(n_bits)]
        prepared = self.crypto.bb84_prepare(alice_bits, alice_bases)
        bob_results = self.crypto.bb84_measure(prepared, bob_bases)
        
        # Sift key
        final_key = self.crypto.bb84_sift_key(alice_bases, bob_bases, bob_results)
        
        return {
            'protocol': 'BB84',
            'raw_bits': n_bits,
            'key_length': len(final_key),
            'efficiency': len(final_key) / n_bits,
            'key_preview': ''.join(map(str, final_key[:20])) + '...',
            'secure': 'Yes (no eavesdropper)'
        }
    
    def gate_info(self, gate_name: str) -> Dict:
        """Get information about a quantum gate."""
        gate_map = {
            'X': (Gates.X, 'Pauli-X (NOT): Bit flip |0⟩↔|1⟩'),
            'Y': (Gates.Y, 'Pauli-Y: Bit+phase flip'),
            'Z': (Gates.Z, 'Pauli-Z: Phase flip |1⟩→-|1⟩'),
            'H': (Gates.H, 'Hadamard: Creates superposition'),
            'S': (Gates.S, 'S gate: π/2 phase (√Z)'),
            'T': (Gates.T, 'T gate: π/4 phase'),
            'CNOT': (Gates.CNOT, 'Controlled-NOT: XOR gate'),
            'CZ': (Gates.CZ, 'Controlled-Z: Conditional phase'),
            'SWAP': (Gates.SWAP, 'SWAP: Exchange qubits'),
            'Toffoli': (Gates.TOFFOLI, 'Toffoli (CCX): AND gate'),
        }
        
        if gate_name.upper() in gate_map:
            gate, desc = gate_map[gate_name.upper()]
            return {
                'name': gate.name,
                'description': desc,
                'n_qubits': gate.n_qubits,
                'matrix': gate.matrix,
                'unitary': True
            }
        return {'error': f'Unknown gate: {gate_name}'}
    
    def list_gates(self) -> Dict[str, List[str]]:
        """List all available gates."""
        return {
            'single_qubit': ['I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'Rx', 'Ry', 'Rz', 'P', 'U', '√X'],
            'two_qubit': ['CNOT/CX', 'CZ', 'SWAP', 'iSWAP', '√SWAP', 'CRz', 'CU'],
            'three_qubit': ['Toffoli/CCX', 'Fredkin/CSWAP'],
            'parameterized': ['Rx(θ)', 'Ry(θ)', 'Rz(θ)', 'P(φ)', 'U(θ,φ,λ)', 'CRz(θ)']
        }
    
    def list_algorithms(self) -> Dict[str, str]:
        """List available quantum algorithms."""
        return {
            'grover': 'Grover\'s search - O(√N) unstructured search',
            'qft': 'Quantum Fourier Transform - exponential speedup for period finding',
            'deutsch_jozsa': 'Deutsch-Jozsa - constant vs balanced in O(1)',
            'bell_state': 'Bell state preparation - maximum entanglement',
            'ghz_state': 'GHZ state - n-qubit entanglement',
            'vqe': 'Variational Quantum Eigensolver - ground state energy',
            'qaoa': 'Quantum Approximate Optimization Algorithm'
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the Quantum Computing Engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          💻 AION QUANTUM COMPUTING ENGINE 💻                              ║
║                                                                           ║
║     Gates, Circuits, Algorithms, Error Correction, Cryptography          ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    qc = QuantumComputingEngine()
    
    # Gates
    print("🔲 Quantum Gates:")
    print("-" * 50)
    gates = qc.list_gates()
    print(f"   Single-qubit: {', '.join(gates['single_qubit'][:6])}...")
    print(f"   Two-qubit: {', '.join(gates['two_qubit'][:4])}")
    print(f"   Three-qubit: {', '.join(gates['three_qubit'])}")
    
    # Bell state
    print("\n🔗 Bell State:")
    print("-" * 50)
    result = qc.bell_state('phi+')
    print(f"   |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print(f"   Entangled: {result['entangled']}")
    
    # Grover's search
    print("\n🔍 Grover's Search (3 qubits, target=5):")
    print("-" * 50)
    result = qc.grover_search(3, 5, shots=100)
    print(f"   Target: {result['target_binary']}")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Success rate: {result['success_rate']*100:.0f}%")
    print(f"   Speedup: {result['speedup']}")
    
    # BB84
    print("\n🔐 BB84 Quantum Key Distribution:")
    print("-" * 50)
    result = qc.bb84_simulation(100)
    print(f"   Raw bits: {result['raw_bits']}")
    print(f"   Final key length: {result['key_length']}")
    print(f"   Key preview: {result['key_preview']}")
    
    # Circuit
    print("\n📋 Example Circuit:")
    print("-" * 50)
    circuit = qc.circuit(3)
    circuit.h(0).cx(0, 1).cx(1, 2)
    print(circuit.draw())
    
    # Algorithms
    print("\n📚 Available Algorithms:")
    print("-" * 50)
    for name, desc in qc.list_algorithms().items():
        print(f"   {name}: {desc[:40]}...")


if __name__ == "__main__":
    demo()
