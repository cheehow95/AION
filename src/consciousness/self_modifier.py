"""
AION Safe Self-Modification Engine
===================================

Enables AION to modify its own source code safely with:
- Sandboxed execution for testing changes
- Rollback capability for failed modifications
- Safety constraints to prevent harmful changes
- Improvement measurement to validate enhancements

"Change is essential, but controlled change prevents catastrophe."
"""

import os
import sys
import ast
import hashlib
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum


# =============================================================================
# MODIFICATION TYPES
# =============================================================================

class ModificationType(Enum):
    """Types of code modifications."""
    ADD_FUNCTION = "add_function"
    MODIFY_FUNCTION = "modify_function"
    ADD_CLASS = "add_class"
    MODIFY_CLASS = "modify_class"
    ADD_IMPORT = "add_import"
    ADD_DOCSTRING = "add_docstring"
    REFACTOR = "refactor"
    OPTIMIZE = "optimize"
    FIX_BUG = "fix_bug"


class ModificationStatus(Enum):
    """Status of a modification attempt."""
    PROPOSED = "proposed"
    VALIDATED = "validated"
    TESTED = "tested"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


# =============================================================================
# CODE CHANGE REPRESENTATION
# =============================================================================

@dataclass
class CodeChange:
    """Represents a proposed change to source code."""
    id: str
    file_path: str
    modification_type: ModificationType
    description: str
    
    # The actual change
    original_content: str = ""
    new_content: str = ""
    
    # For targeted changes
    target_name: str = ""  # Function/class name
    diff: str = ""
    
    # Validation
    syntax_valid: bool = False
    tests_pass: bool = False
    improvement_score: float = 0.0
    
    # Status tracking
    status: ModificationStatus = ModificationStatus.PROPOSED
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackPoint:
    """A point to which we can roll back."""
    id: str
    timestamp: datetime
    file_path: str
    content: str
    reason: str


# =============================================================================
# SAFETY CONSTRAINTS
# =============================================================================

class SafetyConstraints:
    """
    Defines safety constraints for self-modification.
    Prevents dangerous or harmful changes.
    """
    
    # Maximum lines changed in a single modification
    MAX_LINES_CHANGED = 100
    
    # Maximum number of modifications per hour
    MAX_MODIFICATIONS_PER_HOUR = 10
    
    # Protected files that cannot be modified
    PROTECTED_FILES = {
        "self_modifier.py",  # Can't modify itself
        "__init__.py",       # Package structure
    }
    
    # Protected patterns in code
    PROTECTED_PATTERNS = [
        "os.system",         # Shell commands
        "subprocess.Popen",  # Process spawning
        "eval(",             # Dynamic evaluation
        "exec(",             # Dynamic execution
        "__import__",        # Dynamic imports
        "open(",             # File operations (restricted)
        "shutil.rmtree",     # Recursive deletion
    ]
    
    # Required patterns (must preserve)
    REQUIRED_PATTERNS = [
        "class",             # Must have classes
        "def",               # Must have functions
    ]
    
    def __init__(self):
        self.modification_count = 0
        self.last_hour_start = datetime.now()
        self.violations: List[str] = []
    
    def check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        if (now - self.last_hour_start).seconds > 3600:
            self.modification_count = 0
            self.last_hour_start = now
        
        return self.modification_count < self.MAX_MODIFICATIONS_PER_HOUR
    
    def check_protected_file(self, file_path: str) -> bool:
        """Check if file is protected."""
        filename = os.path.basename(file_path)
        return filename not in self.PROTECTED_FILES
    
    def check_dangerous_patterns(self, content: str) -> Tuple[bool, List[str]]:
        """Check for dangerous patterns in code."""
        found = []
        for pattern in self.PROTECTED_PATTERNS:
            if pattern in content:
                found.append(pattern)
        return len(found) == 0, found
    
    def check_size_limit(self, original: str, new: str) -> bool:
        """Check if change size is within limits."""
        original_lines = len(original.splitlines())
        new_lines = len(new.splitlines())
        return abs(new_lines - original_lines) <= self.MAX_LINES_CHANGED
    
    def validate_change(self, change: CodeChange) -> Tuple[bool, List[str]]:
        """
        Validate a proposed change against all safety constraints.
        Returns (is_safe, list_of_violations)
        """
        violations = []
        
        # Rate limit
        if not self.check_rate_limit():
            violations.append("Rate limit exceeded")
        
        # Protected file
        if not self.check_protected_file(change.file_path):
            violations.append(f"Protected file: {change.file_path}")
        
        # Dangerous patterns
        safe, patterns = self.check_dangerous_patterns(change.new_content)
        if not safe:
            violations.append(f"Dangerous patterns found: {patterns}")
        
        # Size limit
        if not self.check_size_limit(change.original_content, change.new_content):
            violations.append("Change exceeds size limit")
        
        self.violations = violations
        return len(violations) == 0, violations


# =============================================================================
# SANDBOX EXECUTOR
# =============================================================================

class SandboxExecutor:
    """
    Executes code changes in a sandboxed environment.
    Tests changes before applying to production code.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.sandbox_dir: Optional[Path] = None
    
    def create_sandbox(self) -> Path:
        """Create a sandboxed copy of the project."""
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix="aion_sandbox_"))
        
        # Copy relevant Python files
        for py_file in self.project_root.rglob("*.py"):
            if ".git" in str(py_file) or "__pycache__" in str(py_file):
                continue
            
            rel_path = py_file.relative_to(self.project_root)
            target = self.sandbox_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_file, target)
        
        return self.sandbox_dir
    
    def apply_change_in_sandbox(self, change: CodeChange) -> bool:
        """Apply a change in the sandbox."""
        if not self.sandbox_dir:
            self.create_sandbox()
        
        # Calculate relative path
        original_path = Path(change.file_path)
        try:
            rel_path = original_path.relative_to(self.project_root)
        except ValueError:
            rel_path = original_path.name
        
        target_file = self.sandbox_dir / rel_path
        
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(change.new_content, encoding='utf-8')
            return True
        except Exception as e:
            return False
    
    def run_syntax_check(self, change: CodeChange) -> Tuple[bool, str]:
        """Check if the modified code has valid syntax."""
        try:
            ast.parse(change.new_content)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    
    def run_tests(self, test_command: str = None) -> Tuple[bool, str]:
        """Run tests in the sandbox."""
        if not self.sandbox_dir:
            return False, "No sandbox created"
        
        test_command = test_command or f"{sys.executable} -m pytest -x --tb=short"
        
        try:
            result = subprocess.run(
                test_command.split(),
                cwd=str(self.sandbox_dir),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return True, "All tests passed"
            else:
                return False, f"Tests failed: {result.stdout[:500]}"
                
        except subprocess.TimeoutExpired:
            return False, "Tests timed out"
        except Exception as e:
            return False, f"Test error: {e}"
    
    def cleanup(self):
        """Clean up the sandbox."""
        if self.sandbox_dir and self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            self.sandbox_dir = None


# =============================================================================
# ROLLBACK MANAGER
# =============================================================================

class RollbackManager:
    """
    Manages rollback points for safe modification.
    Enables reverting to previous versions.
    """
    
    def __init__(self, storage_dir: str = None):
        self.storage_dir = Path(storage_dir or tempfile.gettempdir()) / "aion_rollbacks"
        self.storage_dir.mkdir(exist_ok=True)
        
        self.rollback_points: Dict[str, List[RollbackPoint]] = {}
        self.rollback_counter = 0
    
    def create_rollback_point(self, file_path: str, reason: str = "pre-modification") -> RollbackPoint:
        """Create a rollback point before modification."""
        self.rollback_counter += 1
        
        # Read current content
        content = ""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        point = RollbackPoint(
            id=f"R{self.rollback_counter:04d}",
            timestamp=datetime.now(),
            file_path=file_path,
            content=content,
            reason=reason
        )
        
        # Store
        if file_path not in self.rollback_points:
            self.rollback_points[file_path] = []
        self.rollback_points[file_path].append(point)
        
        # Also save to disk
        rollback_file = self.storage_dir / f"{point.id}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}.bak"
        rollback_file.write_text(content, encoding='utf-8')
        
        return point
    
    def rollback(self, file_path: str, point_id: str = None) -> bool:
        """
        Roll back a file to a previous state.
        If point_id is None, rolls back to most recent point.
        """
        if file_path not in self.rollback_points or not self.rollback_points[file_path]:
            return False
        
        if point_id:
            point = next((p for p in self.rollback_points[file_path] if p.id == point_id), None)
        else:
            point = self.rollback_points[file_path][-1]
        
        if not point:
            return False
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(point.content)
            return True
        except Exception:
            return False
    
    def get_history(self, file_path: str) -> List[RollbackPoint]:
        """Get rollback history for a file."""
        return self.rollback_points.get(file_path, [])


# =============================================================================
# IMPROVEMENT MEASURER
# =============================================================================

class ImprovementMeasurer:
    """
    Measures the improvement from a code change.
    Uses various metrics to quantify enhancement.
    """
    
    def __init__(self):
        self.metrics_history: List[Dict[str, float]] = []
    
    def measure_code_quality(self, content: str) -> Dict[str, float]:
        """Measure basic code quality metrics."""
        lines = content.splitlines()
        
        metrics = {
            "lines": len(lines),
            "functions": content.count("def "),
            "classes": content.count("class "),
            "docstrings": content.count('"""'),
            "comments": sum(1 for line in lines if line.strip().startswith("#")),
            "blank_lines": sum(1 for line in lines if not line.strip()),
        }
        
        # Calculate ratios
        if metrics["lines"] > 0:
            metrics["docstring_ratio"] = metrics["docstrings"] / max(1, metrics["functions"] + metrics["classes"])
            metrics["comment_ratio"] = metrics["comments"] / metrics["lines"]
        else:
            metrics["docstring_ratio"] = 0
            metrics["comment_ratio"] = 0
        
        return metrics
    
    def compare(self, original: str, modified: str) -> Dict[str, float]:
        """Compare original and modified code."""
        original_metrics = self.measure_code_quality(original)
        modified_metrics = self.measure_code_quality(modified)
        
        comparison = {}
        
        for key in original_metrics:
            if key in modified_metrics:
                original_val = original_metrics[key]
                modified_val = modified_metrics[key]
                
                if original_val > 0:
                    comparison[f"{key}_change"] = (modified_val - original_val) / original_val
                else:
                    comparison[f"{key}_change"] = 0.0 if modified_val == 0 else 1.0
        
        return comparison
    
    def calculate_improvement_score(self, original: str, modified: str, 
                                    tests_before: bool, tests_after: bool) -> float:
        """
        Calculate overall improvement score.
        Range: -1 (regression) to +1 (significant improvement)
        """
        score = 0.0
        
        # Test improvement (most important)
        if not tests_before and tests_after:
            score += 0.5  # Fixed tests
        elif tests_before and not tests_after:
            score -= 0.8  # Broke tests (very bad)
        
        # Quality metrics comparison
        comparison = self.compare(original, modified)
        
        # Positive: more docstrings, more comments
        score += comparison.get("docstring_ratio_change", 0) * 0.1
        score += comparison.get("comment_ratio_change", 0) * 0.1
        
        # Lines added should be justified
        lines_change = comparison.get("lines_change", 0)
        if lines_change > 0.5:
            score -= 0.1  # Too much bloat
        elif lines_change < -0.5:
            score -= 0.1  # Too much removal
        
        return max(-1.0, min(1.0, score))


# =============================================================================
# SELF-MODIFIER (MAIN CLASS)
# =============================================================================

class SelfModifier:
    """
    The self-modification engine for AION.
    Enables safe, controlled modification of source code.
    
    Key safety principles:
    1. All changes are sandboxed and tested first
    2. Rollback points are created before any modification
    3. Safety constraints prevent dangerous operations
    4. Improvements must be measurable
    5. Rate limiting prevents runaway modifications
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        
        self.safety = SafetyConstraints()
        self.sandbox = SandboxExecutor(str(self.project_root))
        self.rollback = RollbackManager()
        self.improvement = ImprovementMeasurer()
        
        # Change history
        self.changes: List[CodeChange] = []
        self.change_counter = 0
        
        # Configuration
        self.require_test_improvement = True
        self.require_human_approval = True  # For structural changes
        self.auto_approve_threshold = 0.5   # Auto-approve if improvement > threshold
    
    def _generate_change_id(self) -> str:
        """Generate unique change ID."""
        self.change_counter += 1
        return f"CHG{self.change_counter:04d}"
    
    def propose_change(self, file_path: str, modification_type: ModificationType,
                       description: str, new_content: str,
                       target_name: str = "") -> CodeChange:
        """
        Propose a code change.
        Does not apply the change, just creates a proposal.
        """
        file_path = str(Path(file_path).resolve())
        
        # Read original content
        original_content = ""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        
        change = CodeChange(
            id=self._generate_change_id(),
            file_path=file_path,
            modification_type=modification_type,
            description=description,
            original_content=original_content,
            new_content=new_content,
            target_name=target_name
        )
        
        self.changes.append(change)
        return change
    
    def validate_change(self, change: CodeChange) -> Tuple[bool, List[str]]:
        """Validate a change against safety constraints."""
        # Run safety checks
        is_safe, violations = self.safety.validate_change(change)
        
        if not is_safe:
            change.status = ModificationStatus.REJECTED
            return False, violations
        
        # Check syntax
        syntax_ok, syntax_msg = self.sandbox.run_syntax_check(change)
        change.syntax_valid = syntax_ok
        
        if not syntax_ok:
            change.status = ModificationStatus.REJECTED
            return False, [syntax_msg]
        
        change.status = ModificationStatus.VALIDATED
        return True, []
    
    def test_change(self, change: CodeChange) -> Tuple[bool, str]:
        """Test a change in the sandbox."""
        if change.status not in [ModificationStatus.VALIDATED, ModificationStatus.PROPOSED]:
            return False, "Change not validated"
        
        # Create sandbox and apply change
        self.sandbox.create_sandbox()
        self.sandbox.apply_change_in_sandbox(change)
        
        # Run tests
        tests_ok, test_msg = self.sandbox.run_tests()
        change.tests_pass = tests_ok
        
        if tests_ok:
            change.status = ModificationStatus.TESTED
        
        # Measure improvement
        change.improvement_score = self.improvement.calculate_improvement_score(
            change.original_content,
            change.new_content,
            True,  # Assume original passed
            tests_ok
        )
        
        self.sandbox.cleanup()
        
        return tests_ok, test_msg
    
    def apply_change(self, change: CodeChange, force: bool = False) -> Tuple[bool, str]:
        """
        Apply a change to the actual codebase.
        
        Args:
            change: The change to apply
            force: If True, skip validation/testing (dangerous!)
        """
        if not force:
            # Ensure change has been validated and tested
            if change.status not in [ModificationStatus.TESTED, ModificationStatus.VALIDATED]:
                return False, f"Change not ready (status: {change.status.value})"
            
            # Check improvement threshold for auto-approval
            if change.improvement_score < self.auto_approve_threshold and self.require_human_approval:
                return False, f"Requires human approval (improvement: {change.improvement_score:.2f})"
        
        # Create rollback point
        rollback_point = self.rollback.create_rollback_point(
            change.file_path,
            f"Before change {change.id}: {change.description}"
        )
        
        try:
            # Apply the change
            with open(change.file_path, 'w', encoding='utf-8') as f:
                f.write(change.new_content)
            
            change.status = ModificationStatus.APPLIED
            change.applied_at = datetime.now()
            self.safety.modification_count += 1
            
            return True, f"Change applied successfully (rollback: {rollback_point.id})"
            
        except Exception as e:
            # Rollback on failure
            self.rollback.rollback(change.file_path, rollback_point.id)
            change.status = ModificationStatus.ROLLED_BACK
            return False, f"Failed to apply change: {e}"
    
    def rollback_change(self, change: CodeChange) -> Tuple[bool, str]:
        """Roll back a previously applied change."""
        if change.status != ModificationStatus.APPLIED:
            return False, "Change was not applied"
        
        success = self.rollback.rollback(change.file_path)
        
        if success:
            change.status = ModificationStatus.ROLLED_BACK
            return True, "Change rolled back successfully"
        else:
            return False, "Failed to rollback"
    
    def get_pending_changes(self) -> List[CodeChange]:
        """Get all pending changes awaiting application."""
        return [c for c in self.changes if c.status in [
            ModificationStatus.PROPOSED,
            ModificationStatus.VALIDATED,
            ModificationStatus.TESTED
        ]]
    
    def get_status_report(self) -> str:
        """Generate a status report of self-modification activity."""
        lines = [
            "=" * 60,
            "SELF-MODIFICATION ENGINE STATUS",
            "=" * 60,
            "",
            f"Project Root: {self.project_root}",
            f"Total Changes Proposed: {len(self.changes)}",
            f"Modifications This Hour: {self.safety.modification_count}/{self.safety.MAX_MODIFICATIONS_PER_HOUR}",
            "",
            "Changes by Status:"
        ]
        
        for status in ModificationStatus:
            count = sum(1 for c in self.changes if c.status == status)
            if count > 0:
                lines.append(f"  {status.value}: {count}")
        
        # Recent changes
        recent = sorted(self.changes, key=lambda c: c.created_at, reverse=True)[:5]
        if recent:
            lines.append("")
            lines.append("Recent Changes:")
            for change in recent:
                lines.append(f"  [{change.status.value}] {change.description[:50]}")
                lines.append(f"      Type: {change.modification_type.value}, Improvement: {change.improvement_score:+.2f}")
        
        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the self-modification engine."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          ⚙️ AION SELF-MODIFICATION ENGINE ⚙️                              ║
║                                                                           ║
║     Safe • Sandboxed • Reversible • Improvement-Driven                    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    modifier = SelfModifier(".")
    
    # Demo 1: Safety Constraints
    print("🔒 Safety Constraints:")
    print("-" * 50)
    print(f"  Max lines changed: {modifier.safety.MAX_LINES_CHANGED}")
    print(f"  Max modifications/hour: {modifier.safety.MAX_MODIFICATIONS_PER_HOUR}")
    print(f"  Protected files: {modifier.safety.PROTECTED_FILES}")
    
    # Demo 2: Propose a Safe Change
    print("\n📝 Proposing a Change:")
    print("-" * 50)
    
    test_content = '''
def example_function():
    """This is an example function with docstring."""
    return "Hello from AION"
'''
    
    change = modifier.propose_change(
        file_path="example_module.py",
        modification_type=ModificationType.ADD_FUNCTION,
        description="Add example function with docstring",
        new_content=test_content
    )
    
    print(f"  Change ID: {change.id}")
    print(f"  Type: {change.modification_type.value}")
    print(f"  Status: {change.status.value}")
    
    # Demo 3: Validate the Change
    print("\n✅ Validating Change:")
    print("-" * 50)
    
    is_valid, violations = modifier.validate_change(change)
    print(f"  Valid: {is_valid}")
    print(f"  Status: {change.status.value}")
    if violations:
        print(f"  Violations: {violations}")
    
    # Demo 4: Code Quality Metrics
    print("\n📊 Code Quality Metrics:")
    print("-" * 50)
    
    metrics = modifier.improvement.measure_code_quality(test_content)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Demo 5: Status Report
    print("\n📋 Status Report:")
    print(modifier.get_status_report())


if __name__ == "__main__":
    demo()
