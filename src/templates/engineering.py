"""
AION Industry Templates - Engineering
======================================

Engineering-specific agent templates:
- Code Review: Automated code quality analysis
- Architecture Analysis: System design evaluation
- Technical Documentation: Auto-generated docs
- DevOps Support: CI/CD and infrastructure

Auto-generated for Phase 5: Scale
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import re


class Severity(Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(Enum):
    """Code issue categories."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    STYLE = "style"
    DOCUMENTATION = "documentation"


@dataclass
class CodeIssue:
    """A code review issue."""
    id: str = ""
    category: IssueCategory = IssueCategory.MAINTAINABILITY
    severity: Severity = Severity.MEDIUM
    message: str = ""
    file: str = ""
    line: int = 0
    suggestion: str = ""
    rule_id: str = ""


@dataclass
class ReviewResult:
    """Code review result."""
    file: str = ""
    issues: List[CodeIssue] = field(default_factory=list)
    score: float = 100.0
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class CodeReviewer:
    """Automated code review system."""
    
    # Common code patterns to check
    RULES = {
        'SEC001': {
            'pattern': r'password\s*=\s*["\'][^"\']+["\']',
            'message': 'Hardcoded password detected',
            'category': IssueCategory.SECURITY,
            'severity': Severity.CRITICAL
        },
        'SEC002': {
            'pattern': r'eval\s*\(',
            'message': 'Use of eval() is dangerous',
            'category': IssueCategory.SECURITY,
            'severity': Severity.CRITICAL
        },
        'PERF001': {
            'pattern': r'for.*in.*range\(len\(',
            'message': 'Use enumerate() instead of range(len())',
            'category': IssueCategory.PERFORMANCE,
            'severity': Severity.LOW
        },
        'MAINT001': {
            'pattern': r'#\s*TODO',
            'message': 'TODO comment found',
            'category': IssueCategory.MAINTAINABILITY,
            'severity': Severity.INFO
        },
        'MAINT002': {
            'pattern': r'except\s*:',
            'message': 'Bare except clause catches all exceptions',
            'category': IssueCategory.RELIABILITY,
            'severity': Severity.MEDIUM
        },
        'STYLE001': {
            'pattern': r'^\s{0,3}[a-z]',
            'message': 'Line may have incorrect indentation',
            'category': IssueCategory.STYLE,
            'severity': Severity.LOW
        },
        'DOC001': {
            'pattern': r'def\s+\w+\([^)]*\):\s*\n\s*[^"\']',
            'message': 'Function missing docstring',
            'category': IssueCategory.DOCUMENTATION,
            'severity': Severity.LOW
        }
    }
    
    def __init__(self):
        self.custom_rules: Dict[str, Dict] = {}
        self.review_history: List[ReviewResult] = []
    
    def add_custom_rule(self, rule_id: str, pattern: str, message: str,
                        category: IssueCategory, severity: Severity):
        """Add a custom review rule."""
        self.custom_rules[rule_id] = {
            'pattern': pattern,
            'message': message,
            'category': category,
            'severity': severity
        }
    
    def review_file(self, file_path: str, content: str) -> ReviewResult:
        """Review a single file."""
        result = ReviewResult(file=file_path)
        lines = content.split('\n')
        
        all_rules = {**self.RULES, **self.custom_rules}
        issue_count = 0
        
        for rule_id, rule in all_rules.items():
            pattern = rule['pattern']
            
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issue = CodeIssue(
                        id=f"{file_path}:{line_num}:{rule_id}",
                        category=rule['category'],
                        severity=rule['severity'],
                        message=rule['message'],
                        file=file_path,
                        line=line_num,
                        rule_id=rule_id
                    )
                    result.issues.append(issue)
                    issue_count += 1
        
        # Calculate score
        severity_weights = {
            Severity.CRITICAL: 25,
            Severity.HIGH: 10,
            Severity.MEDIUM: 5,
            Severity.LOW: 2,
            Severity.INFO: 0
        }
        
        penalty = sum(severity_weights[i.severity] for i in result.issues)
        result.score = max(0, 100 - penalty)
        
        # Calculate metrics
        result.metrics = {
            'lines': len(lines),
            'issues': issue_count,
            'issues_per_100_lines': (issue_count / max(len(lines), 1)) * 100
        }
        
        self.review_history.append(result)
        return result
    
    def review_project(self, files: Dict[str, str]) -> Dict[str, Any]:
        """Review an entire project."""
        results = []
        total_lines = 0
        total_issues = 0
        
        for file_path, content in files.items():
            result = self.review_file(file_path, content)
            results.append(result)
            total_lines += result.metrics.get('lines', 0)
            total_issues += len(result.issues)
        
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        
        return {
            'files_reviewed': len(results),
            'total_lines': total_lines,
            'total_issues': total_issues,
            'average_score': avg_score,
            'results': results
        }


@dataclass
class ArchitectureComponent:
    """An architecture component."""
    name: str = ""
    type: str = ""  # service, database, gateway, etc.
    dependencies: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class ArchitectureAnalyzer:
    """System architecture analysis."""
    
    def __init__(self):
        self.components: Dict[str, ArchitectureComponent] = {}
    
    def add_component(self, component: ArchitectureComponent):
        """Add an architecture component."""
        self.components[component.name] = component
    
    def analyze_coupling(self) -> Dict[str, Any]:
        """Analyze component coupling."""
        coupling_metrics = {}
        
        for name, comp in self.components.items():
            # Afferent coupling (incoming dependencies)
            afferent = sum(1 for c in self.components.values() if name in c.dependencies)
            # Efferent coupling (outgoing dependencies)
            efferent = len(comp.dependencies)
            
            coupling_metrics[name] = {
                'afferent': afferent,
                'efferent': efferent,
                'instability': efferent / (afferent + efferent) if (afferent + efferent) > 0 else 0
            }
        
        return coupling_metrics
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            comp = self.components.get(node)
            if comp:
                for dep in comp.dependencies:
                    if dep not in visited:
                        if dfs(dep, path + [node]):
                            return True
                    elif dep in rec_stack:
                        cycle_start = path.index(dep) if dep in path else 0
                        cycles.append(path[cycle_start:] + [node, dep])
                        return True
            
            rec_stack.remove(node)
            return False
        
        for name in self.components:
            if name not in visited:
                dfs(name, [])
        
        return cycles
    
    def suggest_improvements(self) -> List[Dict[str, Any]]:
        """Suggest architecture improvements."""
        suggestions = []
        coupling = self.analyze_coupling()
        
        for name, metrics in coupling.items():
            if metrics['efferent'] > 5:
                suggestions.append({
                    'component': name,
                    'issue': 'High efferent coupling',
                    'suggestion': 'Consider splitting into smaller components'
                })
            
            if metrics['instability'] > 0.8:
                suggestions.append({
                    'component': name,
                    'issue': 'High instability',
                    'suggestion': 'Reduce outgoing dependencies or increase incoming'
                })
        
        cycles = self.detect_cycles()
        if cycles:
            suggestions.append({
                'component': 'System',
                'issue': f'{len(cycles)} circular dependencies detected',
                'suggestion': 'Introduce abstraction layers to break cycles'
            })
        
        return suggestions


class TechnicalDocumentor:
    """Technical documentation generator."""
    
    def __init__(self):
        self.docs: Dict[str, str] = {}
    
    def generate_api_doc(self, functions: List[Dict[str, Any]]) -> str:
        """Generate API documentation."""
        doc = "# API Documentation\n\n"
        
        for func in functions:
            doc += f"## `{func.get('name', 'Unknown')}`\n\n"
            doc += f"{func.get('description', 'No description')}\n\n"
            
            if func.get('params'):
                doc += "### Parameters\n\n"
                for param in func['params']:
                    doc += f"- **{param['name']}** ({param.get('type', 'any')}): {param.get('description', '')}\n"
                doc += "\n"
            
            if func.get('returns'):
                doc += f"### Returns\n\n{func['returns']}\n\n"
            
            if func.get('example'):
                doc += f"### Example\n\n```python\n{func['example']}\n```\n\n"
        
        return doc
    
    def generate_readme(self, project: Dict[str, Any]) -> str:
        """Generate project README."""
        readme = f"# {project.get('name', 'Project')}\n\n"
        readme += f"{project.get('description', '')}\n\n"
        
        if project.get('installation'):
            readme += "## Installation\n\n"
            readme += f"```bash\n{project['installation']}\n```\n\n"
        
        if project.get('usage'):
            readme += "## Usage\n\n"
            readme += f"```python\n{project['usage']}\n```\n\n"
        
        if project.get('features'):
            readme += "## Features\n\n"
            for feature in project['features']:
                readme += f"- {feature}\n"
            readme += "\n"
        
        return readme


class EngineeringAgent:
    """Engineering-specialized AION agent."""
    
    def __init__(self, agent_id: str = "engineering-agent"):
        self.agent_id = agent_id
        self.code_reviewer = CodeReviewer()
        self.architecture_analyzer = ArchitectureAnalyzer()
        self.documentor = TechnicalDocumentor()
    
    async def review_code(self, files: Dict[str, str]) -> Dict[str, Any]:
        """Review code files."""
        return self.code_reviewer.review_project(files)
    
    async def analyze_architecture(self) -> Dict[str, Any]:
        """Analyze system architecture."""
        coupling = self.architecture_analyzer.analyze_coupling()
        cycles = self.architecture_analyzer.detect_cycles()
        suggestions = self.architecture_analyzer.suggest_improvements()
        
        return {
            'components': len(self.architecture_analyzer.components),
            'coupling_metrics': coupling,
            'cycles_detected': len(cycles),
            'suggestions': suggestions
        }
    
    async def generate_documentation(self, 
                                     project: Dict[str, Any],
                                     functions: List[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate project documentation."""
        docs = {}
        
        docs['README.md'] = self.documentor.generate_readme(project)
        
        if functions:
            docs['API.md'] = self.documentor.generate_api_doc(functions)
        
        return docs
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            'agent_id': self.agent_id,
            'files_reviewed': len(self.code_reviewer.review_history),
            'components_tracked': len(self.architecture_analyzer.components)
        }


async def demo_engineering():
    """Demonstrate engineering template."""
    print("🔧 Engineering Template Demo")
    print("=" * 50)
    
    agent = EngineeringAgent()
    
    # Sample code to review
    sample_code = '''
import os

def process_data(data):
    result = eval(data)  # Security issue
    password = "secret123"  # Hardcoded password
    
    for i in range(len(data)):  # Performance issue
        print(data[i])
    
    try:
        risky_operation()
    except:  # Bare except
        pass
    
    # TODO: Improve this later
    return result
'''
    
    # Review code
    print("\n📝 Code Review:")
    review = await agent.review_code({'main.py': sample_code})
    
    print(f"  Files Reviewed: {review['files_reviewed']}")
    print(f"  Total Issues: {review['total_issues']}")
    print(f"  Average Score: {review['average_score']:.1f}/100")
    
    print("\n🚨 Issues Found:")
    for result in review['results']:
        for issue in result.issues[:5]:
            print(f"  [{issue.severity.value.upper()}] {issue.message} (line {issue.line})")
    
    # Architecture analysis
    agent.architecture_analyzer.add_component(ArchitectureComponent(
        name="API Gateway", type="gateway", dependencies=["Auth Service", "User Service"]
    ))
    agent.architecture_analyzer.add_component(ArchitectureComponent(
        name="User Service", type="service", dependencies=["Database", "Cache"]
    ))
    agent.architecture_analyzer.add_component(ArchitectureComponent(
        name="Auth Service", type="service", dependencies=["User Service", "Database"]
    ))
    agent.architecture_analyzer.add_component(ArchitectureComponent(
        name="Database", type="database", dependencies=[]
    ))
    agent.architecture_analyzer.add_component(ArchitectureComponent(
        name="Cache", type="cache", dependencies=[]
    ))
    
    print("\n🏗️ Architecture Analysis:")
    arch_analysis = await agent.analyze_architecture()
    print(f"  Components: {arch_analysis['components']}")
    print(f"  Cycles Detected: {arch_analysis['cycles_detected']}")
    
    print("\n💡 Suggestions:")
    for sug in arch_analysis['suggestions'][:3]:
        print(f"  - {sug['component']}: {sug['suggestion']}")
    
    # Generate docs
    print("\n📚 Documentation Generated")
    docs = await agent.generate_documentation({
        'name': 'AION',
        'description': 'Advanced AI Agent Framework',
        'features': ['Multi-agent systems', 'Self-evolution', 'Cloud-native']
    })
    
    print(f"\n📊 Status: {agent.get_status()}")
    print("\n✅ Engineering template demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_engineering())
