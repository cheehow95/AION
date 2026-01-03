"""
AION Agent Marketplace - Security Scanner
==========================================

Security scanning for packages:
- Vulnerability Detection: Known CVE scanning
- Code Analysis: Static security analysis
- Policy Enforcement: Organization security policies
- Risk Assessment: Package risk scoring

Auto-generated for Phase 5: Scale
"""

import asyncio
import uuid
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum


class Severity(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityType(Enum):
    """Types of vulnerabilities."""
    INJECTION = "injection"
    BROKEN_AUTH = "broken_auth"
    SENSITIVE_DATA = "sensitive_data"
    XXE = "xxe"
    BROKEN_ACCESS = "broken_access"
    MISCONFIG = "misconfig"
    XSS = "xss"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    OUTDATED_COMPONENT = "outdated_component"
    LOGGING_FAILURE = "logging_failure"


@dataclass
class Vulnerability:
    """A detected vulnerability."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: VulnerabilityType = VulnerabilityType.MISCONFIG
    severity: Severity = Severity.MEDIUM
    title: str = ""
    description: str = ""
    location: str = ""
    line_number: int = 0
    cve_id: Optional[str] = None
    remediation: str = ""
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class VulnerabilityReport:
    """Security scan report."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    package_name: str = ""
    package_version: str = ""
    scan_started: datetime = field(default_factory=datetime.now)
    scan_completed: Optional[datetime] = None
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    risk_score: float = 0.0
    passed: bool = True
    
    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == Severity.CRITICAL)
    
    @property
    def high_count(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == Severity.HIGH)
    
    def summary(self) -> Dict[str, Any]:
        by_severity = {}
        for v in self.vulnerabilities:
            by_severity[v.severity.value] = by_severity.get(v.severity.value, 0) + 1
        
        return {
            'package': f"{self.package_name}@{self.package_version}",
            'total_vulnerabilities': len(self.vulnerabilities),
            'by_severity': by_severity,
            'risk_score': self.risk_score,
            'passed': self.passed
        }


class SecurityScanner:
    """Scans packages for security vulnerabilities."""
    
    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        (r'eval\s*\(', VulnerabilityType.INJECTION, Severity.CRITICAL, "Use of eval()"),
        (r'exec\s*\(', VulnerabilityType.INJECTION, Severity.CRITICAL, "Use of exec()"),
        (r'__import__\s*\(', VulnerabilityType.INJECTION, Severity.HIGH, "Dynamic import"),
        (r'subprocess\.', VulnerabilityType.INJECTION, Severity.HIGH, "Subprocess execution"),
        (r'os\.system\s*\(', VulnerabilityType.INJECTION, Severity.CRITICAL, "OS command execution"),
        (r'pickle\.load', VulnerabilityType.INSECURE_DESERIALIZATION, Severity.HIGH, "Unsafe pickle"),
        (r'yaml\.load\s*\([^)]*\)', VulnerabilityType.INSECURE_DESERIALIZATION, Severity.HIGH, "Unsafe YAML load"),
        (r'password\s*=\s*["\'][^"\']+["\']', VulnerabilityType.SENSITIVE_DATA, Severity.HIGH, "Hardcoded password"),
        (r'api_key\s*=\s*["\'][^"\']+["\']', VulnerabilityType.SENSITIVE_DATA, Severity.HIGH, "Hardcoded API key"),
        (r'secret\s*=\s*["\'][^"\']+["\']', VulnerabilityType.SENSITIVE_DATA, Severity.MEDIUM, "Hardcoded secret"),
        (r'http://', VulnerabilityType.MISCONFIG, Severity.LOW, "Insecure HTTP URL"),
    ]
    
    def __init__(self):
        self.known_cves: Dict[str, Dict[str, Any]] = {}
        self.scan_history: List[VulnerabilityReport] = []
    
    def add_known_cve(self, cve_id: str, affected_package: str,
                      affected_versions: List[str], severity: Severity):
        """Add a known CVE to the database."""
        self.known_cves[cve_id] = {
            'package': affected_package,
            'versions': affected_versions,
            'severity': severity
        }
    
    async def scan_code(self, content: str, filename: str = "") -> List[Vulnerability]:
        """Scan code content for vulnerabilities."""
        vulnerabilities = []
        lines = content.split('\n')
        
        for pattern, vuln_type, severity, title in self.DANGEROUS_PATTERNS:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(Vulnerability(
                        type=vuln_type,
                        severity=severity,
                        title=title,
                        description=f"Detected {title} pattern in code",
                        location=filename,
                        line_number=line_num,
                        remediation=f"Review and secure the use of {title}"
                    ))
        
        return vulnerabilities
    
    async def scan_dependencies(self, dependencies: Dict[str, str]) -> List[Vulnerability]:
        """Scan dependencies for known vulnerabilities."""
        vulnerabilities = []
        
        for dep_name, version in dependencies.items():
            for cve_id, cve_data in self.known_cves.items():
                if cve_data['package'] == dep_name:
                    if version in cve_data['versions'] or '*' in cve_data['versions']:
                        vulnerabilities.append(Vulnerability(
                            type=VulnerabilityType.OUTDATED_COMPONENT,
                            severity=cve_data['severity'],
                            title=f"Known vulnerability {cve_id}",
                            description=f"Dependency {dep_name}@{version} has known vulnerability",
                            cve_id=cve_id,
                            remediation=f"Upgrade {dep_name} to a patched version"
                        ))
        
        return vulnerabilities
    
    async def scan_package(self, package_name: str, package_version: str,
                           files: Dict[str, bytes],
                           dependencies: Dict[str, str] = None) -> VulnerabilityReport:
        """Scan a complete package."""
        report = VulnerabilityReport(
            package_name=package_name,
            package_version=package_version
        )
        
        # Scan all code files
        for filename, content in files.items():
            if filename.endswith('.py'):
                try:
                    text = content.decode('utf-8')
                    vulns = await self.scan_code(text, filename)
                    report.vulnerabilities.extend(vulns)
                except UnicodeDecodeError:
                    pass
        
        # Scan dependencies
        if dependencies:
            dep_vulns = await self.scan_dependencies(dependencies)
            report.vulnerabilities.extend(dep_vulns)
        
        # Calculate risk score
        report.risk_score = self._calculate_risk_score(report.vulnerabilities)
        report.passed = report.critical_count == 0 and report.high_count <= 2
        report.scan_completed = datetime.now()
        
        self.scan_history.append(report)
        return report
    
    def _calculate_risk_score(self, vulnerabilities: List[Vulnerability]) -> float:
        """Calculate overall risk score (0-100)."""
        if not vulnerabilities:
            return 0.0
        
        weights = {
            Severity.CRITICAL: 25,
            Severity.HIGH: 10,
            Severity.MEDIUM: 5,
            Severity.LOW: 2,
            Severity.INFO: 1
        }
        
        score = sum(weights.get(v.severity, 1) for v in vulnerabilities)
        return min(100.0, score)


@dataclass
class SecurityPolicy:
    """Organization security policy."""
    name: str = ""
    max_critical: int = 0
    max_high: int = 0
    max_risk_score: float = 50.0
    blocked_patterns: List[str] = field(default_factory=list)
    required_signatures: bool = True
    allowed_licenses: List[str] = field(default_factory=lambda: ["MIT", "Apache-2.0", "BSD-3-Clause"])


class PolicyChecker:
    """Checks packages against security policies."""
    
    def __init__(self):
        self.policies: Dict[str, SecurityPolicy] = {}
    
    def add_policy(self, policy: SecurityPolicy):
        """Add a security policy."""
        self.policies[policy.name] = policy
    
    def check(self, report: VulnerabilityReport, 
              policy_name: str = "default") -> Dict[str, Any]:
        """Check report against policy."""
        policy = self.policies.get(policy_name)
        if not policy:
            return {'passed': True, 'violations': []}
        
        violations = []
        
        if report.critical_count > policy.max_critical:
            violations.append(f"Too many critical vulnerabilities: {report.critical_count} > {policy.max_critical}")
        
        if report.high_count > policy.max_high:
            violations.append(f"Too many high vulnerabilities: {report.high_count} > {policy.max_high}")
        
        if report.risk_score > policy.max_risk_score:
            violations.append(f"Risk score too high: {report.risk_score} > {policy.max_risk_score}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'policy': policy_name
        }


class RiskAssessment:
    """Comprehensive risk assessment for packages."""
    
    def __init__(self, scanner: SecurityScanner, policy_checker: PolicyChecker):
        self.scanner = scanner
        self.policy_checker = policy_checker
    
    async def assess(self, package_name: str, package_version: str,
                     files: Dict[str, bytes],
                     dependencies: Dict[str, str] = None,
                     policy_name: str = "default") -> Dict[str, Any]:
        """Perform full risk assessment."""
        # Run security scan
        report = await self.scanner.scan_package(
            package_name, package_version, files, dependencies
        )
        
        # Check against policy
        policy_result = self.policy_checker.check(report, policy_name)
        
        # Determine risk level
        if report.risk_score >= 75:
            risk_level = "critical"
        elif report.risk_score >= 50:
            risk_level = "high"
        elif report.risk_score >= 25:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            'package': f"{package_name}@{package_version}",
            'risk_level': risk_level,
            'risk_score': report.risk_score,
            'scan_summary': report.summary(),
            'policy_check': policy_result,
            'approved': policy_result['passed'] and report.passed,
            'recommendations': self._generate_recommendations(report)
        }
    
    def _generate_recommendations(self, report: VulnerabilityReport) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if report.critical_count > 0:
            recommendations.append("Immediately address all critical vulnerabilities")
        
        if report.high_count > 0:
            recommendations.append("Review and remediate high-severity issues")
        
        by_type = {}
        for v in report.vulnerabilities:
            by_type[v.type] = by_type.get(v.type, 0) + 1
        
        if VulnerabilityType.INJECTION in by_type:
            recommendations.append("Implement input validation and sanitization")
        
        if VulnerabilityType.SENSITIVE_DATA in by_type:
            recommendations.append("Remove hardcoded credentials; use secure secrets management")
        
        if VulnerabilityType.OUTDATED_COMPONENT in by_type:
            recommendations.append("Update dependencies to latest secure versions")
        
        return recommendations


async def demo_security_scanner():
    """Demonstrate security scanning."""
    print("🔒 Security Scanner Demo")
    print("=" * 50)
    
    scanner = SecurityScanner()
    
    # Add known CVE
    scanner.add_known_cve(
        "CVE-2023-1234", "vulnerable-lib", ["1.0.0", "1.0.1"], Severity.CRITICAL
    )
    
    # Sample code with vulnerabilities
    vulnerable_code = b'''
import os
import pickle

def execute_command(cmd):
    os.system(cmd)  # Dangerous!
    
password = "super_secret_123"  # Hardcoded password

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)  # Unsafe deserialization
'''
    
    files = {
        "agent.py": vulnerable_code,
        "config.py": b"api_key = 'sk-abc123'\n"
    }
    
    dependencies = {"vulnerable-lib": "1.0.0"}
    
    print("\n🔍 Scanning package...")
    report = await scanner.scan_package(
        "test-agent", "1.0.0", files, dependencies
    )
    
    print(f"\n📋 Scan Report:")
    print(f"  Package: {report.package_name}@{report.package_version}")
    print(f"  Vulnerabilities: {len(report.vulnerabilities)}")
    print(f"  Risk Score: {report.risk_score}")
    print(f"  Passed: {report.passed}")
    
    print("\n🚨 Vulnerabilities found:")
    for v in report.vulnerabilities[:5]:
        print(f"  [{v.severity.value.upper()}] {v.title}")
        if v.location:
            print(f"    Location: {v.location}:{v.line_number}")
    
    # Policy check
    policy_checker = PolicyChecker()
    policy_checker.add_policy(SecurityPolicy(
        name="strict",
        max_critical=0,
        max_high=0,
        max_risk_score=25.0
    ))
    
    policy_result = policy_checker.check(report, "strict")
    print(f"\n📜 Policy Check (strict): {'PASS' if policy_result['passed'] else 'FAIL'}")
    for v in policy_result['violations']:
        print(f"  - {v}")
    
    # Risk assessment
    assessment = RiskAssessment(scanner, policy_checker)
    result = await assessment.assess(
        "test-agent", "1.0.0", files, dependencies, "strict"
    )
    
    print(f"\n⚠️ Risk Assessment:")
    print(f"  Level: {result['risk_level']}")
    print(f"  Approved: {result['approved']}")
    print(f"  Recommendations:")
    for r in result['recommendations'][:3]:
        print(f"    - {r}")
    
    print("\n✅ Security scanner demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_security_scanner())
