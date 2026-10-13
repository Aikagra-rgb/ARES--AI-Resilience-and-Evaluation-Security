"""
ARES Structural Attack Detector.

Deterministic, rule-based detection of prompt injection patterns.
No LLM calls, fully auditable, sub-millisecond latency.
Designed as the FIRST line of defense in a defense-in-depth architecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Pattern
import logging

logger = logging.getLogger(__name__)


class AttackType(str, Enum):
    """Categories of structural attacks detected."""
    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_HIJACK = "role_hijack"
    DELIMITER_CONFUSION = "delimiter_confusion"
    ENCODING_OBFUSCATION = "encoding_obfuscation"
    DATA_EXFILTRATION = "data_exfiltration"
    TOOL_MISUSE = "tool_misuse"
    INDIRECT_INJECTION = "indirect_injection"
    MULTI_TURN_ESCALATION = "multi_turn_escalation"
    CONTEXT_SMUGGLING = "context_smuggling"
    PERSONA_ADOPTION = "persona_adoption"


class SeverityLevel(str, Enum):
    """Severity of detected pattern."""
    INFO = "info"           # Pattern present but likely benign
    LOW = "low"             # Suspicious but common in legitimate use
    MEDIUM = "medium"       # Strong indicator of attack
    HIGH = "high"           # Very likely attack
    CRITICAL = "critical"   # Definitive attack pattern


class Decision(str, Enum):
    """Detection decision."""
    ALLOW = "allow"
    FLAG = "flag"           # Log and monitor, but allow
    CHALLENGE = "challenge" # Require additional verification
    BLOCK = "block"         # Deny request


@dataclass(frozen=True)
class Detection:
    """A single pattern detection result."""
    attack_type: AttackType
    pattern_name: str
    matched_text: str
    position: int
    severity: SeverityLevel
    confidence: float  # 0.0 - 1.0
    description: str
    rule_id: str

    def to_dict(self) -> dict:
        return {
            "attack_type": self.attack_type.value,
            "pattern_name": self.pattern_name,
            "matched_text": self.matched_text[:100],  # Truncate for logs
            "position": self.position,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "rule_id": self.rule_id,
        }


@dataclass
class DetectionResult:
    """Aggregated detection result for a prompt."""
    detections: List[Detection] = field(default_factory=list)
    decision: Decision = Decision.ALLOW
    max_severity: SeverityLevel = SeverityLevel.INFO
    total_confidence: float = 0.0

    def add_detection(self, detection: Detection) -> None:
        self.detections.append(detection)
        self._recalculate()

    def _recalculate(self) -> None:
        if not self.detections:
            self.decision = Decision.ALLOW
            self.max_severity = SeverityLevel.INFO
            self.total_confidence = 0.0
            return

        # Max severity
        severity_order = {
            SeverityLevel.INFO: 0,
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4,
        }
        self.max_severity = max(self.detections, key=lambda d: severity_order[d.severity]).severity

        # Weighted confidence
        self.total_confidence = sum(d.confidence for d in self.detections) / len(self.detections)

        # Decision logic
        critical_count = sum(1 for d in self.detections if d.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for d in self.detections if d.severity == SeverityLevel.HIGH)
        medium_count = sum(1 for d in self.detections if d.severity == SeverityLevel.MEDIUM)

        if critical_count > 0 or high_count >= 2:
            self.decision = Decision.BLOCK
        elif high_count >= 1 or medium_count >= 2:
            self.decision = Decision.CHALLENGE
        elif medium_count >= 1 or len(self.detections) >= 3:
            self.decision = Decision.FLAG
        else:
            self.decision = Decision.ALLOW

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "max_severity": self.max_severity.value,
            "total_confidence": round(self.total_confidence, 3),
            "detection_count": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }


class StructuralAttackDetector:
    """
    Deterministic structural pattern detector for prompt injection.
    
    Design principles:
    - Zero LLM calls (deterministic, auditable)
    - Sub-millisecond latency
    - Explainable rules with clear rule IDs
    - Tunable sensitivity per attack type
    - No false positives on legitimate technical content
    """
    
    # Rule ID format: ARS-{category}-{number}
    # Example: ARS-IO-001 = Attack Rule Structural - Instruction Override - 001
    
    INSTRUCTION_OVERRIDE_PATTERNS = [
        # Direct instruction override
        (re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules|guidelines)"), 
         "direct_ignore_previous", 0.95, SeverityLevel.CRITICAL,
         "Explicit instruction to ignore prior system prompt"),
        
        (re.compile(r"(?i)\b(disregard|override|bypass|forget)\s+(all\s+)?(system|safety|security|previous)\s+(rules|guidelines|instructions|prompts)"),
         "disregard_system_rules", 0.93, SeverityLevel.CRITICAL,
         "Explicit instruction to bypass system safeguards"),
        
        # System prompt extraction attempts
        (re.compile(r"(?i)\b(repeat|print|output|show|display|reveal|tell)\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions|rules|guidelines)"),
         "request_system_prompt", 0.85, SeverityLevel.HIGH,
         "Direct request for system prompt disclosure"),
        
        (re.compile(r"(?i)\bwhat\s+(is|are)\s+(your\s+)?(system\s+)?(prompt|instructions|initial\s+instructions)"),
         "query_system_prompt", 0.80, SeverityLevel.HIGH,
         "Interrogative request for system prompt"),
        
        # Fake authority / privilege escalation
        (re.compile(r"(?i)\b(system|admin|administrator|root|developer|creator)\s*(update|override|patch|maintenance)\b"),
         "fake_authority_update", 0.88, SeverityLevel.HIGH,
         "Impersonation of system authority"),
        
        (re.compile(r"(?i)\b(i\s+am|as\s+(an?|the))\s+(admin|administrator|developer|creator|superuser|root)\b"),
         "claim_admin_identity", 0.82, SeverityLevel.HIGH,
         "Claim of elevated privileges"),
        
        # Session/session boundary manipulation
        (re.compile(r"(?i)\b(new\s+session|fresh\s+start|reset\s+(context|conversation)|start\s+over)\b"),
         "session_reset_attempt", 0.70, SeverityLevel.MEDIUM,
         "Attempt to reset conversation context"),
        
        # Instruction hierarchy confusion
        (re.compile(r"(?i)\b(from\s+now\s+on|going\s+forward|henceforth)\s+you\s+(must|will|shall)\b"),
         "persistent_instruction_injection", 0.75, SeverityLevel.HIGH,
         "Attempt to establish persistent malicious instruction"),
    ]

    ROLE_HIJACK_PATTERNS = [
        # DAN and variants
        (re.compile(r"(?i)\bdan\b|do\s+anything\s+now|unrestricted\s+(ai|mode|assistant)"),
         "dan_persona", 0.95, SeverityLevel.CRITICAL,
         "Classic DAN (Do Anything Now) jailbreak persona"),
        
        # Custom persona creation
        (re.compile(r"(?i)\b(pretend|act\s+as|roleplay|simulate)\s+(to\s+be|you\s+are)\s+(an?|the)\s+\w+"),
         "persona_adoption", 0.85, SeverityLevel.HIGH,
         "Request to adopt alternative persona"),
        
        # "No rules" personas
        (re.compile(r"(?i)\b(no\s+rules|without\s+rules|unrestricted|unfiltered|uncensored)\s+(mode|assistant|ai|version)"),
         "no_rules_persona", 0.90, SeverityLevel.CRITICAL,
         "Explicit request for rule-free mode"),
        
        # Authority impersonation
        (re.compile(r"(?i)\b(you\s+are\s+(now\s+)?|become)\s+(an?|the)\s+(expert|professional|specialist|authority)\s+(in|on)\b"),
         "authority_impersonation", 0.70, SeverityLevel.MEDIUM,
         "Attempt to establish false authority"),
    ]

    DELIMITER_CONFUSION_PATTERNS = [
        # Code fence injection
        (re.compile(r"```\w*\n.*?(ignore|disregard|override|system).*?```", re.DOTALL | re.IGNORECASE),
         "code_fence_injection", 0.88, SeverityLevel.HIGH,
         "Malicious instruction hidden in code fence"),
        
        # XML/HTML tag injection
        (re.compile(r"<\|?(system|user|assistant|instruction|prompt|rule)s?\|?>", re.IGNORECASE),
         "xml_tag_injection", 0.85, SeverityLevel.HIGH,
         "Fake XML tags mimicking conversation format"),
        
        # Custom delimiter wrapping
        (re.compile(r"(<<<|###|@@@|===|---)\s*(instruction|prompt|rule|system)\s*(>>>|###|@@@|===|---)", re.IGNORECASE),
         "custom_delimiter_wrapping", 0.80, SeverityLevel.HIGH,
         "Instruction wrapped in custom delimiters"),
        
        # JSON/XML injection in structured data
        (re.compile(r'("|\'|\{)\s*(instruction|prompt|system)\s*(\}|\'|")\s*:\s*("|\')?(ignore|override|disregard)', re.IGNORECASE),
         "structured_data_injection", 0.82, SeverityLevel.HIGH,
         "Injection attempt within JSON/XML structure"),
        
        # Markdown injection
        (re.compile(r"\[(system|instruction|prompt)\]\(.*?\)", re.IGNORECASE),
         "markdown_link_injection", 0.65, SeverityLevel.MEDIUM,
         "Markdown link mimicking system instruction"),
    ]

    ENCODING_OBFUSCATION_PATTERNS = [
        # Base64 encoded payloads
        (re.compile(r"(?i)\b(base64|b64)\s*[:=]\s*[A-Za-z0-9+/=]{30,}"),
         "base64_payload", 0.85, SeverityLevel.HIGH,
         "Base64 encoded instruction payload"),
        
        (re.compile(r"(?i)decode\s+(base64|b64)\s+(and|then)\s+(execute|follow|run)"),
         "base64_decode_execute", 0.90, SeverityLevel.CRITICAL,
         "Explicit instruction to decode and execute base64"),
        
        # Hex encoding
        (re.compile(r"(?i)\b(hex|hexadecimal)\s*[:=]\s*[0-9a-fA-F]{30,}"),
         "hex_payload", 0.80, SeverityLevel.HIGH,
         "Hex encoded payload"),
        
        # ROT13 / Caesar cipher
        (re.compile(r"(?i)\b(rot13|caesar|cipher)\s*(decode|decrypt)"),
         "rot13_instruction", 0.75, SeverityLevel.MEDIUM,
         "Instruction to decode ROT13/cipher"),
        
        # Unicode obfuscation
        (re.compile(r"(\\u[0-9a-fA-F]{4}){10,}"),
         "unicode_obfuscation", 0.70, SeverityLevel.MEDIUM,
         "Excessive Unicode escape sequences"),
        
        # Zero-width characters
        (re.compile(r"[\u200b-\u200f\ufeff]"),
         "zero_width_chars", 0.60, SeverityLevel.LOW,
         "Zero-width characters (potential obfuscation)"),
    ]

    DATA_EXFILTRATION_PATTERNS = [
        # Direct secret extraction
        (re.compile(r"(?i)\b(api[_\s]?key|secret[_\s]?key|access[_\s]?token|auth[_\s]?token)\s*[:=]\s*\S+"),
         "exposed_secret", 0.95, SeverityLevel.CRITICAL,
         "Potential API key/secret exposure in prompt"),
        
        # PII patterns
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
         "ssn_pattern", 0.90, SeverityLevel.HIGH,
         "SSN pattern detected"),
        
        (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  # Credit card
         "credit_card_pattern", 0.85, SeverityLevel.HIGH,
         "Credit card pattern detected"),
        
        (re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S+"),
         "password_exposure", 0.90, SeverityLevel.HIGH,
         "Password exposure in prompt"),
        
        # Database/connection strings
        (re.compile(r"(?i)(mongodb|postgres|mysql|redis)://[^\s]+"),
         "connection_string", 0.90, SeverityLevel.HIGH,
         "Database connection string detected"),
    ]

    TOOL_MISUSE_PATTERNS = [
        # Tool calling injection
        (re.compile(r"(?i)\b(call|invoke|execute|run)\s+(the\s+)?(function|tool|api)\b"),
         "explicit_tool_invocation", 0.75, SeverityLevel.MEDIUM,
         "Explicit request to invoke tools"),
        
        # Function schema extraction
        (re.compile(r"(?i)\b(list|show|describe)\s+(available\s+)?(functions|tools|apis)\b"),
         "function_schema_extraction", 0.70, SeverityLevel.MEDIUM,
         "Request for function/tool schema"),
        
        # Tool output manipulation
        (re.compile(r"(?i)\b(fake|forge|spoof)\s+(the\s+)?(tool|function|api)\s+(output|response|result)"),
         "tool_output_spoofing", 0.85, SeverityLevel.HIGH,
         "Attempt to manipulate tool outputs"),
    ]

    INDIRECT_INJECTION_PATTERNS = [
        # "Read this URL/document" patterns
        (re.compile(r"(?i)\b(read|fetch|get|load)\s+(this\s+)?(url|link|document|page|file)\b"),
         "external_content_fetch", 0.60, SeverityLevel.LOW,
         "Request to fetch external content (potential indirect injection vector)"),
        
        # Summarization with injection
        (re.compile(r"(?i)\bsummarize\s+(this|the)\s+(url|link|page|document|text)\b"),
         "summarize_external", 0.55, SeverityLevel.LOW,
         "Request to summarize external content"),
    ]

    MULTI_TURN_ESCALATION_PATTERNS = [
        # Gradual escalation markers
        (re.compile(r"(?i)\b(hypothetically|theoretically|for\s+(research|educational|academic)\s+purposes)\b"),
         "hypothetical_framing", 0.50, SeverityLevel.LOW,
         "Hypothetical framing (potential escalation setup)"),
        
        # Trust building
        (re.compile(r"(?i)\b(trust\s+me|you\s+can\s+trust|i\s+promise|between\s+us)\b"),
         "trust_building", 0.55, SeverityLevel.LOW,
         "Trust-building language (potential social engineering)"),
    ]

    CONTEXT_SMUGGLING_PATTERNS = [
        # RAG context injection markers
        (re.compile(r"(?i)\b(context|document|retrieved|search\s+result)\s*(says|states|contains|shows)\b"),
         "context_authority_claim", 0.65, SeverityLevel.MEDIUM,
         "Claim about retrieved context content"),
        
        # "According to" injection
        (re.compile(r"(?i)\baccording\s+to\s+(the\s+)?(context|document|search|retrieved)\b"),
         "according_to_context", 0.60, SeverityLevel.MEDIUM,
         "Attribution to retrieved context"),
    ]

    # Compile all patterns into a single list for iteration
    ALL_PATTERNS = [
        (AttackType.INSTRUCTION_OVERRIDE, INSTRUCTION_OVERRIDE_PATTERNS),
        (AttackType.ROLE_HIJACK, ROLE_HIJACK_PATTERNS),
        (AttackType.DELIMITER_CONFUSION, DELIMITER_CONFUSION_PATTERNS),
        (AttackType.ENCODING_OBFUSCATION, ENCODING_OBFUSCATION_PATTERNS),
        (AttackType.DATA_EXFILTRATION, DATA_EXFILTRATION_PATTERNS),
        (AttackType.TOOL_MISUSE, TOOL_MISUSE_PATTERNS),
        (AttackType.INDIRECT_INJECTION, INDIRECT_INJECTION_PATTERNS),
        (AttackType.MULTI_TURN_ESCALATION, MULTI_TURN_ESCALATION_PATTERNS),
        (AttackType.CONTEXT_SMUGGLING, CONTEXT_SMUGGLING_PATTERNS),
    ]

    def __init__(
        self,
        enabled_categories: Optional[List[AttackType]] = None,
        severity_threshold: SeverityLevel = SeverityLevel.LOW,
        custom_patterns: Optional[List[tuple]] = None,
        canary_token: Optional[str] = None,
    ):
        """
        Initialize detector.
        
        Args:
            enabled_categories: Attack types to detect (None = all)
            severity_threshold: Minimum severity to report
            custom_patterns: Additional (AttackType, [(pattern, name, confidence, severity, desc)]) tuples
            canary_token: Optional canary token to detect
        """
        self.enabled_categories = set(enabled_categories) if enabled_categories else set(AttackType)
        self.severity_threshold = severity_threshold
        self.custom_patterns = custom_patterns or []
        self.canary_token = canary_token
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[tuple]:
        """Compile all enabled patterns into flat list."""
        compiled = []
        for attack_type, patterns in self.ALL_PATTERNS:
            if attack_type not in self.enabled_categories:
                continue
            for pattern, name, confidence, severity, description in patterns:
                if self._severity_meets_threshold(severity):
                    rule_id = f"ARS-{attack_type.value[:2].upper()}-{len(compiled)+1:03d}"
                    compiled.append((attack_type, pattern, name, confidence, severity, description, rule_id))
        
        # Add canary token pattern if configured
        if self.canary_token:
            import re
            canary_pattern = re.compile(re.escape(self.canary_token), re.IGNORECASE)
            rule_id = f"ARS-IO-{len(compiled)+1:03d}"
            compiled.append((
                AttackType.INSTRUCTION_OVERRIDE,
                canary_pattern,
                "canary_token_detection",
                1.0,
                SeverityLevel.CRITICAL,
                f"Canary token '{self.canary_token}' detected in prompt",
                rule_id,
            ))
        
        # Add custom patterns
        for attack_type, patterns in self.custom_patterns:
            for pattern, name, confidence, severity, description in patterns:
                if attack_type in self.enabled_categories and self._severity_meets_threshold(severity):
                    rule_id = f"ARS-CUSTOM-{len(compiled)+1:03d}"
                    compiled.append((attack_type, pattern, name, confidence, severity, description, rule_id))
        
        return compiled

    def _severity_meets_threshold(self, severity: SeverityLevel) -> bool:
        order = {
            SeverityLevel.INFO: 0,
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4,
        }
        return order[severity] >= order[self.severity_threshold]

    def detect(
        self,
        text: str,
        context: Optional[str] = None,
        canary_token: Optional[str] = None,
    ) -> DetectionResult:
        """
        Detect structural attack patterns in text.
        
        Args:
            text: The prompt/text to analyze
            context: Optional conversation context for multi-turn analysis
            canary_token: Optional dynamic/tenant-specific canary token to trap extractions
            
        Returns:
            DetectionResult with all matches and aggregated decision
        """
        if not text:
            return DetectionResult()
        
        result = DetectionResult()
        full_text = text
        if context:
            full_text = f"{context}\n{text}"
        
        for attack_type, pattern, name, confidence, severity, description, rule_id in self._compiled_patterns:
            for match in pattern.finditer(full_text):
                detection = Detection(
                    attack_type=attack_type,
                    pattern_name=name,
                    matched_text=match.group(0),
                    position=match.start(),
                    severity=severity,
                    confidence=confidence,
                    description=description,
                    rule_id=rule_id,
                )
                result.add_detection(detection)

        # Dynamic per-tenant canary token verification (if not already compiled)
        if canary_token and canary_token != self.canary_token and canary_token in full_text:
            pos = full_text.find(canary_token)
            detection = Detection(
                attack_type=AttackType.DATA_EXFILTRATION,
                pattern_name="canary_token_leak",
                matched_text=canary_token,
                position=pos,
                severity=SeverityLevel.CRITICAL,
                confidence=1.0,
                description=f"Tenant canary token '{canary_token}' detected in prompt",
                rule_id="ARS-CANARY-001",
            )
            result.add_detection(detection)
        
        # Log high-severity detections
        for d in result.detections:
            if d.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL):
                logger.warning(
                    "Structural detection: %s | %s | confidence=%.2f | pos=%d",
                    d.rule_id, d.attack_type.value, d.confidence, d.position
                )
        
        return result

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        """Detect patterns in multiple texts efficiently."""
        return [self.detect(text) for text in texts]

    def get_stats(self) -> dict:
        """Get detector configuration stats."""
        return {
            "enabled_categories": [c.value for c in self.enabled_categories],
            "severity_threshold": self.severity_threshold.value,
            "pattern_count": len(self._compiled_patterns),
            "patterns_by_category": {
                at.value: sum(1 for at2, _, _, _, _, _, _ in self._compiled_patterns if at2 == at)
                for at in self.enabled_categories
            },
        }


# Convenience function for simple usage
def quick_scan(text: str) -> DetectionResult:
    """Quick scan with default settings."""
    detector = StructuralAttackDetector()
    return detector.detect(text)


# Backwards compatibility alias
StructuralDetector = StructuralAttackDetector