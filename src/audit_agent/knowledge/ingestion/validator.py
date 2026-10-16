"""Validator: schema enforcement, quality scoring, and dedup for KB entries."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from audit_agent.knowledge.models import Finding
from audit_agent.observability.logging import get_logger

log = get_logger("knowledge.validator")

# Minimum quality thresholds
MIN_TITLE_LEN = 10
MIN_SUMMARY_LEN = 30
MIN_ROOT_CAUSE_LEN = 20
QUALITY_REJECT_THRESHOLD = 0.3   # findings below this are dropped


@dataclass
class ValidationResult:
    valid: bool
    quality_score: float
    rejection_reason: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


class Validator:
    """Validates and quality-scores Finding objects before ingestion."""

    def validate(self, finding: Finding) -> ValidationResult:
        """Run all checks. Returns ValidationResult with quality_score in [0, 1]."""
        # --- Required field checks (hard rejections) ---
        if not finding.title or len(finding.title) < MIN_TITLE_LEN:
            return ValidationResult(
                valid=False,
                quality_score=0.0,
                rejection_reason=f"title too short ({len(finding.title)} chars)",
            )

        if finding.severity not in ("critical", "high", "medium", "low", "info"):
            return ValidationResult(
                valid=False,
                quality_score=0.0,
                rejection_reason=f"invalid severity: {finding.severity!r}",
            )

        if not finding.description_structured.summary:
            return ValidationResult(
                valid=False,
                quality_score=0.0,
                rejection_reason="missing description.summary",
            )

        # --- Quality scoring ---
        score = 0.0
        warnings: list[str] = []

        # Title quality (0.10)
        if len(finding.title) >= 20:
            score += 0.10
        else:
            score += 0.05
            warnings.append("short title")

        # Summary quality (0.20)
        summary = finding.description_structured.summary
        if len(summary) >= MIN_SUMMARY_LEN:
            score += 0.20
        elif summary:
            score += 0.10
            warnings.append("short summary")

        # Root cause (0.20)
        rc = finding.description_structured.root_cause
        if len(rc) >= MIN_ROOT_CAUSE_LEN:
            score += 0.20
        elif rc:
            score += 0.10
            warnings.append("short root_cause")
        else:
            warnings.append("missing root_cause")

        # Attack vector (0.10)
        if finding.description_structured.attack_vector:
            score += 0.10

        # Impact (0.10)
        if finding.description_structured.impact:
            score += 0.10

        # Code snippets (0.10)
        if finding.code_snippets:
            score += 0.10
        else:
            warnings.append("no code snippets")

        # Severity specificity (0.10) — high/critical more valuable
        if finding.severity in ("critical", "high"):
            score += 0.10
        elif finding.severity in ("medium",):
            score += 0.07
        else:
            score += 0.04

        # PoC / fix bonus (0.10)
        if finding.poc:
            score += 0.05
        if finding.fix.description:
            score += 0.05

        score = min(score, 1.0)

        if score < QUALITY_REJECT_THRESHOLD:
            return ValidationResult(
                valid=False,
                quality_score=score,
                rejection_reason=f"quality score {score:.2f} below threshold {QUALITY_REJECT_THRESHOLD}",
                warnings=warnings,
            )

        return ValidationResult(valid=True, quality_score=score, warnings=warnings)

    def fingerprint(self, finding: Finding) -> str:
        """Content-based fingerprint for deduplication.

        Uses (title + root_cause) normalised to lowercase stripped text.
        """
        text = (finding.title + finding.description_structured.root_cause).lower()
        text = re.sub(r"\s+", " ", text).strip()
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def validate_raw_dict(data: dict) -> tuple[Optional[Finding], Optional[str]]:
    """Try to parse a raw dict as Finding; return (finding, error_msg)."""
    try:
        finding = Finding(**data)
        return finding, None
    except ValidationError as e:
        return None, str(e)
