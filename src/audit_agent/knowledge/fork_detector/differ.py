"""Differential fork analyzer.

Compares project file hashes against known protocol fingerprints to detect
fork lineage and identify modified files.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Optional

from audit_agent.knowledge.fork_detector.known_protocols import (
    ALL_PROTOCOLS,
    ProtocolFingerprint,
)


@dataclass
class ModifiedFile:
    project_path: str
    canonical_path: str
    modification_type: str  # "changed_formula" | "added_function" | "modified_access" | "new_state_var" | "modified"
    similarity: float       # 0.0 – 1.0 (1.0 = identical)


@dataclass
class ForkMatch:
    protocol: ProtocolFingerprint
    similarity_score: float       # fraction of project files that match
    modified_files: list[ModifiedFile] = field(default_factory=list)
    matched_files: int = 0
    total_protocol_files: int = 0


@dataclass
class DiffResult:
    is_fork: bool
    best_match: Optional[ForkMatch] = None
    all_matches: list[ForkMatch] = field(default_factory=list)

    @property
    def base_protocol(self) -> Optional[str]:
        return self.best_match.protocol.name if self.best_match else None

    @property
    def similarity_score(self) -> float:
        return self.best_match.similarity_score if self.best_match else 0.0

    @property
    def risk_multiplier(self) -> float:
        """Modified files in a fork get a 2× attack surface multiplier; added files 3×."""
        if not self.is_fork or not self.best_match:
            return 1.0
        modified = len(self.best_match.modified_files)
        return 1.0 + modified * 0.5  # rough heuristic


FORK_DETECTION_THRESHOLD = 0.50  # ≥50% file match → consider a fork


class DifferentialAnalyzer:
    """Compare project hashes against known protocol fingerprints."""

    def analyze(self, project_hashes: dict[str, str]) -> DiffResult:
        """Return fork detection result for a project.

        Args:
            project_hashes: {relative_file_path -> normalized_hash}
        """
        matches: list[ForkMatch] = []

        for protocol in ALL_PROTOCOLS:
            match = self._match_protocol(project_hashes, protocol)
            if match.similarity_score > 0:
                matches.append(match)

        matches.sort(key=lambda m: m.similarity_score, reverse=True)

        if not matches or matches[0].similarity_score < FORK_DETECTION_THRESHOLD:
            return DiffResult(is_fork=False, all_matches=matches)

        best = matches[0]
        return DiffResult(is_fork=True, best_match=best, all_matches=matches)

    def _match_protocol(
        self, project_hashes: dict[str, str], protocol: ProtocolFingerprint
    ) -> ForkMatch:
        proto_hashes = protocol.file_hashes
        if not proto_hashes:
            return ForkMatch(protocol=protocol, similarity_score=0.0)

        exact_matches = 0
        modified: list[ModifiedFile] = []

        for proto_path, proto_hash in proto_hashes.items():
            # Try to find a matching file in the project (by filename)
            proto_filename = proto_path.split("/")[-1]
            project_match = self._find_by_filename(project_hashes, proto_filename)

            if project_match is None:
                continue

            proj_path, proj_hash = project_match
            if proj_hash == proto_hash:
                exact_matches += 1
            elif proj_hash and proto_hash:
                # File exists but is different — estimate similarity
                sim = self._hash_similarity(proj_hash, proto_hash)
                modified.append(
                    ModifiedFile(
                        project_path=proj_path,
                        canonical_path=proto_path,
                        modification_type=_classify_modification(sim),
                        similarity=sim,
                    )
                )

        total = len(proto_hashes)
        matched = exact_matches + len(modified)
        score = matched / total if total else 0.0

        return ForkMatch(
            protocol=protocol,
            similarity_score=score,
            modified_files=modified,
            matched_files=matched,
            total_protocol_files=total,
        )

    @staticmethod
    def _find_by_filename(
        project_hashes: dict[str, str], filename: str
    ) -> Optional[tuple[str, str]]:
        for path, h in project_hashes.items():
            if path.split("/")[-1] == filename or path.endswith(filename):
                return path, h
        return None

    @staticmethod
    def _hash_similarity(h1: str, h2: str) -> float:
        """Rough similarity between two hex hashes (character overlap)."""
        return difflib.SequenceMatcher(None, h1, h2).ratio()


def _classify_modification(similarity: float) -> str:
    if similarity >= 0.95:
        return "minor_change"
    if similarity >= 0.80:
        return "modified_access"
    if similarity >= 0.60:
        return "changed_formula"
    return "major_rewrite"
