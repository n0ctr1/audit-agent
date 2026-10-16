from audit_agent.knowledge.fork_detector.differ import DifferentialAnalyzer, DiffResult
from audit_agent.knowledge.fork_detector.fingerprints import (
    compute_file_hash,
    compute_text_hash,
    hash_project_files,
    normalize_source,
)
from audit_agent.knowledge.fork_detector.known_protocols import (
    ALL_PROTOCOLS,
    PROTOCOL_BY_NAME,
    get_protocol,
)

__all__ = [
    "DifferentialAnalyzer",
    "DiffResult",
    "compute_file_hash",
    "compute_text_hash",
    "hash_project_files",
    "normalize_source",
    "ALL_PROTOCOLS",
    "PROTOCOL_BY_NAME",
    "get_protocol",
]
