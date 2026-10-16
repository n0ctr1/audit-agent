"""Scope resolution: determines which files to audit and which are context-only."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from audit_agent.observability.logging import get_logger

log = get_logger("scope")

LIBRARY_PATTERNS = [
    "lib/",
    "node_modules/",
    "@openzeppelin/",
    "forge-std/",
    "solady/",
    "solmate/",
    "@layerzerolabs/",
    "@chainlink/",
    "@uniswap/",
    "test/",
    "tests/",
    "mock/",
    "mocks/",
    "script/",
    "scripts/",
    "vendor/",
    "dependencies/",
]

LIBRARY_SUFFIXES = (".t.sol", ".s.sol")

_IMPORT_RE = re.compile(r'import\s+(?:\{[^}]*\}\s+from\s+)?["\']([^"\']+)["\']')


@dataclass
class ScopeResult:
    """Result of scope resolution."""
    in_scope: list[Path]
    context_files: list[Path]
    out_of_scope: list[Path]
    scope_source: str  # "explicit_file" | "auto_detected" | "all_non_library"


def resolve_scope(
    target_dir: Path,
    scope_file: Path | None = None,
) -> ScopeResult:
    """Determine the analysis scope.

    Priority:
    1. Explicit scope_file argument → in_scope from that file
    2. {target_dir}/scope.txt auto-detected → in_scope from that file
    3. Fallback → all non-library .sol files are in_scope

    Import dependencies of in_scope files are placed in context_files so
    agents can read them for type information without auditing them.
    """
    all_sol = _enumerate_all_sol(target_dir)

    if scope_file and scope_file.exists():
        in_scope_paths = _parse_scope_file(scope_file, target_dir)
        scope_source = "explicit_file"
    elif (target_dir / "scope.txt").exists():
        in_scope_paths = _parse_scope_file(target_dir / "scope.txt", target_dir)
        scope_source = "auto_detected"
    else:
        in_scope_paths = _filter_non_library(all_sol, target_dir)
        scope_source = "all_non_library"

    remappings = parse_remappings(target_dir)
    context_files = _resolve_imports(in_scope_paths, all_sol, target_dir, remappings)
    context_files = [f for f in context_files if f not in in_scope_paths]

    in_scope_set = set(in_scope_paths)
    context_set = set(context_files)
    out_of_scope = [f for f in all_sol if f not in in_scope_set and f not in context_set]

    log.info(
        "scope.resolved",
        source=scope_source,
        in_scope=len(in_scope_paths),
        context=len(context_files),
        out_of_scope=len(out_of_scope),
    )

    return ScopeResult(
        in_scope=in_scope_paths,
        context_files=context_files,
        out_of_scope=out_of_scope,
        scope_source=scope_source,
    )


def _enumerate_all_sol(target_dir: Path) -> list[Path]:
    """Return all .sol files under target_dir, sorted."""
    return sorted(p for p in target_dir.rglob("*.sol") if p.is_file())


def _parse_scope_file(scope_file: Path, target_dir: Path) -> list[Path]:
    """Parse scope.txt and return absolute paths. Warns on missing files."""
    paths: list[Path] = []
    for line in scope_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        resolved = (target_dir / line).resolve()
        if resolved.exists() and resolved.suffix == ".sol":
            paths.append(resolved)
        else:
            log.warning("scope.missing_file", path=line, resolved=str(resolved))
    return paths


def _filter_non_library(all_sol: list[Path], target_dir: Path) -> list[Path]:
    """Return files that are not in library/test directories."""
    result: list[Path] = []
    for path in all_sol:
        try:
            rel = str(path.relative_to(target_dir))
        except ValueError:
            rel = str(path)
        if any(rel.endswith(sfx) for sfx in LIBRARY_SUFFIXES):
            continue
        if any(pat in rel for pat in LIBRARY_PATTERNS):
            continue
        result.append(path)
    return result


def _resolve_imports(
    in_scope: list[Path],
    all_sol: list[Path],
    target_dir: Path,
    remappings: dict[str, str],
) -> list[Path]:
    """Recursively find all .sol files imported by in_scope files."""
    context: set[Path] = set()
    visited: set[Path] = set()
    queue = list(in_scope)

    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)

        try:
            content = current.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for match in _IMPORT_RE.finditer(content):
            import_path = match.group(1)
            resolved = _resolve_import_path(
                import_path, current, target_dir, all_sol, remappings
            )
            if resolved and resolved not in visited:
                context.add(resolved)
                queue.append(resolved)

    return list(context)


def _resolve_import_path(
    import_path: str,
    importing_file: Path,
    target_dir: Path,
    all_sol: list[Path],
    remappings: dict[str, str],
) -> Path | None:
    """Resolve an import path string to an actual file."""
    # Apply remappings first
    for prefix, replacement in remappings.items():
        if import_path.startswith(prefix):
            import_path = replacement + import_path[len(prefix):]
            break

    # Relative import
    if import_path.startswith("."):
        candidate = (importing_file.parent / import_path).resolve()
        if candidate.exists():
            return candidate

    # Absolute from project root
    candidate = (target_dir / import_path).resolve()
    if candidate.exists():
        return candidate

    # Search by filename among known .sol files
    filename = Path(import_path).name
    matches = [f for f in all_sol if f.name == filename]
    if len(matches) == 1:
        return matches[0]

    return None


def parse_remappings(target_dir: Path) -> dict[str, str]:
    """Parse remappings from remappings.txt and foundry.toml."""
    remappings: dict[str, str] = {}

    remap_file = target_dir / "remappings.txt"
    if remap_file.exists():
        for line in remap_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                prefix, path = line.split("=", 1)
                remappings[prefix.strip()] = path.strip()

    toml_file = target_dir / "foundry.toml"
    if toml_file.exists():
        try:
            import tomllib
            with open(toml_file, "rb") as f:
                config = tomllib.load(f)

            def _extract_remappings(node: object) -> list[str]:
                """Recursively find all remappings arrays in the TOML tree."""
                found: list[str] = []
                if isinstance(node, dict):
                    if "remappings" in node and isinstance(node["remappings"], list):
                        found.extend(node["remappings"])
                    for v in node.values():
                        found.extend(_extract_remappings(v))
                return found

            for remap in _extract_remappings(config):
                if isinstance(remap, str) and "=" in remap:
                    prefix, path = remap.split("=", 1)
                    remappings[prefix.strip()] = path.strip()
        except Exception as exc:
            log.debug("scope.remappings_parse_error", error=str(exc))

    return remappings
