"""Draft a benchmark contest entry from a local contest checkout + official report.

Inputs are local files only (no network):
  * the contest code directory;
  * the official report as Markdown (Code4rena or Sherlock).

Output is a dataset JSON with one contest marked ``"needs_review": true``.
Ids, titles and severities are copied from the report (High/Medium only).
``affected_functions`` and ``vulnerability_class`` are heuristic guesses:

  * functions — enclosing functions of the GitHub code links in the finding
    (resolved against the local checkout), plus functions named in the title;
    other functions mentioned in the body are listed in ``review_notes`` only;
  * class — canonical class from ``benchmarks.taxonomy`` matched on the title
    (then the first paragraph), or ``"unclassified"``.

A human must check every finding, then remove ``needs_review``; until then the
runner skips the contest.

Usage:
    uv run python -m benchmarks.tools.import_contest \\
        --source-dir /path/to/2026-03-foo --report report.md \\
        --id c4-2026-03-foo --name "Foo" --gt-source https://code4rena.com/reports/2026-03-foo \\
        --end-date 2026-03-20 --merge-into benchmarks/datasets/contests.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from benchmarks.taxonomy import canonical_classes

_EXCLUDED_DIRS = {"lib", "node_modules", "test", "tests", "script", "scripts", "mocks", "mock",
                  "out", "cache", "artifacts", ".git"}

_C4_HEADING = re.compile(
    r"^#{1,4}\s*\[?\[(?P<id>[HM]-\d+)\]\s*(?P<title>.+?)(?:\]\((?P<url>[^)\s]+)\))?\s*$")
_SHERLOCK_HEADING = re.compile(r"^#{1,4}\s*Issue\s+(?P<id>[HM]-\d+):\s*(?P<title>.+?)\s*$")
_SECTION_HEADING = re.compile(r"^#\s+\S")
_CODE_LINK = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/blob/[^/\s]+/(?P<path>[^#\s)\]]+)"
    r"#L(?P<start>\d+)(?:-L(?P<end>\d+))?")
_SOURCE_LINE = re.compile(r"^Source:\s*(?P<url>\S+)", re.MULTILINE)
_CODE_IDENT = re.compile(r"`(?:(?P<contract>[A-Za-z_]\w*)(?:\.|::))?(?P<fn>[A-Za-z_]\w*)\s*\([^`]*\)`")
_FUNC_DEF = re.compile(r"\bfunction\s+(?P<name>[A-Za-z_]\w*)\s*\(")
_CONTRACT_DEF = re.compile(r"^\s*(?:abstract\s+)?(?:contract|library|interface)\s+(?P<name>[A-Za-z_]\w*)",
                           re.MULTILINE)


# ─────────────────────────────────────────────────────────────────────────────
# Source index
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FunctionSpan:
    contract: Optional[str]
    name: str
    start: int  # 1-based, inclusive
    end: int


@dataclass
class SourceFile:
    rel: str
    spans: list[FunctionSpan]


def _strip_comments(text: str) -> str:
    """Blank out comments/strings but keep line structure for line numbers."""
    def blank(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"', blank, text, flags=re.DOTALL)


def parse_functions(text: str) -> list[FunctionSpan]:
    code = _strip_comments(text)
    line_of = [0]
    for ch in code:
        line_of.append(line_of[-1] + (ch == "\n"))

    contracts = [(m.start(), m.group("name")) for m in _CONTRACT_DEF.finditer(code)]
    spans: list[FunctionSpan] = []
    for m in _FUNC_DEF.finditer(code):
        brace = code.find("{", m.end())
        semi = code.find(";", m.end())
        if brace == -1 or (semi != -1 and semi < brace):
            continue  # declaration without body (interface / abstract)
        depth, i = 0, brace
        while i < len(code):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        contract = None
        for pos, name in contracts:
            if pos < m.start():
                contract = name
        spans.append(FunctionSpan(contract, m.group("name"),
                                  line_of[m.start()] + 1, line_of[min(i, len(code) - 1)] + 1))
    return spans


def index_sources(source_dir: Path) -> list[SourceFile]:
    out = []
    for p in sorted(source_dir.rglob("*.sol")):
        rel = p.relative_to(source_dir)
        if any(part in _EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out.append(SourceFile(rel.as_posix(), parse_functions(text)))
    return out


def _find_file(sources: list[SourceFile], link_path: str) -> Optional[SourceFile]:
    """Longest path-suffix match between a GitHub link path and local files."""
    parts = link_path.strip("/").split("/")
    for k in range(len(parts)):
        suffix = "/".join(parts[k:])
        hits = [s for s in sources if s.rel == suffix or s.rel.endswith("/" + suffix)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return None  # ambiguous
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Report parsing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawFinding:
    id: str
    title: str
    body: str
    url: Optional[str] = None


def detect_format(report: str) -> str:
    if re.search(r"^#{1,4}\s*Issue\s+[HM]-\d+:", report, re.MULTILINE):
        return "sherlock"
    if re.search(r"^#{1,4}\s*\[?\[[HM]-\d+\]", report, re.MULTILINE):
        return "c4"
    raise ValueError("could not detect report format (expected Code4rena or Sherlock headings)")


def parse_report(report: str, fmt: str = "auto") -> list[RawFinding]:
    fmt = detect_format(report) if fmt == "auto" else fmt
    heading = {"c4": _C4_HEADING, "sherlock": _SHERLOCK_HEADING}[fmt]
    findings: list[RawFinding] = []
    current: Optional[RawFinding] = None
    lines: list[str] = []

    def close() -> None:
        if current is not None:
            current.body = "\n".join(lines).strip()
            if fmt == "sherlock" and not current.url:
                m = _SOURCE_LINE.search(current.body)
                current.url = m.group("url") if m else None
            findings.append(current)

    for line in report.splitlines():
        m = heading.match(line)
        if m:
            close()
            title = re.sub(r"\s+", " ", m.group("title")).strip()
            current = RawFinding(id=m.group("id"), title=title, body="",
                                 url=m.groupdict().get("url"))
            lines = []
        elif current is not None and _SECTION_HEADING.match(line):
            close()  # e.g. "# Low Risk and Non-Critical Issues"
            current, lines = None, []
        elif current is not None:
            lines.append(line)
    close()
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Heuristics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Extraction:
    functions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _add(out: list[str], fn: str) -> None:
    if fn not in out:
        out.append(fn)


def extract_functions(f: RawFinding, sources: list[SourceFile]) -> Extraction:
    ex = Extraction()
    for m in _CODE_LINK.finditer(f.body):
        sf = _find_file(sources, m.group("path"))
        start = int(m.group("start"))
        end = int(m.group("end") or start)
        if sf is None:
            ex.notes.append(f"code link not resolved locally: {m.group('path')}#L{start}")
            continue
        hit = False
        for span in sf.spans:
            if span.start <= end and start <= span.end:
                _add(ex.functions, f"{span.contract}.{span.name}" if span.contract else span.name)
                hit = True
        if not hit:
            ex.notes.append(f"code link outside any function: {sf.rel}#L{start}-L{end}")

    defined: dict[str, set[Optional[str]]] = {}
    for sf in sources:
        for span in sf.spans:
            defined.setdefault(span.name, set()).add(span.contract)

    def resolve(m: re.Match[str]) -> Optional[str]:
        name, contract = m.group("fn"), m.group("contract")
        if name not in defined:
            return None
        if contract and contract in defined[name]:
            return f"{contract}.{name}"
        owners = defined[name] - {None}
        return f"{next(iter(owners))}.{name}" if len(owners) == 1 else name

    for m in _CODE_IDENT.finditer(f.title):
        fn = resolve(m)
        if fn:
            _add(ex.functions, fn)
    mentioned = [fn for fn in (resolve(m) for m in _CODE_IDENT.finditer(f.body)) if fn]
    extra = [fn for fn in dict.fromkeys(mentioned) if fn not in ex.functions]
    if extra:
        ex.notes.append("also mentioned in body (not added): " + ", ".join(extra[:15]))
    if not ex.functions:
        ex.notes.append("no affected function found — fill affected_functions manually")
    return ex


def guess_class(f: RawFinding) -> str:
    classes = canonical_classes(f.title)
    if not classes:
        first_para = next((p for p in re.split(r"\n\s*\n", f.body)
                           if p.strip() and not p.lstrip().startswith(("#", "*", "http", "Source:"))), "")
        classes = canonical_classes(first_para[:600])
    if not classes:
        return "unclassified"
    return sorted(classes)[0] if len(classes) == 1 else "/".join(sorted(classes))


def _description(body: str, limit: int = 800) -> str:
    text = re.sub(r"```.*?```", " ", body, flags=re.DOTALL)
    text = re.sub(r"^(Source:|\*Submitted by).*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[#*>`]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# Draft assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_draft(
    *,
    source_dir: Path,
    report_text: str,
    contest_id: str,
    name: str,
    gt_source: str,
    end_date: Optional[str],
    split: str = "test",
    fmt: str = "auto",
    source_dir_label: Optional[str] = None,
) -> dict:
    sources = index_sources(source_dir)
    raw = parse_report(report_text, fmt)
    findings = []
    for f in raw:
        ex = extract_functions(f, sources)
        findings.append({
            "id": f.id,
            "title": f.title,
            "severity": "high" if f.id.startswith("H") else "medium",
            "vulnerability_class": guess_class(f),
            "affected_functions": ex.functions,
            "description": _description(f.body),
            "review_notes": ([f"report issue: {f.url}"] if f.url else []) + ex.notes,
        })
    scope = source_dir / "scope.txt"
    sev = Counter(x["severity"] for x in findings)
    return {
        "name": contest_id,
        "contests": [{
            "id": contest_id,
            "name": name,
            "source_dir": source_dir_label or str(source_dir),
            "scope_file": "scope.txt" if scope.exists() else None,
            "docs_dir": None,
            "split": split,
            "contest_end_date": end_date,
            "contaminated": False,
            "gt_source": gt_source,
            "needs_review": True,
            "review_summary": (f"Auto-imported {len(findings)} findings "
                               f"({sev.get('high', 0)} high, {sev.get('medium', 0)} medium). "
                               "Check every affected_functions / vulnerability_class, then "
                               "remove needs_review."),
            "known_issues": [],
            "official_findings": findings,
        }],
    }


def merge_into(dataset_file: Path, contest: dict) -> None:
    """Insert or replace ``contest`` (matched by id) in a dataset JSON file."""
    if dataset_file.exists():
        ds = json.loads(dataset_file.read_text(encoding="utf-8"))
    else:
        ds = {"name": dataset_file.stem, "contests": []}
    ds["contests"] = [c for c in ds.get("contests", []) if c.get("id") != contest["id"]] + [contest]
    dataset_file.write_text(json.dumps(ds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path, help="official report, Markdown")
    ap.add_argument("--id", required=True, dest="contest_id")
    ap.add_argument("--name", required=True)
    ap.add_argument("--gt-source", required=True, help="URL of the official report")
    ap.add_argument("--end-date", default=None, help="contest end date YYYY-MM-DD")
    ap.add_argument("--split", default="test", choices=["test", "dev", "smoke"])
    ap.add_argument("--format", default="auto", choices=["auto", "c4", "sherlock"])
    ap.add_argument("--source-dir-label", default=None,
                    help="path written to the dataset (e.g. relative to $BENCHMARK_CONTESTS_DIR)")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--out", type=Path, default=None, help="write JSON here (default: stdout)")
    out.add_argument("--merge-into", type=Path, default=None,
                     help="add/replace this contest (by id) in an existing or new dataset file")
    args = ap.parse_args(argv)

    if not args.source_dir.is_dir():
        ap.error(f"source dir not found: {args.source_dir}")
    draft = build_draft(
        source_dir=args.source_dir, report_text=args.report.read_text(encoding="utf-8"),
        contest_id=args.contest_id, name=args.name, gt_source=args.gt_source,
        end_date=args.end_date, split=args.split, fmt=args.format,
        source_dir_label=args.source_dir_label,
    )
    if args.merge_into:
        n = len(draft["contests"][0]["official_findings"])
        merge_into(args.merge_into, draft["contests"][0])
        print(f"merged draft with {n} findings into {args.merge_into} (needs_review: true)",
              file=sys.stderr)
        return 0
    text = json.dumps(draft, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        n = len(draft["contests"][0]["official_findings"])
        print(f"wrote draft with {n} findings to {args.out} (needs_review: true)", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
