"""Code4rena scraper.

Clones code-423n4/*-findings GitHub repos and parses High/Medium severity
Markdown findings into raw text ready for the Structurer.

Offline usage: point `local_findings_dir` at a directory of .md files
(as in tests/fixtures/findings/) — no network required.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

import aiofiles

from audit_agent.knowledge.ingestion.structurer import extract_severity_from_markdown
from audit_agent.observability.logging import get_logger

log = get_logger("knowledge.scrapers.code4rena")

# Repos follow: https://github.com/code-423n4/{contest}-findings
C4_ORG = "code-423n4"
# Only ingest H and M severity findings (filenames contain the prefix)
_SEVERITY_PREFIXES = ("H-", "M-", "H_", "M_")


@dataclass
class RawFinding:
    text: str
    url: str
    source: str = "code4rena"
    contest: str = ""
    filename: str = ""
    suggested_severity: str = ""


@dataclass
class ScrapeResult:
    findings: list[RawFinding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    contests_processed: int = 0


class Code4renaScraper:
    """Scrapes Code4rena findings repos from GitHub or local fixture directory."""

    def __init__(
        self,
        clone_base_dir: Optional[Path] = None,
        max_concurrent: int = 3,
    ) -> None:
        self._clone_base = clone_base_dir or Path("/tmp/c4_findings_cache")
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def scrape_local_dir(self, findings_dir: Path) -> ScrapeResult:
        """Load all .md files from a local directory — offline, no network."""
        result = ScrapeResult()
        md_files = sorted(findings_dir.glob("*.md"))
        for path in md_files:
            raw = await self._read_md_file(path, contest="fixture")
            if raw:
                result.findings.append(raw)
        result.contests_processed = 1
        log.info("code4rena.scrape_local", files=len(result.findings))
        return result

    async def scrape_contests(self, contest_names: list[str]) -> ScrapeResult:
        """Clone and scrape a list of contest repo names from GitHub.

        Each name should be the repo suffix, e.g. "2023-07-basin".
        The full repo URL is: https://github.com/code-423n4/2023-07-basin-findings
        """
        result = ScrapeResult()
        tasks = [self._scrape_one_contest(name, result) for name in contest_names]
        await asyncio.gather(*tasks, return_exceptions=True)
        return result

    async def scrape_repo(self, repo_path: Path) -> ScrapeResult:
        """Scrape an already-cloned findings repo directory."""
        result = ScrapeResult()
        for path in self._iter_hm_files(repo_path):
            contest = repo_path.name.removesuffix("-findings")
            raw = await self._read_md_file(path, contest=contest)
            if raw:
                result.findings.append(raw)
        result.contests_processed = 1
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _scrape_one_contest(self, contest: str, result: ScrapeResult) -> None:
        async with self._semaphore:
            repo_name = f"{contest}-findings"
            clone_dir = self._clone_base / repo_name
            try:
                await self._clone_or_pull(repo_name, clone_dir)
                sub = await self.scrape_repo(clone_dir)
                result.findings.extend(sub.findings)
                result.contests_processed += 1
                log.info("code4rena.contest_done", contest=contest, n=len(sub.findings))
            except Exception as exc:
                err = f"contest={contest}: {exc}"
                result.errors.append(err)
                log.warning("code4rena.contest_error", error=err)

    async def _clone_or_pull(self, repo_name: str, dest: Path) -> None:
        url = f"https://github.com/{C4_ORG}/{repo_name}.git"
        if dest.exists():
            log.debug("code4rena.git_pull", dest=str(dest))
            await asyncio.to_thread(
                subprocess.run,
                ["git", "-C", str(dest), "pull", "--quiet"],
                check=True, capture_output=True,
            )
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            log.debug("code4rena.git_clone", url=url)
            await asyncio.to_thread(
                subprocess.run,
                ["git", "clone", "--depth=1", "--quiet", url, str(dest)],
                check=True, capture_output=True,
            )

    @staticmethod
    def _iter_hm_files(repo_dir: Path) -> list[Path]:
        """Return all H/M severity markdown files in the repo."""
        results: list[Path] = []
        for md in repo_dir.rglob("*.md"):
            name = md.name
            if any(name.startswith(p) for p in _SEVERITY_PREFIXES):
                results.append(md)
        return sorted(results)

    async def _read_md_file(self, path: Path, contest: str) -> Optional[RawFinding]:
        try:
            async with aiofiles.open(path, encoding="utf-8", errors="replace") as f:
                text = await f.read()
        except OSError as e:
            log.warning("code4rena.read_error", path=str(path), error=str(e))
            return None

        if len(text.strip()) < 100:
            return None

        severity = extract_severity_from_markdown(text)
        # Build a best-effort GitHub URL from the path
        url = _infer_github_url(path, contest)
        return RawFinding(
            text=text,
            url=url,
            contest=contest,
            filename=path.name,
            suggested_severity=severity,
        )


def _main() -> None:
    import argparse
    import json

    from audit_agent.observability.logging import configure_logging

    parser = argparse.ArgumentParser(
        description="Code4rena findings scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Scrape a local directory of .md files (offline):\n"
            "  uv run python -m audit_agent.knowledge.ingestion.scrapers.code4rena"
            " --local tests/fixtures/findings/\n\n"
            "  # Clone and scrape contests from GitHub:\n"
            "  uv run python -m audit_agent.knowledge.ingestion.scrapers.code4rena"
            " --contests 2023-07-basin 2024-01-salty\n"
        ),
    )
    parser.add_argument(
        "--local", metavar="DIR",
        help="Scrape all H/M .md files from a local directory (no network)",
    )
    parser.add_argument(
        "--contests", metavar="NAME", nargs="+",
        help="Contest repo suffixes to clone from GitHub (e.g. 2023-07-basin)",
    )
    parser.add_argument(
        "--out", metavar="FILE",
        help="Write findings JSON to FILE instead of stdout",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    if not args.local and not args.contests:
        parser.error("Provide --local <dir> or --contests <name...>")

    scraper = Code4renaScraper()

    async def _run() -> ScrapeResult:
        if args.local:
            return await scraper.scrape_local_dir(Path(args.local))
        return await scraper.scrape_contests(args.contests)

    result = asyncio.run(_run())

    payload = {
        "contests_processed": result.contests_processed,
        "findings_count": len(result.findings),
        "errors": result.errors,
        "findings": [
            {
                "contest": f.contest,
                "filename": f.filename,
                "severity": f.suggested_severity,
                "url": f.url,
                "text_preview": f.text[:200],
            }
            for f in result.findings
        ],
    }

    out_str = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(out_str, encoding="utf-8")
        print(f"Wrote {len(result.findings)} findings to {args.out}")
    else:
        print(out_str)

    if result.errors:
        import sys
        print(f"\n{len(result.errors)} error(s):", file=sys.stderr)
        for err in result.errors:
            print(f"  {err}", file=sys.stderr)


def _infer_github_url(path: Path, contest: str) -> str:
    """Best-effort GitHub URL — works for cloned repos."""
    # path example: /tmp/c4_findings_cache/2023-07-basin-findings/data/H-01.md
    try:
        parts = path.parts
        # Find the repo root by looking for the -findings suffix
        for i, part in enumerate(parts):
            if part.endswith("-findings") or part == "fixture":
                rel = "/".join(parts[i + 1:])
                repo = part if part != "fixture" else f"{contest}-findings"
                return f"https://github.com/{C4_ORG}/{repo}/blob/main/{rel}"
    except Exception:
        pass
    return ""


if __name__ == "__main__":
    _main()
