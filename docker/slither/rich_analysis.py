#!/usr/bin/env python3
"""Rich Slither analysis — runs INSIDE the slither Docker image.

Unlike `slither . --json` (which only emits functions that appear in detector
findings), this script walks the full Slither Python API and emits a complete
code model in the *normalized* JSON format that
`audit_agent.knowledge.graph.builders.parse_slither_json` already understands,
enriched with data the detector JSON never had:

  • EVERY function (public / external / internal / private / inherited), not
    just the ones a detector flagged.
  • Real internal call graph (resolved through inheritance) → `calls`.
  • Real state reads / writes per function → `reads` / `writes`.
  • External-call targets with the *state variable* that holds the callee
    (e.g. `priceOracle.getPrice(...)`) — the substrate for value-flow / oracle
    reasoning.
  • Taint: which written state variables and which external-call sites are
    reachable from user-controlled input (`is_tainted`).

Output: a single JSON file. Stdout is kept clean (Slither logs go to stderr).

Usage (inside container):
    python3 rich_analysis.py <target_dir_or_file> <output_json_path>
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

ORACLE_FN_NAMES = {
    "latestrounddata", "latestanswer", "getrounddata", "getprice",
    "consult", "getamountout", "price", "peek", "read", "current",
}
ORACLE_DEST_HINTS = ("oracle", "price", "feed", "aggregator", "twap")
PRIV_MODIFIER_HINTS = ("only", "auth", "admin", "owner", "governance", "role", "restricted")


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _contract_of(fn) -> str:
    c = getattr(fn, "contract_declarer", None) or getattr(fn, "contract", None)
    return c.name if c is not None else "Unknown"


def _func_id(fn) -> str:
    return f"fn-{_contract_of(fn)}-{getattr(fn, 'full_name', getattr(fn, 'name', '?'))}"


def _sv_id(sv) -> str:
    c = getattr(sv, "contract", None)
    cname = c.name if c is not None else "Unknown"
    return f"sv-{cname}-{sv.name}"


def _state_mutability(fn) -> str:
    if _safe(lambda: fn.payable):
        return "payable"
    if _safe(lambda: fn.pure):
        return "pure"
    if _safe(lambda: fn.view):
        return "view"
    return "nonpayable"


def _natspec(obj) -> dict[str, str]:
    notice, dev = "", ""
    try:
        # Slither exposes raw NatSpec lines on the source comments; the parsed
        # structure is not uniformly available across versions, so we keep it
        # best-effort and let the LLM read bodies for real semantics.
        doc = getattr(obj, "documentation", None)
        if isinstance(doc, str):
            dev = doc[:2000]
    except Exception:
        pass
    return {"notice": notice, "dev": dev}


def _read_source(fn, file_cache: dict[str, list[str]]) -> tuple[dict[str, Any], str]:
    """Return (source_location dict, source_code text) for a function."""
    sm = getattr(fn, "source_mapping", None)
    if sm is None:
        return {"file": "", "start_line": 0, "end_line": 0}, ""
    lines = list(getattr(sm, "lines", []) or [])
    rel = ""
    try:
        rel = sm.filename.relative or sm.filename.short or ""
    except Exception:
        rel = ""
    start = lines[0] if lines else 0
    end = lines[-1] if lines else 0
    src_text = ""
    if rel and start and end:
        if rel not in file_cache:
            try:
                with open(sm.filename.absolute, errors="ignore") as fh:
                    file_cache[rel] = fh.readlines()
            except Exception:
                file_cache[rel] = []
        flines = file_cache[rel]
        if flines:
            src_text = "".join(flines[start - 1:end])[:8000]
    return {"file": rel, "start_line": start, "end_line": end}, src_text


def _classify_external_calls(fn) -> tuple[list[dict], list[dict], bool, bool, bool]:
    """Return (external_calls, external_targets, has_external, handles_value, reads_oracle)."""
    from slither.slithir.operations import (
        HighLevelCall, LowLevelCall, Send, Transfer,
    )
    try:
        from slither.analyses.data_dependency.data_dependency import is_tainted
    except Exception:
        is_tainted = None

    contract_ctx = getattr(fn, "contract", None)
    external_calls: list[dict] = []
    targets: list[dict] = []
    has_external = False
    handles_value = False
    reads_oracle = False

    def _line(ir) -> int:
        node = getattr(ir, "node", None)
        sm = getattr(node, "source_mapping", None) if node else None
        ls = list(getattr(sm, "lines", []) or []) if sm else []
        return ls[0] if ls else 0

    def _tainted(var) -> bool:
        if is_tainted is None or var is None or contract_ctx is None:
            return False
        try:
            return bool(is_tainted(var, contract_ctx))
        except Exception:
            return False

    for ir in _safe(lambda: list(fn.all_slithir_operations()), []) or []:
        if isinstance(ir, HighLevelCall):
            has_external = True
            dest = getattr(ir, "destination", None)
            dest_name = getattr(dest, "name", None) or str(dest)
            callee = getattr(ir, "function", None)
            callee_name = getattr(callee, "name", None) or str(callee)
            if getattr(ir, "call_value", None) is not None:
                handles_value = True
            if (callee_name or "").lower() in ORACLE_FN_NAMES or any(
                h in (dest_name or "").lower() for h in ORACLE_DEST_HINTS
            ):
                reads_oracle = True
            external_calls.append({
                "id": f"hlc-{dest_name}-{callee_name}-{_line(ir)}",
                "call_type": "call",
                "target_type": "user_controlled" if _tainted(dest) else "known_contract",
                "line": _line(ir),
            })
            targets.append({
                "dest": dest_name, "fn": callee_name,
                "kind": "high_level", "line": _line(ir),
            })
        elif isinstance(ir, LowLevelCall):
            has_external = True
            tc = (getattr(ir, "type_call", "call") or "call").lower()
            if getattr(ir, "call_value", None) is not None:
                handles_value = True
            dest = getattr(ir, "destination", None)
            dest_name = getattr(dest, "name", None) or str(dest)
            external_calls.append({
                "id": f"llc-{dest_name}-{_line(ir)}",
                "call_type": tc if tc in ("call", "delegatecall", "staticcall") else "call",
                "target_type": "user_controlled" if _tainted(dest) else "known_contract",
                "line": _line(ir),
            })
            targets.append({
                "dest": dest_name, "fn": "<low_level>",
                "kind": tc, "line": _line(ir),
            })
        elif isinstance(ir, (Send, Transfer)):
            has_external = True
            handles_value = True
            dest = getattr(ir, "destination", None)
            dest_name = getattr(dest, "name", None) or str(dest)
            kind = "send" if isinstance(ir, Send) else "transfer"
            external_calls.append({
                "id": f"{kind}-{dest_name}-{_line(ir)}",
                "call_type": kind,
                "target_type": "user_controlled" if _tainted(dest) else "known_contract",
                "line": _line(ir),
            })

    return external_calls, targets, has_external, handles_value, reads_oracle


def _internal_calls(fn, known_func_ids: set[str]) -> list[str]:
    from slither.slithir.operations import InternalCall, LibraryCall
    out: list[str] = []
    seen: set[str] = set()
    for ir in _safe(lambda: list(fn.all_slithir_operations()), []) or []:
        if isinstance(ir, (InternalCall, LibraryCall)):
            callee = getattr(ir, "function", None)
            if callee is None or not hasattr(callee, "full_name"):
                continue
            cid = _func_id(callee)
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out


def _tainted_writes(fn) -> list[str]:
    try:
        from slither.analyses.data_dependency.data_dependency import is_tainted
    except Exception:
        return []
    ctx = getattr(fn, "contract", None)
    if ctx is None:
        return []
    out: list[str] = []
    for sv in _safe(lambda: list(fn.state_variables_written), []) or []:
        try:
            if is_tainted(sv, ctx):
                out.append(sv.name)
        except Exception:
            pass
    return out


def _is_privileged(fn) -> bool:
    for m in _safe(lambda: list(fn.modifiers), []) or []:
        if any(h in (m.name or "").lower() for h in PRIV_MODIFIER_HINTS):
            return True
    return False


def build_model(target: str) -> dict[str, Any]:
    from slither import Slither

    sl = Slither(target)
    file_cache: dict[str, list[str]] = {}

    # First pass: collect every function ID so call resolution can validate targets.
    known_func_ids: set[str] = set()
    for c in sl.contracts:
        for fn in _safe(lambda: list(c.functions_declared), []) or []:
            known_func_ids.add(_func_id(fn))

    contracts_out: list[dict[str, Any]] = []
    for c in sl.contracts:
        try:
            functions_out: list[dict[str, Any]] = []
            for fn in _safe(lambda: list(c.functions_declared), []) or []:
                try:
                    loc, src_text = _read_source(fn, file_cache)
                    ext_calls, ext_targets, has_ext, handles_val, reads_oracle = \
                        _classify_external_calls(fn)
                    functions_out.append({
                        "id": _func_id(fn),
                        "name": fn.name,
                        "signature": getattr(fn, "full_name", fn.name),
                        "visibility": _safe(lambda: fn.visibility) or "internal",
                        "state_mutability": _state_mutability(fn),
                        "is_constructor": bool(_safe(lambda: fn.is_constructor)),
                        "modifiers": [m.name for m in (_safe(lambda: list(fn.modifiers), []) or [])],
                        "parameters": [
                            {"name": p.name, "type": str(p.type)}
                            for p in (_safe(lambda: list(fn.parameters), []) or [])
                        ],
                        "returns": [
                            {"name": getattr(p, "name", ""), "type": str(p.type)}
                            for p in (_safe(lambda: list(fn.returns), []) or [])
                        ],
                        "source_location": loc,
                        "source_code": src_text,
                        "natspec": _natspec(fn),
                        "calls": _internal_calls(fn, known_func_ids),
                        "external_calls": ext_calls,
                        "external_targets": ext_targets,
                        "reads": [_sv_id(v) for v in (_safe(lambda: list(fn.state_variables_read), []) or [])],
                        "writes": [_sv_id(v) for v in (_safe(lambda: list(fn.state_variables_written), []) or [])],
                        "tainted_state_writes": _tainted_writes(fn),
                        "emits": [],
                        "has_external_calls": has_ext,
                        "handles_value": handles_val or bool(_safe(lambda: fn.payable)),
                        "reads_oracle": reads_oracle,
                        "is_privileged": _is_privileged(fn),
                    })
                except Exception:
                    sys.stderr.write(f"[rich_analysis] fn fail {c.name}.{getattr(fn,'name','?')}\n")
                    sys.stderr.write(traceback.format_exc())

            state_vars_out = []
            for sv in _safe(lambda: list(c.state_variables_declared), []) or []:
                state_vars_out.append({
                    "id": _sv_id(sv),
                    "name": sv.name,
                    "type": str(_safe(lambda: sv.type) or ""),
                    "visibility": _safe(lambda: sv.visibility) or "internal",
                    "is_constant": bool(_safe(lambda: sv.is_constant)),
                    "is_immutable": bool(_safe(lambda: sv.is_immutable)),
                    "initial_value": "",
                })

            contracts_out.append({
                "id": f"contract-{c.name}",
                "name": c.name,
                "source_file": _safe(lambda: c.source_mapping.filename.relative) or "",
                "kind": "interface" if _safe(lambda: c.is_interface) else "contract",
                "is_abstract": bool(_safe(lambda: c.is_abstract)),
                "is_interface": bool(_safe(lambda: c.is_interface)),
                "inherits": [p.name for p in (_safe(lambda: list(c.inheritance), []) or [])],
                "natspec": _natspec(c),
                "state_variables": state_vars_out,
                "functions": functions_out,
                "modifiers": [
                    {"id": f"mod-{c.name}-{m.name}", "name": m.name, "source_code": ""}
                    for m in (_safe(lambda: list(c.modifiers_declared), []) or [])
                ],
                "events": [],
            })
        except Exception:
            sys.stderr.write(f"[rich_analysis] contract fail {getattr(c,'name','?')}\n")
            sys.stderr.write(traceback.format_exc())

    return {
        "format": "rich",
        "slither_version": "0.11.5",
        "contracts": contracts_out,
    }


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write("usage: rich_analysis.py <target> <output.json>\n")
        return 2
    target, out_path = sys.argv[1], sys.argv[2]
    try:
        model = build_model(target)
    except Exception as exc:
        sys.stderr.write(f"[rich_analysis] FATAL: {exc}\n")
        sys.stderr.write(traceback.format_exc())
        # Emit an empty-but-valid model so the host can distinguish "ran but
        # found nothing" from "file missing".
        model = {"format": "rich", "error": str(exc), "contracts": []}
    with open(out_path, "w") as fh:
        json.dump(model, fh)
    n_fns = sum(len(c.get("functions", [])) for c in model.get("contracts", []))
    sys.stderr.write(
        f"[rich_analysis] wrote {len(model.get('contracts', []))} contracts, "
        f"{n_fns} functions to {out_path}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
