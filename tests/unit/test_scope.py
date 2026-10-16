"""Unit tests for scope resolution module."""

from __future__ import annotations

from pathlib import Path

import pytest

from audit_agent.scope import (
    ScopeResult,
    _enumerate_all_sol,
    _filter_non_library,
    _parse_scope_file,
    _resolve_import_path,
    _resolve_imports,
    parse_remappings,
    resolve_scope,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write(path: Path, content: str = "// SPDX-License-Identifier: MIT\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. resolve_scope — explicit scope file
# ─────────────────────────────────────────────────────────────────────────────

class TestExplicitScopeFile:
    def test_explicit_scope_file_sets_in_scope(self, tmp_path: Path) -> None:
        vault = _write(tmp_path / "src" / "Vault.sol")
        token = _write(tmp_path / "src" / "Token.sol")
        _write(tmp_path / "src" / "MockOracle.sol")

        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("src/Vault.sol\nsrc/Token.sol\n")

        result = resolve_scope(tmp_path, scope_file=scope_txt)

        assert result.scope_source == "explicit_file"
        assert vault.resolve() in result.in_scope
        assert token.resolve() in result.in_scope
        assert len(result.in_scope) == 2

    def test_explicit_scope_excludes_unlisted(self, tmp_path: Path) -> None:
        _write(tmp_path / "Vault.sol")
        oracle = _write(tmp_path / "Oracle.sol")
        scope_txt = tmp_path / "my_scope.txt"
        scope_txt.write_text("Vault.sol\n")

        result = resolve_scope(tmp_path, scope_file=scope_txt)

        assert oracle.resolve() not in result.in_scope
        assert oracle.resolve() in result.out_of_scope or oracle.resolve() in result.context_files

    def test_explicit_scope_missing_file_logs_warning(self, tmp_path: Path) -> None:
        _write(tmp_path / "Real.sol")
        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("Real.sol\nDoesNotExist.sol\n")

        result = resolve_scope(tmp_path, scope_file=scope_txt)
        # Should still include the real file, just skip the missing one
        assert len(result.in_scope) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. resolve_scope — auto-detected scope.txt
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoDetectedScope:
    def test_auto_detect_scope_txt_in_target_dir(self, tmp_path: Path) -> None:
        vault = _write(tmp_path / "Vault.sol")
        _write(tmp_path / "Other.sol")
        (tmp_path / "scope.txt").write_text("Vault.sol\n")

        result = resolve_scope(tmp_path)

        assert result.scope_source == "auto_detected"
        assert vault.resolve() in result.in_scope
        assert len(result.in_scope) == 1

    def test_auto_detect_takes_priority_over_all_non_library(self, tmp_path: Path) -> None:
        _write(tmp_path / "A.sol")
        vault = _write(tmp_path / "B.sol")
        (tmp_path / "scope.txt").write_text("B.sol\n")

        result = resolve_scope(tmp_path)

        assert result.scope_source == "auto_detected"
        assert vault.resolve() in result.in_scope


# ─────────────────────────────────────────────────────────────────────────────
# 3. resolve_scope — fallback to all non-library
# ─────────────────────────────────────────────────────────────────────────────

class TestAllNonLibraryFallback:
    def test_no_scope_returns_all_non_library(self, tmp_path: Path) -> None:
        src = _write(tmp_path / "src" / "Token.sol")
        _write(tmp_path / "lib" / "OZ" / "ERC20.sol")

        result = resolve_scope(tmp_path)

        assert result.scope_source == "all_non_library"
        assert src.resolve() in result.in_scope

    def test_no_scope_excludes_test_files(self, tmp_path: Path) -> None:
        prod = _write(tmp_path / "Token.sol")
        _write(tmp_path / "test" / "Token.t.sol")
        _write(tmp_path / "Token.t.sol")

        result = resolve_scope(tmp_path)

        assert prod.resolve() in result.in_scope
        for p in result.in_scope:
            assert not str(p).endswith(".t.sol"), f"test file in scope: {p}"

    def test_no_scope_excludes_script_files(self, tmp_path: Path) -> None:
        prod = _write(tmp_path / "Vault.sol")
        _write(tmp_path / "scripts" / "Deploy.s.sol")
        _write(tmp_path / "Deploy.s.sol")

        result = resolve_scope(tmp_path)

        assert prod.resolve() in result.in_scope
        for p in result.in_scope:
            assert not str(p).endswith(".s.sol"), f"script file in scope: {p}"

    def test_no_scope_empty_dir_gives_empty_result(self, tmp_path: Path) -> None:
        result = resolve_scope(tmp_path)
        assert result.in_scope == []
        assert result.context_files == []


# ─────────────────────────────────────────────────────────────────────────────
# 4. Import resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestImportResolution:
    def test_scope_resolves_direct_import(self, tmp_path: Path) -> None:
        base = _write(tmp_path / "Base.sol")
        vault = _write(
            tmp_path / "Vault.sol",
            content='// SPDX-License-Identifier: MIT\nimport "./Base.sol";\n',
        )
        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("Vault.sol\n")

        result = resolve_scope(tmp_path, scope_file=scope_txt)

        assert vault.resolve() in result.in_scope
        assert base.resolve() in result.context_files

    def test_scope_recursive_imports(self, tmp_path: Path) -> None:
        c = _write(tmp_path / "C.sol")
        b = _write(tmp_path / "B.sol", content='import "./C.sol";\n')
        a = _write(tmp_path / "A.sol", content='import "./B.sol";\n')

        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("A.sol\n")

        result = resolve_scope(tmp_path, scope_file=scope_txt)

        assert a.resolve() in result.in_scope
        assert b.resolve() in result.context_files
        assert c.resolve() in result.context_files

    def test_context_files_not_duplicated_in_in_scope(self, tmp_path: Path) -> None:
        base = _write(tmp_path / "Base.sol")
        vault = _write(tmp_path / "Vault.sol", content='import "./Base.sol";\n')

        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("Vault.sol\nBase.sol\n")

        result = resolve_scope(tmp_path, scope_file=scope_txt)

        # Base.sol is in_scope, so it must NOT appear in context_files
        assert base.resolve() in result.in_scope
        assert base.resolve() not in result.context_files

    def test_unresolvable_import_does_not_crash(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "Vault.sol",
            content='import "@openzeppelin/contracts/token/ERC20/ERC20.sol";\n',
        )
        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("Vault.sol\n")

        result = resolve_scope(tmp_path, scope_file=scope_txt)
        assert len(result.in_scope) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. parse_remappings
# ─────────────────────────────────────────────────────────────────────────────

class TestParseRemappings:
    def test_remappings_txt(self, tmp_path: Path) -> None:
        (tmp_path / "remappings.txt").write_text(
            "@openzeppelin/=lib/openzeppelin-contracts/\n"
            "solady/=lib/solady/src/\n"
        )
        remappings = parse_remappings(tmp_path)
        assert remappings["@openzeppelin/"] == "lib/openzeppelin-contracts/"
        assert remappings["solady/"] == "lib/solady/src/"

    def test_remappings_txt_ignores_comments(self, tmp_path: Path) -> None:
        (tmp_path / "remappings.txt").write_text(
            "# comment\n"
            "forge-std/=lib/forge-std/src/\n"
        )
        remappings = parse_remappings(tmp_path)
        assert "forge-std/" in remappings
        assert len(remappings) == 1

    def test_no_remappings_files_returns_empty(self, tmp_path: Path) -> None:
        remappings = parse_remappings(tmp_path)
        assert remappings == {}

    def test_foundry_toml_remappings(self, tmp_path: Path) -> None:
        (tmp_path / "foundry.toml").write_text(
            '[profile.default]\n'
            'remappings = ["@oz/=lib/openzeppelin/", "ds-test/=lib/ds-test/"]\n'
        )
        remappings = parse_remappings(tmp_path)
        assert "@oz/" in remappings
        assert "ds-test/" in remappings


# ─────────────────────────────────────────────────────────────────────────────
# 6. Scope with remappings
# ─────────────────────────────────────────────────────────────────────────────

class TestScopeWithRemappings:
    def test_remapping_resolves_import(self, tmp_path: Path) -> None:
        oz_erc20 = _write(tmp_path / "lib" / "openzeppelin" / "ERC20.sol")
        _write(
            tmp_path / "Token.sol",
            content='import "@oz/ERC20.sol";\n',
        )
        (tmp_path / "remappings.txt").write_text("@oz/=lib/openzeppelin/\n")
        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("Token.sol\n")

        result = resolve_scope(tmp_path, scope_file=scope_txt)

        assert oz_erc20.resolve() in result.context_files


# ─────────────────────────────────────────────────────────────────────────────
# 7. _parse_scope_file edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestParseScopeFile:
    def test_ignores_blank_lines(self, tmp_path: Path) -> None:
        _write(tmp_path / "A.sol")
        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("\nA.sol\n\n")
        paths = _parse_scope_file(scope_txt, tmp_path)
        assert len(paths) == 1

    def test_ignores_comment_lines(self, tmp_path: Path) -> None:
        _write(tmp_path / "A.sol")
        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("# comment\nA.sol\n")
        paths = _parse_scope_file(scope_txt, tmp_path)
        assert len(paths) == 1

    def test_dotslash_prefix_resolved(self, tmp_path: Path) -> None:
        _write(tmp_path / "src" / "A.sol")
        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("./src/A.sol\n")
        paths = _parse_scope_file(scope_txt, tmp_path)
        assert len(paths) == 1

    def test_nonexistent_file_skipped(self, tmp_path: Path) -> None:
        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("DoesNotExist.sol\n")
        paths = _parse_scope_file(scope_txt, tmp_path)
        assert paths == []

    def test_non_sol_file_skipped(self, tmp_path: Path) -> None:
        _write(tmp_path / "config.json")
        scope_txt = tmp_path / "scope.txt"
        scope_txt.write_text("config.json\n")
        paths = _parse_scope_file(scope_txt, tmp_path)
        assert paths == []
