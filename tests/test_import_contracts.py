"""Enforce the module dependency rules declared in CLAUDE.md.

Layering rules (checked per top-level package under src/rainier):
- backtest/ imports only from core/
- signals/ imports only from core/ and analysis/
- features/ imports only from core/
- trader/ imports only from core/

Additionally, the import graph between top-level packages must be acyclic.

Known pre-existing violations are pinned in the ALLOWLIST constants below so
that new violations fail while existing ones pass. Shrink the allowlists as
the violations are fixed.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "rainier"

# package -> top-level packages it may import from (itself and "core" are
# always allowed and implied)
LAYER_RULES: dict[str, frozenset[str]] = {
    "backtest": frozenset(),
    "signals": frozenset({"analysis"}),
    "features": frozenset(),
    "trader": frozenset(),
}

# Exact pre-existing violations: file (relative to src/rainier) -> the extra
# imported modules that are tolerated. Do not add entries for new code.
ALLOWLIST: dict[str, frozenset[str]] = {
    "backtest/qu100_portfolio.py": frozenset(
        {"rainier.paper.calendar", "rainier.paper.ingest"}
    ),
    "features/extractor.py": frozenset(
        {"rainier.analysis.pivots", "rainier.analysis.regime"}
    ),
}

# Pre-existing cycle-forming edges (importer package -> imported package)
# that are excluded from the acyclicity check.
CYCLE_EDGE_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {("llm_thesis", "paper")}
)


def _resolve_relative(module: str | None, level: int, file_pkg_parts: list[str]) -> str | None:
    """Resolve a relative import to an absolute rainier.* module path."""
    if level == 0:
        return module
    # file_pkg_parts is the package path of the importing module, e.g.
    # ["rainier", "backtest"] for src/rainier/backtest/foo.py
    if level > len(file_pkg_parts):
        return None
    base = file_pkg_parts[: len(file_pkg_parts) - (level - 1)]
    if module:
        base = base + module.split(".")
    return ".".join(base)


def _iter_module_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def _imports_of(path: Path) -> set[str]:
    """All absolute rainier.* modules imported by *path*."""
    rel = path.relative_to(SRC_ROOT)
    file_pkg_parts = ["rainier", *rel.parent.parts]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative(node.module, node.level, file_pkg_parts)
            if resolved:
                found.add(resolved)
    return {m for m in found if m == "rainier" or m.startswith("rainier.")}


def _top_package(module: str) -> str | None:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else None


def test_layer_rules() -> None:
    violations: list[str] = []
    for path in _iter_module_files():
        rel = path.relative_to(SRC_ROOT).as_posix()
        pkg = rel.split("/")[0]
        if pkg not in LAYER_RULES:
            continue
        allowed = LAYER_RULES[pkg] | {pkg, "core"}
        allowlisted = ALLOWLIST.get(rel, frozenset())
        for module in sorted(_imports_of(path)):
            top = _top_package(module)
            if top is None or top in allowed:
                continue
            if module in allowlisted:
                continue
            violations.append(f"{rel}: imports {module} (allowed: {sorted(allowed)})")
    assert not violations, (
        "Layering violations (see CLAUDE.md module dependency rules):\n"
        + "\n".join(violations)
    )


def test_allowlist_is_exact() -> None:
    """Every allowlisted entry must still be a real violation; prune stale ones."""
    stale: list[str] = []
    for rel, modules in ALLOWLIST.items():
        path = SRC_ROOT / rel
        if not path.exists():
            stale.append(f"{rel}: file no longer exists")
            continue
        imports = _imports_of(path)
        for module in sorted(modules):
            if module not in imports:
                stale.append(f"{rel}: no longer imports {module}")
    assert not stale, "Stale ALLOWLIST entries, remove them:\n" + "\n".join(stale)


def _package_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for path in _iter_module_files():
        rel = path.relative_to(SRC_ROOT)
        importer = rel.parts[0] if len(rel.parts) > 1 else rel.stem
        for module in _imports_of(path):
            top = _top_package(module)
            if top and top != importer:
                graph.setdefault(importer, set()).add(top)
    return graph


def test_no_import_cycles_between_top_level_packages() -> None:
    graph = _package_graph()
    for importer, imported in CYCLE_EDGE_ALLOWLIST:
        graph.get(importer, set()).discard(imported)

    cycles: list[str] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {}

    def dfs(node: str, path: list[str]) -> None:
        color[node] = GRAY
        path.append(node)
        for neighbor in sorted(graph.get(node, set())):
            state = color.get(neighbor, WHITE)
            if state == GRAY:
                cycle = path[path.index(neighbor) :] + [neighbor]
                cycles.append(" -> ".join(cycle))
            elif state == WHITE:
                dfs(neighbor, path)
        color[node] = BLACK
        path.pop()

    for node in sorted(graph):
        if color.get(node, WHITE) == WHITE:
            dfs(node, [])

    assert not cycles, (
        "Import cycles between top-level packages:\n" + "\n".join(cycles)
    )


def test_cycle_edge_allowlist_is_exact() -> None:
    graph = _package_graph()
    stale = [
        f"{importer} -> {imported}"
        for importer, imported in sorted(CYCLE_EDGE_ALLOWLIST)
        if imported not in graph.get(importer, set())
    ]
    assert not stale, (
        "Stale CYCLE_EDGE_ALLOWLIST entries, remove them:\n" + "\n".join(stale)
    )
