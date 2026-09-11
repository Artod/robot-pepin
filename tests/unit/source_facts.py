"""Facts about a source file read from its syntax tree, for the contract tests.

A contract on a launch file or a node must not pass because a comment mentions the right words,
nor fail because a line was re-wrapped: these helpers answer "what does the file call, import,
assign, name" from ``ast``, and ``ast.unparse`` renders a node in one canonical spelling (single
quotes, one space after a comma) so a test can hold an exact call or value as a string. Shell
scripts have no tree; :func:`shell_commands` at least joins their continued lines.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@functools.cache
def tree(rel: str) -> ast.Module:
    """The parsed module at ``rel`` (a path under the repository), parsed once per session."""
    return ast.parse((REPO / rel).read_text(), filename=rel)


def dotted(node: ast.expr) -> str:
    """A callee as written: ``name``, ``obj.attr``, ``a.b.c``, ``super().__init__``; empty for
    anything that is not a name chain (a subscript, a lambda)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    if isinstance(node, ast.Call):
        base = dotted(node.func)
        return f"{base}()" if base else ""
    return ""


def calls(module: ast.AST) -> set[str]:
    """Every callee of the file, as :func:`dotted` writes it."""
    return {dotted(n.func) for n in ast.walk(module) if isinstance(n, ast.Call)} - {""}


def calls_to(module: ast.AST, callee: str) -> list[ast.Call]:
    """The calls whose callee is exactly ``callee`` (:func:`dotted` spelling), in file order."""
    return [n for n in ast.walk(module) if isinstance(n, ast.Call) and dotted(n.func) == callee]


def unparsed(module: ast.AST, kind: type[ast.AST]) -> set[str]:
    """Every node of ``kind`` rendered by ``ast.unparse``: an exact call, tuple or f-string
    to hold as a string, in the canonical spelling (single quotes)."""
    return {ast.unparse(n) for n in ast.walk(module) if isinstance(n, kind)}


def keywords(call: ast.Call) -> dict[str, ast.expr]:
    """The keyword arguments of a call by name (``**spread`` entries left out)."""
    return {k.arg: k.value for k in call.keywords if k.arg is not None}


def strings(module: ast.AST) -> set[str]:
    """Every string literal of the file, the constant parts of f-strings included; never a
    comment."""
    return {
        n.value
        for n in ast.walk(module)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def names(module: ast.AST) -> set[str]:
    """Every bare name the file reads or writes."""
    return {n.id for n in ast.walk(module) if isinstance(n, ast.Name)}


def imported(module: ast.AST) -> set[str]:
    """Every name an import statement binds (``from a import b`` gives ``b``, ``import a.b``
    gives ``a.b``), aliases as written after ``as``."""
    found: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found |= {alias.asname or alias.name for alias in node.names}
    return found


def assignments(module: ast.Module) -> dict[str, str]:
    """The module-level ``NAME = value`` statements, the value rendered by ``ast.unparse``."""
    found: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                found[target.id] = ast.unparse(node.value)
    return found


def dict_items(module: ast.AST) -> dict[str, set[str]]:
    """Over every dict literal of the file: each string key mapped to the set of values it is
    given anywhere, rendered by ``ast.unparse`` (a name stays a name, a literal a literal)."""
    found: dict[str, set[str]] = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                found.setdefault(key.value, set()).add(ast.unparse(value))
    return found


def shell_commands(text: str) -> list[str]:
    """A shell script's lines with backslash continuations joined, so a command wrapped over
    several lines is one string (a docker run with its mounts, say)."""
    joined: list[str] = []
    pending = ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            pending += line.rstrip()[:-1]
            continue
        joined.append(pending + line)
        pending = ""
    if pending:
        joined.append(pending)
    return joined
