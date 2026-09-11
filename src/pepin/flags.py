"""Feature flags: a node's live switches declared once, in one table, portable to any robot.

CLAUDE.md rule 19: every behaviour ships behind a switch with the old behaviour reachable, the
switch is printed in the node's report line, and a flag's name is the feature's name. This
module is the ROS-free half. :class:`Flag` says what one switch is — its kind, its default,
the range of a number or the choices of a choice, whether it changes live; :class:`FlagSet` is
a node's table with the current values: it validates every change by kind and answers with a
reason, and renders the table for the report line (:meth:`FlagSet.state`), for the README
(:meth:`FlagSet.describe`) and for a tool. The ROS half, ``pepin_bringup.node_kit.Switches``,
declares the same table as parameters with descriptors and routes ``ros2 param set`` through
:meth:`FlagSet.set`; another robot writes another adapter over the same table.

Four kinds, told apart by the default and the keywords: a bool (``Flag("align", True)``), a
choice of strings (``Flag("depth_backend", "local", choices=("remote", "local", "auto"))``), a
number with an optional range (``Flag("min_weight", 2.0, range=(0.0, 100.0))``; an int default
makes an integer flag, a float default a double), and a list of choices (``Flag("sources",
("lidar",), choices=("lidar", "depth"))``, written ``lidar,depth`` on the wire). ``env`` names
an environment variable that overrides the default at start (``PEPIN_DEPTH_BACKEND``);
``live=False`` declares a flag the adapter refuses to change once the node runs.
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Kind = Literal["bool", "choice", "number", "list"]

ON_WORDS = ("true", "on", "1", "yes")
OFF_WORDS = ("false", "off", "0", "no")
COLUMNS = ("flag", "kind", "default", "live", "description")


@dataclass(frozen=True)
class Flag:
    """One switch: its name (the feature's), its default, what values it takes, whether it
    changes while the node runs, and the sentence a person reads about it."""

    name: str
    default: Any
    description: str = ""
    choices: tuple[str, ...] = ()
    range: tuple[float, float] | None = None
    env: str | None = None
    live: bool = True
    kind: Kind = field(init=False)

    def __post_init__(self) -> None:
        default = self.default
        if isinstance(default, (list, tuple)):
            default = tuple(default)
            object.__setattr__(self, "default", default)
        if not self.name.isidentifier():
            raise ValueError(f"{self.name!r}: a flag's name is an identifier")
        if isinstance(default, bool):
            kind: Kind = "bool"
        elif isinstance(default, (int, float)):
            kind = "number"
        elif isinstance(default, str):
            kind = "choice"
        elif isinstance(default, tuple):
            kind = "list"
        else:
            raise ValueError(
                f"{self.name}: a flag is a bool, a choice of strings, a number or a list of"
                f" choices, not {type(default).__name__}"
            )
        object.__setattr__(self, "kind", kind)
        if kind in ("choice", "list"):
            if not self.choices or not all(isinstance(c, str) and c for c in self.choices):
                raise ValueError(f"{self.name}: a {kind} flag needs its choices")
            if len(set(self.choices)) != len(self.choices):
                raise ValueError(f"{self.name}: the choices repeat")
        elif self.choices:
            raise ValueError(f"{self.name}: choices belong to a choice or a list flag")
        if self.range is not None:
            if kind != "number":
                raise ValueError(f"{self.name}: a range belongs to a number flag")
            lo, hi = self.range
            if not lo <= hi:
                raise ValueError(f"{self.name}: the range {lo}..{hi} is empty")
        object.__setattr__(self, "default", self.parse(default))  # a default outside its own
        # range or choices is refused here, at import, not at the first set

    @property
    def integer(self) -> bool:
        """Whether a number flag counts (an int default) rather than measures (a float one)."""
        return self.kind == "number" and isinstance(self.default, int)

    @property
    def wire_type(self) -> str:
        """The type a transport carries the flag as: ``bool``, ``integer``, ``double`` or
        ``string`` (a choice is its word, a list its words joined by commas)."""
        if self.kind == "bool":
            return "bool"
        if self.kind == "number":
            return "integer" if self.integer else "double"
        return "string"

    def parse(self, value: Any) -> Any:
        """``value`` as this flag holds it — a bool, an int or float, a choice, a tuple of
        choices — from the value itself or its text (``on``, ``2.5``, ``lidar,depth``);
        ``ValueError`` with the reason when it is not one of the flag's values."""
        if self.kind == "bool":
            return self._parse_bool(value)
        if self.kind == "number":
            return self._parse_number(value)
        if self.kind == "choice":
            if not isinstance(value, str) or value not in self.choices:
                raise ValueError(f"{self.name}: {value!r} is not one of {', '.join(self.choices)}")
            return value
        return self._parse_list(value)

    def _parse_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            word = value.strip().lower()
            if word in ON_WORDS:
                return True
            if word in OFF_WORDS:
                return False
        raise ValueError(f"{self.name}: {value!r} is not on or off")

    def _parse_number(self, value: Any) -> int | float:
        number: int | float
        if isinstance(value, bool):
            raise ValueError(f"{self.name}: {value!r} is not a number")
        if isinstance(value, str):
            try:
                number = int(value) if self.integer else float(value)
            except ValueError as exc:
                raise ValueError(f"{self.name}: {value!r} is not a number") from exc
        elif isinstance(value, (int, float)):
            number = value
        else:
            raise ValueError(f"{self.name}: {value!r} is not a number")
        if self.integer:
            if isinstance(number, float):
                if not number.is_integer():
                    raise ValueError(f"{self.name}: {value!r} is not a whole number")
                number = int(number)
        else:
            number = float(number)
        if self.range is not None:
            lo, hi = self.range
            if not lo <= number <= hi:
                raise ValueError(f"{self.name}: {number} is outside {self._range_text()}")
        return number

    def _parse_list(self, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            items: Sequence[Any] = [w.strip() for w in value.split(",") if w.strip()]
        elif isinstance(value, (list, tuple)):
            items = value
        else:
            raise ValueError(f"{self.name}: {value!r} is not a list of {', '.join(self.choices)}")
        chosen: list[str] = []
        for item in items:
            if not isinstance(item, str) or item not in self.choices:
                raise ValueError(f"{self.name}: {item!r} is not one of {', '.join(self.choices)}")
            if item not in chosen:
                chosen.append(item)
        return tuple(chosen)

    def render(self, value: Any) -> str:
        """``value`` as the report line prints it: ``on``/``off``, the number, the choice, the
        list's words joined by commas (``(none)`` when empty)."""
        if self.kind == "bool":
            return "on" if value else "off"
        if self.kind == "list":
            return ",".join(value) or "(none)"
        return str(value)

    def wire(self, value: Any) -> Any:
        """``value`` as a transport carries it (:attr:`wire_type`): a list as one string."""
        return ",".join(value) if self.kind == "list" else value

    def kind_text(self) -> str:
        """The kind for a table: ``bool``, ``choice: remote, local, auto``, ``number 0..100``,
        ``integer``, ``list of: lidar, depth``."""
        if self.kind == "bool":
            return "bool"
        if self.kind == "choice":
            return f"choice: {', '.join(self.choices)}"
        if self.kind == "list":
            return f"list of: {', '.join(self.choices)}"
        text = "integer" if self.integer else "number"
        return f"{text} {self._range_text()}" if self.range is not None else text

    def help(self) -> str:
        """The description with what a setter must know appended: the choices, the range, the
        variable that overrides the default, and that a flag is not live."""
        parts = [self.description] if self.description else []
        if self.kind == "choice":
            parts.append(f"(one of: {', '.join(self.choices)})")
        elif self.kind == "list":
            parts.append(f"(any of: {', '.join(self.choices)}, comma-separated)")
        elif self.range is not None:
            parts.append(f"({self._range_text()})")
        if self.env:
            parts.append(f"({self.env} overrides the default at start)")
        if not self.live:
            parts.append("(not live: set at the next start)")
        return " ".join(parts)

    def _range_text(self) -> str:
        assert self.range is not None
        lo, hi = self.range
        return f"{int(lo)}..{int(hi)}" if self.integer else f"{lo:g}..{hi:g}"


class FlagSet:
    """A node's flags in declaration order with their current values, which start at the
    defaults: ``flags["align"]`` reads one, :meth:`set` changes one with the flag's own
    validation. ``FlagSet(*other)`` is a fresh copy at the defaults, so a module's table stays
    what it declares while a node works on its own."""

    def __init__(self, *flags: Flag) -> None:
        self._flags: dict[str, Flag] = {}
        for flag in flags:
            if flag.name in self._flags:
                raise ValueError(f"{flag.name}: declared twice")
            self._flags[flag.name] = flag
        self._values: dict[str, Any] = {f.name: f.default for f in flags}

    def __iter__(self) -> Iterator[Flag]:
        return iter(self._flags.values())

    def __len__(self) -> int:
        return len(self._flags)

    def __contains__(self, name: object) -> bool:
        return name in self._flags

    def __getitem__(self, name: str) -> Any:
        return self._values[self.flag(name).name]

    @property
    def names(self) -> tuple[str, ...]:
        """The flags' names, in declaration order."""
        return tuple(self._flags)

    def flag(self, name: str) -> Flag:
        """The declaration of ``name``; ``ValueError`` naming the flags there are otherwise."""
        try:
            return self._flags[name]
        except KeyError:
            raise ValueError(
                f"{name}: not a flag here; the flags are {', '.join(self._flags) or 'none'}"
            ) from None

    def on(self, name: str) -> bool:
        """Whether switch ``name`` is on (any kind: a number is on when non-zero, a list when
        non-empty)."""
        return bool(self[name])

    def defaults(self, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Every flag's default by name, a flag's ``env`` variable overriding it when set in
        ``environ``; a value the flag refuses is a ``ValueError`` naming the variable."""
        out: dict[str, Any] = {}
        for flag in self:
            value = flag.default
            if flag.env and environ is not None and flag.env in environ:
                try:
                    value = flag.parse(environ[flag.env])
                except ValueError as exc:
                    raise ValueError(f"{flag.env}: {exc}") from None
            out[flag.name] = value
        return out

    def set(self, name: str, value: Any) -> Any:
        """Change ``name`` to ``value`` (the value itself or its text), validated by the flag's
        kind, and return the old value; ``ValueError`` with the reason changes nothing."""
        flag = self.flag(name)
        new = flag.parse(value)
        old, self._values[name] = self._values[name], new
        return old

    def as_dict(self) -> dict[str, Any]:
        """Every flag's current value by name (a list flag as a tuple)."""
        return dict(self._values)

    def state(self, live_only: bool = True) -> str:
        """The values in one compact string for a report line: ``align=on backend=remote
        sources=lidar,depth`` — the live flags, or every flag."""
        return " ".join(
            f"{flag.name}={flag.render(self._values[flag.name])}"
            for flag in self
            if flag.live or not live_only
        )

    def rows(self) -> list[list[str]]:
        """One row per flag for a table: name, kind, default (with its env variable), live or
        not, description — the columns of :data:`COLUMNS`."""
        rows = []
        for flag in self:
            default = flag.render(flag.default)
            if flag.env:
                default += f" (env {flag.env})"
            rows.append(
                [
                    f"`{flag.name}`",
                    flag.kind_text(),
                    default,
                    "yes" if flag.live else "at start",
                    flag.description,
                ]
            )
        return rows

    def describe(self) -> str:
        """The table as markdown: one row per flag, :data:`COLUMNS` as the header."""
        return markdown_table(COLUMNS, self.rows())


def markdown_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """A markdown table of ``header`` over ``rows``, pipes in the cells escaped."""
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |")
    return "\n".join(lines)


def load_table(path: Path | str, name: str = "FLAGS") -> FlagSet:
    """The ``FLAGS = FlagSet(...)`` table of a module that cannot be imported here — a ROS node
    on a laptop without rclpy — read from its syntax tree: the expression is evaluated with
    :class:`Flag`, :class:`FlagSet`, the module's own literal constants and the names it imports
    from ``pepin`` (never from ROS). Any other name in the table is a ``ValueError`` saying so:
    a table is data, and this loader is what keeps it readable without the robot."""
    source = Path(path).read_text()
    module = ast.parse(source, filename=str(path))
    namespace: dict[str, Any] = {"Flag": Flag, "FlagSet": FlagSet}
    table: ast.expr | None = None
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "pepin":
            imported = importlib.import_module(node.module or "pepin")
            for alias in node.names:
                namespace[alias.asname or alias.name] = getattr(imported, alias.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id == name:
                table = node.value
                break
            try:
                namespace[target.id] = ast.literal_eval(node.value)
            except ValueError:
                continue  # not a literal: not a name the table may use
    if table is None:
        raise ValueError(f"{path}: no {name} = FlagSet(...) at module level")
    try:
        flags = eval(compile(ast.Expression(table), str(path), "eval"), namespace)
    except NameError as exc:
        raise ValueError(
            f"{path}: {name} uses {exc.name}, which is not a literal of the module or a pepin"
            " import; a flags table is data"
        ) from None
    if not isinstance(flags, FlagSet):
        raise ValueError(f"{path}: {name} is a {type(flags).__name__}, not a FlagSet")
    return flags
