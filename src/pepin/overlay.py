"""Config and asset files looked for in a private tree first, then in the public one.

The robot is public; a household's persona and prompts, its sounds, its own faces and gestures are
not. They live in a private tree that mirrors the public layout (``companion/config/face.json``
beside ``config/face.json``, ``companion/sounds/`` beside ``sounds/``), and every reader names a
file by its path relative to a tree: :func:`resolve` answers the file a reader opens,
:func:`load_json` the parsed content of a JSON file, merged over the trees. :func:`read_json` and
:func:`read_text` take a path inside any tree (what :func:`pepin.deployment.config_file`
answered, ``REPO / "config/x.json"``) and answer the same; a path outside every tree is read as it
is, so a test's temporary file stays its own.

The trees are ``config/overlay.json``'s ``roots``, highest priority first, relative to the public
tree (``"."`` is the public tree itself; a list without it is refused), or ``$PEPIN_OVERLAY``
(the same list, ``:``-separated, for one run: ``PEPIN_OVERLAY=.`` is the public tree alone). A
root that does not exist, or is not a directory, is skipped: the public tree alone is a whole
robot. A root that is a symlink is the directory it points to. Inside the public tree a
``config/`` path is the first of :func:`config_homes` that holds the file
(``$PEPIN_CONFIG_DIR``, then beside the package, then beside the source tree): the roots come
first, so a private tree's copy merges over ``$PEPIN_CONFIG_DIR``'s when that is set.

How the trees combine:

- a JSON file MERGES (:func:`merge`): objects key by key, a list whose items all carry a ``name``
  on both sides by name (an entry of the same name replaces the lower one whole, a new name is
  appended; a name twice in one tree's list is a ``ValueError``), any other value replaced by
  the higher tree's (a list of lists, a list one of whose items carries no name, ``[]``);
  ``null`` is a value like any other (the public files use it: face.json's ``hold_s``), so
  nothing deletes a public key or entry: a private tree turns one off through its own fields,
  empties a list with ``[]``, or supplies the whole file (``override``);
- every other file (markdown, text, audio, binaries) is REPLACED whole by the highest tree that
  has it;
- ``rules`` in overlay.json name exceptions by path or glob: ``"mode": "override"`` for a JSON
  file the highest tree supplies whole, ``"mode": "merge"`` for a text file the trees concatenate
  (the public text first), ``"entries"`` for the objects of a JSON file whose members are named
  entries (animation.json's ``atoms``): there a higher tree's member replaces the lower one's
  whole instead of merging into it.

    uv run python -m pepin.overlay                        the roots in use
    uv run python -m pepin.overlay --where config/face.json   which trees supply a path
    uv run python -m pepin.overlay --merged config/face.json  what a reader of it gets
    uv run python -m pepin.overlay --check                every reference resolves
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    from pepin.expression.markers import Vocabulary

Mode = Literal["merge", "override"]
MODES: tuple[str, ...] = get_args(Mode)
OVERLAY_FILE = "overlay.json"  # in the public config directory
ENV = "PEPIN_OVERLAY"  # the roots for one run, os.pathsep-separated
CONFIG_ENV = "PEPIN_CONFIG_DIR"  # the public config directory, searched first
PUBLIC = "."  # the public tree's name among the roots
DEFAULT_ROOTS = ("companion", PUBLIC)
CONFIG = "config/"
RULE_KEYS = ("mode", "entries", "note")
SOUNDS_REGISTRY = "config/sounds.json"  # the sounds by name: {"sounds": [{name, file, external}]}
SOUNDS = "sounds/"  # where a registry's ``file`` is looked for, in every tree


@dataclass(frozen=True)
class Rule:
    """What overlay.json's ``rules`` say of a path: its mode (None: by its type, JSON merges) and
    the dotted keys of its objects whose members are named entries, replaced whole by name."""

    mode: Mode | None = None
    entries: tuple[str, ...] = ()


@dataclass(frozen=True)
class Overlay:
    """The trees a path is looked for in (highest priority first, the absent ones skipped), the
    public tree, the directories its ``config/`` may be (:func:`config_homes`) and the rules."""

    roots: tuple[Path, ...]
    public: Path
    homes: tuple[Path, ...]
    rules: Mapping[str, Rule] = field(default_factory=dict)
    skipped: tuple[Path, ...] = ()

    def rule(self, rel: str) -> Rule:
        """The rule of ``rel``: its own, else the first glob that matches, else none."""
        rel = clean(rel)
        if rel in self.rules:
            return self.rules[rel]
        found = (r for p, r in self.rules.items() if fnmatch.fnmatchcase(rel, p))
        return next(found, Rule())

    def mode(self, rel: str) -> Mode:
        """``merge`` for JSON and ``override`` for anything else, unless a rule says otherwise."""
        mode = self.rule(rel).mode
        if mode is not None:
            return mode
        return "merge" if rel.endswith(".json") else "override"

    def places(self, root: Path, rel: str) -> tuple[Path, ...]:
        """Where ``rel`` may be in ``root``, in order: the public tree's ``config/`` is each of
        the config homes (the first that has the file is the public one)."""
        rel = clean(rel)
        if root == self.public and rel.startswith(CONFIG):
            return tuple(home / rel[len(CONFIG) :] for home in self.homes)
        return (root / rel,)

    def found(self, root: Path, rel: str) -> Path | None:
        """``rel``'s file in ``root``, or None."""
        return next((p for p in self.places(root, rel) if p.is_file()), None)

    def layers(self, rel: str, own: Path | None = None) -> list[Path]:
        """Every tree's file of ``rel``, highest priority first, one per tree; ``own`` stands for
        the public tree's (the file a caller named by its path)."""
        layers: list[Path] = []
        for root in self.roots:
            if own is not None and root == self.public:
                path: Path | None = own if own.is_file() else None
            else:
                path = self.found(root, rel)
            if path is not None and all(path.resolve() != p.resolve() for p in layers):
                layers.append(path)
        return layers

    def resolve(self, rel: str) -> Path:
        """The file a reader of ``rel`` opens: the highest tree's for a file supplied whole, the
        lowest (the base, the public one when it has it) for a merged one, which
        :func:`read_json` merges the others over. ``FileNotFoundError`` names every place."""
        layers = self._layers_or_raise(rel)
        return layers[-1] if self.mode(rel) == "merge" else layers[0]

    def load_json(self, rel: str, own: Path | None = None) -> Any:
        """``rel`` parsed, the trees merged by its rule (:func:`merge`); ``own``: the public
        tree's file of it (:meth:`layers`)."""
        layers = self._layers_or_raise(rel, own)
        if self.mode(rel) == "override":
            return _parse(layers[0])
        entries = self.rule(rel).entries
        merged = _parse(layers[-1])
        for layer in reversed(layers[:-1]):
            try:
                merged = merge(merged, _parse(layer), entries)
            except ValueError as exc:
                raise ValueError(
                    f"{clean(rel)}, {layer} merged over the trees below: {exc}"
                ) from exc
        return merged

    def load_text(self, rel: str, own: Path | None = None) -> str:
        """``rel``'s text: the highest tree's, or for a ``merge`` rule every tree's, the public
        text first, a blank line between them; ``own``: the public tree's file of it."""
        layers = self._layers_or_raise(rel, own)
        if self.mode(rel) == "override" or len(layers) == 1:
            return layers[0].read_text()
        return "\n\n".join(p.read_text().rstrip("\n") for p in reversed(layers)) + "\n"

    def relative(self, path: str | Path) -> str | None:
        """``path`` relative to the tree it lies in (the most specific one), or None for a path
        outside every tree: ``.../companion/config/x.json`` -> ``config/x.json``."""
        located = self.locate(path)
        return None if located is None else located[0]

    def locate(self, path: str | Path) -> tuple[str, Path] | None:
        """``path`` relative to the tree it lies in (the most specific one) and that tree, or
        None for a path outside every tree."""
        target = Path(path).resolve()
        bases = [(root, root, "") for root in self.roots]
        if self.public in self.roots:
            bases += [(home.resolve(), self.public, CONFIG) for home in self.homes]
        for base, root, prefix in sorted(bases, key=lambda b: len(b[0].parts), reverse=True):
            if target != base and target.is_relative_to(base):
                return prefix + target.relative_to(base).as_posix(), root
        return None

    def _layers_or_raise(self, rel: str, own: Path | None = None) -> list[Path]:
        layers = self.layers(rel, own)
        if not layers:
            looked = [str(p.parent) for root in self.roots for p in self.places(root, rel)]
            raise FileNotFoundError(f"{clean(rel)} is in none of {looked}")
        return layers


def clean(rel: str) -> str:
    """``rel`` as a tree-relative POSIX path; ``ValueError`` for an absolute one or one with
    ``..``."""
    path = PurePosixPath(rel)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{rel!r}: a path relative to a tree, without '..'")
    return path.as_posix()


def config_homes() -> tuple[Path, ...]:
    """Where the public tree's ``config`` directory may be, in order: ``$PEPIN_CONFIG_DIR`` when
    set; the ``config`` directory beside the ``pepin`` package (the board's container:
    /ws/pepin_src/config, put there by ros/sync.sh, which mounts no /ws/config); the one beside
    the source tree (a checkout: src/pepin/../../config; the laptop's containers: /ws/config)."""
    package = Path(__file__).resolve().parent
    override = os.environ.get(CONFIG_ENV)
    homes = [Path(override)] if override else []
    return (*homes, package.parent / "config", package.parents[1] / "config")


def public_tree() -> Path:
    """The public tree: the directory holding the ``config`` this library reads (the checkout's
    root; /ws/pepin_src on the board; /ws in the laptop's containers)."""
    package = Path(__file__).resolve().parent
    for home in (package.parent / "config", package.parents[1] / "config"):
        if home.is_dir():
            return home.parent
    return package.parents[1]


def current() -> Overlay:
    """The overlay in force: the roots from ``$PEPIN_OVERLAY`` or ``config/overlay.json``
    (:data:`DEFAULT_ROOTS` without either), the rules from overlay.json. Read at every call, so
    an edit or an exported variable counts at the next read."""
    homes = config_homes()
    public = public_tree().resolve()
    settings = _settings(homes)
    listed = os.environ.get(ENV, "")
    names: Sequence[str]
    if listed:
        names = [n for n in listed.split(os.pathsep) if n]
    else:
        names = settings.get("roots", DEFAULT_ROOTS)
        if not isinstance(names, (list, tuple)) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"config/{OVERLAY_FILE}: roots is a list of paths")
    roots: list[Path] = []
    skipped: list[Path] = []
    for name in names:
        root = public if name == PUBLIC else (public / Path(name).expanduser()).resolve()
        if not root.is_dir():
            skipped.append(root)
        elif root not in roots:
            roots.append(root)
    if public not in roots:
        source = f"${ENV}" if listed else f"config/{OVERLAY_FILE}'s roots"
        raise ValueError(f"{source} {list(names)}: the public tree {PUBLIC!r} is not among them")
    rules = _rules(settings.get("rules", {}))
    return Overlay(tuple(roots), public, homes, rules, tuple(skipped))


def resolve(rel: str) -> Path:
    """The file a reader of ``rel`` opens (:meth:`Overlay.resolve`)."""
    return current().resolve(rel)


def load_json(rel: str) -> Any:
    """``rel`` parsed and merged over the trees (:meth:`Overlay.load_json`)."""
    return current().load_json(rel)


def read_json(path: str | Path) -> Any:
    """The JSON file at ``path`` as a reader gets it: when ``path`` lies in a tree, merged over
    the trees (a public path stands for the public layer, the private ones merged over it);
    outside them, parsed as it is."""
    overlay = current()
    located = overlay.locate(path)
    if located is None:
        return _parse(Path(path))
    rel, root = located
    return overlay.load_json(rel, Path(path) if root == overlay.public else None)


def read_text(path: str | Path) -> str:
    """The text file at ``path`` as a reader gets it (:meth:`Overlay.load_text` when ``path``
    lies in a tree, a public path standing for the public layer; the file's own text when
    outside them)."""
    overlay = current()
    located = overlay.locate(path)
    if located is None:
        return Path(path).read_text()
    rel, root = located
    return overlay.load_text(rel, Path(path) if root == overlay.public else None)


def merge(base: Any, top: Any, entries: Collection[str] = (), at: str = "") -> Any:
    """``top`` over ``base``: objects key by key (recursively; the members of an object whose
    dotted key is in ``entries`` replaced whole), a non-empty list whose items all carry a
    string ``name`` over a list of such items by name (the same name replaced whole in place,
    new names appended in ``top``'s order; a name twice in either list is a ``ValueError``),
    anything else ``top`` (``null`` too: it sets, never deletes). Neither input is changed."""
    if isinstance(base, dict) and isinstance(top, dict):
        out = dict(base)
        whole = at in entries
        for key, value in top.items():
            inner = f"{at}.{key}" if at else str(key)
            out[key] = value if whole or key not in out else merge(out[key], value, entries, inner)
        return out
    if isinstance(base, list) and isinstance(top, list) and top and named(base) and named(top):
        out_list = list(base)
        index = _names(base, at)
        _names(top, at)
        for item in top:
            at_index = index.get(item["name"])
            if at_index is None:
                index[item["name"]] = len(out_list)
                out_list.append(item)
            else:
                out_list[at_index] = item
        return out_list
    return top


def named(items: list[Any]) -> bool:
    """Whether every item of ``items`` is an object with a string ``name``."""
    return all(isinstance(i, dict) and isinstance(i.get("name"), str) for i in items)


def _names(items: list[Any], at: str) -> dict[str, int]:
    """Each name of a named list to its index; ``ValueError`` for a name given twice (which
    one would win is no rule)."""
    index: dict[str, int] = {}
    for i, item in enumerate(items):
        if item["name"] in index:
            raise ValueError(f"{at or 'the list'}: {item['name']!r} twice in one tree's list")
        index[item["name"]] = i
    return index


def where(rel: str, overlay: Overlay | None = None) -> list[str]:
    """Which trees supply ``rel`` and what a reader gets: one line for the path and its mode,
    one per tree, one for the result."""
    overlay = overlay or current()
    rule = overlay.rule(rel)
    mode = overlay.mode(rel)
    entries = f" (entries {', '.join(rule.entries)})" if rule.entries else ""
    lines = [f"{clean(rel)}: {mode}{entries}"]
    for root in overlay.roots:
        kind = "public " if root == overlay.public else "private"
        path = overlay.found(root, rel)
        lines.append(f"  {kind} {root}: {path if path is not None else '-'}")
    lines += [f"  absent  {root}: skipped" for root in overlay.skipped]
    layers = overlay.layers(rel)
    if not layers:
        lines.append("  -> nowhere")
    elif mode == "merge" and len(layers) > 1:
        lines.append(f"  -> {len(layers)} files merged, {layers[-1]} the base")
    else:
        lines.append(f"  -> {layers[0]}")
    return lines


def check(overlay: Overlay | None = None) -> list[str]:
    """Every reference between the files the overlay in force supplies, as one problem a line
    (none: every name resolves): each JSON file of ``config/`` loads; every ``entries`` key of a
    rule is an object of its file and no private tree carries an overlay.json of its own (it
    would not be read); the gesture book loads and every face it or head.json names is an
    expression of face.json, whose own table checks; every persona file voice_cascade.json
    names resolves and every marker its text writes names a gesture, a face or a style; every
    sound of the registry (:data:`SOUNDS_REGISTRY`) that names a file finds it under
    ``sounds/`` in some tree or is marked ``"external": true`` or ``"synthesized": true`` (one
    whose file is null is :func:`pending_sounds`, a synthesized one not made yet
    :func:`unmade_sounds`)."""
    overlay = overlay or current()
    problems: list[str] = []
    for rel in _config_json(overlay):
        try:
            overlay.load_json(rel)
        except (OSError, ValueError) as exc:
            problems.append(f"{rel}: {exc}")
    problems += _check_rules(overlay)
    problems += _check_faces(overlay)
    problems += _check_personas(overlay)
    problems += _check_sounds(overlay)
    return problems


def _config_json(overlay: Overlay) -> list[str]:
    """Every JSON file under ``config/`` in any tree, as a tree-relative path."""
    names: set[str] = set()
    for root in overlay.roots:
        folders = overlay.homes if root == overlay.public else (root / CONFIG,)
        for folder in (f for f in folders if f.is_dir()):
            names |= {f"{CONFIG}{p.name}" for p in folder.glob("*.json")}
    return sorted(names)


def _check_rules(overlay: Overlay) -> list[str]:
    """Every ``entries`` key of the rule in force for a JSON file names an object of the merged
    file (a misspelt key would merge a private entry into the public one key by key, without a
    word), and no private tree has a ``config/overlay.json`` (only the public one is read)."""
    problems = []
    for root in overlay.roots:
        own = root / CONFIG / OVERLAY_FILE
        if root != overlay.public and own.is_file():
            problems.append(f"{own}: not read (the roots and rules are the public one's)")
    exact = {p for p in overlay.rules if not set(p) & set("*?[")}
    for rel in sorted(set(_config_json(overlay)) | exact):
        rule = overlay.rule(rel)
        if not rule.entries or overlay.mode(rel) == "override" or not overlay.layers(rel):
            continue
        try:
            data = overlay.load_json(rel)
        except (OSError, ValueError):
            continue  # the file's own problem, named by check()
        for key in rule.entries:
            value = data
            for part in key.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if not isinstance(value, dict):
                problems.append(f"config/{OVERLAY_FILE}: {rel} entries {key!r}: no such object")
    return problems


def _check_faces(overlay: Overlay) -> list[str]:
    """The book's faces against face.json's expressions (the step's choices of a face atom's
    ``$name`` key, a face key's literal value, a variant's ``needs.face``), and head.json's
    ``resting_expression``."""
    from pepin.expression.book import book_from
    from pepin.face import parse_face_table

    try:
        table = parse_face_table(overlay.load_json("config/face.json"))
        book = book_from(overlay.load_json("config/animation.json"))
    except (OSError, ValueError, KeyError) as exc:
        return [f"gestures and faces: {exc}"]
    known = set(table.names)
    problems = []
    for name, composition in book.compositions.items():
        for variant in composition.variants:
            wanted = variant.needs.get("face", "any")
            if wanted != "any" and wanted not in known:
                problems.append(f"compositions.{name}: needs face {wanted!r}, not in face.json")
            for step in variant.steps:
                atom = book.atoms.get(step.atom)
                if atom is None or atom.part != "face":
                    continue
                for key in atom.keys:
                    value = key.values.get("expression")
                    if not isinstance(value, str):
                        continue
                    faces = step.choices.get(value[1:], ()) if value.startswith("$") else (value,)
                    problems += [
                        f"compositions.{name}: face {face!r} (atom {atom.name}) not in face.json"
                        for face in faces
                        if face not in known
                    ]
    head = "config/head.json"
    try:
        resting = (
            overlay.load_json(head).get("resting_expression") if overlay.layers(head) else None
        )
    except (OSError, ValueError, AttributeError) as exc:
        problems.append(f"{head}: {exc}")
        resting = None
    if resting is not None and resting not in known:
        problems.append(f"{head}: resting_expression {resting!r} not in face.json")
    return sorted(set(problems))


def _vocabulary(overlay: Overlay) -> Vocabulary | None:
    """What a marker may name with the overlay's book and faces; None when either does not load
    (:func:`_check_faces` names why)."""
    from pepin.expression.book import book_from, vocabulary
    from pepin.face import parse_face_table

    try:
        table = parse_face_table(overlay.load_json("config/face.json"))
        book = book_from(overlay.load_json("config/animation.json"))
    except (OSError, ValueError, KeyError):
        return None
    return vocabulary(book, frozenset(table.names))


def _unknown_marks(text: str, vocabulary: Vocabulary) -> list[str]:
    """The markers of ``text`` (a persona) that name nothing, or carry an argument their
    gesture does not take, as written and in order: what the brain would be taught to write
    and the parser would drop (:func:`pepin.expression.markers.read_mark`)."""
    from pepin.expression.markers import MARKER, read_mark

    bad: dict[str, None] = {}
    for found in MARKER.finditer(text):
        mark = read_mark(found.group(1), vocabulary)
        if mark is None or mark.ignored:
            bad[found.group(0)] = None
    return list(bad)


def _check_personas(overlay: Overlay) -> list[str]:
    """Every ``persona_file`` of voice_cascade.json (the brain's, each preset's) is a name that
    resolves, and every marker its text writes names a gesture of the book, an expression of
    face.json or a style (:func:`_unknown_marks`)."""
    try:
        data = overlay.load_json("config/voice_cascade.json")
    except (OSError, ValueError) as exc:
        return [f"config/voice_cascade.json: {exc}"]
    brain = data.get("brain", {}) if isinstance(data, dict) else {}
    brain = brain if isinstance(brain, dict) else {}
    presets = brain.get("presets", {})
    presets = presets if isinstance(presets, dict) else {}
    blocks = [brain, *(p for p in presets.values() if isinstance(p, dict))]
    problems = []
    named_files: set[str] = set()
    for block in blocks:
        if "persona_file" not in block or (block is not brain and block["persona_file"] is None):
            continue  # none named here: the code's default, or a preset taking the brain's
        value = block["persona_file"]
        if isinstance(value, str):
            named_files.add(value)
        else:
            problems.append(f"voice_cascade.json persona_file: {value!r} is no file name")
    vocabulary = _vocabulary(overlay)
    for name in sorted(named_files):
        rel = f"{CONFIG}{name}"
        try:
            text = overlay.load_text(rel)
        except (OSError, ValueError) as exc:
            problems.append(f"voice_cascade.json persona_file: {exc}")
            continue
        if vocabulary is not None:
            problems += [
                f"{rel}: the marker {raw} names no gesture, face or style (or an argument "
                "its gesture does not take)"
                for raw in _unknown_marks(text, vocabulary)
            ]
    return problems


def _check_sounds(overlay: Overlay) -> list[str]:
    """Every sound of the registry that names a file finds it under ``sounds/`` in some tree,
    unless it is marked external or synthesized (a file still null is :func:`pending_sounds`,
    a synthesized one not made yet :func:`unmade_sounds`: neither is a fault)."""
    problems = []
    for entry in _sounds(overlay):
        name, file = entry.get("name"), entry.get("file")
        if entry.get("external") is True or entry.get("synthesized") is True or file is None:
            continue
        try:
            found = isinstance(file, str) and bool(overlay.layers(f"{SOUNDS}{file}"))
        except ValueError:  # an absolute path or one with '..': no tree holds it
            found = False
        if not found:
            problems.append(f"{SOUNDS_REGISTRY}: sound {name!r} has no file {SOUNDS}{file}")
    return problems


def pending_sounds(overlay: Overlay | None = None) -> list[str]:
    """The sounds of the registry whose ``file`` is still null (named, not yet dropped in)."""
    overlay = overlay or current()
    return [
        str(e.get("name"))
        for e in _sounds(overlay)
        if e.get("file") is None and e.get("external") is not True
    ]


def unmade_sounds(overlay: Overlay | None = None) -> list[str]:
    """The synthesized sounds of the registry whose file is in no tree yet (pepin.sounds makes
    it on first use, or ``python -m pepin.sounds --make-default``)."""
    overlay = overlay or current()
    names = []
    for e in _sounds(overlay):
        file = e.get("file")
        if e.get("synthesized") is not True or not isinstance(file, str):
            continue
        try:
            made = bool(overlay.layers(f"{SOUNDS}{file}"))
        except ValueError:
            made = False
        if not made:
            names.append(str(e.get("name")))
    return names


def _sounds(overlay: Overlay) -> list[dict[str, Any]]:
    """The registry's entries (none without a registry; a broken one is :func:`check`'s)."""
    if not overlay.layers(SOUNDS_REGISTRY):
        return []
    try:
        sounds = overlay.load_json(SOUNDS_REGISTRY).get("sounds", [])
    except (OSError, ValueError):
        return []
    return [e for e in sounds if isinstance(e, dict)]


def _parse(path: Path) -> Any:
    return json.loads(path.read_text())


def _settings(homes: Sequence[Path]) -> dict[str, Any]:
    """The public overlay.json (the first config home that has it), or ``{}``."""
    for home in homes:
        path = home / OVERLAY_FILE
        if path.is_file():
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError(f"{path}: an object of roots and rules")
            return data
    return {}


def _rules(raw: Any) -> dict[str, Rule]:
    """overlay.json's ``rules``: a path or glob -> ``mode``, ``entries``, ``note``."""
    if not isinstance(raw, dict):
        raise ValueError(f"config/{OVERLAY_FILE}: rules is an object of path -> rule")
    rules = {}
    for pattern, body in raw.items():
        if pattern.startswith("_"):
            continue
        where = f"config/{OVERLAY_FILE}: rules.{pattern}"
        if not isinstance(body, dict) or set(body) - set(RULE_KEYS):
            raise ValueError(f"{where}: an object of {', '.join(RULE_KEYS)}")
        mode = body.get("mode")
        if mode is not None and mode not in MODES:
            raise ValueError(f"{where}: mode is one of {', '.join(MODES)}")
        entries = body.get("entries", [])
        if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
            raise ValueError(f"{where}: entries is a list of dotted keys")
        rules[clean(pattern)] = Rule(mode, tuple(entries))
    return rules


def main(argv: Sequence[str] | None = None) -> int:
    """Print the roots, where paths come from, a merged file, or the reference check."""
    parser = argparse.ArgumentParser(description="The private overlay over the public tree.")
    parser.add_argument("--where", nargs="+", metavar="PATH", help="which trees supply PATH")
    parser.add_argument("--merged", metavar="PATH", help="what a reader of PATH gets")
    parser.add_argument("--check", action="store_true", help="every reference resolves")
    args = parser.parse_args(argv)
    overlay = current()
    if args.where:
        for rel in args.where:
            print("\n".join(where(rel, overlay)))
    elif args.merged:
        if overlay.mode(args.merged) == "merge" and args.merged.endswith(".json"):
            print(json.dumps(overlay.load_json(args.merged), indent=2, ensure_ascii=False))
        else:
            sys.stdout.write(overlay.load_text(args.merged))
    elif args.check:
        problems = check(overlay)
        print("\n".join(problems) if problems else "every reference resolves")
        pending = pending_sounds(overlay)
        if pending:
            print(f"sounds with no file yet ({len(pending)}): {', '.join(pending)}")
        unmade = unmade_sounds(overlay)
        if unmade:
            print(f"synthesized sounds not made yet, made on first use or by python -m "
                  f"pepin.sounds --make-default ({len(unmade)}): {', '.join(unmade)}")  # fmt: skip
        return 1 if problems else 0
    else:
        for root in overlay.roots:
            print(f"{'public ' if root == overlay.public else 'private'} {root}")
        for root in overlay.skipped:
            print(f"absent  {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
