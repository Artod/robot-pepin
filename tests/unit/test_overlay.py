"""The private overlay: a private tree searched before the public one, JSON merged, the rest
replaced whole, and the public tree alone a whole robot."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pepin import overlay
from pepin.overlay import Overlay, Rule, merge

REPO = Path(__file__).resolve().parents[2]


def _tree(root: Path, files: dict[str, Any]) -> Path:
    """``root`` holding ``files`` (a path -> a JSON value, or text, or bytes)."""
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        elif isinstance(content, str):
            path.write_text(content)
        else:
            path.write_text(json.dumps(content))
    return root


def _overlay(
    tmp_path: Path,
    private: dict[str, Any],
    public: dict[str, Any],
    rules: dict[str, Rule] | None = None,
) -> Overlay:
    """A private tree over a public one, both under ``tmp_path``."""
    pub = _tree(tmp_path / "public", public).resolve()
    (pub / "config").mkdir(exist_ok=True)
    priv = _tree(tmp_path / "public" / "companion", private).resolve()
    return Overlay((priv, pub), pub, (pub / "config",), rules or {})


def test_json_merges_key_by_key_and_named_lists_by_name(tmp_path: Path) -> None:
    """A private face.json adds an expression, replaces one of a public name whole, and changes
    one number of a block; everything it does not name stays the public file's."""
    public = {
        "colors": {"lip": "#ffffff", "teeth": "#eeeeee"},
        "expressions": [
            {"name": "neutral", "params": {}},
            {"name": "smile", "params": {"smile": 0.6, "open": 0.1}},
        ],
        "idle": {"expressions": ["neutral", "smile"], "every_s": [8, 20]},
    }
    private = {
        "colors": {"lip": "#ff0000"},
        "expressions": [
            {"name": "smile", "params": {"smile": 1.0}},
            {"name": "meme", "params": {}},
        ],
        "idle": {"expressions": ["meme"]},
    }
    o = _overlay(tmp_path, {"config/face.json": private}, {"config/face.json": public})
    merged = o.load_json("config/face.json")
    assert merged["colors"] == {"lip": "#ff0000", "teeth": "#eeeeee"}
    assert merged["expressions"] == [
        {"name": "neutral", "params": {}},
        {"name": "smile", "params": {"smile": 1.0}},  # replaced whole: no "open" left over
        {"name": "meme", "params": {}},
    ]
    assert merged["idle"] == {"expressions": ["meme"], "every_s": [8, 20]}  # a plain list replaced
    assert o.resolve("config/face.json") == tmp_path.resolve() / "public/config/face.json", (
        "the base"
    )


def test_entries_replace_named_members_of_an_object_whole(tmp_path: Path) -> None:
    """animation.json keeps its compositions by name in an object: with ``entries`` a private
    composition of a public name replaces it whole; without, it merges into it."""
    public = {"compositions": {"wave": {"cooldown_s": 4, "variants": [1]}, "nod": {"x": 1}}}
    private = {"compositions": {"wave": {"variants": [2]}, "meme_dance": {"variants": [3]}}}
    files = ({"config/animation.json": private}, {"config/animation.json": public})
    whole = _overlay(
        tmp_path / "a", *files, {"config/animation.json": Rule(entries=("compositions",))}
    )
    assert whole.load_json("config/animation.json")["compositions"] == {
        "wave": {"variants": [2]},
        "nod": {"x": 1},
        "meme_dance": {"variants": [3]},
    }
    keywise = _overlay(tmp_path / "b", *files).load_json("config/animation.json")
    assert keywise["compositions"]["wave"] == {"cooldown_s": 4, "variants": [2]}


def test_markdown_and_audio_are_replaced_whole(tmp_path: Path) -> None:
    """The persona and a sound: the private tree's file is the one a reader opens; a file only
    the public tree has stays the public one."""
    o = _overlay(
        tmp_path,
        {"config/persona.md": "private persona\n", "sounds/rimshot.wav": b"RIFFprivate"},
        {"config/persona.md": "public persona\n", "config/other.md": "public\n"},
    )
    assert o.mode("config/persona.md") == "override"
    assert o.load_text("config/persona.md") == "private persona\n"
    assert o.resolve("sounds/rimshot.wav").read_bytes() == b"RIFFprivate"
    assert o.resolve("config/other.md").read_text() == "public\n"


def test_the_rules_override_a_json_file_and_concatenate_a_text_file(tmp_path: Path) -> None:
    """``override`` gives the private JSON whole; ``merge`` on text puts the public text first;
    a glob rule applies to every path it matches."""
    o = _overlay(
        tmp_path,
        {"config/persona.json": {"who": "private"}, "prompts/judge.md": "private rules\n"},
        {"config/persona.json": {"who": "public", "tone": "dry"}, "prompts/judge.md": "public\n"},
        {"config/persona.json": Rule(mode="override"), "prompts/*.md": Rule(mode="merge")},
    )
    assert o.load_json("config/persona.json") == {"who": "private"}
    assert o.resolve("config/persona.json") == o.roots[0] / "config/persona.json"
    assert o.load_text("prompts/judge.md") == "public\n\nprivate rules\n"


def test_a_path_inside_either_tree_reads_the_merge_and_one_outside_reads_itself(
    tmp_path: Path,
) -> None:
    """The public file's path and the private copy's give the same merge; a file elsewhere is
    read as it is."""
    o = _overlay(tmp_path, {"config/x.json": {"a": 2}}, {"config/x.json": {"a": 1, "b": 1}})
    public_path, private_path = o.roots[1] / "config/x.json", o.roots[0] / "config/x.json"
    assert o.relative(public_path) == o.relative(private_path) == "config/x.json"
    assert o.relative(tmp_path / "elsewhere.json") is None
    assert o.load_json(o.relative(public_path) or "") == {"a": 2, "b": 1}


def test_the_overlay_in_force_reads_the_environment_and_skips_an_absent_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """$PEPIN_OVERLAY names the roots for a run: a private tree in a temporary directory merges
    over this checkout's config through every reader; a root that does not exist is skipped and
    the public tree alone answers as before."""
    from pepin.deployment import config_file
    from pepin.expression.book import face_names

    private = _tree(tmp_path / "private", {"config/face.json": {"expressions": [{"name": "meme"}]}})
    monkeypatch.setenv("PEPIN_OVERLAY", f"{private}:.")
    assert overlay.current().roots == (private.resolve(), REPO)
    assert "meme" in face_names() and "smile" in face_names()
    assert config_file("face.json") == REPO / "config/face.json", "the merge's base"
    assert "meme" in {
        e["name"] for e in overlay.read_json(REPO / "config/face.json")["expressions"]
    }
    monkeypatch.setenv("PEPIN_OVERLAY", f"{tmp_path / 'absent'}:.")
    o = overlay.current()
    assert o.roots == (REPO,) and o.skipped == ((tmp_path / "absent").resolve(),)
    assert "meme" not in face_names()
    monkeypatch.delenv("PEPIN_OVERLAY")
    assert overlay.current().roots[-1] == REPO, "config/overlay.json's roots end in the public tree"


def test_a_path_a_caller_names_is_the_public_layer_whichever_config_home_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """$PEPIN_CONFIG_DIR is searched before the checkout's config/ (config_file finds its copy
    first), yet a reader handed the checkout's own file by its path reads that file."""
    from pepin.deployment import config_file

    elsewhere = _tree(tmp_path / "cfg", {"imu.json": {"elsewhere": True}})
    monkeypatch.setenv("PEPIN_CONFIG_DIR", str(elsewhere))
    assert config_file("imu.json") == elsewhere / "imu.json"
    assert overlay.read_json(elsewhere / "imu.json") == {"elsewhere": True}
    assert overlay.read_json(REPO / "config/imu.json") == json.loads(
        (REPO / "config/imu.json").read_text()
    )


def test_a_missing_file_names_every_place_looked(tmp_path: Path) -> None:
    o = _overlay(tmp_path, {}, {})
    with pytest.raises(FileNotFoundError, match=r"sounds/none\.wav is in none of .*companion"):
        o.resolve("sounds/none.wav")
    with pytest.raises(ValueError, match=r"without '\.\.'"):
        o.resolve("../outside.json")


def test_overlay_json_refuses_a_rule_it_cannot_apply() -> None:
    assert overlay._rules({"_note": "x", "a/*.md": {"mode": "merge"}}) == {"a/*.md": Rule("merge")}
    for bad in (
        {"a.json": {"mdoe": "merge"}},
        {"a.json": {"mode": "replace"}},
        {"a": {"entries": "x"}},
    ):
        with pytest.raises(ValueError, match="rules"):
            overlay._rules(bad)


def test_merge_is_deterministic_and_leaves_its_inputs_alone() -> None:
    base = {"a": {"b": [1, 2], "c": 1}, "named": [{"name": "x", "v": 1}]}
    top = {"a": {"b": [3]}, "named": [{"name": "y"}, {"name": "x", "v": 2}], "new": None}
    before = json.dumps([base, top])
    once = merge(base, top)
    assert (
        once
        == merge(base, top)
        == {
            "a": {"b": [3], "c": 1},
            "named": [{"name": "x", "v": 2}, {"name": "y"}],
            "new": None,
        }
    )
    assert json.dumps([base, top]) == before
    assert merge({"l": [{"name": "x"}]}, {"l": []}) == {"l": []}, "an empty list empties"
    assert merge({"k": {"x": 1}}, {"k": 5}) == {"k": 5}, "another type replaces"


def test_where_names_the_tree_of_each_file(tmp_path: Path) -> None:
    o = _overlay(tmp_path, {"config/x.json": {}}, {"config/x.json": {}, "config/y.md": "y"})
    lines = overlay.where("config/x.json", o)
    assert lines[0] == "config/x.json: merge" and lines[-1].startswith("  -> 2 files merged")
    assert overlay.where("config/y.md", o)[-1] == f"  -> {o.roots[1] / 'config/y.md'}"
    assert overlay.where("config/z.md", o)[-1] == "  -> nowhere"


def test_the_check_finds_a_face_a_gesture_names_and_face_json_lacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private composition that shows a face the merged face.json does not have, and a sound
    of the registry with no file, are named; a sound marked external is not."""
    private = _tree(
        tmp_path / "private",
        {
            "config/animation.json": {
                "compositions": {
                    "meme": {
                        "variants": [{"steps": [{"atom": "face", "params": {"name": "nope"}}]}]
                    }
                }
            },
            "config/sounds.json": {
                "sounds": [
                    {"name": "rimshot", "file": "memes/rimshot.mp3"},
                    {"name": "chime", "file": "chime.mp3", "external": True},
                    {"name": "kept", "file": "kept.wav"},
                    {"name": "later", "file": None},
                ]
            },
            "sounds/kept.wav": b"RIFF",
        },
    )
    monkeypatch.setenv("PEPIN_OVERLAY", f"{private}:.")
    problems = overlay.check()
    assert problems == [
        "compositions.meme: face 'nope' (atom face) not in face.json",
        "config/sounds.json: sound 'rimshot' has no file sounds/memes/rimshot.mp3",
    ]
    assert "later" in overlay.pending_sounds(), "a file still null is listed, not a fault"


def test_the_public_tree_alone_resolves_every_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no overlay at all, every config file loads through the loader and every name one
    file gives another resolves: the book's faces in face.json, face.json's own events and idle,
    the book's poses and nested compositions, the personas voice_cascade.json names, the sounds
    of the registry (the public one names no file but the synthesized default beat's, made on
    demand: no audio is tracked)."""
    monkeypatch.setenv("PEPIN_OVERLAY", ".")
    o = overlay.current()
    assert o.roots == (REPO,)
    assert overlay.check(o) == []
    # config/anime holds the shape engine's SVGs, read by pepin.face_shapes alone
    for path in sorted(p for p in (REPO / "config").iterdir() if p.is_file()):
        rel = f"config/{path.name}"
        if path.suffix == ".json":
            assert overlay.load_json(rel) == json.loads(path.read_text()), rel
        else:
            assert o.load_text(rel) == path.read_text(), rel
    sounds = (
        o.load_json(overlay.SOUNDS_REGISTRY).get("sounds", [])
        if o.layers(overlay.SOUNDS_REGISTRY)
        else []
    )
    assert all(s.get("file") is None or s.get("external") or s.get("synthesized") for s in sounds)


def test_a_name_twice_is_refused_null_sets_and_other_lists_are_replaced(tmp_path: Path) -> None:
    """A name twice in one tree's named list is a ``ValueError`` naming the file (which copy
    would win is no rule); ``null`` sets a key and never deletes it (the public files use it as
    a value); a list of lists, and a named list under one whose item carries no name, are
    replaced whole."""
    with pytest.raises(ValueError, match=r"expressions: 'meme' twice"):
        merge(
            {"expressions": [{"name": "a"}]},
            {"expressions": [{"name": "meme"}, {"name": "meme", "x": 1}]},
        )
    with pytest.raises(ValueError, match=r"the list: 'a' twice"):
        merge([{"name": "a"}, {"name": "a"}], [{"name": "b"}])
    assert merge({"hold_s": 2.0, "k": 1}, {"hold_s": None}) == {"hold_s": None, "k": 1}
    assert merge({"l": [[1, 2], [3]]}, {"l": [[4]]}) == {"l": [[4]]}
    assert merge({"l": [{"name": "a", "v": 1}]}, {"l": [{"v": 2}]}) == {"l": [{"v": 2}]}
    o = _overlay(
        tmp_path,
        {"config/x.json": {"l": [{"name": "b"}, {"name": "b"}]}},
        {"config/x.json": {"l": [{"name": "a"}]}},
    )
    with pytest.raises(ValueError, match=r"config/x\.json, \S+/companion/config/x\.json merged"):
        o.load_json("config/x.json")


def test_roots_without_the_public_tree_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A roots list that leaves the public tree out would read a private tree alone (and read a
    public path as it is): refused, naming the fix; ``./`` is the public tree too."""
    private = _tree(tmp_path / "private", {"config/face.json": {}})
    monkeypatch.setenv("PEPIN_OVERLAY", str(private))
    with pytest.raises(ValueError, match=r"the public tree '\.' is not among them"):
        overlay.current()
    monkeypatch.setenv("PEPIN_OVERLAY", f"{private}:./")
    assert overlay.current().roots == (private.resolve(), REPO)


def test_a_symlinked_root_is_its_target_and_a_root_that_is_a_file_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private tree reached through a symlink (a worktree's companion pointing at the main
    checkout's) is read, and a path named through the link merges the same; a root that is a
    file (a submodule's ``.git`` placeholder) is skipped like an absent one."""
    target = _tree(tmp_path / "real", {"config/face.json": {"expressions": [{"name": "meme"}]}})
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("PEPIN_OVERLAY", f"{link}:.")
    assert overlay.current().roots == (target.resolve(), REPO)
    names = {e["name"] for e in overlay.read_json(link / "config/face.json")["expressions"]}
    assert {"meme", "smile"} <= names
    placeholder = tmp_path / "placeholder"
    placeholder.write_text("gitdir: ../.git/modules/companion\n")
    monkeypatch.setenv("PEPIN_OVERLAY", f"{placeholder}:.")
    o = overlay.current()
    assert o.roots == (REPO,) and o.skipped == (placeholder.resolve(),)


def test_a_private_copy_merges_over_pepin_config_dir_and_the_checkout_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order of places: the roots first (the private tree over the public one); inside the
    public tree the first config home that holds the file ($PEPIN_CONFIG_DIR before the
    checkout's config/). The private copy merges over $PEPIN_CONFIG_DIR's file, which is the
    merge's base, and the checkout's copy is not read at all."""
    from pepin.deployment import config_file

    elsewhere = _tree(tmp_path / "cfg", {"imu.json": {"from": "config_dir", "a": 1}})
    private = _tree(tmp_path / "private", {"config/imu.json": {"from": "private"}})
    monkeypatch.setenv("PEPIN_CONFIG_DIR", str(elsewhere))
    monkeypatch.setenv("PEPIN_OVERLAY", f"{private}:.")
    assert config_file("imu.json") == elsewhere / "imu.json", "the merge's base"
    merged = {"from": "private", "a": 1}
    assert overlay.read_json(config_file("imu.json")) == merged
    assert overlay.read_json(private / "config/imu.json") == merged


def test_a_sibling_named_beside_a_resolved_path_reads_through_the_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reader that takes ``.parent`` of what config_file answered and names a sibling
    (camera.json -> its calibration) reads the sibling through the trees as well: beside a
    private-only file the public sibling is found, beside the public tree a private-only one."""
    from pepin.deployment import config_file

    private = _tree(tmp_path / "private", {"config/only_private.json": {"p": 1}})
    monkeypatch.setenv("PEPIN_OVERLAY", f"{private}:.")
    beside = config_file("only_private.json").parent
    assert beside == (private / "config").resolve()
    public_imu = json.loads((REPO / "config/imu.json").read_text())
    assert overlay.read_json(beside / "imu.json") == public_imu
    assert overlay.read_json(REPO / "config/only_private.json") == {"p": 1}


def test_the_check_finds_a_misspelt_entries_key_an_unread_overlay_json_and_a_bad_sound_path(
    tmp_path: Path,
) -> None:
    """A rule's ``entries`` key that is no object of its file (the private entry would merge
    key by key without a word), a private overlay.json (never read) and a sound file outside
    every tree are named, not raised."""
    o = _overlay(
        tmp_path,
        {
            "config/overlay.json": {"rules": {}},
            "config/sounds.json": {"sounds": [{"name": "x", "file": "../x.wav"}]},
        },
        {"config/animation.json": {"compositions": {}}},
        {"config/animation.json": Rule(entries=("compositons",))},
    )
    unread = o.roots[0] / "config/overlay.json"
    assert overlay._check_rules(o) == [
        f"{unread}: not read (the roots and rules are the public one's)",
        "config/overlay.json: config/animation.json entries 'compositons': no such object",
    ]
    assert overlay._check_sounds(o) == ["config/sounds.json: sound 'x' has no file sounds/../x.wav"]


def test_the_check_finds_a_face_lost_to_an_override_and_not_to_a_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private face.json that renames a public expression: merged by name, the public one
    stays and a private composition showing it is fine; supplied whole (``override``), the
    composition's face is named as missing."""
    face = json.loads((REPO / "config/face.json").read_text())
    inside = json.dumps({k: v for k, v in face.items() if k != "expressions"})
    likes = {e.get("like") for e in face["expressions"]}
    lost = next(
        e["name"]
        for e in face["expressions"]
        if f'"{e["name"]}"' not in inside and e["name"] not in likes
    )
    face["expressions"] = [
        dict(e, name="renamed") if e["name"] == lost else e for e in face["expressions"]
    ]
    step = {"atom": "face", "params": {"name": lost}}
    private = _tree(
        tmp_path / "private",
        {
            "config/face.json": face,
            "config/animation.json": {"compositions": {"meme": {"variants": [{"steps": [step]}]}}},
        },
    )
    monkeypatch.setenv("PEPIN_OVERLAY", f"{private}:.")
    missing = f"compositions.meme: face {lost!r} (atom face) not in face.json"
    merged = overlay.current()
    assert missing not in overlay.check(merged)
    whole = replace(merged, rules={**merged.rules, "config/face.json": Rule(mode="override")})
    assert missing in overlay.check(whole)


def test_the_check_finds_a_persona_marker_and_a_resting_face_that_name_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private persona that teaches a gesture the book lacks, a face face.json lacks or an
    argument a gesture does not take is named (the parser would drop each); so is head.json's
    resting face when face.json lacks it."""
    rel = "config/voice_cascade_persona.md"
    private = _tree(
        tmp_path / "private",
        {
            rel: "Hi {laugh} {meme_dance} {face:nope} {laugh:zzz} {whispering} {}\n",
            "config/head.json": {"resting_expression": "nope"},
        },
    )
    monkeypatch.setenv("PEPIN_OVERLAY", f"{private}:.")
    problems = overlay.check()
    assert [p.split(" names ")[0] for p in problems if p.startswith(rel)] == [
        f"{rel}: the marker {m}" for m in ("{meme_dance}", "{face:nope}", "{laugh:zzz}")
    ]
    assert "config/head.json: resting_expression 'nope' not in face.json" in problems
