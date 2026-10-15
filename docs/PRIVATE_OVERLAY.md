# Private overlay

The robot's code and every default it runs on live in this public tree. A deployment can add a
private tree that mirrors the same layout and holds what belongs to one home only: the persona and
the LLM prompts, sounds and music with their registry, faces and gestures of its own, and later
the profiles of the people the robot knows. Every reader of a config or asset file goes through
one loader, `pepin.overlay`, which looks in the private tree first and in the public tree after
it. Without a private tree the public one is a whole robot.

## Layout

```
./                                   the public tree (this repository)
  config/overlay.json                the roots, in priority order, and the per-path rules
  config/face.json                   public defaults, as everywhere in config/
  sounds/                            ignored by git: sound files never enter the public tree
  companion/                         the private tree (a git submodule; empty in a public clone)
    config/face.json                 merged over config/face.json
    config/animation.json            merged over config/animation.json
    config/voice_cascade_persona.md  replaces the public persona whole
    config/sounds.json               merged over the public sound registry by name
    sounds/                          the files the registry names
```

A file in the private tree sits at the same relative path as the public file it extends or
replaces. A file only the private tree has is found there the same way.

## Rules

| File | What a reader gets |
| --- | --- |
| JSON | the merge: objects key by key; a list whose items all carry a `name`, on both sides, by name (an item of the same name replaces the public one whole, a new name is appended; a name twice in one tree's list is an error); any other value replaced by the private one (a list of lists, a list with an item that carries no name, `[]`) |
| anything else (markdown, text, audio, binaries) | the private file whole, when the private tree has it |

`null` is a value like any other: the public files use it as one (`face.json`'s `hold_s`,
`voice_cascade.json`'s `thinking`), so a private `null` sets a key to null and never deletes it.
Nothing removes a public key or entry: a private tree turns one off through its own fields (a
composition's `"callable": false`), empties a list with `[]`, or supplies the whole file with
`"mode": "override"`. An expression of `face.json` is never removed in any case: its position in
the list is its id on the wire to the head.

`config/overlay.json` sets the exceptions by path or glob in `rules`:

```json
"rules": {
  "config/animation.json": {"entries": ["poses", "atoms", "compositions"]},
  "config/persona.json": {"mode": "override"},
  "prompts/*.md": {"mode": "merge"}
}
```

- `entries`: the keys (dotted for nested ones) of objects whose members are named entries, such as
  `animation.json`'s compositions: a private member of a public name replaces it whole instead of
  merging into it.
- `"mode": "override"`: a JSON file the private tree supplies whole.
- `"mode": "merge"`: a text file the trees concatenate, the public text first.

## Roots

`config/overlay.json`'s `roots` lists the trees, highest priority first, relative to the public
tree (`"."` is the public tree itself, and a list without it is refused); the default is
`["companion", "."]`. `PEPIN_OVERLAY` replaces the list for one run, `:`-separated:
`PEPIN_OVERLAY=. ...` runs the public tree alone, `PEPIN_OVERLAY=/path/to/other:. ...` another
private tree. A root that does not exist, or is a file (a submodule's `.git` placeholder), is
skipped, so the board and the containers, which carry no private tree, read the public files as
before. A root that is a symlink counts as the directory it points to. Only the public
`config/overlay.json` is read: a private tree's own copy is ignored, and `--check` says so.

A path is looked for in the roots in their order. Inside the public tree, a `config/` path is
the first of these that holds the file: `$PEPIN_CONFIG_DIR` when set, the `config` directory
beside the `pepin` package (the board's ROS container), the one beside the source tree (a
checkout; `/ws/config` in the laptop's containers). So with `$PEPIN_CONFIG_DIR` set, a private
copy merges over the file there, and the checkout's copy is not read.

The head draws an expression by its position in the table compiled into its firmware from the
public `face.json` (`python -m pepin.face --write`). A private expression, or a private change to
a public one, reaches the laptop's readers (the brain's vocabulary, the gestures) but not the
face the head draws.

## Reading a file

- `pepin.deployment.config_file(name)`: the path of `config/<name>`; for a file supplied whole,
  the private copy when there is one; for a merged JSON file, the public base.
- `pepin.overlay.read_json(path)`: a JSON file's merged content, for a path inside either tree;
  a path outside both (a test's temporary file) is read as it is.
- `pepin.overlay.read_text(path)`, `resolve(rel)`, `load_json(rel)`: the same by path or by
  tree-relative name.

Two kinds of code read the public file alone, on purpose: tools that rewrite a config file
(calibrations, `ros/tools/doa_calibrate.py`) read and write the file they were given, and
generators of tracked files (the face tables, `python -m pepin.face --write`; the flags table,
`ros/tools/flags_doc.py`) build them from the public tree, so nothing private ever enters a
tracked file.

## Sounds

The registry is `config/sounds.json` (`pepin.sounds`): a `sounds` list of named entries, merged
by name like any named list. An entry's `file` is relative to `sounds/` and is looked for in every
tree; `null` means the file is not dropped in yet (`--check` lists those, they are not a fault),
`"external": true` marks a sound that is fetched separately and not expected on disk, and
`"synthesized": true` one that `pepin.sounds` makes itself under the public tree's `sounds/`, on
first use or by `python -m pepin.sounds --make-default` (`--check` lists it until it is made).
The public registry holds one sound, `default_beat`: a 4-bar drum loop at 120 BPM synthesized
from numpy, its beat grid known by construction, `rock` its style. Recordings live only in a
private tree, with their entries in its copy of the registry, and `.gitignore` keeps `sounds/`
and music files out of the public one.

## Commands

```bash
uv run python -m pepin.overlay                              # the roots in use, the absent ones
uv run python -m pepin.overlay --where config/face.json     # which tree supplies a path
uv run python -m pepin.overlay --merged config/face.json    # what a reader of it gets
uv run python -m pepin.overlay --check                      # every name one file gives another resolves
uv run python -m pepin.sounds                               # the sounds in force, the brain's list
uv run python -m pepin.sounds --make-default                # sounds/default_beat.wav, the synthesized beat
```

`--check` loads every JSON file of `config/` through the loader and checks the references between
them: every `entries` key of a rule is an object of its file, the faces the gestures and
`head.json`'s `resting_expression` show exist in `face.json`, `face.json`'s events and idle name
its own expressions, the gesture book's poses and nested compositions exist, the persona files
`voice_cascade.json` names resolve and every marker their text writes (`{laugh}`, `{face:grin}`,
`{whispering}`) names a gesture, a face or a style, and every sound of the registry that names a
file finds it (or is external, or synthesized).
The unit tests run with `PEPIN_OVERLAY=.` (`tests/conftest.py`), and
`tests/unit/test_overlay.py` runs the same check on the public tree alone.
