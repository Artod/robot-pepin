"""RAFT-Stereo, vendored: the model files the stereo matcher needs and nothing else.

UPSTREAM princeton-vl/RAFT-Stereo at 6e93ed2 (2026-08-03), MIT licensed — see LICENSE beside
this file. Vendored rather than pip-installed because the project is not a package: its modules
live in a top-level ``core/`` that is imported by path, so a ``uv`` dependency on it would put
``core`` and ``utils`` on this repo's import path. Five files are copied VERBATIM —
``raft_stereo.py``, ``extractor.py``, ``update.py``, ``corr.py`` and ``core/utils/utils.py`` as
``utils.py`` — with three edits and no others, so a diff against upstream stays readable:

* ``from core.x import`` became ``from .x import`` (the package moved);
* ``update.py``'s ``from opt_einsum import contract`` is gone — it was imported and never used,
  and it was the only reason to depend on opt_einsum;
* ``utils.py``'s ``from scipy import interpolate`` moved inside ``forward_interpolate``, the one
  function that uses it and the one function the model never calls.

Nothing else is edited: no formatting, no typing, no renames. The files are excluded from ruff
and from mypy for the same reason (pyproject.toml).

The CHECKPOINTS are not vendored — 40 MB of binary does not belong in git. ``models/`` at the
repo root is where they live (gitignored); upstream's ``download_models.sh`` fetches them, and
``raftstereo-realtime.pth`` is the one this stack measured (see :mod:`pepin.stereo_host`).
"""

from .raft_stereo import RAFTStereo
from .utils import InputPadder

__all__ = ["RAFTStereo", "InputPadder"]
