"""The people the robot knows (config/people.json's store, data/people/people.db): list them,
forget one, or forget every guest at once (after a party). Forgetting deletes the name and every
face and voice sample; the rows are overwritten on disk (SQLite ``secure_delete``).

    uv run python scripts/people.py list
    uv run python scripts/people.py forget Саша
    uv run python scripts/people.py forget-guests --yes

The voice cascade may run meanwhile: it reads the store at each utterance, so a forgotten person
is a stranger from the next one on.
"""

from __future__ import annotations

import argparse
import sys
import time

from pepin.people.config import PeopleConfig
from pepin.people.store import SqliteStore


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="everyone known, with their samples")
    forget = sub.add_parser("forget", help="forget one person by name")
    forget.add_argument("name")
    guests = sub.add_parser("forget-guests", help="forget everyone who is not family")
    guests.add_argument("--yes", action="store_true", help="really (else only list them)")
    args = ap.parse_args()
    config = PeopleConfig.load()
    path = config.path(config.store)
    if not path.exists():
        print(f"nobody known yet ({path} does not exist)")
        return 0
    store = SqliteStore(path)
    try:
        people = store.people()
        if args.cmd == "list":
            for p in people:
                counts = store.counts(p.id)
                seen = "never" if p.last_seen_at is None else _when(p.last_seen_at)
                print(f"{p.name:20s} {p.kind:6s} met {_when(p.created_at)}, last seen {seen};"
                      f" face {counts['face']}, voice {counts['voice']}")  # fmt: skip
            print(f"{len(people)} people in {path}")
            return 0
        if args.cmd == "forget":
            found = store.named(args.name)
            if len(found) != 1:
                print(f"{len(found)} people called {args.name!r}: nothing forgotten")
                return 1
            store.forget(found[0].id)
            print(f"forgot {found[0].name}")
            return 0
        chosen = [p for p in people if p.kind == "guest"]
        for p in chosen:
            if args.yes:
                store.forget(p.id)
            print(("forgot " if args.yes else "would forget ") + p.name)
        if not args.yes:
            print(f"{len(chosen)} guests; --yes forgets them")
        return 0
    finally:
        store.close()


def _when(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


if __name__ == "__main__":
    sys.exit(main())
