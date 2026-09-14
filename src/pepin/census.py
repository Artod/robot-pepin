"""What the board actually runs, against what ``config/board_manifest.json`` says it may run.

The Orange Pi has four A53 cores and 1.5 GB, and has been overloaded (load 6-15) often enough
that "who is eating the board" must be answerable in one place instead of by reading ``top``
with a memory of what belongs there. This module is that answer: pure text in, a verdict out.

No ROS, no ssh, no board knowledge — it parses the output of
``ps -eo pid,ppid,ni,pcpu,rss,etime,args`` and ``/proc/loadavg`` (or ``uptime``), matches every
process against the manifest's regexes, and reports per entry OK / OVER / MISSING, every
unlisted process above the manifest's CPU threshold, the zombies, the load against the core
count, and what the manifest promises in total. ``ros/board.sh census`` is the shell face that
brings the two texts over ssh; a test feeds it a recorded dump instead.

Two warnings about the numbers, both of them ps's and not ours: ``%CPU`` is the process's
average over its whole life (a stack up for a minute reads its start-up cost, which is why the
report prints the age of the youngest expected process), and it is a percentage of ONE core, so
400 % is the whole board.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pepin.deployment import config_file

MANIFEST_FILE = "board_manifest.json"

# What a census reads off the board, in one ssh and with no ros2 CLI at all: a `ros2 node list`
# costs seconds of CPU on four A53 cores (2026-09-13). ros/board.sh and pepin.health both ask
# for this exact string, so the dump the parser sees is the dump the parser was written for.
PS_COMMAND = "ps -eo pid,ppid,ni,pcpu,rss,etime,args"
CENSUS_COMMAND = f'echo "### load"; cat /proc/loadavg; echo "### ps"; {PS_COMMAND}'

# One entry's status against the live ps.
OK = "OK"
OVER = "OVER"  # running, but above its budget
MISSING = "MISSING"  # expected always, not running
IDLE = "IDLE"  # `when: sometimes` and not running now: a drive is not on, nothing is wrong
FORBIDDEN = "FORBIDDEN"  # `expected: false` and running anyway
RED_STATUSES = frozenset({OVER, MISSING, FORBIDDEN})


@dataclass(frozen=True)
class Process:
    """One line of ``ps -eo pid,ppid,ni,pcpu,rss,etime,args``: what it is and what it costs."""

    pid: int
    ppid: int
    nice: int | None  # None for kernel threads, which ps prints as "-"
    cpu_percent: float  # ps's average over the process's life, in percent of ONE core
    rss_mb: float
    elapsed_s: float
    args: str

    @property
    def zombie(self) -> bool:
        """True when the kernel is holding a dead process for a parent that has not reaped it."""
        return self.args.endswith("<defunct>")


@dataclass(frozen=True)
class Budget:
    """A ceiling for one manifest entry: percent of one core, and resident megabytes."""

    cpu_percent: float
    rss_mb: float


@dataclass(frozen=True)
class Entry:
    """One process the board is expected (or forbidden) to run, as the manifest declares it."""

    name: str
    match: str
    role: str
    owner: str
    budget: Budget
    on_board_because: tuple[str, ...] = ()
    components: tuple[tuple[str, str], ...] = ()  # (name, role) of the nodes inside a container
    when: str = "always"  # "always" (missing is red) or "sometimes" (a drive, a timer, a mode)
    expected: bool = True
    note: str = ""

    def matches(self, process: Process) -> bool:
        """True when this entry's regex is found anywhere in the process's command line."""
        return re.search(self.match, process.args) is not None


@dataclass(frozen=True)
class Manifest:
    """The board's process registry: its entries, what to ignore, and how big the board is."""

    entries: tuple[Entry, ...]
    cores: int = 4
    unlisted_cpu_percent: float = 1.0  # an unlisted process below this is noise, not a finding
    ignore: tuple[tuple[str, str], ...] = ()  # (regex, why) — never counted, never reported
    note: str = ""

    def ignored(self, process: Process) -> bool:
        """True when the process is one the manifest deliberately does not account for."""
        return any(re.search(pattern, process.args) for pattern, _ in self.ignore)

    @property
    def promised_cpu_percent(self) -> float:
        """The sum of the budgets of everything expected to run always, in percent of one core."""
        return sum(e.budget.cpu_percent for e in self.entries if e.expected and e.when == "always")

    @property
    def promised_rss_mb(self) -> float:
        """The sum of the memory budgets of everything expected to run always."""
        return sum(e.budget.rss_mb for e in self.entries if e.expected and e.when == "always")


@dataclass(frozen=True)
class Measured:
    """One manifest entry against the live ps: what it costs now, and the verdict on it."""

    entry: Entry
    status: str
    processes: tuple[Process, ...]

    @property
    def cpu_percent(self) -> float:
        """The CPU of every process this entry matched, summed (one role may be several PIDs)."""
        return sum(p.cpu_percent for p in self.processes)

    @property
    def rss_mb(self) -> float:
        """The resident memory of every process this entry matched, summed."""
        return sum(p.rss_mb for p in self.processes)

    @property
    def youngest_s(self) -> float | None:
        """Age of the youngest matched process in seconds; None when nothing matched."""
        return min((p.elapsed_s for p in self.processes), default=None)


@dataclass(frozen=True)
class Load:
    """The kernel's own verdict on how deep the run queue is: /proc/loadavg or uptime."""

    one: float
    five: float
    fifteen: float


@dataclass(frozen=True)
class Census:
    """A whole census: every manifest entry measured, what was unlisted, and the load."""

    manifest: Manifest
    measured: tuple[Measured, ...]
    unlisted: tuple[Process, ...]
    zombies: int
    load: Load

    @property
    def green(self) -> bool:
        """True when nothing is over budget, missing, forbidden or unlisted."""
        return not self.unlisted and not any(m.status in RED_STATUSES for m in self.measured)

    @property
    def problems(self) -> list[str]:
        """One short line per finding, in the order a reader should worry about them."""
        lines = [
            f"{m.entry.name} {m.status}"
            + (
                f" ({m.cpu_percent:.0f} % / {m.rss_mb:.0f} MB vs "
                f"{m.entry.budget.cpu_percent:.0f} % / {m.entry.budget.rss_mb:.0f} MB)"
                if m.status == OVER
                else ""
            )
            for m in self.measured
            if m.status in RED_STATUSES
        ]
        lines += [
            f"UNLISTED {p.cpu_percent:.0f} % pid {p.pid}: {p.args[:60]}" for p in self.unlisted
        ]
        return lines

    @property
    def measured_cpu_percent(self) -> float:
        """Everything the census accounted for, summed, in percent of one core."""
        return sum(m.cpu_percent for m in self.measured) + sum(p.cpu_percent for p in self.unlisted)

    @property
    def measured_rss_mb(self) -> float:
        """The resident memory of everything the census accounted for, summed."""
        return sum(m.rss_mb for m in self.measured) + sum(p.rss_mb for p in self.unlisted)

    @property
    def youngest_s(self) -> float | None:
        """Age of the youngest expected process: under a minute, the CPU numbers are start-up."""
        ages = [m.youngest_s for m in self.measured if m.youngest_s is not None]
        return min(ages) if ages else None


# -- parsing ------------------------------------------------------------------------------------


def parse_elapsed(text: str) -> float:
    """``[[DD-]HH:]MM:SS`` as ps prints it, in seconds ("1-03:30:38" -> 99038.0)."""
    days, _, rest = text.strip().rpartition("-")
    parts = [float(p) for p in rest.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds + (float(days) * 86400 if days else 0.0)


def parse_ps(text: str) -> list[Process]:
    """Every process of a ``ps -eo pid,ppid,ni,pcpu,rss,etime,args`` dump, header and junk skipped.

    Lines that do not start with a PID (the header, a truncated tail) are ignored rather than
    raising: a census must survive a dump that came back short over a flaky link.
    """
    processes = []
    for line in text.splitlines():
        fields = line.split(maxsplit=6)
        if len(fields) < 7 or not fields[0].isdigit():
            continue
        pid, ppid, nice, pcpu, rss, etime, args = fields
        try:
            processes.append(
                Process(
                    pid=int(pid),
                    ppid=int(ppid),
                    nice=int(nice) if nice.lstrip("-").isdigit() else None,
                    cpu_percent=float(pcpu),
                    rss_mb=float(rss) / 1024.0,
                    elapsed_s=parse_elapsed(etime),
                    args=args.strip(),
                )
            )
        except ValueError:
            continue
    return processes


def parse_load(text: str) -> Load:
    """The three load averages out of ``/proc/loadavg`` or an ``uptime`` line.

    Raises ValueError when neither shape is there: a census with no load figure would quietly
    report a calm board.
    """
    stripped = text.strip()
    match = re.search(r"load average[s]?:\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)", stripped)
    if match is None:
        match = re.match(r"^([\d.]+)\s+([\d.]+)\s+([\d.]+)\s", stripped + " ")
    if match is None:
        raise ValueError(f"no load average in {stripped[:80]!r}")
    return Load(float(match.group(1)), float(match.group(2)), float(match.group(3)))


def load_manifest(path: Path | None = None) -> Manifest:
    """Read ``config/board_manifest.json`` (or the file given) into a :class:`Manifest`."""
    path = path or config_file(MANIFEST_FILE)
    return manifest_from_dict(json.loads(path.read_text()))


def manifest_from_dict(data: dict[str, Any]) -> Manifest:
    """Build a manifest from parsed JSON; an entry missing a required field raises KeyError."""
    entries = tuple(
        Entry(
            name=item["name"],
            match=item["match"],
            role=item["role"],
            owner=item["owner"],
            budget=Budget(
                cpu_percent=float(item["budget"]["cpu_percent"]),
                rss_mb=float(item["budget"]["rss_mb"]),
            ),
            on_board_because=tuple(item.get("on_board_because", ())),
            components=tuple((c["name"], c["role"]) for c in item.get("components", ())),
            when=item.get("when", "always"),
            expected=bool(item.get("expected", True)),
            note=item.get("note", ""),
        )
        for item in data["processes"]
    )
    return Manifest(
        entries=entries,
        cores=int(data.get("cores", 4)),
        unlisted_cpu_percent=float(data.get("unlisted_cpu_percent", 1.0)),
        ignore=tuple((i["match"], i["why"]) for i in data.get("ignore", ())),
        note=data.get("note", ""),
    )


# -- the census ---------------------------------------------------------------------------------


def take_census(manifest: Manifest, ps_text: str, load_text: str) -> Census:
    """Match a ``ps`` dump against the manifest and return the verdict.

    A process belongs to the FIRST entry whose regex matches it, so a specific entry placed
    above a general one (the zenoh bridge above ``docker``) keeps its own budget. Zombies are
    counted and then left out of the accounting: they cost no CPU, only a slot.
    """
    processes = parse_ps(ps_text)
    # Zombies are counted over the whole dump: a dead process wears its parent's name in
    # brackets and would otherwise be filtered away as a kernel thread.
    zombies = sum(1 for p in processes if p.zombie)
    alive = [p for p in processes if not p.zombie and not manifest.ignored(p)]
    taken: set[int] = set()
    measured = []
    for entry in manifest.entries:
        mine = tuple(p for p in alive if p.pid not in taken and entry.matches(p))
        taken.update(p.pid for p in mine)
        measured.append(Measured(entry=entry, status=_status(entry, mine), processes=mine))
    unlisted = tuple(
        p for p in alive if p.pid not in taken and p.cpu_percent >= manifest.unlisted_cpu_percent
    )
    return Census(
        manifest=manifest,
        measured=tuple(measured),
        unlisted=unlisted,
        zombies=zombies,
        load=parse_load(load_text),
    )


def _status(entry: Entry, processes: tuple[Process, ...]) -> str:
    """The verdict on one entry: OK, OVER, MISSING, IDLE or FORBIDDEN."""
    if not processes:
        return IDLE if (entry.when == "sometimes" or not entry.expected) else MISSING
    if not entry.expected:
        return FORBIDDEN
    cpu = sum(p.cpu_percent for p in processes)
    rss = sum(p.rss_mb for p in processes)
    over = cpu > entry.budget.cpu_percent or rss > entry.budget.rss_mb
    return OVER if over else OK


# -- reports ------------------------------------------------------------------------------------


def format_census(census: Census) -> str:
    """The census as a table for a terminal: one line per entry, then the findings and a verdict."""
    manifest = census.manifest
    lines = [
        f"{'PROCESS':<18} {'STATUS':<9} {'CPU%':>6} {'/ BUDGET':>9} {'RSS MB':>7} {'/ BUDGET':>9}",
        "-" * 62,
    ]
    for m in census.measured:
        if m.status == IDLE:  # nothing to measure: a drive is not on, a mode is not in use
            lines.append(
                f"{m.entry.name:<18} {m.status:<9} {'-':>6} {m.entry.budget.cpu_percent:>9.0f}"
                f" {'-':>7} {m.entry.budget.rss_mb:>9.0f}"
            )
            continue
        lines.append(
            f"{m.entry.name:<18} {m.status:<9} {m.cpu_percent:>6.1f} "
            f"{m.entry.budget.cpu_percent:>9.0f} {m.rss_mb:>7.0f} {m.entry.budget.rss_mb:>9.0f}"
        )
    for p in census.unlisted:
        lines.append(
            f"{'(unlisted)':<18} {'UNLISTED':<9} {p.cpu_percent:>6.1f} {'-':>9}"
            f" {p.rss_mb:>7.0f} {'-':>9}"
        )
        lines.append(f"{'':<18} pid {p.pid}: {p.args[:70]}")
    cores = manifest.cores
    lines += [
        "-" * 62,
        f"load {census.load.one:.2f} / {census.load.five:.2f} / {census.load.fifteen:.2f}"
        f" on {cores} cores"
        f" ({100 * census.load.one / cores:.0f} % of the board over the last minute)",
        f"the manifest promises {manifest.promised_cpu_percent:.0f} % of {100 * cores} %"
        f" ({manifest.promised_cpu_percent / cores:.0f} % of the board)"
        f" and {manifest.promised_rss_mb:.0f} MB",
        f"measured {census.measured_cpu_percent:.0f} % and {census.measured_rss_mb:.0f} MB"
        f", zombies {census.zombies}",
    ]
    youngest = census.youngest_s
    if youngest is not None and youngest < 60:
        lines.append(
            f"the youngest expected process is {youngest:.0f} s old: these CPU numbers are"
            " start-up averages, not the running cost"
        )
    if census.green:
        lines.append("VERDICT: green — every process accounted for, every budget kept")
    else:
        lines.append("VERDICT: red")
        lines += [f"  - {problem}" for problem in census.problems]
    return "\n".join(lines)


def census_json(census: Census) -> dict[str, Any]:
    """The same census as data for a tool: entries, unlisted processes, load, verdict."""
    return {
        "green": census.green,
        "load": {
            "one": census.load.one,
            "five": census.load.five,
            "fifteen": census.load.fifteen,
            "cores": census.manifest.cores,
        },
        "zombies": census.zombies,
        "promised_cpu_percent": census.manifest.promised_cpu_percent,
        "promised_rss_mb": census.manifest.promised_rss_mb,
        "measured_cpu_percent": census.measured_cpu_percent,
        "measured_rss_mb": census.measured_rss_mb,
        "processes": [
            {
                "name": m.entry.name,
                "status": m.status,
                "cpu_percent": round(m.cpu_percent, 1),
                "rss_mb": round(m.rss_mb, 1),
                "budget_cpu_percent": m.entry.budget.cpu_percent,
                "budget_rss_mb": m.entry.budget.rss_mb,
                "pids": [p.pid for p in m.processes],
            }
            for m in census.measured
        ],
        "unlisted": [
            {
                "pid": p.pid,
                "cpu_percent": p.cpu_percent,
                "rss_mb": round(p.rss_mb, 1),
                "args": p.args,
            }
            for p in census.unlisted
        ],
        "problems": census.problems,
    }


def format_manifest(manifest: Manifest) -> str:
    """The registry itself as a table: what runs on the board, why it is there, what it may cost."""
    lines = [
        f"{'PROCESS':<18} {'CPU%':>5} {'RSS MB':>7}  {'WHY ON THE BOARD':<34} ROLE / OWNER",
        "-" * 110,
    ]
    for e in manifest.entries:
        why = ", ".join(e.on_board_because) or ("not expected" if not e.expected else "-")
        mark = "" if e.expected else "  [NOT EXPECTED]"
        when = "" if e.when == "always" else f"  [{e.when}]"
        lines.append(
            f"{e.name:<18} {e.budget.cpu_percent:>5.0f} {e.budget.rss_mb:>7.0f}  {why:<34}"
            f" {e.role}{mark}{when}"
        )
        lines.append(f"{'':<18} {'':>5} {'':>7}  {'':<34} owner: {e.owner}")
        for name, role in e.components:
            lines.append(f"{'':<18} {'':>5} {'':>7}  {'':<34}   - {name}: {role}")
    lines += [
        "-" * 110,
        f"{len(manifest.entries)} entries; the ones expected always promise"
        f" {manifest.promised_cpu_percent:.0f} % of {100 * manifest.cores} %"
        f" and {manifest.promised_rss_mb:.0f} MB on {manifest.cores} cores",
    ]
    return "\n".join(lines)


def main() -> int:
    """Read a ``ps`` dump and a load line from stdin, print the census; 1 on a red verdict.

    The input is what ``ros/board.sh census`` brings over ssh: a ``### load`` section and a
    ``### ps`` section. ``--json`` prints the data instead of the table, ``--manifest`` prints
    the registry alone and reads no input at all, and ``--command`` prints the shell command a
    caller must run on the board to produce that input (one source of truth for the dump).
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="The board's process census.")
    parser.add_argument("--json", action="store_true", help="the census as JSON")
    parser.add_argument("--manifest", action="store_true", help="print the registry, read nothing")
    parser.add_argument("--file", type=Path, default=None, help="a manifest other than config/")
    parser.add_argument(
        "--command", action="store_true", help="print the board-side shell command and exit"
    )
    args = parser.parse_args()
    if args.command:
        print(CENSUS_COMMAND)
        return 0
    manifest = load_manifest(args.file)
    if args.manifest:
        print(format_manifest(manifest))
        return 0
    sections = split_sections(sys.stdin.read())
    census = take_census(manifest, sections.get("ps", ""), sections.get("load", ""))
    print(json.dumps(census_json(census), indent=2) if args.json else format_census(census))
    return 0 if census.green else 1


def split_sections(text: str) -> dict[str, str]:
    """Split a ``### name``-delimited dump into its named sections (what board.sh sends)."""
    sections: dict[str, list[str]] = {}
    current = ""
    for line in text.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line)
    return {name: "\n".join(body) for name, body in sections.items()}


if __name__ == "__main__":
    raise SystemExit(main())
