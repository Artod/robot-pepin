"""The board census over a recorded ps dump: parsing, matching, budgets and the verdict."""
# ruff: noqa: E501 — the dump below is a real ps output and its lines are as long as they were.

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pepin.census import (
    FORBIDDEN,
    IDLE,
    MISSING,
    OK,
    OVER,
    Budget,
    Entry,
    Manifest,
    census_json,
    format_census,
    format_manifest,
    load_manifest,
    manifest_from_dict,
    parse_elapsed,
    parse_load,
    parse_ps,
    split_sections,
    take_census,
)

# A trimmed real dump from the board (2026-09-14 12:40 EDT), plus a kernel thread, the census's
# own ps, a zombie and an intruder that no manifest entry claims.
PS_DUMP = """    PID    PPID  NI %CPU   RSS     ELAPSED COMMAND
 122778  122777   0  400  3556       00:00 ps -eo pid,ppid,ni,pcpu,rss,etime,args --sort=-pcpu
 122289  122098   5 79.9 101652      02:22 /opt/ros/jazzy/lib/rclcpp_components/component_container_isolated --ros-args -r __node:=nav2_container -r __ns:=/
 122285  122098 -10 21.1 59872       02:22 /opt/ros/jazzy/lib/rclcpp_components/component_container_isolated --ros-args -r __node:=sensors_container -r __ns:=/
   9098       1   0 10.3 17852  1-01:38:33 /opt/pepin/bin/python -m pepin.base_server --config /opt/pepin/config/base.json
    464       2 -20  1.3     0  1-03:31:58 [SPRDWL_TX_QUEUE]
 122900  122098   0 44.0 40000       00:30 /usr/bin/python3 /opt/ros/jazzy/bin/ros2 topic hz /scan
 122901       1   0  0.0     0       00:10 [python3] <defunct>
"""

LOAD = "7.30 11.71 13.43 1/415 122861"

MANIFEST = Manifest(
    entries=(
        Entry(
            name="nav2_container",
            match="component_container_isolated .*__node:=nav2_container",
            role="Nav2 in one process",
            owner="pepin-ros.service",
            budget=Budget(cpu_percent=120.0, rss_mb=152.0),
            on_board_because=("real-time",),
            components=(("controller_server", "the 8 Hz loop"),),
        ),
        Entry(
            name="sensors_container",
            match="component_container_isolated .*__node:=sensors_container",
            role="the sensing half",
            owner="pepin-ros.service",
            budget=Budget(cpu_percent=32.0, rss_mb=40.0),  # deliberately under the measured 58 MB
        ),
        Entry(
            name="base_server",
            match=r"pepin\.base_server",
            role="the wheels next to the UART",
            owner="pepin-base.service",
            budget=Budget(cpu_percent=16.0, rss_mb=27.0),
        ),
        Entry(
            name="session_logger",
            match=r"session_logger\.py",
            role="the per-drive recorder",
            owner="ros/goto.sh",
            budget=Budget(cpu_percent=25.0, rss_mb=90.0),
            when="sometimes",
        ),
        Entry(
            name="tof_bridge",
            match=r"pepin_bringup\.tof_bridge",
            role="the ToF ranges as ROS messages",
            owner="pepin-ros.service",
            budget=Budget(cpu_percent=20.0, rss_mb=102.0),
        ),
        Entry(
            name="foxglove_bridge",
            match="foxglove_bridge",
            role="the Foxglove websocket, being removed",
            owner="was a component of sensors_container",
            budget=Budget(cpu_percent=0.0, rss_mb=0.0),
            expected=False,
        ),
    ),
    cores=4,
    unlisted_cpu_percent=1.0,
    ignore=(
        (r"^\[", "kernel threads"),
        ("ps -eo pid,ppid,ni,pcpu,rss,etime,args", "the census's own ps"),
    ),
)


def census():
    return take_census(MANIFEST, PS_DUMP, LOAD)


def test_parse_elapsed_reads_every_ps_shape() -> None:
    assert parse_elapsed("00:30") == 30.0
    assert parse_elapsed("02:22") == 142.0
    assert parse_elapsed("01:13:40") == 4420.0
    assert parse_elapsed("1-03:30:38") == 99038.0


def test_parse_ps_reads_fields_and_skips_the_header() -> None:
    processes = parse_ps(PS_DUMP)
    assert len(processes) == 7
    nav = next(p for p in processes if "nav2_container" in p.args)
    assert (nav.pid, nav.ppid, nav.nice) == (122289, 122098, 5)
    assert nav.cpu_percent == 79.9
    assert round(nav.rss_mb) == 99
    assert nav.elapsed_s == 142.0
    kernel = next(p for p in processes if p.args.startswith("["))
    assert kernel.nice == -20
    assert kernel.rss_mb == 0.0


def test_parse_ps_survives_a_truncated_dump() -> None:
    assert parse_ps("garbage\n  12  13\n") == []


def test_parse_load_reads_proc_and_uptime() -> None:
    assert parse_load(LOAD).one == 7.30
    uptime = " 12:38:13 up 1 day,  3:30,  3 users,  load average: 9.75, 13.15, 14.00"
    assert parse_load(uptime) == parse_load("9.75 13.15 14.00 30/412 1")
    with pytest.raises(ValueError):
        parse_load("no load here")


def test_statuses_cover_ok_over_missing_idle_and_forbidden() -> None:
    by_name = {m.entry.name: m for m in census().measured}
    assert by_name["nav2_container"].status == OK  # 80 % of a 120 % budget
    assert by_name["sensors_container"].status == OVER  # 58 MB of a 40 MB budget
    assert by_name["base_server"].status == OK
    assert by_name["tof_bridge"].status == MISSING  # expected always, not in the dump
    assert by_name["session_logger"].status == IDLE  # only during a drive
    assert by_name["foxglove_bridge"].status == IDLE  # not expected, not there: fine


def test_a_forbidden_process_that_runs_is_red() -> None:
    dump = (
        PS_DUMP + "  333  1 0 5.0 20000 01:00 /opt/ros/jazzy/lib/foxglove_bridge/foxglove_bridge\n"
    )
    report = take_census(MANIFEST, dump, LOAD)
    assert next(m for m in report.measured if m.entry.name == "foxglove_bridge").status == FORBIDDEN
    assert not report.green


def test_an_unlisted_hog_is_reported_and_the_ignored_are_not() -> None:
    report = census()
    assert [p.pid for p in report.unlisted] == [122900]  # the stray `ros2 topic hz`
    assert "ros2 topic hz" in report.problems[-1]
    args = [p.args for p in report.unlisted]
    assert not any(a.startswith("[") or a.startswith("ps -eo") for a in args)


def test_a_quiet_unlisted_process_is_noise_not_a_finding() -> None:
    dump = PS_DUMP + "  777  1 0 0.4 2000 05:00 /usr/sbin/irrelevant-daemon\n"
    assert 777 not in [p.pid for p in take_census(MANIFEST, dump, LOAD).unlisted]


def test_zombies_are_counted_and_left_out_of_the_accounting() -> None:
    report = census()
    assert report.zombies == 1
    assert 122901 not in [p.pid for p in report.unlisted]


def test_one_process_belongs_to_the_first_matching_entry_only() -> None:
    manifest = Manifest(
        entries=(
            MANIFEST.entries[0],
            Entry(
                name="everything",
                match="component_container_isolated",
                role="a general entry below a specific one",
                owner="-",
                budget=Budget(cpu_percent=400.0, rss_mb=1000.0),
            ),
        ),
        ignore=MANIFEST.ignore,
    )
    report = take_census(manifest, PS_DUMP, LOAD)
    nav, rest = report.measured
    assert [p.pid for p in nav.processes] == [122289]
    assert [p.pid for p in rest.processes] == [122285]


def test_the_verdict_is_green_only_when_everything_fits() -> None:
    assert not census().green
    roomy = replace(MANIFEST.entries[1], budget=Budget(cpu_percent=32.0, rss_mb=90.0))
    clean = Manifest(
        entries=(MANIFEST.entries[0], roomy, MANIFEST.entries[2]),
        ignore=(*MANIFEST.ignore, ("ros2 topic hz", "the stray tool, for this test")),
    )
    report = take_census(clean, PS_DUMP, LOAD)
    assert report.green
    assert report.problems == []
    assert "VERDICT: green" in format_census(report)


def test_the_table_carries_the_numbers_and_the_findings() -> None:
    text = format_census(census())
    assert "nav2_container" in text and "79.9" in text
    assert "VERDICT: red" in text
    assert "sensors_container OVER" in text and "tof_bridge MISSING" in text
    assert "load 7.30" in text and "on 4 cores" in text
    assert "the manifest promises 188 %" in text  # 120 + 32 + 16 + 20, the always-on entries
    assert "start-up averages" not in text  # the youngest expected process is 142 s old


def test_a_fresh_stack_is_marked_as_start_up_numbers() -> None:
    dump = PS_DUMP.replace("      02:22", "      00:20")
    assert "start-up averages" in format_census(take_census(MANIFEST, dump, LOAD))


def test_json_is_the_same_verdict_as_the_table() -> None:
    data = census_json(census())
    assert data["green"] is False
    assert data["load"]["cores"] == 4
    assert data["zombies"] == 1
    statuses = {p["name"]: p["status"] for p in data["processes"]}
    assert statuses["sensors_container"] == OVER
    assert data["unlisted"][0]["pid"] == 122900
    json.dumps(data)  # a tool must be able to read it


def test_the_manifest_table_says_why_each_process_is_on_the_board() -> None:
    text = format_manifest(MANIFEST)
    assert "nav2_container" in text and "real-time" in text
    assert "controller_server: the 8 Hz loop" in text
    assert "[NOT EXPECTED]" in text and "[sometimes]" in text
    assert "owner: pepin-base.service" in text


def test_the_shipped_manifest_parses_and_covers_the_stack() -> None:
    manifest = load_manifest()
    names = {e.name for e in manifest.entries}
    assert {"nav2_container", "sensors_container", "relocalizer", "base_server", "docker"} <= names
    assert manifest.cores == 4
    for entry in manifest.entries:
        assert entry.role and entry.owner, entry.name
        assert entry.when in {"always", "sometimes"}, entry.name
        assert entry.expected == bool(entry.on_board_because), entry.name  # rule 20: a reason each
    # The board has four cores. The promise is already 94 % of them (measured +50 % on every
    # entry): a manifest that promised MORE than the whole board would be a plan to overload it.
    assert manifest.promised_cpu_percent < 100 * manifest.cores


def test_the_shipped_manifest_matches_a_real_dump() -> None:
    report = take_census(load_manifest(), PS_DUMP, LOAD)
    by_name = {m.entry.name: m for m in report.measured}
    assert by_name["nav2_container"].status == OK
    assert by_name["sensors_container"].status == OK
    assert by_name["base_server"].status == OK
    assert [p.pid for p in report.unlisted] == [122900]  # the stray ros2 CLI tool, nothing else


def test_split_sections_reads_what_board_sh_sends() -> None:
    sections = split_sections("### load\n1.0 2.0 3.0 1/2 3\n### ps\n  1 2 0 0.1 100 00:01 init\n")
    assert sections["load"].strip().startswith("1.0")
    assert parse_ps(sections["ps"])[0].args == "init"


def test_manifest_from_dict_defaults_what_it_may() -> None:
    manifest = manifest_from_dict(
        {
            "processes": [
                {
                    "name": "x",
                    "match": "x",
                    "role": "r",
                    "owner": "o",
                    "budget": {"cpu_percent": 1, "rss_mb": 2},
                }
            ]
        }
    )
    entry = manifest.entries[0]
    assert (entry.when, entry.expected, entry.components) == ("always", True, ())
    assert (manifest.cores, manifest.unlisted_cpu_percent) == (4, 1.0)
