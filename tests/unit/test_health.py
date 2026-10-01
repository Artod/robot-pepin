"""Health probes: the parts that need no board — parsing what ssh brought back."""

from pepin.health import _parse_local_ports


def test_busy_ports_come_from_the_local_address_column() -> None:
    out = (
        "0      0      10.0.0.187:3333   10.0.0.42:52110\n"
        "0      0      [::ffff:10.0.0.187]:3334   [::ffff:10.0.0.42]:52111\n"
    )
    assert _parse_local_ports(out) == {3333, 3334}


def test_busy_ports_ignore_a_header_and_empty_output() -> None:
    assert _parse_local_ports("Recv-Q Send-Q Local Address:Port Peer Address:Port\n") == set()
    assert _parse_local_ports("") == set()


def test_the_boards_own_loopback_connection_is_not_a_busy_port() -> None:
    out = (
        "0      0      127.0.0.1:3333   127.0.0.1:41234\n"  # the base server owning the bus
        "0      0      10.0.0.187:3334  10.0.0.42:52111\n"  # a laptop driving the lidar
    )
    assert _parse_local_ports(out) == {3334}


def test_a_red_result_is_rechecked_soon_and_only_a_settled_green_one_waits_five_minutes() -> None:
    """A ten-second Wi-Fi flap once painted the tray red for five minutes (2026-09-09)."""
    from pepin.health import FAST_POLL_S, SLOW_POLL_S, is_stale, next_poll_s

    assert next_poll_s(all_go=False, fast_for_left_s=-1.0) == FAST_POLL_S  # NO GO: look again soon
    assert next_poll_s(all_go=None, fast_for_left_s=-1.0) == FAST_POLL_S  # unreachable: likewise
    assert next_poll_s(all_go=True, fast_for_left_s=10.0) == FAST_POLL_S  # just started / refreshed
    assert next_poll_s(all_go=True, fast_for_left_s=-1.0) == SLOW_POLL_S  # settled: slow
    assert not is_stale(age_s=59.0, cadence_s=30.0) and is_stale(age_s=61.0, cadence_s=30.0)


def test_the_laptop_stack_is_green_only_with_the_mapper_up() -> None:
    """The laptop's half without ssh: docker ps' name+status lines, the mapper decides."""
    from pepin.health import probe_laptop_containers

    both = "pepin-vslam Up 2 hours\npepin-macnav Up 35 minutes\npepin-zrouter-laptop Up 2 hours\n"
    green = probe_laptop_containers(ps_output=both)
    assert green.ok and green.detail == "pepin-vslam up 2 hours, pepin-macnav up 35 minutes"
    red = probe_laptop_containers(ps_output="pepin-macnav Up 35 minutes\n")
    assert not red.ok and "pepin-vslam not running" in red.detail and "pepin-macnav" in red.detail
    assert not probe_laptop_containers(ps_output="").ok
    exited = probe_laptop_containers(ps_output="pepin-vslam Exited (0) 3 hours ago\n")
    assert not exited.ok


def test_a_docker_daemon_that_is_off_is_a_red_line_not_an_exception() -> None:
    import subprocess

    from pepin.health import DOCKER_PS, probe_laptop_containers

    def dead(argv: list[str]) -> subprocess.CompletedProcess[str]:
        assert argv == DOCKER_PS
        return subprocess.CompletedProcess(argv, 1, "", "Cannot connect to the Docker daemon")

    probe = probe_laptop_containers(run=dead)
    assert not probe.ok and probe.detail == "docker daemon not running"

    def missing(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("docker")

    assert not probe_laptop_containers(run=missing).ok


def test_the_goal_server_probe_names_the_side_nav2_runs_on() -> None:
    from unittest import mock

    from pepin import goal_link, health

    with mock.patch.object(goal_link, "find_server", lambda board, port: "127.0.0.1"):
        local = health.probe_goal_server("10.0.0.187")
    assert local.ok and local.detail == "127.0.0.1:3337 (Nav2 on this Mac)"
    with mock.patch.object(goal_link, "find_server", lambda board, port: board):
        board = health.probe_goal_server("10.0.0.187")
    assert board.ok and board.detail == "10.0.0.187:3337 (Nav2 on the board)"
    with mock.patch.object(goal_link, "find_server", lambda board, port: None):
        nobody = health.probe_goal_server("10.0.0.187")
        alone = health.probe_goal_server(None)
    assert not nobody.ok and nobody.detail == "nobody listens on 127.0.0.1:3337 or 10.0.0.187:3337"
    assert alone.detail == "nobody listens on 127.0.0.1:3337"
