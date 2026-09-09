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
