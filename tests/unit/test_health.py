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
