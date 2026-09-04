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
