"""Board address resolution: prefer IPv4, remember the answer, fall back to it."""

import socket

import pytest

from pepin import transport


def test_ipv4_answer_is_used_and_cached(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PEPIN_HOST", raising=False)
    monkeypatch.setattr(transport, "_CACHE", tmp_path / "board_ip")
    monkeypatch.setattr(transport, "ipv4_from_arp", lambda *a, **k: None)
    infos = [(socket.AF_INET, 0, 0, "", ("10.0.0.187", 0))]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: infos)
    assert transport.board_address("pepin.local") == "10.0.0.187"
    assert (tmp_path / "board_ip").read_text() == "10.0.0.187"


ARP = """? (10.0.0.1) at 3c:2d:9e:bc:3a:ca on en0 ifscope [ethernet]
? (10.0.0.187) at 6c:35:cd:1:44:cb on en0 ifscope [ethernet]
? (10.0.0.255) at ff:ff:ff:ff:ff:ff on en0 ifscope [ethernet]
"""


def test_arp_lookup_finds_the_board_despite_macos_octet_formatting() -> None:
    assert transport.ipv4_from_arp("6c:35:cd:01:44:cb", ARP) == "10.0.0.187"
    assert transport.ipv4_from_arp("aa:bb:cc:dd:ee:ff", ARP) is None


def test_arp_answers_before_mdns_is_even_asked(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PEPIN_HOST", raising=False)
    monkeypatch.setattr(transport, "_CACHE", tmp_path / "board_ip")

    def never(*a, **k):
        raise AssertionError("mDNS must not be consulted when ARP knows the board")

    monkeypatch.setattr(socket, "getaddrinfo", never)
    monkeypatch.setattr(transport, "ipv4_from_arp", lambda *a, **k: "10.0.0.187")
    assert transport.board_address("pepin.local", attempts=1, pause_s=0.0) == "10.0.0.187"


def test_env_override_wins(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PEPIN_HOST", "10.0.0.50")
    monkeypatch.setattr(transport, "_CACHE", tmp_path / "board_ip")
    monkeypatch.setattr(transport, "ipv4_from_arp", lambda *a, **k: "10.0.0.187")
    assert transport.board_address() == "10.0.0.50"


def test_falls_back_to_the_cached_address_when_mdns_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PEPIN_HOST", raising=False)
    cache = tmp_path / "board_ip"
    cache.write_text("10.0.0.187")
    monkeypatch.setattr(transport, "_CACHE", cache)

    def fail(*a, **k):
        raise socket.gaierror("nodename nor servname provided")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    monkeypatch.setattr(transport, "ipv4_from_arp", lambda *a, **k: None)
    assert transport.board_address("pepin.local", attempts=2, pause_s=0.0) == "10.0.0.187"


def test_raises_without_any_address(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PEPIN_HOST", raising=False)
    monkeypatch.setattr(transport, "_CACHE", tmp_path / "missing")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    monkeypatch.setattr(transport, "ipv4_from_arp", lambda *a, **k: None)
    with pytest.raises(ConnectionError):
        transport.board_address("pepin.local", attempts=1, pause_s=0.0)
