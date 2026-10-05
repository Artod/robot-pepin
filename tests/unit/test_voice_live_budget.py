"""The caps that refuse a paid session and the cost estimate behind them."""

import json
import time
from pathlib import Path

import pytest

from pepin.voice_live.budget import Ledger, SessionCost, Usage
from pepin.voice_live.config import LiveConfig, Prices

NOON = time.mktime((2026, 10, 2, 12, 0, 0, 0, 0, -1))  # local noon: no midnight inside a test


def test_the_checked_in_config_loads_and_its_caps_hold() -> None:
    config = LiveConfig.load()
    assert config.model == "gemini-3.8-live"
    assert 0 < config.max_session_s <= 900
    assert config.daily_budget_cad > 0 and config.max_sessions_per_hour >= 1
    assert config.daily_budget_usd == pytest.approx(
        config.daily_budget_cad / config.prices.usd_to_cad
    )
    # the persona's lines are named in the config, not in the code
    assert config.persona_acknowledge and config.persona_arrived
    assert config.mic_while_driving == "gated" and config.report_drive_start
    assert config.idle_after_drive_s < config.idle_close_s
    assert config.resume_valid_s <= 300  # a resumed context is re-billed at every pass


@pytest.mark.parametrize(
    "change",
    [
        {"caps": {"max_session_s": 0}},
        {"caps": {"max_session_s": 1000}},
        {"caps": {"daily_budget_cad": 0}},
        {"caps": {"max_sessions_per_hour": 0}},
        {"session": {"compression_trigger_tokens": 100, "compression_target_tokens": 200}},
        {"session": {"idle_after_drive_s": 0}},
        {"session": {"mic_while_driving": "muted"}},
    ],
)
def test_a_cap_without_its_point_is_refused(change: dict[str, dict[str, float]]) -> None:
    with pytest.raises(ValueError):
        LiveConfig.from_dict(change)


def test_overrides_are_checked_too() -> None:
    config = LiveConfig()
    assert config.with_overrides(idle_close_s=6.0, max_session_s=None).idle_close_s == 6.0
    with pytest.raises(ValueError):
        config.with_overrides(max_session_s=-1.0)


def test_usage_prices_each_modality() -> None:
    prices = Prices(audio_in=3.0, audio_out=12.0, text_in=0.75, text_out=4.5)
    usage = Usage(audio_in=1_000_000, text_in=1_000_000, audio_out=1_000_000, text_out=1_000_000)
    assert usage.usd(prices) == pytest.approx(3.0 + 0.75 + 12.0 + 4.5)
    usage.add(Usage(audio_in=10, reports=1))
    assert usage.audio_in == 1_000_010 and usage.reports == 1


def test_a_session_costs_its_passes_as_counted_plus_every_second_streamed() -> None:
    prices = Prices(audio_in=3.0, audio_out=12.0, text_in=0.75, text_out=4.5)
    cost = SessionCost(prices)
    cost.audio_in_s = 60.0  # a minute streamed, silence or not: 0.0045 USD (the docs' 0.005/min)
    assert cost.stream_usd == pytest.approx(0.0045) and cost.usd == pytest.approx(0.0045)
    # 2026-10-05's first drive in a fresh session: two passes (2.9k and 3.0k tokens of context)
    cost.usage.add(Usage(audio_in=486, text_in=5256, text_out=154, reports=1))
    cost.usage.add(Usage(audio_in=243, text_in=2721, audio_out=43, text_out=74, reports=1))
    assert cost.usage_usd == pytest.approx(0.0097, abs=1e-4)
    assert cost.usd == pytest.approx(cost.usage_usd + 0.0045)


def test_the_ledger_refuses_past_the_hourly_count_and_the_daily_budget(tmp_path: Path) -> None:
    now = [NOON]
    config = LiveConfig(max_sessions_per_hour=2, daily_budget_cad=1.38)  # 1.00 USD a day
    ledger = Ledger(tmp_path / "ledger.jsonl", config, clock=lambda: now[0])
    assert ledger.refusal() is None
    ledger.record("a", now[0], 0.10, closed=True)
    ledger.record("b", now[0], 0.10, closed=True)
    assert "2 sessions in the last hour" in str(ledger.refusal())
    now[0] += 3601  # an hour later the count is clear; today's spend is not
    assert ledger.refusal() is None
    ledger.record("c", now[0], 0.80)
    assert "daily budget 1.38 CAD" in str(ledger.refusal())
    now[0] += 86400  # tomorrow
    assert ledger.refusal() is None


def test_a_crashed_session_still_counts_its_last_progress_and_a_new_process_reads_it(
    tmp_path: Path,
) -> None:
    config = LiveConfig(daily_budget_cad=1.38)
    path = tmp_path / "ledger.jsonl"
    t = NOON
    first = Ledger(path, config, clock=lambda: t)
    first.record("x", t, 0.0)
    first.record("x", t, 0.4)  # progress; the process then dies without the closing line
    second = Ledger(path, config, clock=lambda: t)
    assert second.today_usd() == pytest.approx(0.4)
    assert second.over_budget("y", 0.6) and not second.over_budget("x", 0.9)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["closed"] for r in rows] == [False, False]


def test_an_unreadable_ledger_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text('not json\n{"sid": "a", "t_open": 1.0, "usd": 0.5}\n')
    assert Ledger(path, LiveConfig()).opened_last_hour(now=2.0) == 1
