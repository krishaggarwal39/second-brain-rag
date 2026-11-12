"""R-39: daily counters reset at UTC midnight; telemetry never exports on the request path."""
from datetime import datetime, timezone

import pytest

from app.core import cache


def _ts(y, m, d, h=0, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc).timestamp()


def test_daily_key_rolls_over_at_utc_midnight():
    before = cache.daily_key("queries_today", now=_ts(2026, 11, 12, 23, 59))
    after = cache.daily_key("queries_today", now=_ts(2026, 11, 13, 0, 0))
    assert before == "queries_today:2026-11-12"
    assert after == "queries_today:2026-11-13"


@pytest.mark.asyncio
async def test_counter_for_new_day_starts_at_zero(monkeypatch):
    monkeypatch.setattr(cache, "_redis_client", None)
    day1 = cache.daily_key("q_test", now=_ts(2030, 1, 1, 12))
    day2 = cache.daily_key("q_test", now=_ts(2030, 1, 2, 0, 1))
    await cache.increment_metric(day1, 3, ttl=3600)
    assert await cache.get_metric(day1) == 3
    assert await cache.get_metric(day2) == 0


@pytest.mark.asyncio
async def test_local_ttl_expires(monkeypatch):
    monkeypatch.setattr(cache, "_redis_client", None)
    now = [1000.0]
    monkeypatch.setattr(cache.time, "time", lambda: now[0])
    await cache.increment_metric("ttl_metric_test", 1, ttl=10)
    assert await cache.get_metric("ttl_metric_test") == 1
    now[0] += 11
    assert await cache.get_metric("ttl_metric_test") == 0
    await cache.increment_metric("ttl_metric_test", 1, ttl=10)  # restarts after expiry
    assert await cache.get_metric("ttl_metric_test") == 1


def test_telemetry_defaults_to_no_export(monkeypatch):
    from app.core import telemetry

    monkeypatch.setattr(telemetry, "_initialized", False)
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    added = []
    monkeypatch.setattr(telemetry.TracerProvider, "add_span_processor",
                        lambda self, p: added.append(type(p).__name__))
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", lambda p: None)
    telemetry.init_telemetry(app=None)
    assert added == []

    monkeypatch.setattr(telemetry, "_initialized", False)
    monkeypatch.setenv("OTEL_EXPORTER", "console")
    telemetry.init_telemetry(app=None)
    assert added == ["BatchSpanProcessor"]
    monkeypatch.setattr(telemetry, "_initialized", False)
