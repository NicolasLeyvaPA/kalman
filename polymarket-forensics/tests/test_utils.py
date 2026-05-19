"""Tests for shared utility helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from utils.time import to_utc, utc_now


class TestToUtc:
    def test_none(self):
        assert to_utc(None) is None

    def test_naive_datetime_treated_as_utc(self):
        d = datetime(2026, 5, 18, 12, 0, 0)
        result = to_utc(d)
        assert result is not None
        assert result.tzinfo is not None
        assert result.tzinfo.utcoffset(result).total_seconds() == 0

    def test_aware_datetime_preserved(self):
        d = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
        assert to_utc(d) == d

    def test_unix_timestamp_int(self):
        result = to_utc(1747560000)
        assert result is not None
        assert result.tzinfo is not None

    def test_unix_timestamp_float(self):
        result = to_utc(1747560000.123)
        assert result is not None

    def test_iso_string(self):
        result = to_utc("2026-05-18T12:00:00Z")
        assert result is not None
        assert result.year == 2026

    def test_iso_string_with_offset(self):
        result = to_utc("2026-05-18T12:00:00+02:00")
        assert result is not None
        assert result.hour == 10  # converted to UTC

    def test_numeric_string(self):
        result = to_utc("1747560000")
        assert result is not None

    def test_garbage_string(self):
        assert to_utc("not a date") is None

    def test_empty_string(self):
        assert to_utc("") is None


def test_utc_now_is_aware():
    assert utc_now().tzinfo is not None
