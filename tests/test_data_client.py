"""Tests for the data layer: API client, storage, and synthetic data."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.models import MarketInfo, MarketObservation, PriceHistory, PricePoint
from src.data.polymarket_client import PolymarketClient
from src.data.storage import MarketStorage
from src.data.synthetic import (
    generate_random_walk,
    generate_sine_wave,
    generate_step_change,
    synthetic_to_price_history,
)


class TestPolymarketClient:
    """Tests for the Polymarket API client."""

    def test_cache_key_deterministic(self) -> None:
        """Same URL and params should produce the same cache key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = PolymarketClient(cache_dir=tmpdir)
            key1 = client._cache_key("https://example.com", {"a": 1})
            key2 = client._cache_key("https://example.com", {"a": 1})
            assert key1 == key2

    def test_cache_key_different_for_different_params(self) -> None:
        """Different params should produce different cache keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = PolymarketClient(cache_dir=tmpdir)
            key1 = client._cache_key("https://example.com", {"a": 1})
            key2 = client._cache_key("https://example.com", {"a": 2})
            assert key1 != key2

    @patch("src.data.polymarket_client.requests.Session.get")
    def test_get_with_caching(self, mock_get: MagicMock) -> None:
        """First call should hit the API, second should use cache."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"id": "test"}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            client = PolymarketClient(cache_dir=tmpdir, rate_limit=0)
            result1 = client._get("https://example.com/api", {"key": "val"})
            result2 = client._get("https://example.com/api", {"key": "val"})

            assert result1 == [{"id": "test"}]
            assert result2 == [{"id": "test"}]
            assert mock_get.call_count == 1  # Second call served from cache

    @patch("src.data.polymarket_client.requests.Session.get")
    def test_get_markets(self, mock_get: MagicMock) -> None:
        """get_markets should return list of market dicts."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "1", "question": "Test?", "volume": "1000000"}
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            client = PolymarketClient(cache_dir=tmpdir, rate_limit=0)
            markets = client.get_markets(limit=5)
            assert len(markets) == 1
            assert markets[0]["question"] == "Test?"


class TestMarketStorage:
    """Tests for SQLite storage."""

    def test_save_and_retrieve_market(self) -> None:
        """Saving a market and retrieving its data should round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.db"
            storage = MarketStorage(db_path=db_path)

            market = MarketInfo(
                market_id="test-123",
                question="Will it rain?",
                token_id="token-abc",
                slug="will-it-rain",
                category="weather",
            )
            storage.save_market(market)

            # Save and retrieve price history
            history = PriceHistory(
                market_id="test-123",
                question="Will it rain?",
                points=[
                    PricePoint(datetime(2025, 1, 1, tzinfo=timezone.utc), 0.5),
                    PricePoint(datetime(2025, 1, 2, tzinfo=timezone.utc), 0.6),
                ],
            )
            storage.save_price_history(history)

            retrieved = storage.get_price_history("test-123")
            assert len(retrieved.points) == 2
            assert retrieved.points[0].price == 0.5
            assert retrieved.question == "Will it rain?"
            storage.close()

    def test_save_and_retrieve_observation(self) -> None:
        """Observations should round-trip through the database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.db"
            storage = MarketStorage(db_path=db_path)

            obs = MarketObservation(
                timestamp=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
                market_id="test-obs",
                market_question="Test question",
                yes_price=0.65,
                no_price=0.35,
                midpoint=0.65,
                spread=0.02,
                volume_24h=500000.0,
                best_bid=0.64,
                best_ask=0.66,
                bid_depth=10000.0,
                ask_depth=8000.0,
                total_depth=18000.0,
                imbalance=10000.0 / 18000.0,
                num_trades_1h=42,
            )
            storage.save_observation(obs)

            results = storage.get_observations("test-obs")
            assert len(results) == 1
            assert abs(results[0].yes_price - 0.65) < 1e-10
            assert results[0].num_trades_1h == 42
            storage.close()

    def test_export_csv(self) -> None:
        """CSV export should write valid data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.db"
            csv_path = f"{tmpdir}/export.csv"
            storage = MarketStorage(db_path=db_path)

            history = PriceHistory(
                market_id="csv-test",
                question="CSV test",
                points=[
                    PricePoint(datetime(2025, 1, 1, tzinfo=timezone.utc), 0.42),
                ],
            )
            storage.save_price_history(history)
            storage.export_to_csv("csv-test", csv_path)

            assert Path(csv_path).exists()
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 2  # header + 1 row
            assert "0.42" in lines[1]
            storage.close()


class TestSyntheticData:
    """Tests for synthetic data generation."""

    def test_random_walk_shape(self) -> None:
        """Generated data should have correct dimensions."""
        data = generate_random_walk(n_steps=100, seed=42)
        assert len(data.true_states) == 100
        assert len(data.observations) == 100
        assert len(data.timestamps) == 100

    def test_random_walk_bounded(self) -> None:
        """All prices should be in [0, 1]."""
        data = generate_random_walk(n_steps=1000, Q=1e-3, R=1e-2, seed=42)
        assert data.observations.min() >= 0.0
        assert data.observations.max() <= 1.0
        assert data.true_states.min() >= 0.0
        assert data.true_states.max() <= 1.0

    def test_random_walk_noise_levels(self) -> None:
        """Observations should be noisier than true states."""
        data = generate_random_walk(n_steps=5000, Q=1e-5, R=1e-2, seed=42)
        obs_var = np.var(np.diff(data.observations))
        true_var = np.var(np.diff(data.true_states))
        assert obs_var > true_var

    def test_step_change_has_jump(self) -> None:
        """Step change data should show a clear shift."""
        data = generate_step_change(
            n_steps=500, step_time=250,
            step_from=0.3, step_to=0.7, Q=1e-6, seed=42
        )
        mean_before = data.true_states[:250].mean()
        mean_after = data.true_states[250:].mean()
        assert abs(mean_before - 0.3) < 0.05
        assert abs(mean_after - 0.7) < 0.05

    def test_sine_wave_shape(self) -> None:
        """Sine wave should oscillate around center."""
        data = generate_sine_wave(n_steps=200, center=0.5, amplitude=0.2, seed=42)
        assert data.true_states.min() < 0.4
        assert data.true_states.max() > 0.6

    def test_synthetic_to_price_history(self) -> None:
        """Conversion to PriceHistory should preserve data."""
        data = generate_random_walk(n_steps=50, seed=42)
        history = synthetic_to_price_history(data)
        assert len(history.points) == 50
        assert history.market_id == data.market_id

    def test_reproducibility(self) -> None:
        """Same seed should produce identical data."""
        data1 = generate_random_walk(n_steps=100, seed=42)
        data2 = generate_random_walk(n_steps=100, seed=42)
        np.testing.assert_array_equal(data1.observations, data2.observations)
        np.testing.assert_array_equal(data1.true_states, data2.true_states)

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different data."""
        data1 = generate_random_walk(n_steps=100, seed=42)
        data2 = generate_random_walk(n_steps=100, seed=99)
        assert not np.array_equal(data1.observations, data2.observations)
