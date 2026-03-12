# Polymarket Kalman Filter

A production-grade Kalman filter system for estimating the true underlying probability of Polymarket prediction market events by filtering out noise from raw market prices.

## Why This Exists

Prediction market prices are noisy observations of an underlying truth. A market asking "Will X happen?" might show a price of 0.65, but that number is distorted by vig (platform margin), thin order books, behavioral biases (YES bias, favorite-longshot bias), stale quotes, temporary liquidity imbalances, and panic buying/selling. The *true* probability might be 0.62 — or 0.68. You can't tell from a single price.

The Kalman filter is a Bayesian state estimator that separates signal from noise in real time. It maintains an estimate of the true probability along with a measure of uncertainty, updating both as new price observations arrive. When the market is liquid and actively traded, the filter trusts observations more. When spreads are wide and books are thin, it trusts its own estimate more.

This project implements the Kalman filter from scratch (no `filterpy` or `pykalman`), building from the simplest scalar case through adaptive, logit-space, and multivariate variants that track correlated market clusters.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run the basic filter on synthetic data
python -c "
from src.data.synthetic import generate_random_walk
from src.filters.scalar_kalman import ScalarKalmanFilter

data = generate_random_walk(n_steps=500, Q=1e-4, R=1e-3, seed=42)
kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
result = kf.filter(data.observations)
print(f'Final estimate: {result.states[-1]:.4f} ± {result.covariances[-1]**0.5:.4f}')
"

# Run tests
pytest tests/ -v
```

## Filter Variants

| Filter | Description | Use Case |
|--------|-------------|----------|
| **Scalar Kalman** | Basic predict-update cycle | Single stable market |
| **Adaptive Kalman** | Dynamic Q (innovation monitoring) + dynamic R (microstructure) | Markets with regime changes |
| **Logit Kalman** | Operates in log-odds space | Markets near 0 or 1 (approaching resolution) |
| **Multivariate Kalman** | Tracks n correlated markets | Fed/macro clusters, crypto basket |

## Architecture

```
Polymarket API ──→ Data Layer ──→ Kalman Filter ──→ Analysis
  │                  │               │                │
  ├─ Gamma API       ├─ SQLite       ├─ Scalar        ├─ Brier Score
  ├─ CLOB API        ├─ CSV export   ├─ Adaptive      ├─ Log Loss
  ├─ Data API        ├─ Caching      ├─ Logit         ├─ Calibration
  └─ WebSocket       └─ Synthetic    ├─ Multivariate  └─ Visualization
                                     └─ MLE params
```

## Notebooks

Each phase has a Jupyter notebook with visualizations and analysis:

| Notebook | Phase | Key Demonstration |
|----------|-------|-------------------|
| `01_basic_filter.ipynb` | Scalar filter | Filtered vs raw, parameter sensitivity, MLE |
| `02_adaptive_filter.ipynb` | Adaptive noise | Regime change tracking, Q/R dynamics |
| `03_logit_filter.ipynb` | Logit space | Bounded estimates, asymmetric CIs |
| `04_multivariate_filter.ipynb` | Correlated markets | Cross-market propagation, missing data |
| `05_regime_and_realtime.ipynb` | Detection + backtest | CUSUM/chi2, Brier scores, calibration |

## Mathematical Reference

### State-Space Model

**State transition** (random walk):
$$x_t = x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$

**Observation model**:
$$z_t = x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

### Filter Equations

**Predict:**
- $\hat{x}_{t|t-1} = \hat{x}_{t-1}$
- $P_{t|t-1} = P_{t-1} + Q$

**Update:**
- Innovation: $y_t = z_t - \hat{x}_{t|t-1}$
- Innovation covariance: $S_t = P_{t|t-1} + R$
- Kalman gain: $K_t = P_{t|t-1} / S_t$
- State update: $\hat{x}_t = \hat{x}_{t|t-1} + K_t y_t$
- Covariance update: $P_t = (1 - K_t) P_{t|t-1}$

### Parameter Estimation (MLE)

Optimal Q and R are found by maximizing the log-likelihood of the innovation sequence:
$$\log L = -\frac{1}{2} \sum_t \left[ \log(2\pi S_t) + \frac{y_t^2}{S_t} \right]$$

### Logit Transform

Maps probabilities to unbounded space: $\text{logit}(p) = \log\frac{p}{1-p}$

Observation noise transforms via the delta method: $R_{\text{logit}} = \frac{R_{\text{prob}}}{(p(1-p))^2}$

## Project Structure

```
src/
├── data/               # API client, storage, synthetic data
├── filters/            # Scalar, adaptive, logit, multivariate filters
├── detection/          # Regime change detection (CUSUM, chi2, autocorrelation)
├── pipeline/           # Real-time filtering and backtesting
├── analysis/           # Metrics, visualization, correlation estimation
└── utils/              # Transforms, math helpers
tests/                  # 85 tests covering all filter variants
notebooks/              # 5 demonstration notebooks
```

## Testing

```bash
# Run full suite
pytest tests/ -v

# Run specific filter tests
pytest tests/test_scalar_kalman.py -v
pytest tests/test_multivariate_kalman.py -v
```

## Key Results

- **Scalar Kalman** reduces observation noise variance by 3-10x depending on Q/R ratio
- **Adaptive filter** tracks step changes 5-10x faster than the basic filter
- **Logit filter** guarantees bounded estimates — critical near market resolution
- **Multivariate filter** improves MSE on correlated markets vs independent filtering
- **MLE** recovers true Q and R parameters within a factor of 2 from 2000+ observations

## References

- Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
- Mehra, R.K. (1972). "Approaches to Adaptive Filtering"
- Mohamed & Schwarz (1999). "Adaptive Kalman Filtering for INS/GPS"
- Ledoit & Wolf (2004). "Honey, I Shrunk the Sample Covariance Matrix"

## License

MIT
