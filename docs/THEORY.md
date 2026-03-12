# Mathematical Theory

This document provides the mathematical derivations underlying the Kalman filter implementations in this project.

## 1. Scalar Kalman Filter

### State-Space Model

The true probability of a prediction market event follows a random walk:

$$x_t = x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$

We observe a noisy version of this probability (the market price):

$$z_t = x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

where:
- $x_t \in [0, 1]$ is the true (latent) probability
- $z_t \in [0, 1]$ is the observed market price
- $Q$ is the process noise variance (how fast the true probability changes)
- $R$ is the measurement noise variance (how noisy the market price is)

### Predict Step

Project the state and its uncertainty forward:

$$\hat{x}_{t|t-1} = \hat{x}_{t-1|t-1}$$
$$P_{t|t-1} = P_{t-1|t-1} + Q$$

The state estimate doesn't change (random walk has zero drift), but uncertainty grows by Q.

### Update Step

Incorporate the new observation:

1. **Innovation**: $y_t = z_t - \hat{x}_{t|t-1}$ (prediction error)
2. **Innovation covariance**: $S_t = P_{t|t-1} + R$
3. **Kalman gain**: $K_t = P_{t|t-1} / S_t$
4. **State update**: $\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t y_t$
5. **Covariance update**: $P_{t|t} = (1 - K_t) P_{t|t-1}$

### Steady-State Gain

For constant Q and R, the Kalman gain converges to a steady-state value. The steady-state covariance satisfies the algebraic Riccati equation:

$$P_{ss} = P_{ss} + Q - \frac{P_{ss}^2}{P_{ss} + R}$$

Solving: $P_{ss}^2 - Q \cdot P_{ss} - Q \cdot R = 0$

$$P_{ss} = \frac{Q + \sqrt{Q^2 + 4QR}}{2}$$

$$K_{ss} = \frac{P_{ss}}{P_{ss} + R}$$

## 2. Maximum Likelihood Estimation

Under correct model specification, the innovation sequence $\{y_t\}$ is Gaussian white noise with zero mean and variance $S_t$. The log-likelihood is:

$$\log L(\theta) = -\frac{1}{2} \sum_{t=1}^{T} \left[ \log(2\pi S_t) + \frac{y_t^2}{S_t} \right]$$

where $\theta = (Q, R)$.

We optimize in log-space (log Q, log R) to enforce positivity, using L-BFGS-B.

## 3. Adaptive Process Noise

When a regime change occurs, the innovation sequence exhibits:
- Large magnitude (the filter is "surprised")
- Serial correlation (same-sign innovations persist)

The normalized innovation $\tilde{y}_t = y_t / \sqrt{S_t}$ should be $\mathcal{N}(0, 1)$ under correct specification. When $|\tilde{y}_t| > \tau$ (threshold):

$$Q_t = Q_{base} \cdot \alpha$$

where $\alpha$ is the inflation factor. After detection, $\alpha$ decays exponentially:

$$\alpha_{t+1} = 1 + (\alpha_t - 1) \cdot \lambda$$

where $\lambda \in (0, 1)$ is the decay rate.

## 4. Dynamic Observation Noise

The observation noise $R_t$ is estimated from market microstructure:

$$R_t = w_1 R_{spread} + w_2 R_{depth} + w_3 R_{imbalance} + w_4 R_{stale}$$

where:
- $R_{spread} = (s/2)^2$ (half-spread squared)
- $R_{depth} = k_1 / \log(1 + D)$ (inverse log-depth)
- $R_{imbalance} = (I - 0.5)^2$ (deviation from balance)
- $R_{stale} = k_2 / (1 + n)$ (inverse trade count)

## 5. Logit-Space Filter

### Transform

$$\ell = \text{logit}(p) = \log\frac{p}{1-p}$$
$$p = \text{sigmoid}(\ell) = \frac{1}{1 + e^{-\ell}}$$

### Delta Method for Noise Transform

If $z \sim \mathcal{N}(p, R_{prob})$ in probability space, then:

$$\text{logit}(z) \approx \mathcal{N}\left(\text{logit}(p), \frac{R_{prob}}{(p(1-p))^2}\right)$$

This follows from $\text{Var}[g(X)] \approx (g'(\mu))^2 \text{Var}[X]$ where $g' = (\text{logit})' = 1/(p(1-p))$.

### Confidence Intervals

In logit space, the CI is symmetric: $\ell \pm k\sqrt{P}$

In probability space, this maps to asymmetric bounds:
$$[\ \text{sigmoid}(\ell - k\sqrt{P}),\ \text{sigmoid}(\ell + k\sqrt{P})\ ]$$

These bounds are automatically in $(0, 1)$ regardless of P.

## 6. Multivariate Filter

### State-Space Model

$$\mathbf{x}_t = \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$$
$$\mathbf{z}_t = \mathbf{H}_t \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_t)$$

where $\mathbf{x}_t \in \mathbb{R}^n$, $\mathbf{Q}$ is $n \times n$, and $\mathbf{H}_t$ is a selection matrix for observed markets.

### Update Equations

$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}_t^T (\mathbf{H}_t \mathbf{P}_{t|t-1} \mathbf{H}_t^T + \mathbf{R}_t)^{-1}$$

$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t (\mathbf{z}_t - \mathbf{H}_t \hat{\mathbf{x}}_{t|t-1})$$

Joseph form for covariance (numerically stable):
$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t|t-1} (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t)^T + \mathbf{K}_t \mathbf{R}_t \mathbf{K}_t^T$$

### Ledoit-Wolf Shrinkage

The sample covariance $\mathbf{S}$ is shrunk toward a scaled identity:

$$\hat{\mathbf{\Sigma}} = \alpha \cdot \frac{\text{tr}(\mathbf{S})}{n} \mathbf{I} + (1 - \alpha) \cdot \mathbf{S}$$

where $\alpha \in [0, 1]$ is the optimal shrinkage intensity estimated from the data.

## 7. Regime Detection

### CUSUM Test

Accumulate deviations from expected squared innovations:

$$C_t = C_{t-1} + (\tilde{y}_t^2 - 1)$$

Detection when $C_t - \min_{s \leq t} C_s > h$ (threshold).

### Chi-Square Test

Over a window of W observations:

$$\chi^2 = \sum_{t=T-W+1}^{T} \tilde{y}_t^2$$

Under $H_0$: $\chi^2 \sim \chi^2(W)$. Reject if exceeds critical value.

### Autocorrelation Test

Compute lag-1 autocorrelation of the innovation sequence over the window.
Under correct specification, innovations are white noise (zero autocorrelation).
