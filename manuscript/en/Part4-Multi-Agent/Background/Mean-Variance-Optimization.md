# Background: Mean-Variance Portfolio Optimization

> "Don't put all your eggs in one basket—but calculate exactly how many to put in each."

---

## Core Idea

**Problem**: Given N assets, how to allocate capital to maximize returns and minimize risk?

**Markowitz's Insight**: Returns can be weighted averaged, but risk isn't simply additive—**correlation determines the effectiveness of diversification**.

---

## Basic Math

### Portfolio Return

```
Expected Portfolio Return = Sum(wi x mui)

Where:
  wi = weight of asset i
  mui = expected return of asset i
```

**Example**: Two assets
```
AAPL: weight 60%, expected return 12%
MSFT: weight 40%, expected return 10%

Expected Portfolio Return = 0.6 x 12% + 0.4 x 10% = 11.2%
```

### Portfolio Risk

```
Portfolio Variance = Sum_i Sum_j (wi x wj x sigma_ij)

Where:
  sigma_ij = covariance between assets i and j
  sigma_ii = variance of asset i
```

**Simplified form for two assets**:
```
sigma_p^2 = w1^2*sigma1^2 + w2^2*sigma2^2 + 2*w1*w2*rho*sigma1*sigma2

Where:
  rho = correlation coefficient between the two assets
```

---

## The Power of Correlation

**Example**: Two assets, each with 50% weight
- Asset A: volatility 20%
- Asset B: volatility 20%

| Correlation rho | Portfolio Volatility | Diversification Effect |
|-----------------|---------------------|------------------------|
| +1.0 | 20.0% | None |
| +0.5 | 17.3% | 13.5% reduction |
| 0.0 | 14.1% | 29.5% reduction |
| -0.5 | 10.0% | 50% reduction |
| -1.0 | 0% | Perfect hedge |

**Key Insight**: Lower correlation means better diversification.

---

## Efficient Frontier

**Definition**: The set of portfolios that achieve the highest expected return for a given level of risk.

```
Return
  |                    *------- Highest return point
  |                 ***
  |              ***
  |           ***    <- Efficient Frontier
  |        ***
  |     ***
  |  ***------------- Minimum variance point
  |
  +--------------------> Risk

Points below the efficient frontier are "inefficient":
Same risk could achieve higher return
```

### Computing the Efficient Frontier

```
Optimization Problem:

Maximize: Portfolio Return = w'mu
Subject to:
  1. Portfolio Risk = sqrt(w'Sigma*w) <= sigma_target
  2. Sum(wi) = 1 (weights sum to 1)
  3. wi >= 0 (optional: no short selling)
```

---

## Practical Calculation Example

**Three-asset portfolio**: AAPL, MSFT, GOOGL

**Input Data**:
```
Expected Returns (annualized):
  AAPL: 15%
  MSFT: 12%
  GOOGL: 18%

Volatility (annualized):
  AAPL: 25%
  MSFT: 20%
  GOOGL: 30%

Correlation Matrix:
       AAPL  MSFT  GOOGL
AAPL   1.0   0.7   0.6
MSFT   0.7   1.0   0.5
GOOGL  0.6   0.5   1.0
```

**Minimum Variance Portfolio**:
```
Weights: AAPL 25%, MSFT 55%, GOOGL 20%
Return: 13.5%
Volatility: 17.2%
```

**Maximum Sharpe Portfolio** (assuming risk-free rate 2%):
```
Weights: AAPL 30%, MSFT 30%, GOOGL 40%
Return: 15.6%
Volatility: 20.1%
Sharpe: 0.68
```

---

## Common Optimization Objectives

| Objective | Optimization Function | Characteristics |
|-----------|----------------------|-----------------|
| Minimum Variance | min w'Sigma*w | Most conservative, lowest volatility |
| Maximum Sharpe | max (w'mu - rf) / sqrt(w'Sigma*w) | Highest risk-adjusted return |
| Target Return | min w'Sigma*w s.t. w'mu = target | Minimum risk for target return |
| Risk Parity | Equal risk contribution from each asset | More balanced risk allocation |
| Maximum Diversification | max Sum(wi*sigma_i) / sqrt(w'Sigma*w) | Maximize diversification effect |

---

## Real-World Implementation Issues

### Issue 1: Estimation Error

**Theory requires**: Precise expected returns and covariance matrix

**Reality**: Estimated from historical data, with large errors

```
Estimation Error Impact:

Return estimation error → Large weight fluctuations
  - Historical 5-year AAPL return 18%
  - But future could be 10% or 25%
  - Small return prediction changes → Dramatic weight changes

Covariance estimation error → Unstable correlations
  - Normal periods: AAPL-MSFT correlation 0.6
  - Crisis periods: Correlation spikes to 0.9
  - Diversification effect disappears
```

### Issue 2: Extreme Weights

Unconstrained optimization often produces extreme results:

```
Theoretical optimum:
  Asset A: +250% (long)
  Asset B: -150% (short)

Problems:
  - High leverage risk
  - Short selling costs
  - Liquidity constraints
```

**Solution**: Add constraints
```
Common constraints:
  - 0 <= wi <= 30% (single asset cap)
  - Sum(wi) = 1 (fully invested)
  - wi >= 0 (no short selling)
```

### Issue 3: High Turnover

Optimization results are sensitive to inputs, each reoptimization may produce large rebalancing:

```
This month optimal: AAPL 40%, MSFT 30%, GOOGL 30%
Next month optimal: AAPL 20%, MSFT 50%, GOOGL 30%

Turnover: |40-20| + |30-50| + |30-30| = 40%
Cost: 40% x 0.2% x 2 = 0.16%

Annualized cost can consume most of the excess returns
```

**Solutions**:
```
1. Turnover penalty: Objective function - lambda x Turnover
2. Only rebalance when deviation exceeds threshold
3. Use more stable estimation methods
```

---

## Improvement Methods

### 1. Shrinkage Estimation

"Shrink" sample estimates toward more stable priors:

```
Shrunk Covariance = alpha x Sample Covariance + (1-alpha) x Structured Estimate

Common structured estimates:
  - Diagonal matrix (assume uncorrelated)
  - Single-factor model
  - Equal correlation model
```

### 2. Black-Litterman Model

Combines market equilibrium with subjective views:

```
Inputs:
  1. Market equilibrium returns (implied from market cap weights)
  2. Investor views (e.g., "I believe AAPL will outperform MSFT by 3%")
  3. View confidence levels

Output:
  Adjusted expected returns → More stable weights
```

### 3. Risk Parity

Don't predict returns, just balance risk contributions:

```
Goal: Each asset contributes equally to portfolio risk

Three-asset example:
  Total risk = 15%
  Each asset contribution = 5%

Result: Low volatility assets get higher weights, high volatility assets get lower weights
```

---

## Multi-Agent Perspective

In a multi-agent architecture, portfolio optimization can be applied as follows:

```
Signal Agents (multiple)
  |
  +-- Agent A: Output AAPL expected return
  +-- Agent B: Output MSFT expected return
  +-- Agent C: Output GOOGL expected return
       |
       v
Portfolio Agent (optimizer)
  |
  +-- Input: Return predictions from each Agent
  +-- Estimate covariance matrix
  +-- Execute mean-variance optimization
  +-- Output: Target weights
       |
       v
Risk Agent
  |
  +-- Check if weights violate risk limits
  +-- Check if turnover is too high
  +-- Adjust or reject proposed weights
```

---

## Common Misconceptions

**Misconception 1: Historically optimal portfolio will be optimal in the future**

Wrong. Optimization is the perfect overfitting tool:
- Over-reliance on historical data noise
- Historical correlations can change
- Expected return estimates are unreliable

**Misconception 2: More assets means better diversification**

There's an upper limit. Marginal benefits diminish:
```
Number of assets vs diversification effect:
  2 -> 10: Significant risk reduction
  10 -> 30: Moderate effect
  30 -> 100: Limited effect, increased complexity
```

**Misconception 3: Covariance matrix is stable**

Dangerous assumption. Covariance changes dramatically across regimes:
- Normal periods: Diversification works
- Crisis periods: Correlations approach 1, diversification fails

---

## Practical Recommendations

### 1. Start Simple

```
Starting choices:
- Equal weight (1/N): Robust, no estimates needed
- Risk parity: No return prediction needed
- Minimum variance: Only covariance estimate needed
```

### 2. Add Reasonable Constraints

```
Recommended constraints:
- Single asset weight <= 30%
- No short selling (unless you have a clear short strategy)
- Turnover penalty
```

### 3. Rebalance Periodically

```
Rebalancing strategies:
- Fixed period: Monthly/Quarterly
- Threshold trigger: When deviation from target weight >5%
- Combination: Threshold trigger + minimum interval
```

---

## Summary

| Key Point | Description |
|-----------|-------------|
| Core Idea | Diversify risk through low-correlated assets |
| Basic Formula | Portfolio Variance = w'Sigma*w |
| Key Challenges | Estimation error, extreme weights, turnover costs |
| Improvement Methods | Shrinkage estimation, Black-Litterman, risk parity |
| Practical Advice | Add constraints, start simple |
