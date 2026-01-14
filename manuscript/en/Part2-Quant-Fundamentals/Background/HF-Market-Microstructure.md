# Background: High-Frequency Market Microstructure

> "At the millisecond level, the market is not a random walk, but a battle between buyers and sellers."

---

## What is Market Microstructure?

**Traditional view**: Stock prices are determined by fundamentals—earnings, macroeconomics, industry trends

**Microstructure view**: Stock prices are determined by order flow—who's buying, who's selling, at what prices they're posting orders

```
Time scale comparison:

Fundamental Analysis: Months → Years
Technical Analysis: Days → Weeks
Market Microstructure: Milliseconds → Seconds
```

---

## Core Concept: Order Book

**Order Book**: All currently unexecuted buy and sell orders

![Order Book Structure](assets/order-book.svg)

### Key Terminology

| Term | Definition | Significance |
|------|------------|--------------|
| Bid (Level 1) | Highest buy price | Best price to sell now |
| Ask (Level 1) | Lowest sell price | Best price to buy now |
| Spread | Ask - Bid | Implicit trading cost |
| Mid | (Bid + Ask) / 2 | Fair price estimate |
| Depth | Order quantity at each level | Market absorption capacity |

---

## Order Types

### Market Order

```
Instruction: "Buy 100 shares immediately, any price"

Execution:
  - Fills at current ask level 1 price $185.35
  - If ask level 1 only has 50 shares, remaining 50 fills at ask level 2 $185.40

Pros: Guaranteed execution
Cons: May slip (adverse price)
```

### Limit Order

```
Instruction: "Buy 100 shares at $185.30 or lower"

Execution:
  - If there's a sell order at ≤$185.30 → Immediate fill
  - Otherwise → Enters order book and waits

Pros: Price control
Cons: May not execute
```

### Other Order Types

| Type | Function | Scenario |
|------|----------|----------|
| Stop | Becomes market order when price triggers | Risk control |
| Stop-Limit | Becomes limit order when triggered | Precise stop-loss |
| IOC (Immediate or Cancel) | Fill what you can | Urgent execution |
| FOK (Fill or Kill) | All or nothing | Avoid partial fills |
| Iceberg | Display only partial quantity | Hide large order intent |

---

## Meaning of Bid-Ask Spread

### Spread Components

| Component | Explanation | Proportion |
|-----------|-------------|------------|
| Inventory cost | Compensation for market maker holding inventory risk | 20-30% |
| Adverse selection | Protection against losses from informed traders | 40-50% |
| Order processing cost | Trading system and labor costs | 10-20% |

**Adverse selection** is key: Market makers worry about losing when trading with informed traders

```
Scenario:
  Someone knows Apple is about to release good news, quietly buys heavily
  Market maker doesn't know, sells to them at low price
  After news releases, market maker loses money

Response:
  Market makers widen spreads to compensate for these expected losses
```

### Spread and Liquidity

```
Good Liquidity:
  - Tight spread ($0.01)
  - Large order sizes
  - Large orders don't move price
  - Examples: AAPL, MSFT, SPY

Poor Liquidity:
  - Wide spread ($0.10+)
  - Small order sizes
  - Large orders significantly move price
  - Examples: Small caps, illiquid ETFs
```

---

## Market Impact

**Definition**: The effect of your trade on price

```
You want to buy 10,000 shares of AAPL
Current order book:
  Ask Level 1: $185.35 × 200 shares
  Ask Level 2: $185.40 × 300 shares
  Ask Level 3: $185.45 × 500 shares
  ...

If buying all at once:
  Fill 200 @ $185.35
  Fill 300 @ $185.40
  Fill 500 @ $185.45
  ...

Average execution price might be $185.60
$0.25 higher than mid price → Market impact cost
```

### Impact Factors

| Factor | Impact | Reason |
|--------|--------|--------|
| Order size | Larger = bigger impact | Consumes more depth |
| Market liquidity | Low liquidity = bigger impact | Thin order book |
| Execution speed | Faster = bigger impact | Cannot hide intent |
| Information content | Informed trading = more persistent impact | Price discovery |

### Market Impact Model

```
Simplified square-root rule:

Market Impact ≈ σ × √(Q / V)

Where:
  σ = Daily volatility
  Q = Trade quantity
  V = Average daily volume

Example:
  Volatility = 2%
  Trade size = 1% of average daily volume

  Impact ≈ 2% × √0.01 = 0.2%
```

---

## Role of Market Makers

**Market Maker**: Simultaneously places buy and sell orders, earns spread

```
Market Maker Strategy:
  Place buy order @ $185.30
  Place sell order @ $185.35

  If both sides fill:
    Sell price - Buy price = $0.05 profit

  Risks:
    Only one side fills → Inventory risk exposure
    Directional market → Losses
```

### Market Maker vs Directional Trader

| Dimension | Market Maker | Directional Trader |
|-----------|--------------|-------------------|
| Goal | Earn spread | Earn price movement |
| Holding period | Very short (seconds/minutes) | Longer (hours/days) |
| Risk exposure | Minimize directional | Has directional exposure |
| Profit source | Spread − Adverse selection − Inventory cost | Prediction accuracy |

---

## Price Discovery Process

**Price Discovery**: How markets reflect information in prices

```
Information release → Informed traders order → Order flow changes → Price adjusts

Timeline example:
  T+0ms: Positive news released
  T+1ms: High-speed traders detect news
  T+5ms: Large buy orders flood in
  T+10ms: Ask level 1 consumed, price rises
  T+100ms: Price mostly reflects new information
  T+1000ms: Regular investors see price change
```

### Order Flow Toxicity

**Toxic Order Flow**: Order flow dominated by informed traders

```
Indicator: VPIN (Volume-Synchronized Probability of Informed Trading)

High VPIN → Informed traders active → High risk for market makers
Low VPIN → Noise traders dominate → Market makers safe

Applications:
  - Market makers adjust spreads based on VPIN
  - Warn of market stress (high VPIN may precede crash)
```

---

## High-Frequency Strategy Categories

| Strategy Type | Description | Profit Source |
|---------------|-------------|---------------|
| Market Making | Two-sided quotes earn spread | Spread revenue − Adverse selection − Inventory risk |
| Statistical Arbitrage* | Exploit price deviation mean reversion | Price dislocation correction |
| News Trading | Fast news interpretation (ms to minutes) | Information advantage |
| Latency Arbitrage | Exploit inter-exchange delays | Speed advantage |
| Structural Arbitrage | Exploit market structure mechanics (rebates, auctions, ETF creation/redemption) | System understanding |

> **\* About Statistical Arbitrage**: Stat arb is NOT exclusive to high-frequency trading. By holding period:
> - **High-frequency stat arb** (ms to seconds): Cross-exchange microstructure signals
> - **Intraday stat arb** (minutes to hours): Pairs trading, ETF-constituent arbitrage
> - **Medium/low-frequency stat arb** (days to weeks): Multi-factor, sector-neutral strategies
>
> Whether to go high-frequency depends on: **Signal decay speed vs. trading costs**. Fast decay "forces" high-frequency.

**Latency Arbitrage Example**:

```
NYSE Price: $185.35
BATS Price: $185.30 (delayed update)

Strategy:
  Buy on BATS @ $185.30
  Sell on NYSE @ $185.35
  Theoretical spread $0.05

⚠️ Real-world risks:
  - Leg risk: May only fill one side
  - Queue risk: Your order may not get filled at posted price
  - Adverse selection: Price may reverse instantly
  - Fees/slippage: May eat into profit

Prerequisite: Be faster than everyone else, and profit must cover fees and slippage
```

---

## Trading Cost Breakdown

Total cost of a trade:

```
Total Cost = Explicit Cost + Implicit Cost

Explicit Cost:
  - Commission: $0 for most US retail brokers (since 2019)
  - Exchange fees (~$0.003/share, often absorbed by broker)

Implicit Cost:
  - Bid-ask spread ($0.02/share)
  - Market impact ($0.05/share)
  - Timing cost (price change from decision to execution)
  - PFOF cost (see note below)

Example:
  Buy 1,000 shares @ $185.30
  Commission: $0
  Spread cost: $20
  Impact cost: $50

  Total cost: $70 = 0.038%
```

> **Note: The Hidden Cost of "Free" Trading**
>
> Since 2019, major US retail brokers (Schwab, Fidelity, TD Ameritrade, Robinhood) eliminated trading commissions. But there's no free lunch—the cost shifted to **Payment for Order Flow (PFOF)**.
>
> **How PFOF works:**
> - Your broker sells your order to a market maker (Citadel, Virtu, etc.)
> - Market maker pays broker $0.002-0.004 per share
> - Market maker profits by filling your order at slightly worse prices
>
> **What this means for you:**
> - You might get filled at $185.32 instead of the $185.30 best price
> - Cost: ~$0.01-0.02 per share (hidden in execution quality)
> - For small orders (<$10,000), zero commission still wins
> - For large orders, consider brokers with direct market access
>
> **Bottom line**: Commission-free doesn't mean cost-free. The cost moved from visible (commission) to invisible (worse execution).

---

## Multi-Agent Perspective

Market microstructure knowledge applied in multi-agent architecture:

```
Execution Agent
  │
  ├─ Monitor order book depth
  ├─ Evaluate market impact cost
  ├─ Decide execution strategy:
  │    - Large order → Split execution
  │    - Small order → Immediate execution
  │    - Urgent → Accept slippage
  │
  ↓
Risk Agent
  │
  ├─ Monitor toxicity indicators like VPIN
  ├─ Pause trading during high toxicity
  └─ Alert when liquidity dries up

Market State Agent
  │
  ├─ Track spread changes
  ├─ Identify liquidity regimes
  └─ Adjust strategy parameters
```

---

## Common Misconceptions

**Misconception 1: Execution price is true cost**

Incomplete. True cost includes:
- The portion you pushed the price (impact cost)
- Price change while your order was resting (timing cost)

**Misconception 2: Liquidity is always available**

Liquidity disappears in crises:
- Market makers withdraw
- Spreads widen dramatically
- Orders cannot execute

During the 2010 Flash Crash, some stocks had spreads widen to several dollars.

**Misconception 3: All HFT is manipulation**

Most HFT strategies provide liquidity:
- Market makers tighten spreads
- Arbitrageurs eliminate price deviations
- Increase market efficiency

Of course predatory strategies exist, but not all.

---

## Practical Advice

### For Low-Frequency Traders

```
1. Avoid large market orders
   - Split execution
   - Use limit orders

2. Avoid high-volatility periods
   - First and last half-hour have wide spreads
   - Major news releases have poor liquidity

3. Mind liquidity
   - Single trade size < 1% of daily volume
   - Otherwise market impact cost too high
```

### For Strategy Developers

```
1. Include realistic costs in backtests
   - Not just commission
   - Include spread and impact

2. Monitor volume
   - Strategy capacity = 1-5% of average daily volume
   - Returns degrade significantly beyond that

3. Use slippage models
   - Simple: Fixed percentage slippage
   - Advanced: Order book simulation
```

---

## Summary

| Key Point | Description |
|-----------|-------------|
| Core concepts | Order book, Bid-ask spread, Market depth |
| Key costs | Spread + Market impact + Timing cost |
| Market maker role | Provide liquidity, earn spread |
| Price discovery | Order flow reflects information |
| Multi-agent application | Execution Agent optimizes execution |
