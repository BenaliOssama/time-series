# Pandas Time Series — Reference Document
> Exercises 1–4 | Pandas · NumPy · Plotly

---

## 1. Core Concepts

### Why a DateTime Index?

A plain list of numbers has positions (0, 1, 2…). A DateTime index replaces those positions with actual dates so Pandas knows *when* each value happened. This unlocks time-specific operations: slicing by month, resampling by frequency, and computing rolling windows.

### Key Methods at a Glance

| Method / Object | What it does | When to use it |
|---|---|---|
| `pd.date_range()` | Generates a sequence of dates | Building a date index from scratch |
| `pd.to_datetime()` | Converts strings to datetime type | After loading a CSV with date strings |
| `df.set_index('Date')` | Makes a column the row label | Right after converting to datetime |
| `df.dropna()` | Removes rows with missing values | Before any computation |
| `series.shift(n)` | Moves values up (n<0) or down (n>0) | Computing future or past values |
| `series.pct_change()` | Computes (t - t-1) / t-1 per row | Daily returns without a for loop |
| `series.rolling(n).mean()` | Average of last n rows, sliding | Smoothing noisy time series |
| `df.resample('BM')` | Groups rows by time period | Collapsing daily data to monthly |
| `df.pivot_table()` | Reshapes long data to wide | Multi-asset: one column per ticker |

---

## 2. Exercise 1 — Series & Rolling Average

### Building a DateTime Series

The goal: a Series where dates are the index and values count days since the start (0, 1, 2…).

```python
import pandas as pd

index = pd.date_range(start='2010-01-01', end='2020-12-31', freq='D')
# freq='D' = one entry per Day. 'M' = monthly, 'H' = hourly, etc.

integer_series = pd.Series(
    data=range(len(index)),
    index=index,
    name='integer_series'
)
```

### 7-Day Rolling Average

A rolling average replaces each value with the mean of itself and the N-1 values before it. The window "rolls" forward one day at a time. The first N-1 values are `NaN` because there is not yet a full window of history.

```python
rolling_avg = integer_series.rolling(7).mean()
```

---

## 3. Exercise 2 — Financial Data (AAPL)

### Preprocessing Pipeline

Always follow this order when loading financial CSVs:

1. Load the file and inspect types with `df.info()` and `df.describe()`
2. Convert the date column from string to datetime with `pd.to_datetime()`
3. Set the date column as the index with `df.set_index('Date')`
4. Drop missing rows with `df.dropna()`

```python
df = pd.read_csv('AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.dropna()
```

### Candlestick Chart (Plotly)

A candlestick chart shows four prices per day: Open, High, Low, Close. Use Plotly Graph Objects (not Express) — Express does not have a Candlestick type.

```python
import plotly.graph_objects as go

fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])
fig.show()
```

### Monthly Resampling with `.resample()`

`resample()` groups rows by a time frequency — like `groupby()` but for time. `'BM'` = last **B**usiness day of the **M**onth. The aggregation function must match the meaning of each column:

| Column | Aggregation | Why |
|---|---|---|
| Open, Close | `mean` | Average price over the month |
| High | `max` | Highest of all daily highs = monthly high |
| Low | `min` | Lowest of all daily lows = monthly low |
| Volume | `sum` | Total shares traded = sum of daily volumes |

```python
transformed_df = df.resample('BM').agg({
    'Open':   'mean',
    'Close':  'mean',
    'Volume': 'sum',
    'High':   'max',
    'Low':    'min'
})
```

### Daily Returns (two methods)

The return at time t: `(Price(t) - Price(t-1)) / Price(t-1)`. The first row is always `NaN` — no previous day exists.

**Method 1 — `pct_change` (built-in shortcut):**
```python
returns = df['Open'].pct_change()
```

**Method 2 — vectorized formula using `shift`:**

`shift(1)` moves all values DOWN by one row, so each row now contains yesterday's price. Subtracting and dividing applies the formula to every row at once without a loop.

```python
returns = (df['Open'] - df['Open'].shift(1)) / df['Open'].shift(1)
```

---

## 4. Exercise 3 — Multi-Asset Returns

### The Problem with MultiIndex Data

When you have many tickers stacked in one column, `pct_change()` computes changes between consecutive rows regardless of ticker — AAPL's return would be computed relative to the previous row which belongs to a different company. Nonsense.

The fix: reshape from **long format** (one row per date+ticker) to **wide format** (one row per date, one column per ticker) using `pivot_table`.

### Solution

```python
pivoted = market_data.pivot_table(
    values='Price',
    index='Date',
    columns='Ticker'
)

returns = pivoted.pct_change()
```

Now `pct_change()` operates column by column — each ticker's returns are computed independently.

---

## 5. Exercise 4 — Backtest

### What is a Backtest?

A backtest answers: *"if I had followed this trading signal historically, how much money would I have made?"* It applies a signal to historical returns mathematically — no real trades.

### Adj Close vs Close

`Adj Close` corrects for stock splits and dividends — events that change the price artificially without reflecting a real gain or loss. Always use `Adj Close` for return calculations so the series has no artificial jumps.

### Future Return

Unlike the past return (how much did price change since yesterday?), the **future return** asks: how much will price change tomorrow? The signal at day `t` tells you to buy before close — you sell the next day. Your profit depends on what happens tomorrow.

```python
future_returns = (df['Adj Close'].shift(-1) - df['Adj Close']) / df['Adj Close']
future_returns.name = 'Daily_futur_returns'
```

`shift(-1)` moves values UP by one row — each row now contains tomorrow's price. The last row becomes `NaN` because there is no tomorrow.

### `shift()` Direction Reference

| Call | Values move | Each row now contains | Edge becomes |
|---|---|---|---|
| `shift(1)` | DOWN | Yesterday's value | First row → NaN |
| `shift(-1)` | UP | Tomorrow's value | Last row → NaN |

### Random Signal

```python
import numpy as np
np.random.seed(2712)

signal = pd.Series(
    data=np.random.randint(0, 2, len(df.index)),
    index=df.index,
    name='long_only_signal'
)
```

`np.random.randint(0, 2, n)` generates n integers that are either 0 or 1 with equal probability.

### PnL Computation

When signal = 1 you bought → PnL equals the future return. When signal = 0 you did nothing → PnL = 0. This is just multiplication — both Series share the same index so Pandas multiplies row by row.

```python
pnl = signal * future_returns
pnl.name = 'PnL'
```

### Strategy Return

Total return = (Total earned − Total invested) / Total invested. Since you invest $1 each time signal = 1, total invested = `signal.sum()`. Total earned − invested = `pnl.sum()`.

```python
total_return = pnl.sum() / signal.sum()
```

### Always-Buy Baseline

```python
always_buy = pd.Series(
    data=np.ones(len(df.index), dtype=int),
    index=df.index
)
always_buy_pnl = always_buy * future_returns
```

### Plotting Both Strategies

```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=pnl.index, y=pnl, name='Random signal'))
fig.add_trace(go.Scatter(x=always_buy_pnl.index, y=always_buy_pnl, name='Always buy'))
fig.show()
```

---

## 6. Common Pitfalls

| Mistake | Problem | Fix |
|---|---|---|
| Using for loops | Slow — defeats Pandas vectorisation | Use `shift()`, `pct_change()`, `rolling()` |
| `pct_change()` on MultiIndex | Computes across tickers, not within | `pivot_table()` first, then `pct_change()` |
| Not setting datetime index | Time operations fail or give wrong results | `pd.to_datetime()` then `set_index()` |
| Wrong aggregation in resample | e.g. summing prices instead of averaging | `mean` for prices, `sum` for volume, `min`/`max` for low/high |
| Using `Close` instead of `Adj Close` | Splits/dividends create artificial price jumps | Always use `Adj Close` for return calculations |
| `shift(1)` vs `shift(-1)` confusion | Past return vs future return — opposite meanings | `shift(-1)` for future (backtest), `shift(1)` for past (returns) |
