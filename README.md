# K-ADVISOR

A portfolio suggestion workbench for relationship managers talking to high
net-worth clients. Built on the K-Asset fund universe (136 funds, daily returns
from 1992 to July 2026) with live macro, news and search-trend data.

**The interface is in Thai.** Terminology follows the SEC/AIMC suitability form
the app implements, so what appears on screen matches the paperwork an RM
already uses — `เงินฝากและตราสารหนี้ระยะสั้น`, `ตราสารทุน`, `การลงทุนทางเลือก`.
Fund names stay in their Latin K-* form, as they do on every Thai factsheet, and
technical terms carry the English in parentheses on first use. Text is set in
Sarabun; only figures keep Georgia's print-like numerals. News is pulled from
the Thai Google News locale, so headlines come from Thai financial press.

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8570
```

The workbook is read from `data/Fund Return.xlsx`, or from
`$FUND_RETURN_XLSX`, or from `~/Downloads/Fund Return.xlsx`.

Deep-link to a client with `?client=C08`.

---

## The three sections

### 1 · Present portfolio

What the client owns today, for each of the 11 mock books:

- **Suitability check** against both SEC tables — the product risk level of
  every holding, and look-through exposure to the four allocation buckets
  against the band for that investor type.
- **Allocation** by fund, asset class and suitability bucket. Multi-asset funds
  are decomposed, so a balanced fund counts partly as equity.
- **Performance** against a choice of benchmarks including a policy benchmark
  built from the client's own suitability band, with beta, alpha, tracking
  error, information ratio and up/down capture.
- **Attribution** — per-holding contribution measured on drifting weights, so
  the contributions sum *exactly* to the portfolio's compounded return.
- **Risk contribution** — Euler decomposition of volatility, plotted against
  capital weight. The gap between the two bars is the point.
- **Stress testing** — 17 historical windows from the 1997 crisis to the 2026
  gold drawdown, plus factor-based hypothetical shocks for funds too young to
  have lived through them.
- **Monte Carlo** — 10,000 paths, up to 120 periods, five generators.

### 2 · Suggested portfolio

Eleven objectives (max Sharpe, min volatility, target return, target
volatility, min drawdown, min CVaR, inverse volatility, risk parity, max
diversification, max Sortino, equal weight), each solved under the *same*
suitability constraints so the results are comparable and every one of them is
a portfolio the client is allowed to hold.

Output: allocation by fund, asset class and core/satellite; a like-for-like
statistical comparison on the window both portfolios share; the efficient
frontier drawn inside the client's bands; stress and Monte Carlo side by side;
and a costed trade list.

### 3 · Market & financial literacy

A macro dashboard (12 indicators, each with change and three-year percentile),
mechanically derived market signals, twelve explained themes with live
headlines, Thai search trends, and six plain-language lessons.

Crucially, this section feeds back into the first two: each theme carries
exposure tags, each fund carries exposure tags, and the intersection produces
the caution ribbon at the top of sections 1 and 2.

---

## Things worth knowing

**The two SEC tables contradict each other, and the app says so.** Every
alternative fund carries product risk level 8. A profile-7 client may hold
products up to level 7, yet their allocation band allows up to 20%
alternatives — which they can never reach. Worse, their remaining caps
(10% cash + 40% fixed + 40% equity) sum to 90%, so *no* portfolio exists.

The app handles this in two ways rather than silently returning something
plausible:

1. An **acknowledgement budget** admits a capped sleeve of above-level
   products, which is how the exception is handled in practice. Those holdings
   are reported as a disclosed exception, not a breach.
2. A **band-repair LP** finds the smallest relaxation that makes the constraint
   set feasible, priced so that over-holding cash is cheap and over-holding
   equity or alternatives is expensive. Whatever it changed is printed.

If nothing works, the optimiser returns an error — never a filler portfolio.

**Expected returns are assumptions, not history.** Trailing five-year returns
would have the optimiser short China and avoid Thai bonds. Expected returns
default to forward-looking capital market assumptions built from a cash rate
plus a term premium plus an equity risk premium, all listed and editable in
section 3. Volatility and correlation still come from the workbook, because
those persist.

**Estimation windows are honest.** Most K-Asset funds launched after 2018.
Including a fund launched last year would truncate the shared history for every
other fund, so the eligible universe is filtered by minimum track record and
the app states the window and observation count it actually used. Stress
replays label how much of the portfolio was stood in for by a proxy.

**One observation is quarantined.** K-FIXED-A carries a −54% print on
1997-12-26 that is a NAV restatement, not a bond-market move. It is removed and
the removal is disclosed in the sidebar.

---

## Layout

```
app.py                  Streamlit UI — three tabs, presentation only
advisor/
  data.py               workbook loading, cleaning, portfolio return series
  universe.py           136 funds → asset class, SEC risk level, look-through, tags
  clients.py            11 client books + the SEC suitability framework
  assumptions.py        forward-looking capital market assumptions
  metrics.py            performance and risk statistics
  risk.py               covariance (Ledoit-Wolf / sample / EWMA), risk contribution
  attribution.py        contribution and Brinson allocation-vs-selection
  stress.py             historical replay + factor shocks
  montecarlo.py         five path generators, fan charts
  optimizer.py          objectives, constraints, band repair, holding shrink
  market.py             macro, news, trends, themes, signals
  cautions.py           theme × holding intersection
  engine.py             one-call analytics bundle
  charts.py             Plotly builders
  theme.py              palette, CSS, Plotly template, components
  th.py                 Thai display strings for values kept in English as keys
test_app.py             end-to-end checks — run with `python3 test_app.py`
```

`test_app.py` runs every client through the analytics, every objective through
every risk profile, and every Monte Carlo generator, checking that
contributions reconcile to the total return, risk contributions sum to
portfolio volatility, and no objective quietly exceeds its constraints. It also
asserts the localisation holds — that asset classes, client records, objectives,
stress events, market themes and suitability findings all render in Thai, so an
untranslated string added later fails the suite rather than reaching a client.

A note on the code: strings that double as dictionary keys — suitability status,
severity level, core/satellite role — stay English internally and are translated
at the point of display by `advisor/th.py`. Translating those constants would
mean chasing every lookup that depends on them.

---

## Data sources

| Source | Used for | Failure behaviour |
|---|---|---|
| `data/Fund Return.xlsx` | all fund returns | required |
| Yahoo Finance | macro dashboard, signals | dashboard hidden, structural themes remain |
| Google News RSS | theme headlines | headlines omitted |
| Google Trends RSS | Thai search interest | panel omitted |

Every network call is optional and cached for 30 minutes. The app works
offline.

---

Client portfolios are illustrative mock-ups. Expected returns are stated
assumptions, not forecasts. Historical simulation is not a prediction of future
performance. This tool supports a suitability conversation; it does not replace
one.
