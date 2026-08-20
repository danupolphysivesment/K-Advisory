# K-ADVISOR

A portfolio suggestion workbench for relationship managers talking to high
net-worth clients. Built on the K-Asset fund universe (136 funds, daily returns
from 1992 to July 2026) with live macro, news and search-trend data.

**The interface is in Thai, in the register Thai investment professionals
actually write in:** Thai for the connective and explanatory prose, English for
the industry's own technical nouns. So it is *Core / Satellite*, *Max Drawdown*,
*Efficient Frontier*, *Effective Positions*, *Rebalancing* and *Capital Market
Assumptions* — never a laboured Thai calque of those — while everything
carrying the argument stays Thai.

Two exceptions to that rule, both deliberate:

- **Regulatory vocabulary is Thai and verbatim**, lifted from the SEC/AIMC
  suitability form the app implements, so what appears on screen matches the
  paperwork an RM already has in front of them:
  `เงินฝากและตราสารหนี้ระยะสั้น`, `ตราสารทุน`, `การลงทุนทางเลือก`.
- **Terms with a settled, natural Thai form keep it** — `ความผันผวน`,
  `ผลตอบแทนต่อปี`, `สินทรัพย์ทางเลือก`, `กองทุนผสม`. Swapping those to English
  would be affectation, not clarity.

Fund names stay in their Latin K-* form, as on every Thai factsheet. Text is set
in Sarabun; only figures keep Georgia's print-like numerals. News is pulled from
the Thai Google News locale, so headlines come from Thai financial press.

Thai has no inter-word spaces, so an English term dropped into a Thai sentence
runs straight into it (`กองทุน Sectorและธีม`). `test_app.py` checks every piece
of generated prose for a Thai/Latin boundary with no space, since this is
invisible in code review and obvious on screen.

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8570
```

The workbook is read from `data/Fund Return.xlsx`, or from
`$FUND_RETURN_XLSX`, or from `~/Downloads/Fund Return.xlsx`.

Deep-link to a client with `?client=C08`.

---

## Deploying to Streamlit Community Cloud

1. **Commit `data/Fund Return.xlsx`.** It is 4.3 MB, well inside GitHub's
   100 MB file limit, so no Git LFS is needed. Check it is not caught by a
   `*.xlsx` rule in `.gitignore` — that is the most common cause of a deploy
   that boots and then immediately errors. If the file is missing the app now
   says so in Thai and tells you which of the three things to fix, rather than
   showing a traceback.
2. **Push the whole `advisor/` folder, not just the files you changed.** The
   modules share display fields — `advisor/cautions.py` reads `Theme.caution`
   from `advisor/market.py` — so a partial push fails deep inside a render with
   an `AttributeError` that Cloud redacts, which is impossible to diagnose from
   the browser. The app now guards against this twice: those fields are read
   defensively so a mismatched checkout degrades to plainer cards instead of
   dying, and a startup check compares `market.SCHEMA_VERSION` against
   `cautions.REQUIRED_SCHEMA` and names the stale file in Thai. The version in
   use is printed in the sidebar's *แหล่งที่มาของข้อมูล* panel.
3. Point the app at `app.py` and leave the Python version at the default.
4. `requirements.txt` states floors rather than pins, because Cloud installs
   current versions and the app is written to run on either generation of
   pandas — see below.

**The pandas version on Cloud is not the one on your laptop, and two of its
changes break code silently or loudly.** `advisor/compat.py` handles both, and
the sidebar's *แหล่งที่มาของข้อมูล* panel prints the versions actually in use,
which is the first thing to check when a deployment behaves differently:

- **Copy-on-Write.** From pandas 2.x, `Series.to_numpy()` may return a
  *read-only view* of the Series' buffer instead of a copy. Assigning to one
  element raises `ValueError: assignment destination is read-only` — which is
  exactly what took down section 1 on the first deploy. Three places built a
  rebalancing mask this way; they now go through `compat.writable()`.
- **Renamed offset aliases.** `"M"`, `"Q"` and `"A"` became `"ME"`, `"QE"` and
  `"YE"` for resampling (deprecated in 2.2, removed in 3.0), while `to_period`
  still wants the short forms — so the two APIs no longer accept the same
  string. Every frequency is resolved through `compat.resample_alias()` /
  `compat.period_alias()`, which probe once at import rather than comparing
  version strings.

There is also no `matplotlib` dependency: the monthly-return heat scale is drawn
from the app's own palette, because `Styler.background_gradient` needs
matplotlib and it is easy to leave out of `requirements.txt` and only discover
as an `ImportError` in production.

`test_app.py` covers all of this — it monkey-patches `Series.to_numpy()` to
return read-only arrays and runs every rebalancing rule through the analytics,
so this class of failure is caught on a laptop instead of on Cloud.

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

**The caution cards carry copy written for the card.** Each theme has a
dedicated one-sentence line saying what the theme does to *this client's money*
— "ถ้าดอกเบี้ยไทยขึ้น 1% กองทุนตราสารหนี้อายุยาวอาจติดลบหลายเปอร์เซ็นต์
ขณะที่กองทุนอายุสั้นแทบไม่ขยับ" — followed by the live number behind it and the
funds that carry the exposure. An earlier version truncated the long theme
explainer to fit, which does not work in Thai: the language has almost no
inter-word spaces, so a word-count truncation returns the entire paragraph and
the card becomes unreadable. `test_app.py` asserts the caution line is neither a
prefix of the explainer nor longer than 110 characters, because the cards sit
three-across and equalise off the tallest.

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
