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

**Settings are sticky and shareable.** Every control is remembered in the query
string, so switching client, changing tab, or losing the websocket and
reconnecting all come back to the same view — and the URL of a configured
screen can be sent to a colleague as-is. `?client=C08&t2_obj=Max Sharpe Ratio&lookback=10`
opens exactly that, and a hand-built portfolio travels the same way
(`&t2_custom=K-GOLD-A(A):40,K-SF-A:60`).

Only settings that differ from the default are written, so the common case
stays a clean `?client=C06`. The portfolio value is remembered *per client*
(`aum_C06`), so editing it for one client neither leaks into the next nor is
lost on the way back. **ล้างค่าที่ตั้งไว้** in the sidebar clears everything.

This matters most on Streamlit Community Cloud, where an idle or flaky session
is dropped and restarted: `st.session_state` is wiped, but the URL is not.

Every numeric slider is paired with a number box so an exact figure can be
typed rather than dragged to. Both drive one remembered value: Streamlit has no
widget that is both and two widgets cannot share a key, so each has its own key
and an `on_change` callback that writes the canonical value and its twin. The
box is free text, so its callback clamps before the value reaches the slider,
which raises if handed something out of range.

**Client notes.** Each client has a free-text note under the client header,
stored in `data/client_notes.json` — too long for the query string, and the one
thing in the app the user authors rather than derives, so losing it matters more
than losing a setting. The file is gitignored. On Community Cloud the
filesystem is ephemeral, so a container restart clears it and the UI says so; a
read-only filesystem degrades to "notes work until you reload" rather than
raising.

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
- **Performance** against the WealthPLUS glide path (Balanced / SpeedUp /
  Ultimate), with beta, alpha, tracking error, information ratio and up/down
  capture. Those are the one-ticket funds an RM would otherwise have
  recommended, so the comparison answers the question the client actually
  asks: would I have been better off just buying the ready-made fund?
- **Attribution** — per-holding contribution measured on drifting weights, so
  the contributions sum *exactly* to the portfolio's compounded return, plus
  the same split by asset class.
- **Risk contribution** — Euler decomposition of volatility, plotted against
  capital weight. The gap between the two bars is the point.
- **Stress testing** — 17 historical windows from the 1997 crisis to the 2026
  gold drawdown, plus factor-based hypothetical shocks for funds too young to
  have lived through them.
- **Monte Carlo** — 10,000 paths, up to 120 periods, block bootstrap.
  Five generators are implemented and tested; the app runs the one that
  keeps volatility clustering and trend intact, because those are what make
  a drawdown number worth quoting to a client.

### 2 · Suggested portfolio

Six objectives (max Sharpe, min volatility, target return, target volatility,
min drawdown, max diversification), each solved under the *same* suitability
constraints so the results are comparable and every one of them is a portfolio
the client is allowed to hold. Five more — min CVaR, inverse volatility, risk
parity, max Sortino, equal weight — remain in `optimizer.ALL_OBJECTIVES` and
under test, but are not offered: an RM choosing between eleven near-identical
low-volatility answers is choosing noise.

There is also a **Custom mode**, because an RM often arrives at the meeting
already knowing what they want to present. Pick funds, type weights, and the
same engine scores the result — `optimizer.evaluate()` shares every formula
with `optimise()`, so a hand-built book and a solved one are directly
comparable rather than merely adjacent. Seed it from the client's current
holdings or from the optimiser's answer and edit from there; weights that do
not add to 100% are reported and can be rescaled in one click, and the
statistics below are always computed on the normalised book because volatility
and drawdown are only defined on a fully invested portfolio. The mandate
becomes a checklist instead of a constraint: a hand-built portfolio may break
the per-fund cap, the satellite cap or the holding count, and the app says
which and by how much rather than quietly moving the position. Suitability is
checked exactly as it is for a solved portfolio. The whole book travels in the
URL, so a proposal is a link.

Output, in both modes: allocation by fund, asset class and core/satellite; a
like-for-like statistical comparison on the window both portfolios share; the
efficient frontier drawn inside the client's bands; stress and Monte Carlo side
by side; and a costed trade list.

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

**Expected returns blend the house view with what actually happened.**
Trailing five-year returns alone would have the optimiser short China and avoid
Thai bonds; a pure house view alone ignores what these particular funds have
delivered. So expected returns are a weighted mix of the two, set by one slider
on each tab and defaulting to 70% Kasikorn Asset capital market assumptions
(cash rate plus term premium plus equity risk premium, all listed in section 3)
and 30% trailing history. Push it to 0% and the numbers become pure history —
for C06 that lifts expected return from 4.9% to 10.1% and Sharpe from 0.92 to
1.90, which is the extrapolation trap made visible rather than hidden.
Volatility and correlation always come from the workbook, because those
persist and average returns do not.

**The analysis window is two dates, not a round number of years.** The sidebar
carries a year count as a quick preset, but the performance block on tab 1 has
explicit start and end date pickers, because a review is written against the
period the client actually held the book. The window governs the headline
statistics, historical performance, return attribution and risk contribution,
*and* the proposal's like-for-like comparison — nothing on screen is measured
over a different period from anything else. Moving the preset reseats both
dates; typing a date holds until the preset moves again. Both ends travel in
the URL (`&t1_from=2024-01-02&t1_to=2025-12-30`), and a start date after the end
date is reported rather than analysed.

**Each portfolio tab ends in a PDF export.** The RM ticks what to include —
tables and text on one side (client profile, statistics, holdings, allocation,
yearly returns, attribution, risk contribution, suitability findings, stress,
Monte Carlo, market cautions, their own notes), charts on the other (growth,
drawdown, allocation, band compliance, attribution, risk contribution,
correlation, stress, Monte Carlo, efficient frontier, weight comparison, theme
exposure) — and gets a document to take into the meeting. Every page carries
the window the numbers were measured over and a disclaimer, because the
document leaves the building.

Charts in the PDF are the *same figures the tab renders*, rasterised through
kaleido — there is no second set of chart builders to fall out of step. What
changes is the paint: `report.to_print()` walks the figure and swaps every
screen colour for a print one, because the app is drawn mint-on-near-black and
that is invisible on paper. Type is scaled by the ratio between render width
and printed width, or Plotly clips the tick labels it no longer has room for.
If kaleido is missing the chart blocks are shown disabled with the reason, and
the text sections still export — a chart that will not render costs its chart,
not the document.

Thai text is why `assets/fonts` exists. Every PDF base font is Latin-only and
renders Thai as black boxes *without raising*, so Sarabun (OFL, the same face
the UI uses) is embedded from the repo. Push that folder along with
`requirements.txt`, or the export degrades to squares.

**Deploying?** See [DEPLOY.md](DEPLOY.md). `app.py` and `advisor/` ship
together or the app stops on a self-check naming the files that did not arrive —
Streamlit Cloud redacts exception text, so a partial upload otherwise surfaces
as a traceback with no cause.

**The analysis window is floored by the data, not by the workbook.** A
portfolio is only as long as its newest holding, so the lookback slider's
ceiling and the date pickers' floor both move with the client: asking for
twenty years of a book whose newest fund launched in 2023 opens the window on
its first day and says so, instead of printing "2006 – 2026" over 2.8 years of
numbers. A saved link carrying a longer period is pulled into range before the
slider renders, so the slider, the URL and the dates never disagree.

**Growth curves start on the same day or they lie.** The WealthPLUS benchmarks
launched in Aug 2021 while most client books are older, and two curves both
based at ฿10M but opening on different dates hand the later one whatever the
earlier one did in between — for C06 that was a 7.8% fall the benchmark simply
skipped. The default window is therefore seeded where the portfolio *and* its
benchmark both have history; an explicitly chosen period is honoured instead,
and the chart trims itself to the common first day with a caption saying what
it cost. The statistics above keep using the full selected window.

**One covariance estimator, chosen rather than offered.** Ledoit-Wolf
shrinkage, always. A sample covariance over seventy-odd funds and five years of
daily data is dominated by estimation noise, and an RM has no basis for picking
between estimators in front of a client — so the app picks the defensible one
and says so. The alternatives stay in `risk.py` under test.

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
  assumptions.py        capital market assumptions, and the blend with history
  metrics.py            performance and risk statistics
  report.py             PDF assembly: switchable sections, print palette,
                        chart rasterising, embedded Thai font
  risk.py               covariance (Ledoit-Wolf, the one the app uses; sample
                        and EWMA are implemented and tested), risk contribution
  attribution.py        contribution and Brinson allocation-vs-selection
  stress.py             historical replay + factor shocks
  montecarlo.py         five path generators, fan charts
  optimizer.py          objectives, constraints, band repair, holding shrink,
                        and evaluate() for portfolios built by hand
  market.py             macro, news, trends, themes, signals
  cautions.py           theme × holding intersection
  engine.py             one-call analytics bundle
  charts.py             Plotly builders
  theme.py              palette, CSS, Plotly template, components
  th.py                 Thai display strings for values kept in English as keys
  prefs.py              sticky settings, and the slider-with-number-box widget
  notes.py              per-client RM notes, stored as JSON beside the workbook
test_app.py             end-to-end checks — run with `python3 test_app.py`
```

`test_app.py` runs every client through the analytics, all eleven objectives
through every risk profile, and all five Monte Carlo generators — including the
ones the UI no longer offers, so re-offering one is a single-line change rather
than a leap of faith. It checks that
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
