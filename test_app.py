"""End-to-end checks for K-ADVISOR.

Run with:  python3 test_app.py

Exercises every client through the same code paths the UI uses, so a change
that breaks one profile's constraint set shows up here rather than in front of
a client.
"""

from __future__ import annotations

import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from advisor import assumptions as cma
from advisor import attribution, cautions, clients, data as dataio, engine
from advisor import market, metrics, montecarlo as mc, optimizer as opt
from advisor import risk as risklib, stress, universe as uni
from advisor import th as thmod

PASS, FAIL = "  ok  ", " FAIL "
failures: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{PASS if condition else FAIL} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# --------------------------------------------------------------------------- #
section("Data and universe")
fd = dataio.load_workbook()
UNI = uni.build_universe(fd.codes)

check("workbook loads", len(fd.returns) > 1000, f"{len(fd.returns):,} rows")
check("every fund classified", all(u"unclassified" not in f.tags for f in UNI.values()),
      f"{len(UNI)} funds")
check("look-through sums to 1", all(
    abs(sum(f.lookthrough.values()) - 1.0) < 1e-9 for f in UNI.values()))
check("risk levels in 1-8", all(1 <= f.risk_level <= 8 for f in UNI.values()))
check("artefacts quarantined", (fd.returns.abs() > 0.35).sum().sum() == 0,
      f"{len(fd.artefacts)} removed")
check("every client alias resolves",
      all(c in fd.returns.columns for cl in clients.CLIENTS for c in cl.codes))

# --------------------------------------------------------------------------- #
section("Portfolio analytics — all 11 clients")
WINDOW = pd.Timestamp(fd.end) - pd.DateOffset(years=5)
analyses = {}
for cl in clients.CLIENTS:
    try:
        a = engine.analyse(fd, cl.holdings, UNI, WINDOW, None, "Q", 0.0175)
        analyses[cl.id] = a
        contrib_sum = a.contributions["contribution"].sum()
        total = a.contributions.attrs["portfolio_total"]
        risk_sum = a.risk_frame["risk_contribution"].sum()
        port_vol = risklib.portfolio_volatility(
            np.array([a.weights[c] for c in a.risk_frame.index]),
            a.cov.loc[a.risk_frame.index, a.risk_frame.index].to_numpy())
        ok = (abs(contrib_sum - total) < 1e-9
              and abs(risk_sum - port_vol) < 1e-9
              and abs(sum(a.bucket_exposure.values()) - 1.0) < 1e-9)
        check(f"{cl.id} analytics reconcile", ok,
              f"CAGR {a.stats.cagr:+.1%}, vol {a.stats.volatility:.1%}, "
              f"maxDD {a.stats.max_drawdown:.1%}, from {a.window[0]:%b %Y}")
    except Exception as exc:
        check(f"{cl.id} analytics reconcile", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

# --------------------------------------------------------------------------- #
section("Suitability findings")
for cl in clients.CLIENTS:
    f = clients.check_suitability(cl.holdings, cl.profile, UNI)
    status = clients.suitability_status(f)
    ex = clients.bucket_exposure(cl.holdings, UNI)
    check(f"{cl.id} profile {cl.risk_profile} → {status}", True,
          f"cash {ex[uni.CASH]:.0%} / fi {ex[uni.FIXED]:.0%} / "
          f"eq {ex[uni.EQUITY]:.0%} / alt {ex[uni.ALT]:.0%} · {len(f)} finding(s)")

_book = {"K-GOLD-A(A)": 0.1, "K-STAR-A(A)": 0.9}


def _gold_finding(**kw):
    found = [f for f in clients.check_suitability(
        _book, clients.RISK_PROFILES[7], UNI, **kw)
        if f.subject == "K-GOLD-A(A)"]
    return found[0] if found else None


check("an unacknowledged level-8 holding is a breach",
      _gold_finding().severity == "breach")
check("acknowledgement downgrades that finding to a watch",
      _gold_finding(acknowledged={"K-GOLD-A(A)"}).severity == "watch",
      _gold_finding(acknowledged={"K-GOLD-A(A)"}).detail[-52:])

# --------------------------------------------------------------------------- #
section("Stress testing")
for cid in ["C02", "C06", "C08", "C10"]:
    cl = clients.get_client(cid)
    frame = stress.results_frame(stress.run_all(fd.returns, cl.holdings, UNI))
    check(f"{cid} stress replay", len(frame) >= 10,
          f"{len(frame)} events, worst {frame['ผลตอบแทน'].min():.1%}")

panel = stress.factor_panel(fd.returns, uni.FACTOR_PROXIES, start="2018-01-01")
shocks = stress.shock_summary(fd.returns, clients.get_client("C06").holdings, panel)
check("factor shocks produce finite impacts",
      shocks["ผลกระทบต่อพอร์ต"].notna().all(),
      f"equity -20% → {shocks['ผลกระทบต่อพอร์ต'].iloc[0]:.1%}")

# --------------------------------------------------------------------------- #
section("Monte Carlo — every generator")
a = analyses["C06"]
for method in mc.METHODS:
    try:
        sim = mc.simulate(a.panel, a.weights, method=method, n_paths=3000,
                          n_periods=120, period="เดือน")
        ts = sim.terminal_stats()
        sane = (sim.paths.shape == (3000, 121)
                and np.isfinite(sim.terminal).all()
                and 0.0 <= ts["prob_loss"] <= 1.0
                and sim.diagnostics["sim_vol_ann"] > 0)
        check(f"{method}", sane,
              f"median {ts['median']:.2f}x, P(loss) {ts['prob_loss']:.0%}, "
              f"vol {sim.diagnostics['sim_vol_ann']:.1%} vs "
              f"{sim.diagnostics['hist_vol_ann']:.1%} realised")
    except Exception as exc:
        check(f"{method}", False, f"{type(exc).__name__}: {exc}")

mu_fwd = cma.cma_series(list(a.weights), UNI)
sim_cma = mc.simulate(a.panel, a.weights, n_paths=2000, n_periods=60,
                      mu_override=mu_fwd.to_dict())
sim_hist = mc.simulate(a.panel, a.weights, n_paths=2000, n_periods=60)
check("forward CMA drift beats the trailing-loss drift",
      sim_cma.terminal_stats()["median"] > sim_hist.terminal_stats()["median"],
      f"{sim_cma.terminal_stats()['median']:.2f}x vs "
      f"{sim_hist.terminal_stats()['median']:.2f}x")
check("horizon of 120 months is supported",
      mc.simulate(a.panel, a.weights, n_paths=500, n_periods=120).n_periods == 120)

# --------------------------------------------------------------------------- #
section("Optimiser — every objective, every profile")
for level in sorted({c.risk_profile for c in clients.CLIENTS}):
    prof = clients.RISK_PROFILES[level]
    since = pd.Timestamp(fd.end) - pd.DateOffset(years=5)
    elig = opt.eligible_universe(fd.returns, UNI, level, require_since=since,
                                 allow_above_level=True)
    above = frozenset(c for c in elig if UNI[c].risk_level > level)
    pnl = fd.slice(since, None, elig).dropna(how="any")
    cov = risklib.covariance(pnl)
    hist = (1.0 + pnl).prod() ** (252 / len(pnl)) - 1.0
    mu = cma.blended_mu(list(pnl.columns), UNI, hist, cma.CMA)
    band = opt.resolve_bands(elig, UNI, prof.bands, 0.35, above, 0.20)
    cons = opt.Constraints(max_weight=0.35, bands=band.bands, max_satellite=0.35,
                           max_funds=8, min_position=0.03,
                           above_level_codes=above, above_level_budget=0.20)
    print(f"\n  profile {level}: {len(elig)} eligible, bands "
          f"{'as printed' if band.feasible_as_printed else 'relaxed: ' + '; '.join(band.relaxations)}")
    for objective in opt.OBJECTIVES:
        target = {opt.TARGET_RETURN: 0.06, opt.TARGET_VOL: 0.08}.get(objective)
        try:
            sol = opt.optimise(pnl, list(pnl.columns), UNI, objective, cons, mu,
                               cov, target=target)
            ok = (bool(sol.weights)
                  and abs(sum(sol.weights.values()) - 1.0) < 1e-6
                  and all(w >= -1e-9 for w in sol.weights.values())
                  and max(sol.weights.values()) <= 0.35 + 1e-4
                  and sol.diagnostics.get("feasible", False)
                  and len(sol.weights) <= 8)
            check(f"    {objective}", ok,
                  f"n={len(sol.weights)}, er {sol.expected_return:.1%}, "
                  f"vol {sol.volatility:.1%}, sharpe {sol.sharpe:.2f}")
        except Exception as exc:
            check(f"    {objective}", False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

# --------------------------------------------------------------------------- #
section("Infeasibility is reported, not faked")
tiny = ["K-GOLD-A(A)", "K-OIL"]
strict = opt.Constraints(max_weight=0.35, bands=clients.RISK_PROFILES[4].bands,
                         max_satellite=0.0)
bad = opt.optimise(fd.slice("2021-01-01", None, tiny).dropna(), tiny, UNI,
                   opt.MIN_VOL, strict)
check("an impossible mandate returns an error, not a portfolio",
      bad.status == "error" and not bad.weights, bad.message[:70])

res = opt.resolve_bands(["K-STAR-A(A)", "K-FIXED-A", "K-SFPLUS-A"], UNI,
                        clients.RISK_PROFILES[7].bands, 0.35)
check("band repair fires when alternatives are unreachable",
      res.adjusted and "Alternative" not in " ".join(res.relaxations),
      "; ".join(res.relaxations) or "no relaxation")

# --------------------------------------------------------------------------- #
section("Market intelligence")
view = market.build_market_view(with_news=False)
check("macro snapshot", view.macro.status in ("live", "offline"),
      f"{view.macro.status}, {len(view.macro.table)} indicators")
check("signals derived", len(view.signals) >= 3 if view.macro.ok else True,
      ", ".join(f"{s.label}={s.state}" for s in view.signals[:3]))
check("themes ranked", len(view.themes) == len(market.THEMES))
check("severity follows the live signal, not just the baseline",
      len({t.severity for t in view.themes}) > 1,
      f"levels present: {sorted({t.severity_key for t in view.themes})}")

for cid in ["C02", "C06", "C08"]:
    cl = clients.get_client(cid)
    cs = cautions.build_cautions(cl.holdings, UNI, view)
    max_exp = max((c.exposure for c in cs), default=0.0)
    check(f"{cid} cautions link to holdings", max_exp <= 1.0 + 1e-9 and bool(cs),
          f"{len(cs)} themes, top exposure {max_exp:.0%}")

check("trend classifier rejects place names",
      not market._is_finance("สนามยิงปืน บางบัวทอง")
      and market._is_finance("ตลาดหุ้น"))

# ---- Thai localisation ----------------------------------------------------
section("Thai localisation")


def _has_thai(text: str) -> bool:
    return any("\u0e00" <= ch <= "\u0e7f" for ch in str(text))


check("SEC buckets use the form's Thai wording",
      all(_has_thai(b) for b in uni.SEC_BUCKETS), " · ".join(uni.SEC_BUCKETS))
check("asset classes are Thai",
      all(_has_thai(f.asset_class) for f in UNI.values()))
check("regions are Thai", all(_has_thai(f.region) for f in UNI.values()))
check("client names and personas are Thai",
      all(_has_thai(c.name) and _has_thai(c.persona) and _has_thai(c.objective)
          for c in clients.CLIENTS))
check("risk profile descriptions are Thai",
      all(_has_thai(clients.RISK_PROFILES[l].description) for l in (1, 4, 5, 7, 8)))
check("optimiser objectives are Thai",
      all(_has_thai(o) for o in opt.OBJECTIVES))
check("Monte Carlo methods and periods are Thai",
      all(_has_thai(m) for m in mc.METHODS)
      and all(_has_thai(k) for k in mc.PERIOD_LABELS))
check("stress event names are Thai",
      all(_has_thai(e.name) and _has_thai(e.blurb) for e in stress.EVENTS))
check("market themes are Thai",
      all(_has_thai(t.title) and _has_thai(t.what) and _has_thai(t.why)
          and all(_has_thai(w) for w in t.watch) for t in market.THEMES))
check("suitability findings render in Thai",
      all(_has_thai(f.detail)
          for c in clients.CLIENTS
          for f in clients.check_suitability(c.holdings, c.profile, UNI)))
check("Thai month helper formats correctly",
      thmod.month(pd.Timestamp("2026-07-15")) == "ก.ค. 2026"
      and thmod.date(pd.Timestamp("2026-07-15")) == "15 ก.ค. 2026")
check("status and severity map to Thai",
      thmod.status("Breach") == "ไม่ผ่านเกณฑ์"
      and thmod.severity("Elevated") == "สูง"
      and thmod.role("Satellite") == "ส่วนเสริม")

# --------------------------------------------------------------------------- #
section("Current vs proposed comparison")
cur = analyses["C06"]
prop = engine.analyse(fd, {"K-PLAN1": 0.35, "K-CBOND-A": 0.2, "K-GLOBE": 0.25,
                           "K-GOLD-A(D)": 0.2}, UNI, WINDOW, None, "Q", 0.0175)
table = engine.comparison_table(cur, prop)
check("comparison table builds", len(table) == 12,
      f"window {table.attrs['window'][0]:%b %Y}–{table.attrs['window'][1]:%b %Y}")
check("improvement flags computed", len(engine.improvement_flags(table)) == 12)
check("comparison table rows are Thai",
      all(_has_thai(m) or m in ("Sharpe ratio", "Sortino ratio", "Calmar ratio")
          for m in table.index),
      " · ".join(list(table.index)[:4]))
check("turnover in [0, 1]",
      0.0 <= opt.turnover(cur.weights, prop.weights) <= 1.0,
      f"{opt.turnover(cur.weights, prop.weights):.0%}")
trades = opt.trade_list(cur.weights, prop.weights, UNI, 85_000_000)
check("trade list nets to zero",
      abs(trades["เปลี่ยนแปลง"].sum()) < 1e-9, f"{len(trades)} trades")

# --------------------------------------------------------------------------- #
print(f"\n{'=' * 74}")
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  · {f.strip()}")
    sys.exit(1)
print("All checks passed.")
