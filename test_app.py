"""End-to-end checks for K-ADVISOR.

Run with:  python3 test_app.py

Exercises every client through the same code paths the UI uses, so a change
that breaks one profile's constraint set shows up here rather than in front of
a client.
"""

from __future__ import annotations

import ast as _ast
import dataclasses as _dc
import pathlib
import re as _re
import re
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
from advisor import charts as chartsmod
from advisor import compat as compatmod
from advisor import theme as themod
from advisor import notes as noteslib
import importlib as _importlib
import inspect as _inspect
import os as _os

PASS, FAIL = "  ok  ", " FAIL "
failures: list = []


def _has_thai(text: str) -> bool:
    return any("\u0e00" <= ch <= "\u0e7f" for ch in str(text))


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{PASS if condition else FAIL} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def _assignable_after_writable() -> bool:
    frozen = np.zeros(3)
    frozen.flags.writeable = False
    thawed = compatmod.writable(frozen)
    thawed[0] = 1.0
    return thawed[0] == 1.0


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

check("the yearly table reports each year's return and the risk behind it",
      (lambda t: not t.empty
       and list(t.index) == sorted(t.index, reverse=True)
       and {"ผลตอบแทน", "ความผันผวน", "Max Drawdown", "วันทำการ"} <= set(t.columns)
       and (t["Max Drawdown"] <= 1e-12).all()
       and t["วันทำการ"].sum() == len(analyses["C06"].returns.dropna()))(
          metrics.annual_return_table(analyses["C06"].returns)),
      f"{len(metrics.annual_return_table(analyses['C06'].returns))} years, "
      f"newest first")
check("a yearly return matches compounding that year's daily returns",
      (lambda t, r: abs(t.loc[2024, "ผลตอบแทน"]
                        - ((1.0 + r[r.index.year == 2024]).prod() - 1.0)) < 1e-9)(
          metrics.annual_return_table(analyses["C06"].returns),
          analyses["C06"].returns.dropna()))
check("growth curves are based on an investable amount, not a unit NAV",
      chartsmod.GROWTH_BASE == 10_000_000.0
      and chartsmod.baht_label(chartsmod.GROWTH_BASE) == "10 ล้านบาท",
      chartsmod.baht_label(chartsmod.GROWTH_BASE))
check("amounts read the way an RM says them",
      chartsmod.baht_label(1_000_000) == "1 ล้านบาท"
      and chartsmod.baht_label(2_500_000) == "2.5 ล้านบาท"
      and chartsmod.baht_label(500_000) == "500,000 บาท")
check("the curve actually starts at the base",
      (lambda f: abs(f.data[0].y[0]
                     - chartsmod.GROWTH_BASE
                     * (1.0 + analyses["C06"].returns.dropna().iloc[0])) < 1e-6)(
          chartsmod.growth_chart({"พอร์ต": analyses["C06"].returns})),
      f"{chartsmod.growth_chart({'พอร์ต': analyses['C06'].returns}).data[0].y[-1]:,.0f} บาท "
      f"ปลายทาง")

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
for method in mc.ALL_METHODS:
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
check("only Block Bootstrap is offered in the app",
      mc.METHODS == [mc.BLOCK] and mc.DEFAULT_METHOD == mc.BLOCK
      and set(mc.METHODS) <= set(mc.ALL_METHODS),
      " · ".join(mc.METHODS))

# --------------------------------------------------------------------------- #
section("Expected returns — CMA / historical blend")
codes = list(a.weights)
hist_a = (1.0 + a.panel).prod() ** (252 / len(a.panel)) - 1.0
pure_cma = cma.mixed_mu(codes, UNI, hist_a, 1.0)
pure_hist = cma.mixed_mu(codes, UNI, hist_a, 0.0)
half = cma.mixed_mu(codes, UNI, hist_a, 0.5)
check("weight 1.0 reproduces the pure house view",
      np.allclose(pure_cma.to_numpy(),
                  cma.blended_mu(codes, UNI, hist_a, cma.CMA).to_numpy()))
check("weight 0.0 reproduces trailing history",
      np.allclose(pure_hist.to_numpy(), hist_a.reindex(codes).to_numpy()))
check("intermediate weights interpolate linearly",
      np.allclose(half.to_numpy(),
                  0.5 * (pure_cma.to_numpy() + pure_hist.to_numpy())),
      f"C06: CMA {pure_cma.mean():+.1%} · hist {pure_hist.mean():+.1%} · "
      f"50/50 {half.mean():+.1%}")
check("the blend is monotone between the two sources",
      all(min(pure_cma[c], pure_hist[c]) - 1e-12 <= half[c]
          <= max(pure_cma[c], pure_hist[c]) + 1e-12 for c in codes))
check("weights outside 0-1 are clamped, not extrapolated",
      np.allclose(cma.mixed_mu(codes, UNI, hist_a, 2.5).to_numpy(),
                  pure_cma.to_numpy())
      and np.allclose(cma.mixed_mu(codes, UNI, hist_a, -1.0).to_numpy(),
                      pure_hist.to_numpy()))
check("the default leans on the house view",
      0.5 < cma.DEFAULT_CMA_WEIGHT <= 1.0,
      f"{cma.DEFAULT_CMA_WEIGHT:.0%} CMA")
check("blend labels name both sources and their weights",
      cma.blend_label(0.7) == "Kasikorn Asset CMA 70% + Historical 30%"
      and cma.blend_label(1.0) == "Kasikorn Asset CMA 100%"
      and cma.blend_label(0.0) == "Historical 100%"
      and cma.blend_short(0.7) == "CMA 70% + Hist 30%",
      cma.blend_label(0.7))
check("every requested code comes back with a finite number",
      list(half.index) == codes and np.isfinite(half.to_numpy()).all())
_cov_flat = risklib.covariance(a.panel).to_numpy()
check("the blend changes only mu — covariance is read from the panel",
      np.allclose(_cov_flat, risklib.covariance(a.panel).to_numpy())
      and not np.allclose(pure_cma.to_numpy(), pure_hist.to_numpy()),
      "vol and correlation always come from history")

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
    for objective in opt.ALL_OBJECTIVES:
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
section("Hand-built portfolios")
from advisor import prefs as prefs_mod


def clientlib_status(weights, cl) -> str:
    return clients.suitability_status(
        clients.check_suitability(weights, cl.profile, UNI))


_c = clients.CLIENTS[5]
_hand = opt.evaluate(fd.returns, _c.holdings, UNI)
check("a hand-built book is scored, not re-optimised",
      _hand.ok and set(_hand.weights) == set(_c.holdings)
      and all(abs(_hand.weights[k] - v) < 1e-9 for k, v in _c.holdings.items()),
      f"ER {_hand.expected_return:+.1%} · vol {_hand.volatility:.1%} · "
      f"maxDD {_hand.diagnostics['max_drawdown_sample']:.1%}")

# The whole point of sharing the engine is that the two modes are comparable:
# run the optimiser's own answer back through evaluate() and the numbers must
# land in the same place, or every current-vs-proposed table is lying.
_since = pd.Timestamp(fd.end) - pd.DateOffset(years=5)
_elig = opt.eligible_universe(fd.returns, UNI, _c.profile.level,
                              require_since=_since, allow_above_level=True)
_pnl = fd.slice(_since, None, _elig).dropna(how="any")
_cov = risklib.covariance(_pnl)
_hist = (1.0 + _pnl).prod() ** (252 / len(_pnl)) - 1.0
_mu = cma.mixed_mu(list(_pnl.columns), UNI, _hist)
_bands = opt.resolve_bands(_elig, UNI, _c.profile.bands, 0.35, frozenset(), 0.0)
_cons = opt.Constraints(max_weight=0.35, bands=_bands.bands, max_satellite=0.35,
                        max_funds=8, min_position=0.03, rf=0.02)
_sol = opt.optimise(_pnl, list(_pnl.columns), UNI, opt.MIN_VOL, _cons, _mu, _cov)
_scored = opt.evaluate(fd.returns, _sol.weights, UNI, _mu,
                       _cov.loc[sorted(_sol.weights), sorted(_sol.weights)],
                       rf=0.02)
check("scoring the optimiser's own answer reproduces its statistics",
      abs(_scored.expected_return - _sol.expected_return) < 1e-9
      and abs(_scored.volatility - _sol.volatility) < 1e-9
      and abs(_scored.sharpe - _sol.sharpe) < 1e-9,
      f"ER {_scored.expected_return:.4%} vs {_sol.expected_return:.4%} · "
      f"vol {_scored.volatility:.4%} vs {_sol.volatility:.4%}")

check("weights that do not sum to 100% are normalised, and the raw total kept",
      (lambda r: abs(sum(r.weights.values()) - 1.0) < 1e-9
       and abs(r.diagnostics["raw_total"] - 0.6) < 1e-9)(
          opt.evaluate(fd.returns,
                       {c: v * 0.6 for c, v in _c.holdings.items()}, UNI)))
check("an empty book is refused instead of invented",
      opt.evaluate(fd.returns, {}, UNI).status == "error"
      and not opt.evaluate(fd.returns, {}, UNI).weights)
check("unknown or zero-weight codes are dropped quietly",
      set(opt.evaluate(fd.returns, {**_c.holdings, "NOT-A-FUND": 0.2,
                                    list(_c.holdings)[0] + "x": 0.0},
                       UNI).weights) == set(_c.holdings))

# A hand-built book may break the mandate — that is the RM's prerogative. What
# it may not do is break it silently.
_tight = opt.Constraints(max_weight=0.25, max_satellite=0.10, max_funds=2,
                         rf=0.02, above_level_codes=frozenset(_c.holdings),
                         above_level_budget=0.05)
_checked = opt.evaluate(fd.returns, _c.holdings, UNI, cons=_tight)
_breaches = _checked.diagnostics["mandate_breaches"]
check("mandate breaches are reported, never silently repaired",
      _checked.status == "warning" and len(_breaches) == 4
      and _checked.weights == _hand.weights
      and not _checked.diagnostics["feasible"],
      f"{len(_breaches)} breaches, weights untouched")
check("each breach names the rule, the holding and the limit",
      all(_has_thai(b) and len(_re.findall(r"\d+", b)) >= 2 for b in _breaches),
      " | ".join(b[:44] for b in _breaches[1:3]))
check("a book inside the mandate reports no breach",
      (lambda r: r.ok and not r.diagnostics["mandate_breaches"])(
          opt.evaluate(fd.returns, _c.holdings, UNI,
                       cons=opt.Constraints(max_weight=0.6, max_satellite=0.8,
                                            max_funds=10, rf=0.02))))

# Suitability is checked on hand-built books exactly as on solved ones, which
# is the reason the mode is worth having: the RM sees the breach before the
# client does.
check("suitability still applies to a hand-built book",
      clientlib_status(_hand.weights, _c) == clientlib_status(_c.holdings, _c),
      clientlib_status(_hand.weights, _c))

_enc = prefs_mod.encode_holdings(_hand.weights)
_dec = prefs_mod.decode_holdings(_enc)
check("a hand-built book round-trips through the URL",
      set(_dec) == set(_hand.weights)
      and all(abs(_dec[c] - w) < 5e-4 for c, w in _hand.weights.items()),
      _enc)
check("a malformed holdings link drops the bad pair, not the session",
      prefs_mod.decode_holdings("K-GOLD-A(A):abc,K-SF-A:10") == {"K-SF-A": 0.1}
      and prefs_mod.decode_holdings("nonsense") == {}
      and prefs_mod.decode_holdings(None) == {}
      and prefs_mod.decode_holdings("K-SF-A:-5") == {})

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
check("the app offers a narrowed objective menu",
      set(opt.OBJECTIVES) <= set(opt.ALL_OBJECTIVES)
      and not ({opt.MIN_CVAR, opt.INVERSE_VOL, opt.RISK_PARITY,
                opt.MAX_SORTINO, opt.EQUAL_WEIGHT} & set(opt.OBJECTIVES)),
      f"{len(opt.OBJECTIVES)} of {len(opt.ALL_OBJECTIVES)} shown")
check("optimiser objectives use industry English",
      all(o.isascii() for o in opt.ALL_OBJECTIVES),
      " · ".join(opt.OBJECTIVES[:3]))
check("Core / Satellite stay English",
      thmod.role("Core") == "Core" and thmod.role("Satellite") == "Satellite")
check("Monte Carlo period units are Thai",
      all(_has_thai(k) for k in mc.PERIOD_LABELS), " · ".join(mc.PERIOD_LABELS))
check("stress event names are Thai",
      all(_has_thai(e.name) and _has_thai(e.blurb) for e in stress.EVENTS))
check("market themes are Thai",
      all(_has_thai(t.title) and _has_thai(t.what) and _has_thai(t.why)
          and all(_has_thai(w) for w in t.watch) for t in market.THEMES))
check("suitability findings render in Thai",
      all(_has_thai(f.detail)
          for c in clients.CLIENTS
          for f in clients.check_suitability(c.holdings, c.profile, UNI)))
_BOUNDARY = re.compile(r"[ก-๙][A-Za-z]|[A-Za-z][ก-๙]")


def _runs_together(text) -> bool:
    return bool(_BOUNDARY.search(str(text)))


check("no Thai word runs into an English word without a space",
      not any(_runs_together(t) for t in (
          [c.description for c in clients.RISK_PROFILES.values()]
          + [c.notes for c in clients.CLIENTS]
          + [o for o in opt.OBJECTIVE_NOTES.values()]
          + [m for m in mc.METHOD_NOTES.values()]
          + [e.blurb for e in stress.EVENTS]
          + [t.what for t in market.THEMES] + [t.why for t in market.THEMES])),
      "checked profiles, clients, objectives, MC methods, events and themes")

# The constants above are only half of it. app.py builds most of its prose in
# f-strings, and a missing space there survives every module-level scan — that
# is exactly how the last batch of run-together text shipped. Read the source
# instead of the modules.
_app_src = pathlib.Path(__file__).with_name("app.py").read_text()
_app_literals = [n.value for n in _ast.walk(_ast.parse(_app_src))
                 if isinstance(n, _ast.Constant) and isinstance(n.value, str)]
_app_bad = [t for t in _app_literals if _runs_together(t)]
check("app.py's own prose keeps Thai and English apart",
      not _app_bad,
      f"{len(_app_literals)} string literals scanned"
      if not _app_bad else _app_bad[0][:70])

# Values interpolated straight after Thai need a leading space of their own,
# and these are the ones the Monte Carlo and optimiser captions splice in.
check("interpolated labels do not fuse with the Thai around them",
      not any(_runs_together(f"ผลตอบแทนคาดหวังจาก {lab}")
              for w in (0.0, 0.35, 0.7, 1.0)
              for lab in (cma.blend_label(w), cma.blend_short(w)))
      and not _runs_together(f"วิธีจำลอง · {mc.DEFAULT_METHOD}"),
      f"ผลตอบแทนคาดหวังจาก {cma.blend_label(0.7)}")

check("every theme carries a purpose-written caution line",
      all(t.caution and _has_thai(t.caution) for t in market.THEMES))
# The cards sit three-across and equalise their height off the tallest, so a
# caution much longer than its neighbours overflows its own card. Cap the copy
# rather than clamp it visually — a truncated warning is worse than none.
check("caution lines fit a three-across card",
      all(len(t.caution) <= 110 for t in market.THEMES),
      f"longest {max(len(t.caution) for t in market.THEMES)} chars")
check("caution lines are not a truncation of the long explainer",
      all(t.caution not in t.why and not t.why.startswith(t.caution[:40])
          for t in market.THEMES))

_cards = cautions.build_cautions(clients.get_client("C06").holdings, UNI, view)
# A half-pushed checkout on Streamlit Cloud once served a new cautions.py
# against an old market.py, and the app died on Theme.caution with an error
# Cloud redacts. The display fields are now read defensively; prove it.
_StaleTheme = _dc.make_dataclass(
    "_StaleTheme", [(f.name, f.type) for f in _dc.fields(market.Theme)
                    if f.name != "caution"])
_StaleSignal = _dc.make_dataclass(
    "_StaleSignal", [(f.name, f.type) for f in _dc.fields(market.Signal)
                     if f.name != "reading"])


def _stale(obj, cls, drop):
    return cls(**{f.name: getattr(obj, f.name)
                  for f in _dc.fields(type(obj)) if f.name != drop})


_stale_view = market.MarketView(
    view.macro, view.signals,
    [market.ThemeView(_stale(tv.theme, _StaleTheme, "caution"), tv.severity,
                      None if tv.signal is None
                      else _stale(tv.signal, _StaleSignal, "reading"),
                      tv.headlines)
     for tv in view.themes],
    view.trends, view.trends_status, view.news_status, view.fetched_at)

try:
    _stale_cards = cautions.build_cautions(
        clients.get_client("C06").holdings, UNI, _stale_view)
    _stale_ok = bool(_stale_cards) and all(c.headline for c in _stale_cards)
    _stale_detail = f"{len(_stale_cards)} cards still render"
except Exception as exc:
    _stale_ok, _stale_detail = False, f"{type(exc).__name__}: {exc}"
check("cautions survive an out-of-date market.py", _stale_ok, _stale_detail)

check("module schema versions agree",
      market.SCHEMA_VERSION >= cautions.REQUIRED_SCHEMA,
      f"market {market.SCHEMA_VERSION} ≥ cautions {cautions.REQUIRED_SCHEMA}")

check("caution cards expose headline and reading separately",
      all(c.headline and not _runs_together(c.headline) for c in _cards),
      f"{len(_cards)} cards")

check("Thai month helper formats correctly",
      thmod.month(pd.Timestamp("2026-07-15")) == "ก.ค. 2026"
      and thmod.date(pd.Timestamp("2026-07-15")) == "15 ก.ค. 2026")
check("status and severity map to Thai",
      thmod.status("Breach") == "ไม่ผ่านเกณฑ์"
      and thmod.severity("Elevated") == "สูง")

# --------------------------------------------------------------------------- #
section("Current vs proposed comparison")
cur = analyses["C06"]
prop = engine.analyse(fd, {"K-PLAN1": 0.35, "K-CBOND-A": 0.2, "K-GLOBE": 0.25,
                           "K-GOLD-A(D)": 0.2}, UNI, WINDOW, None, "Q", 0.0175)
table = engine.comparison_table(cur, prop)
check("comparison table builds", len(table) == 12,
      f"window {table.attrs['window'][0]:%b %Y}–{table.attrs['window'][1]:%b %Y}")
check("improvement flags computed", len(engine.improvement_flags(table)) == 12)
# Metric names are deliberately English where that is the industry's own term
# ("Max Drawdown", "Sharpe Ratio"); only the descriptive rows must be Thai.
_ENGLISH_BY_DESIGN = {
    "Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
    "Calmar Ratio", "Hit Rate", "Effective Positions", "Diversification Ratio",
    "VaR 95% (1 วัน)", "CVaR 95% (1 วัน)",
}
check("comparison table rows read as Thai or industry English",
      all(_has_thai(m) or m in _ENGLISH_BY_DESIGN for m in table.index),
      " · ".join(list(table.index)[:4]))
check("turnover in [0, 1]",
      0.0 <= opt.turnover(cur.weights, prop.weights) <= 1.0,
      f"{opt.turnover(cur.weights, prop.weights):.0%}")
trades = opt.trade_list(cur.weights, prop.weights, UNI, 85_000_000)
check("trade list nets to zero",
      abs(trades["เปลี่ยนแปลง"].sum()) < 1e-9, f"{len(trades)} trades")



# --------------------------------------------------------------------------- #
section("Sticky settings")

_app_src = pathlib.Path("app.py").read_text()
_app_tree = _ast.parse(_app_src)
_WIDGETS = {"selectbox", "slider", "select_slider", "number_input", "checkbox",
            "toggle", "radio", "text_input", "multiselect", "date_input",
            "text_area", "time_input", "color_picker"}
_calls = [n for n in _ast.walk(_app_tree)
          if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
          and n.func.attr in _WIDGETS and isinstance(n.func.value, _ast.Name)
          and n.func.value.id == "st"]


def _kwargs(call):
    return {k.arg for k in call.keywords}


# Without a key a widget is re-created from its default arguments whenever they
# change, which is what made settings snap back to defaults.
check("every widget declares a key",
      all("key" in _kwargs(c) for c in _calls),
      f"{len(_calls)} widgets")

# A literal value=/index= alongside a key fights session state: Streamlit warns
# and the remembered value loses. Passing `st.session_state[key]` back in is a
# different thing — it is equal to the remembered value by construction, and
# st.date_input needs it, because that widget range-checks its implicit "today"
# default before it ever looks at session state.
def _literal_default(call) -> bool:
    for kw in call.keywords:
        if kw.arg not in ("value", "index"):
            continue
        node = kw.value
        is_state_read = (
            isinstance(node, _ast.Subscript)
            and isinstance(node.value, _ast.Attribute)
            and node.value.attr == "session_state")
        if not is_state_read:
            return True
    return False


check("no widget mixes a key with a literal default",
      all(not _literal_default(c) for c in _calls),
      "value= is allowed only when it reads st.session_state back")

_keys = {c.keywords[[k.arg for k in c.keywords].index("key")].value.value
         for c in _calls
         if "key" in _kwargs(c)
         and isinstance(c.keywords[[k.arg for k in c.keywords].index("key")].value,
                        _ast.Constant)}
_registered = set(_re.findall(r'"([a-z0-9_]+)",', _app_src.split("REMEMBERED: List[str] = [")[1].split("]")[0]))
# Keys starting with "_" are session scratch, not settings: the hand-built
# portfolio's picker and per-fund boxes are rebuilt from "t2_custom" on every
# render, and that is the key the URL carries.
_static_keys = {k for k in _keys if not k.startswith("_")}
check("every static widget key is registered in REMEMBERED",
      _static_keys <= _registered | {"aum"},
      f"unregistered: {sorted(_static_keys - _registered - {'aum'}) or 'none'}")

from advisor import prefs as prefsmod


def _as_date(raw):
    import datetime as _dt
    if isinstance(raw, _dt.date):
        return raw
    return _dt.date.fromisoformat(str(raw).strip())


import datetime as _dt

check("an analysis window round-trips through the query string",
      prefsmod.resolve("2024-01-02", _dt.date(2021, 7, 15), cast=_as_date)
      == _dt.date(2024, 1, 2)
      and str(_dt.date(2024, 1, 2)) == "2024-01-02")
check("a malformed date falls back to the preset window",
      all(prefsmod.resolve(bad, _dt.date(2021, 7, 15), cast=_as_date)
          == _dt.date(2021, 7, 15)
          for bad in ("yesterday", "2024-13-45", "", "2024/01/02", "1700000000")))

# The window governs the headline stats, the three history sections and the
# proposal's comparison, so the analytics have to honour both ends of it —
# an end date that is silently ignored would overstate a client's returns.
_win = engine.analyse(fd, clients.CLIENTS[5].holdings, UNI,
                      pd.Timestamp("2024-01-02"), pd.Timestamp("2025-12-30"),
                      "Q", 0.02, "Ledoit-Wolf shrinkage", "Windowed")
check("analysis honours both ends of the window",
      pd.Timestamp("2024-01-02") <= _win.window[0]
      and _win.window[1] <= pd.Timestamp("2025-12-30"),
      f"{_win.window[0]:%d %b %Y} – {_win.window[1]:%d %b %Y}, "
      f"{len(_win.returns):,} days")
_full = engine.analyse(fd, clients.CLIENTS[5].holdings, UNI,
                       pd.Timestamp("2021-07-15"), None,
                       "Q", 0.02, "Ledoit-Wolf shrinkage", "Full")
check("a narrower window gives different figures, not the same ones",
      len(_win.returns) < len(_full.returns)
      and abs(_win.stats.cagr - _full.stats.cagr) > 1e-6,
      f"CAGR {_win.stats.cagr:+.1%} on the window vs "
      f"{_full.stats.cagr:+.1%} on five years")
check("attribution and risk contribution use the windowed panel",
      _win.contributions.attrs["portfolio_total"]
      != _full.contributions.attrs["portfolio_total"]
      and not _win.risk_frame.empty
      and abs(_win.risk_frame["risk_share"].sum() - 1.0) < 1e-6,
      f"contributions total {_win.contributions.attrs['portfolio_total']:+.1%} "
      f"over the window")

check("a query value is parsed and used",
      prefsmod.resolve("M", "Q", valid=["Q", "M", "A"]) == "M"
      and prefsmod.resolve("2500", 10000, cast=int) == 2500
      and prefsmod.resolve("0", True, cast=prefsmod.as_bool) is False)

check("a missing value falls back to the default",
      prefsmod.resolve(None, "Q", valid=["Q", "M", "A"]) == "Q")

# A shared or hand-edited link must never crash the app or feed a widget an
# option it does not offer.
# Sliders are paired with a number box; the box is free text, so its callback
# must clamp before the value reaches the slider, which raises out of range.
check("number_slider clamps a typed value into range",
      hasattr(prefsmod, "number_slider")
      and "min(max(" in _inspect.getsource(prefsmod.number_slider))

# ---- client notes ---------------------------------------------------------
import tempfile as _tempfile

_os.environ["KADVISOR_NOTES_PATH"] = str(
    pathlib.Path(_tempfile.mkdtemp()) / "notes.json")
_importlib.reload(noteslib)

_note = "ลูกค้ากังวลสัดส่วนจีน นัดทบทวนไตรมาสหน้า"
check("a note saves and reads back",
      noteslib.save("C06", _note)[0] and noteslib.get("C06") == _note)
check("notes are separate per client",
      noteslib.save("C02", "เน้นสภาพคล่อง")[0]
      and noteslib.get("C06") == _note
      and noteslib.get("C02") == "เน้นสภาพคล่อง"
      and noteslib.get("C01") == "")
check("clearing a note leaves the others intact",
      noteslib.save("C06", "   ")[0] and noteslib.get("C06") == ""
      and noteslib.get("C02") == "เน้นสภาพคล่อง")

_os.environ["KADVISOR_NOTES_PATH"] = "/nonexistent/dir/notes.json"
_importlib.reload(noteslib)
check("a read-only filesystem degrades instead of raising",
      noteslib.save("C01", "x")[0] is False and noteslib.load() == {})
_os.environ.pop("KADVISOR_NOTES_PATH", None)
_importlib.reload(noteslib)

check("a malformed or unknown query value falls back safely",
      prefsmod.resolve("NOPE", "Q", valid=["Q", "M", "A"]) == "Q"
      and prefsmod.resolve("abc", 10000, cast=int) == 10000
      and prefsmod.resolve("", 6.0, cast=float) == 6.0)

# --------------------------------------------------------------------------- #
section("PDF export")
from advisor import report as reportlib

_pdf_ok, _pdf_why = reportlib.available()
check("the PDF toolchain is present", _pdf_ok, _pdf_why or "reportlab + Sarabun")

# Latin-only base fonts render Thai as black boxes rather than raising, so the
# bundled face is the difference between a usable document and one an RM hands
# to a client full of squares.
check("a Thai-capable font ships with the repo",
      all(_os.path.isfile(_os.path.join(reportlib.FONT_DIR, n))
          for n in ("Sarabun-Regular.ttf", "Sarabun-Bold.ttf"))
      and _os.path.isfile(_os.path.join(reportlib.FONT_DIR, "OFL.txt")),
      reportlib.FONT_DIR)
_reqs = pathlib.Path("requirements.txt").read_text()
check("reportlab is declared for deployment", "reportlab" in _reqs)

_charts_ok, _charts_why = reportlib.charts_available()
check("chart rasterising is available", _charts_ok, _charts_why or "kaleido")
# kaleido 1.x delegates to a separately installed Chrome, which Streamlit Cloud
# does not have; 0.2.x carries its own. Pin it or charts vanish on deploy.
check("kaleido is pinned to the self-contained 0.2 line",
      "kaleido==0.2" in _reqs, _reqs.strip().splitlines()[-1])

if _charts_ok:
    import plotly.graph_objects as _go
    _screen = chartsmod.growth_chart({"พอร์ต": analyses["C06"].returns})
    _paper = reportlib.to_print(_screen)
    _colours = {t.line.color for t in _paper.data if getattr(t, "line", None)}
    check("figures are recoloured for white paper, not left dark-theme",
          _paper.layout.paper_bgcolor == "#FFFFFF"
          and themod.MINT not in _colours
          and reportlib.ACCENT in _colours,
          f"line colours {sorted(c for c in _colours if c)}")
    # Only the colours that carry data or text. Gridlines and panel fills are
    # meant to be faint — holding a hairline to a text contrast bar would just
    # force it dark enough to compete with the series drawn on top of it.
    _furniture = {"#FFFFFF", reportlib.BAND, reportlib.RULE}
    _data_colours = [c for c in reportlib.PRINT_MAP.values()
                     if c not in _furniture]
    check("every print colour that carries meaning is legible on white",
          all(themod.contrast_ratio(c, "#FFFFFF") >= 3.0 for c in _data_colours),
          f"worst {min(themod.contrast_ratio(c, '#FFFFFF') for c in _data_colours):.2f}:1 "
          f"across {len(_data_colours)} colours")
    check("gridlines stay faint enough to sit behind the data",
          themod.contrast_ratio(reportlib.RULE, "#FFFFFF") < 2.0)
    check("translucent fills keep their alpha through the remap",
          reportlib._print_colour(themod.rgba(themod.MINT, 0.18))
          .startswith("rgba(15, 122, 87,")
          and reportlib._print_colour("#123456") == "#123456")
    _png = reportlib.figure_png(_screen, height=380)
    check("a figure rasterises to a PNG the document can place",
          _png is not None and _png[:4] == b"\x89PNG" and len(_png) > 5000,
          f"{len(_png):,} bytes")
    _text_only = reportlib.build_pdf(
        "ข้อความ", "C06", [reportlib.Section("ผลการดำเนินงาน",
                                             bullets=["ทดสอบ"])])
    _with_chart = reportlib.build_pdf(
        "กราฟ", "C06", [reportlib.Section("ผลการดำเนินงาน", image=_png)])
    check("a chart section lands in the PDF",
          _with_chart.startswith(b"%PDF")
          and len(_with_chart) > len(_text_only) + 10_000,
          f"{len(_with_chart):,} bytes with the chart vs "
          f"{len(_text_only):,} without")
    check("a chart that will not render costs its chart, not the document",
          reportlib.figure_png(None) is None)

if _pdf_ok:
    _c = clients.CLIENTS[5]
    _secs = [
        reportlib.Section("ข้อมูลลูกค้า",
                          bullets=[f"{_c.name} ({_c.id})", "ระดับความเสี่ยง 7"]),
        reportlib.Section("สรุปสถิติพอร์ต", lead="ช่วง 15 ก.ค. 2021 – 15 ก.ค. 2026",
                          columns=["รายการ", "ค่า"],
                          rows=[["ผลตอบแทนต่อปี", "-4.5%"],
                                ["ความผันผวน", "18.0%"]],
                          align_right=[1]),
    ]
    _bytes = reportlib.build_pdf("พอร์ตปัจจุบัน", "C06 · ทดสอบ", _secs,
                                 footer="K-ADVISOR", disclaimer="ทดสอบ")
    check("a PDF is produced, with Thai text and the font embedded",
          _bytes.startswith(b"%PDF") and len(_bytes) > 8000
          and b"Sarabun" in _bytes,
          f"{len(_bytes):,} bytes")
    check("an empty selection still yields a valid file, not a crash",
          reportlib.build_pdf("ว่าง", "—", []).startswith(b"%PDF"))
    check("every offered block has a Thai title",
          all(_has_thai(b.title) or b.title.isascii()
              for b in reportlib.CURRENT_BLOCKS + reportlib.PROPOSED_BLOCKS)
          and all(b.key and b.title
                  for b in reportlib.CURRENT_BLOCKS + reportlib.PROPOSED_BLOCKS))
    check("block keys are unique within each document",
          len({b.key for b in reportlib.CURRENT_BLOCKS}) == len(reportlib.CURRENT_BLOCKS)
          and len({b.key for b in reportlib.PROPOSED_BLOCKS})
          == len(reportlib.PROPOSED_BLOCKS))
    check("the default selection names blocks that exist",
          set(reportlib.DEFAULT_CURRENT) <= {b.key for b in reportlib.CURRENT_BLOCKS}
          and set(reportlib.DEFAULT_PROPOSED)
          <= {b.key for b in reportlib.PROPOSED_BLOCKS},
          f"{len(reportlib.DEFAULT_CURRENT)} + "
          f"{len(reportlib.DEFAULT_PROPOSED)} ticked by default")

# The app builds one Section per ticked block, so a block the app cannot render
# would silently vanish from a document an RM believes is complete.
_app_body = pathlib.Path("app.py").read_text()
check("app.py renders every block it offers",
      all(f'"{b.key}" in want' in _app_body
          for b in reportlib.CURRENT_BLOCKS + reportlib.PROPOSED_BLOCKS),
      f"{len(reportlib.CURRENT_BLOCKS) + len(reportlib.PROPOSED_BLOCKS)} blocks")
check("both documents carry the same disclaimer",
      _app_body.count("disclaimer=DISCLAIMER") == 2
      and _has_thai(_re.search(r'DISCLAIMER = \(\n(.*?)\n\)',
                               _app_body, _re.S).group(1)))


def _srgb(channel: float) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _srgb(r) + 0.7152 * _srgb(g) + 0.0722 * _srgb(b)


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# --------------------------------------------------------------------------- #
section("Analysis window floors")

# A portfolio is only as long as its newest holding. Before this, a 20-year
# request on a book whose newest fund launched in 2023 printed "2006 – 2026" in
# the header while analysing 2.8 years.
_floors = {c.id: dataio.common_floor(fd.returns, c.holdings)
           for c in clients.CLIENTS}
check("every client's floor is its newest holding's launch",
      all(f == max(dataio.first_day(fd.returns, code)
                   for code in c.holdings)
          for c in clients.CLIENTS for f in [_floors[c.id]]),
      ", ".join(f"{k} {v.date()}" for k, v in list(_floors.items())[:4]) + " …")
check("a fund with no data does not sink the floor",
      dataio.first_day(fd.returns, "NOT-A-FUND") is None
      and dataio.common_floor(fd.returns, []) is None
      and dataio.common_floor(fd.returns, ["NOT-A-FUND"]) is None)

# The benchmarks launched together in Aug 2021, which is why the default window
# has to be seeded from them rather than from a round number of years back.
# The same three funds app.py offers, read from the app rather than restated
# here, so a benchmark added there is covered without touching this file.
_bench_src = _re.search(r"BENCH_FUNDS = \{(.*?)\}",
                        pathlib.Path("app.py").read_text(), _re.S).group(1)
_bench_codes = dict(_re.findall(r'"([^"]+)":\s*"([^"]+)"', _bench_src))
_bench_starts = {name: dataio.first_day(fd.returns, code)
                 for name, code in _bench_codes.items()}
check("every benchmark has a launch date the window can be seeded from",
      all(d is not None for d in _bench_starts.values()),
      ", ".join(f"{n.split()[-1]} {d.date()}" for n, d in _bench_starts.items()))
check("the benchmarks start later than most portfolios, so alignment matters",
      sum(1 for c in clients.CLIENTS
          if _floors[c.id] < min(_bench_starts.values())) >= 4,
      f"{sum(1 for c in clients.CLIENTS if _floors[c.id] < min(_bench_starts.values()))}"
      f" of {len(clients.CLIENTS)} clients start before the benchmarks do")

# The bug this fixes: two curves both based at 10M but starting on different
# days hand the later one whatever the earlier one did in between.
_c06 = next(c for c in clients.CLIENTS if c.id == "C06")
_a06 = analyses["C06"]
_bench06 = fd.returns["K-WPBALANCED"].reindex(_a06.returns.index).dropna()
_common = max(_a06.returns.index[0], _bench06.index[0])
check("trimming to the common first day removes the benchmark's free head start",
      _common > _a06.returns.index[0]
      and abs((1.0 + _a06.returns.loc[:_common]).prod() - 1.0) > 0.01,
      f"portfolio moved {(1.0 + _a06.returns.loc[:_common]).prod() - 1.0:+.2%} "
      f"before the benchmark existed")
check("after trimming, both series open on the same day",
      _a06.returns.loc[_common:].index[0] == _bench06.loc[_common:].index[0])

_win_src = pathlib.Path("app.py").read_text()
check("the date pickers cannot reach behind the portfolio's own data",
      "DATA_FLOOR = _portfolio_floor.date()" in _win_src)
check("a period longer than the data lands on the first day there is data",
      "_over_run or window_start < _portfolio_floor" in _win_src
      and "window_start = _portfolio_floor" in _win_src)
check("a stale saved period is pulled into range before the slider renders",
      'st.session_state["lookback"] = _MAX_YEARS' in _win_src)
check("an explicitly chosen period is not overridden by the benchmark",
      "window_start if _lookback_explicit" in _win_src)
check("the benchmark only floors the default, never the analysis itself",
      "_compare_floor" in _win_src
      and "_compare_floor" not in _win_src.split("analysis_start =")[1])


# --------------------------------------------------------------------------- #
section("Tabs")

_app_src = pathlib.Path("app.py").read_text()
_tab_labels = _re.search(r"st\.tabs\(\[(.*?)\]\)", _app_src, _re.S).group(1)
# The three tabs are views of one client, not steps in a sequence. Numbering
# them implies an order an RM does not follow.
check("tab labels carry no step numbers",
      not _re.search(r'"\s*\d+\s*·', _tab_labels)
      and _tab_labels.count('"') == 6,
      ", ".join(_re.findall(r'"([^"]+)"', _tab_labels)))
check("tab labels are not padded with spaces to fake indentation",
      all(v == v.strip() for v in _re.findall(r'"([^"]+)"', _tab_labels)))

# Streamlit renders each tab label inside its own <p>, which carries the body
# text colour. Styling only the tab leaves every label looking selected, which
# is the bug this pins.
check("the selected tab is distinguished on the label element itself",
      '.stTabs [aria-selected="true"] p' in themod.CSS
      and 'font-weight: 600' in themod.CSS.split(
          '.stTabs [aria-selected="true"] p')[1][:120])
check("unselected tab labels sit at the muted weight and colour",
      themod.MUTED in themod.CSS.split(
          '.stTabs [data-baseweb="tab"] p')[1][:160])
check("the active indicator can animate, so it is not a border",
      '.stTabs [data-baseweb="tab"]::after' in themod.CSS
      and "transition: width" in themod.CSS)
# Slice the whole tab block rather than a fixed number of characters, so
# adding a rule cannot quietly move a check's evidence out of its window.
_tab_css = themod.CSS.split("---------- tabs")[1].split("/* ---------- widgets")[0]
check("tabs answer to hover and to keyboard focus",
      ':hover p' in _tab_css and ':focus-visible' in _tab_css)
check("the tab row stays usable when the window is narrow",
      "@media (max-width: 1200px)" in _tab_css
      and "scrollbar-width: none" in _tab_css)
check("selected tab label clears 4.5:1 on the canvas",
      _contrast(themod.TEXT, themod.BG) >= 4.5)
check("unselected tab label clears 4.5:1 on the canvas",
      _contrast(themod.MUTED, themod.BG) >= 4.5,
      f"{_contrast(themod.MUTED, themod.BG):.2f}:1")


# --------------------------------------------------------------------------- #
section("Colour contrast")


# Every foreground/background pair the app actually renders. The mint-filled
# button and the multiselect tag both shipped at ~1.2:1 because Streamlit puts
# the label in its own element and the colour was only inherited — so the pairs
# are pinned here, where a palette edit cannot quietly undo the fix.
_PAIRS = [
    ("body text on canvas", themod.TEXT, themod.BG),
    ("body text on card", themod.TEXT, themod.CARD),
    ("muted text on card", themod.MUTED, themod.CARD),
    ("dim label on card", themod.DIM, themod.CARD),
    ("dim label on canvas", themod.DIM, themod.BG),
    ("dim label on sidebar", themod.DIM, themod.PANEL),
    ("outlined button label", themod.MINT, themod.CARD),
    ("primary button label", themod.BG, themod.MINT),
    ("multiselect tag label", themod.MINT, themod.CARD_HI),
    ("link on card", themod.MINT, themod.CARD),
    ("negative figure on card", themod.CORAL, themod.CARD),
    ("warning figure on card", themod.AMBER, themod.CARD),
    ("informational on card", themod.SKY, themod.CARD),
    ("accent on canvas", themod.MINT, themod.BG),
    ("sand accent on card", themod.SAND, themod.CARD),
]
for _name, _fg, _bg in _PAIRS:
    _r = _contrast(_fg, _bg)
    check(f"{_name} clears 4.5:1", _r >= 4.5, f"{_r:.2f}:1  {_fg} on {_bg}")

# A diverging scale cannot carry one text colour: light text disappears on the
# mint end and dark text on the middle. Every correlation a matrix can show has
# to be legible over whatever the ramp puts under it.
_worst_rho, _worst_ratio = None, 99.0
for _i in range(-100, 101):
    _rho = _i / 100.0
    _fill = themod.sample_scale(themod.CORRELATION_SCALE, (_rho + 1.0) / 2.0)
    _ratio = themod.contrast_ratio(themod.on_scale(themod.CORRELATION_SCALE,
                                                   (_rho + 1.0) / 2.0), _fill)
    if _ratio < _worst_ratio:
        _worst_rho, _worst_ratio = _rho, _ratio
check("every correlation value is legible on its own cell",
      _worst_ratio >= 4.5,
      f"worst is ρ={_worst_rho:+.2f} at {_worst_ratio:.2f}:1")
check("the correlation ramp is diverging around a dark zero",
      themod.luminance(themod.sample_scale(themod.CORRELATION_SCALE, 0.5))
      < themod.luminance(themod.sample_scale(themod.CORRELATION_SCALE, 0.0))
      and themod.luminance(themod.sample_scale(themod.CORRELATION_SCALE, 0.5))
      < themod.luminance(themod.sample_scale(themod.CORRELATION_SCALE, 1.0)),
      f"ρ=-1 {themod.sample_scale(themod.CORRELATION_SCALE, 0.0)} · "
      f"ρ=0 {themod.sample_scale(themod.CORRELATION_SCALE, 0.5)} · "
      f"ρ=+1 {themod.sample_scale(themod.CORRELATION_SCALE, 1.0)}")
check("scale sampling is clamped, not extrapolated",
      themod.sample_scale(themod.CORRELATION_SCALE, -3.0)
      == themod.sample_scale(themod.CORRELATION_SCALE, 0.0)
      and themod.sample_scale(themod.CORRELATION_SCALE, 9.0)
      == themod.sample_scale(themod.CORRELATION_SCALE, 1.0))

check("the de-emphasis ladder still reads as a ladder",
      _luminance(themod.DIM) < _luminance(themod.MUTED) < _luminance(themod.TEXT),
      f"DIM {themod.DIM} < MUTED {themod.MUTED} < TEXT {themod.TEXT}")

# A label colour set only on the button, not on the <p> Streamlit nests inside
# it, is the bug that started this. Assert both halves are present.
_css = _inspect.getsource(themod)
_app_txt = pathlib.Path("app.py").read_text()
check("the current book leaves Core / Satellite unstated",
      '"Core / Satellite": thmod.NO_DATA'.replace("thmod", "th") in _app_txt
      and _has_thai(thmod.NO_DATA),
      f"{thmod.NO_DATA} — the role is a design intent the existing book "
      f"never had")
check("the covariance estimator is fixed, not an RM-facing control",
      "Covariance Estimator" not in _app_txt
      and "COV_METHOD = risklib.COV_METHODS[0]" in _app_txt
      and '"cov"' not in _app_txt.split("REMEMBERED: List[str] = [")[1].split("]")[0],
      themod and risklib.COV_METHODS[0])
check("the lookback window is a typed number of years",
      'prefs.number_slider(\n        "ข้อมูลย้อนหลัง (ปี)", "lookback"' in _app_txt
      and 'prefs.remember("lookback", 5, cast=int)' in _app_txt)

check("button and tag labels set colour on the nested element",
      ".stButton > button p" in _css
      and '.stButton > button[kind="primary"] p' in _css
      and 'span[data-baseweb="tag"] span' in _css)


# --------------------------------------------------------------------------- #
section("Deployment compatibility")
print(f"  {compatmod.REPORT}")

check("offset aliases resolve for this pandas",
      compatmod.resample_alias("M") in ("M", "ME")
      and compatmod.resample_alias("Q") in ("Q", "QE")
      and compatmod.resample_alias("A") in ("A", "YE")
      and compatmod.period_alias("A") in ("A", "Y"))

check("writable() hands back an assignable array",
      _assignable_after_writable())

# Streamlit Community Cloud installs a pandas with Copy-on-Write, where
# Series.to_numpy() can return a read-only view. Writing to one raised
# "assignment destination is read-only" and took down the whole first tab.
# Force that behaviour here so a future in-place write is caught locally.
_real_to_numpy = pd.Series.to_numpy


def _readonly_to_numpy(self, *args, **kwargs):
    arr = _real_to_numpy(self, *args, **kwargs)
    try:
        arr = arr.view()
        arr.flags.writeable = False
    except (ValueError, AttributeError):
        pass
    return arr


pd.Series.to_numpy = _readonly_to_numpy
try:
    _cl = clients.get_client("C06")
    for _reb in ("Q", "M", "A", "D", "none"):
        engine.analyse(fd, _cl.holdings, UNI, WINDOW, None, _reb, 0.0175)
    _ok, _detail = True, "every rebalancing rule survives read-only arrays"
except Exception as exc:
    _ok, _detail = False, f"{type(exc).__name__}: {exc}"
finally:
    pd.Series.to_numpy = _real_to_numpy
check("analytics run when to_numpy() is read-only (Copy-on-Write)", _ok, _detail)

check("monthly return grid needs no matplotlib",
      "matplotlib" not in themod.style_returns_grid(
          metrics.monthly_return_table(analyses["C06"].returns)).to_html())

# --------------------------------------------------------------------------- #
print(f"\n{'=' * 74}")
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  · {f.strip()}")
    sys.exit(1)
print("All checks passed.")
