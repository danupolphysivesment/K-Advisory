"""Portfolio construction under suitability constraints.

Every objective is solved subject to the *same* constraint set, so the
resulting portfolios are directly comparable and all of them are suitable for
the client by construction:

* long only, fully invested
* every fund's product risk level ≤ the client's risk profile
* look-through exposure to each of the four SEC buckets inside the band the
  suitability form prescribes for that investor type
* an optional per-fund cap and a cap on the satellite sleeve

Only SciPy is used. Convex objectives go to SLSQP from several starting points
so a bad start cannot silently produce a poor local optimum; minimum-CVaR is
solved exactly as a linear program (Rockafellar-Uryasev).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from .data import periods_per_year
from .risk import covariance, portfolio_volatility
from .universe import ALT, CASH, EQUITY, FIXED, SATELLITE, SEC_BUCKETS

# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
# Objective names stay in the industry's English. A Thai rendering of
# "Inverse Volatility" or "Minimum Drawdown" is longer, less precise, and not
# what anyone in a Thai investment committee actually says out loud.
MAX_SHARPE = "Max Sharpe Ratio"
MIN_VOL = "Min Volatility"
TARGET_RETURN = "Target Return"
TARGET_VOL = "Target Volatility"
MIN_DRAWDOWN = "Min Drawdown"
MIN_CVAR = "Min CVaR (95%)"
INVERSE_VOL = "Inverse Volatility"
RISK_PARITY = "Risk Parity"
MAX_DIVERSIFICATION = "Max Diversification"
MAX_SORTINO = "Max Sortino Ratio"
EQUAL_WEIGHT = "Equal Weight"

OBJECTIVES: List[str] = [
    MAX_SHARPE, MIN_VOL, TARGET_RETURN, TARGET_VOL, MIN_DRAWDOWN, MIN_CVAR,
    INVERSE_VOL, RISK_PARITY, MAX_DIVERSIFICATION, MAX_SORTINO, EQUAL_WEIGHT,
]

OBJECTIVE_NOTES: Dict[str, str] = {
    MAX_SHARPE: "ผลตอบแทนสูงสุดต่อหนึ่งหน่วยความผันผวน อ่อนไหวต่อค่าประมาณ"
                "ผลตอบแทนคาดหวังมาก จึงควรใช้ตัวประมาณแบบ shrunk หรือ equilibrium "
                "เว้นแต่มีมุมมองที่ชัดเจนจริง",
    MIN_VOL: "พอร์ตที่มีความแปรปรวนต่ำที่สุดภายใต้เกณฑ์ความเหมาะสม "
             "ไม่ต้องพยากรณ์ผลตอบแทนเลย จึงมักให้ผลดีเมื่อออกไปนอกช่วงข้อมูล",
    TARGET_RETURN: "วิธีที่ใช้ความผันผวนน้อยที่สุดเพื่อไปถึงผลตอบแทนที่ตั้งไว้ "
                   "เหมาะเมื่อลูกค้ามีตัวเลขในใจอยู่แล้ว",
    TARGET_VOL: "ผลตอบแทนคาดหวังสูงสุดที่ระดับความเสี่ยงของลูกค้าซื้อได้ "
                "เป็นภาพสะท้อนของผลตอบแทนเป้าหมาย",
    MIN_DRAWDOWN: "ลด Drawdown ที่เกิดขึ้นจริงในข้อมูล "
                  "เป็นปัญหาไม่คอนเวกซ์ จึงแก้จากหลายจุดเริ่มต้น "
                  "คำตอบสื่อสารตรงกับความกลัวการขาดทุนของลูกค้า",
    MIN_CVAR: "ลดค่าเฉลี่ยการขาดทุนในช่วงที่แย่ที่สุด 5% แก้เป็นโปรแกรมเชิงเส้น "
              "ได้คำตอบแม่นตรง มุ่งจัดการหางของการแจกแจง ไม่ใช่ความแปรปรวน"
              "แบบสมมาตร",
    INVERSE_VOL: "น้ำหนักแปรผันกับ 1/σ ไม่ต้องใช้ Optimiser และไม่มี"
                 "ความคลาดเคลื่อนจากการประมาณสหสัมพันธ์ เป็นเกณฑ์เปรียบเทียบ"
                 "ที่ทนทาน",
    RISK_PARITY: "ทุกกองทุนมีส่วนร่วมในความเสี่ยงของพอร์ตเท่ากัน "
                 "จึงไม่มีสถานะใดครอบงำผลลัพธ์",
    MAX_DIVERSIFICATION: "เพิ่มอัตราส่วนความผันผวนถ่วงน้ำหนักเฉลี่ยต่อความผันผวน"
                         "ของพอร์ตให้สูงสุด คือซื้อประโยชน์จากสหสัมพันธ์ให้มากที่สุด"
                         "เท่าที่ Fund Universe จะให้ได้",
    MAX_SORTINO: "คล้าย Sharpe สูงสุด แต่ลงโทษเฉพาะความเบี่ยงเบนด้านลบ "
                 "จึงไม่นับความผันผวนขาขึ้นเป็นความเสี่ยง",
    EQUAL_WEIGHT: "แบ่งเท่ากัน 1/N ทั่ว Fund Universe ที่ลงทุนได้ "
                  "เอาชนะได้ยากเมื่อออกนอกช่วงข้อมูล และเป็นไม้วัดที่ยุติธรรม"
                  "สำหรับทุกวิธีอื่น",
}

NEEDS_TARGET = {TARGET_RETURN, TARGET_VOL}
NEEDS_EXPECTED_RETURN = {MAX_SHARPE, TARGET_RETURN, TARGET_VOL, MAX_SORTINO}


# --------------------------------------------------------------------------- #
# Constraint set
# --------------------------------------------------------------------------- #
@dataclass
class Constraints:
    max_weight: float = 0.35
    min_weight: float = 0.0            # applied only to funds that get selected
    bands: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    max_satellite: float = 0.35
    max_funds: Optional[int] = 8
    min_position: float = 0.03         # positions below this are pruned
    rf: float = 0.0175
    above_level_budget: float = 0.0    # weight allowed in funds above the client's level
    above_level_codes: frozenset = field(default_factory=frozenset)

    def bucket_matrix(self, codes: Sequence[str], universe: Mapping) -> np.ndarray:
        """(4 × n) look-through loading of each fund onto the SEC buckets."""
        rows = []
        for bucket in SEC_BUCKETS:
            rows.append([universe[c].lookthrough.get(bucket, 0.0) if c in universe else 0.0
                         for c in codes])
        return np.array(rows, dtype=float)

    def satellite_vector(self, codes: Sequence[str], universe: Mapping) -> np.ndarray:
        return np.array(
            [1.0 if (c in universe and universe[c].role == SATELLITE) else 0.0 for c in codes],
            dtype=float,
        )

    def above_level_vector(self, codes: Sequence[str]) -> np.ndarray:
        return np.array([1.0 if c in self.above_level_codes else 0.0 for c in codes],
                        dtype=float)


# --------------------------------------------------------------------------- #
# Feasibility: the two SEC tables can contradict each other
# --------------------------------------------------------------------------- #
# Relaxing a band upward is not equally acceptable in every direction. Holding
# more cash than the form suggests is prudent; holding more equity or more
# alternatives than it allows is the thing the form exists to prevent. The
# repair LP therefore prices the slacks: cheap to over-hold cash, expensive to
# over-hold risk.
_RELAX_COST: Dict[str, float] = {CASH: 1.0, FIXED: 2.0, EQUITY: 25.0, ALT: 40.0}


@dataclass
class BandResolution:
    bands: Dict[str, Tuple[float, float]]
    feasible_as_printed: bool
    relaxations: List[str]
    reason: str = ""

    @property
    def adjusted(self) -> bool:
        return bool(self.relaxations)


def resolve_bands(
    codes: Sequence[str],
    universe: Mapping,
    bands: Mapping[str, Tuple[float, float]],
    max_weight: float = 1.0,
    above_level_codes: frozenset = frozenset(),
    above_level_budget: float = 0.0,
) -> BandResolution:
    """Check whether the suitability bands admit any portfolio, and repair them.

    A profile-7 client illustrates the problem: the allocation table allows up
    to 20% alternatives, but every alternative fund carries product risk level
    8, so a level-7 ceiling leaves the alternatives bucket unreachable. The
    remaining caps (10% cash + 40% fixed + 40% equity) sum to 90%, and no
    portfolio can be built. Rather than failing, this finds the smallest,
    cheapest relaxation that makes the set feasible and reports exactly what
    it changed so the RM can explain it.
    """
    n = len(codes)
    if n == 0:
        return BandResolution(dict(bands), False, [], "ไม่มีกองทุนที่ลงทุนได้")

    B = np.array(
        [[universe[c].lookthrough.get(b, 0.0) if c in universe else 0.0 for c in codes]
         for b in SEC_BUCKETS],
        dtype=float,
    )
    order = list(SEC_BUCKETS)
    n_slack = 2 * len(order)

    # Variables: [w (n), slack_up (4), slack_dn (4)]
    c_vec = np.zeros(n + n_slack)
    for i, bucket in enumerate(order):
        c_vec[n + i] = _RELAX_COST.get(bucket, 5.0)              # up
        c_vec[n + len(order) + i] = _RELAX_COST.get(bucket, 5.0)  # down

    rows, rhs = [], []
    for i, bucket in enumerate(order):
        lo, hi = bands.get(bucket, (0.0, 1.0))
        up = np.zeros(n + n_slack)
        up[:n] = B[i]
        up[n + i] = -1.0
        rows.append(up); rhs.append(hi)
        if lo > 0.0:
            dn = np.zeros(n + n_slack)
            dn[:n] = -B[i]
            dn[n + len(order) + i] = -1.0
            rows.append(dn); rhs.append(-lo)

    if above_level_codes and any(c in above_level_codes for c in codes):
        row = np.zeros(n + n_slack)
        row[:n] = [1.0 if c in above_level_codes else 0.0 for c in codes]
        rows.append(row); rhs.append(above_level_budget)

    A_eq = np.zeros((1, n + n_slack)); A_eq[0, :n] = 1.0
    bounds = [(0.0, max_weight)] * n + [(0.0, None)] * n_slack

    res = linprog(c_vec, A_ub=np.array(rows), b_ub=np.array(rhs),
                  A_eq=A_eq, b_eq=np.array([1.0]), bounds=bounds, method="highs")

    if not res.success:
        return BandResolution(dict(bands), False, [],
                              "ไม่มีพอร์ตใดผ่านเงื่อนไขเหล่านี้แม้จะผ่อนคลาย"
                              "กรอบสัดส่วนแล้ว กรุณาเพิ่มเพดานน้ำหนักต่อกองทุน "
                              "หรือขยาย Fund Universe ที่ลงทุนได้")

    slack_up = res.x[n:n + len(order)]
    slack_dn = res.x[n + len(order):]
    out: Dict[str, Tuple[float, float]] = {}
    notes: List[str] = []
    for i, bucket in enumerate(order):
        lo, hi = bands.get(bucket, (0.0, 1.0))
        new_hi, new_lo = hi, lo
        if slack_up[i] > 1e-6:
            new_hi = min(hi + slack_up[i] + 1e-9, 1.0)
            notes.append(f"ขยายเพดาน{bucket} จาก {hi:.0%} เป็น {new_hi:.0%}")
        if slack_dn[i] > 1e-6:
            new_lo = max(lo - slack_dn[i] - 1e-9, 0.0)
            notes.append(f"ลดขั้นต่ำ{bucket} จาก {lo:.0%} เป็น {new_lo:.0%}")
        out[bucket] = (new_lo, new_hi)

    reason = ""
    if notes:
        reachable = {b for i, b in enumerate(order) if B[i].max() > 0}
        missing = [b for b in order
                   if b not in reachable and bands.get(b, (0, 1))[1] > 0]
        if missing:
            reason = (f"ไม่มีกองทุนใดในระดับความเสี่ยงของลูกค้าที่ให้ความเสี่ยง"
                      f"ในกลุ่ม{', '.join(missing)} ทำให้กรอบสัดส่วนตามแบบฟอร์ม"
                      f"รวมกันไม่ถึง 100%")
        else:
            reason = ("เพดานตามแบบฟอร์มรวมกันไม่ถึง 100% สำหรับชุดกองทุน"
                      "ที่ลงทุนได้ ณ เพดานน้ำหนักต่อกองทุนปัจจุบัน")
    return BandResolution(out, not notes, notes, reason)


@dataclass
class Solution:
    weights: Dict[str, float]
    objective: str
    expected_return: float
    volatility: float
    sharpe: float
    status: str
    message: str
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# --------------------------------------------------------------------------- #
# Eligible universe
# --------------------------------------------------------------------------- #
def eligible_universe(
    returns: pd.DataFrame,
    universe: Mapping,
    risk_profile_level: int,
    min_history_periods: int = 250,
    exclude_rmf: bool = True,
    extra_exclude: Sequence[str] = (),
    include_always: Sequence[str] = (),
    require_since=None,
    allow_above_level: bool = False,
) -> List[str]:
    """Funds a client of this risk profile may hold, with enough history to model.

    ``require_since`` keeps only funds that were already trading on that date,
    which is what makes the estimation window honest: without it, adding one
    fund launched last year truncates the common history for every other fund
    down to a year and every covariance in the problem becomes noise.

    RMF share classes are excluded by default: they are tax-wrapper duplicates
    of funds already in the list and would otherwise crowd the optimiser with
    near-identical columns.

    ``allow_above_level`` admits products above the client's risk level. They
    are still capped by ``Constraints.above_level_budget`` and flagged in the
    UI as requiring a signed risk acknowledgement.
    """
    keep: List[str] = []
    forced = set(include_always)
    banned = set(extra_exclude)
    cutoff = pd.Timestamp(require_since) if require_since is not None else None
    for code in returns.columns:
        fund = universe.get(code)
        if fund is None or code in banned:
            continue
        if code in forced:
            keep.append(code)
            continue
        if fund.risk_level > risk_profile_level and not allow_above_level:
            continue
        if exclude_rmf and fund.is_rmf:
            continue
        if returns[code].notna().sum() < min_history_periods:
            continue
        if cutoff is not None:
            first = returns[code].first_valid_index()
            if first is None or first > cutoff:
                continue
            if returns[code].loc[cutoff:].isna().mean() > 0.05:
                continue
        keep.append(code)

    # Drop near-duplicate share classes of the same fund family: keep the one
    # with the longest history so the optimiser sees one column per strategy.
    by_name: Dict[str, str] = {}
    for code in keep:
        name = universe[code].name.replace(" (A)", "").replace(" (D)", "")
        current = by_name.get(name)
        if current is None or returns[code].notna().sum() > returns[current].notna().sum():
            by_name[name] = code
    deduped = sorted(set(by_name.values()) | (forced & set(keep)))
    return deduped


# --------------------------------------------------------------------------- #
# Core solvers
# --------------------------------------------------------------------------- #
def _scipy_constraints(codes, universe, cons: Constraints):
    out = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    B = cons.bucket_matrix(codes, universe)
    for i, bucket in enumerate(SEC_BUCKETS):
        lo, hi = cons.bands.get(bucket, (0.0, 1.0))
        row = B[i]
        if hi < 1.0:
            out.append({"type": "ineq", "fun": (lambda w, r=row, h=hi: float(h - r @ w))})
        if lo > 0.0:
            out.append({"type": "ineq", "fun": (lambda w, r=row, l=lo: float(r @ w - l))})
    sat = cons.satellite_vector(codes, universe)
    if cons.max_satellite < 1.0 and sat.any():
        out.append({"type": "ineq",
                    "fun": (lambda w, s=sat, h=cons.max_satellite: float(h - s @ w))})
    above = cons.above_level_vector(codes)
    if above.any():
        out.append({"type": "ineq",
                    "fun": (lambda w, a=above, h=cons.above_level_budget: float(h - a @ w))})
    return out


def _starting_points(n: int, cons: Constraints, B: np.ndarray) -> List[np.ndarray]:
    rng = np.random.default_rng(7)
    starts = [np.ones(n) / n]
    for _ in range(6):
        w = rng.dirichlet(np.ones(n) * 1.5)
        starts.append(np.clip(w, 0.0, cons.max_weight))
    return [s / s.sum() for s in starts]


def _solve_slsqp(
    n: int,
    codes: Sequence[str],
    universe: Mapping,
    cons: Constraints,
    objective: Callable[[np.ndarray], float],
    extra: Optional[List[dict]] = None,
    multistart: int = 5,
) -> Tuple[np.ndarray, bool, str]:
    bounds = [(0.0, cons.max_weight)] * n
    constraints = _scipy_constraints(codes, universe, cons) + (extra or [])
    B = cons.bucket_matrix(codes, universe)

    best_w, best_f, ok, msg = None, np.inf, False, "no feasible solution"
    for start in _starting_points(n, cons, B)[:multistart]:
        try:
            res = minimize(objective, start, method="SLSQP", bounds=bounds,
                           constraints=constraints,
                           options={"maxiter": 400, "ftol": 1e-10})
        except Exception as exc:  # pragma: no cover - solver blow-ups
            msg = str(exc)
            continue
        if not np.all(np.isfinite(res.x)):
            continue
        w = np.clip(res.x, 0.0, None)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        if not _feasible(w, codes, universe, cons):
            continue
        f = objective(w)
        if res.success and f < best_f:
            best_w, best_f, ok, msg = w, f, True, res.message
        elif best_w is None and f < best_f:
            best_w, best_f, msg = w, f, res.message

    if best_w is None:
        # Nothing feasible was found. Return the flag, not a plausible-looking
        # 1/N portfolio that would be presented to a client as a solution.
        return None, False, msg
    return best_w, ok, msg


def _feasible(w: np.ndarray, codes, universe, cons: Constraints, tol: float = 1e-4) -> bool:
    if w.min() < -tol or w.max() > cons.max_weight + tol:
        return False
    B = cons.bucket_matrix(codes, universe)
    exposure = B @ w
    for i, bucket in enumerate(SEC_BUCKETS):
        lo, hi = cons.bands.get(bucket, (0.0, 1.0))
        if exposure[i] > hi + tol or exposure[i] < lo - tol:
            return False
    sat = cons.satellite_vector(codes, universe)
    if sat.any() and sat @ w > cons.max_satellite + tol:
        return False
    above = cons.above_level_vector(codes)
    if above.any() and above @ w > cons.above_level_budget + tol:
        return False
    return True


def _min_cvar_lp(
    panel: pd.DataFrame,
    codes: Sequence[str],
    universe: Mapping,
    cons: Constraints,
    alpha: float = 0.95,
    min_return: Optional[float] = None,
    mu: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, str]:
    """Rockafellar-Uryasev minimum-CVaR as a linear program.

    Variables: [w (n), zeta (1), u (T)].
    Minimise  zeta + 1/((1-alpha) T) Σ u_t
    s.t.      u_t ≥ -r_t·w - zeta,  u_t ≥ 0
    """
    R = panel[list(codes)].to_numpy(dtype=float)
    T, n = R.shape
    n_vars = n + 1 + T

    c = np.zeros(n_vars)
    c[n] = 1.0
    c[n + 1:] = 1.0 / ((1.0 - alpha) * T)

    # -R w - zeta - u <= 0
    A_ub = np.zeros((T, n_vars))
    A_ub[:, :n] = -R
    A_ub[:, n] = -1.0
    A_ub[:, n + 1:] = -np.eye(T)
    b_ub = np.zeros(T)

    B = cons.bucket_matrix(codes, universe)
    rows, rhs = [], []
    for i, bucket in enumerate(SEC_BUCKETS):
        lo, hi = cons.bands.get(bucket, (0.0, 1.0))
        if hi < 1.0:
            row = np.zeros(n_vars); row[:n] = B[i]; rows.append(row); rhs.append(hi)
        if lo > 0.0:
            row = np.zeros(n_vars); row[:n] = -B[i]; rows.append(row); rhs.append(-lo)
    sat = cons.satellite_vector(codes, universe)
    if cons.max_satellite < 1.0 and sat.any():
        row = np.zeros(n_vars); row[:n] = sat; rows.append(row); rhs.append(cons.max_satellite)
    above = cons.above_level_vector(codes)
    if above.any():
        row = np.zeros(n_vars); row[:n] = above
        rows.append(row); rhs.append(cons.above_level_budget)
    if min_return is not None and mu is not None:
        row = np.zeros(n_vars); row[:n] = -mu; rows.append(row); rhs.append(-min_return)
    if rows:
        A_ub = np.vstack([A_ub, np.array(rows)])
        b_ub = np.concatenate([b_ub, np.array(rhs)])

    A_eq = np.zeros((1, n_vars)); A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, cons.max_weight)] * n + [(None, None)] + [(0.0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return None, False, res.message
    w = np.clip(res.x[:n], 0.0, None)
    return w / w.sum(), True, "optimal"


def _max_drawdown_of(w: np.ndarray, R: np.ndarray) -> float:
    port = R @ w
    curve = np.cumprod(1.0 + port)
    return float((curve / np.maximum.accumulate(curve) - 1.0).min())


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def optimise(
    returns: pd.DataFrame,
    codes: Sequence[str],
    universe: Mapping,
    objective: str,
    cons: Constraints,
    mu: Optional[pd.Series] = None,
    cov: Optional[pd.DataFrame] = None,
    target: Optional[float] = None,
    prune: bool = True,
) -> Solution:
    """Solve one objective over ``codes`` and return cleaned weights."""
    codes = [c for c in codes if c in returns.columns]
    if len(codes) < 2:
        return Solution({}, objective, 0.0, 0.0, 0.0, "error",
                        "ต้องมีกองทุนที่ลงทุนได้อย่างน้อย 2 กองทุน")

    panel = returns[codes].dropna(how="any")
    if len(panel) < 30:
        return Solution({}, objective, 0.0, 0.0, 0.0, "error",
                        "ข้อมูลย้อนหลังที่ทับซ้อนกันของกองทุนกลุ่มนี้ไม่เพียงพอ")

    ann = periods_per_year(panel.index)
    if cov is None:
        cov = covariance(panel)
    cov = cov.loc[codes, codes]
    Sigma = cov.to_numpy()

    if mu is None:
        mu = (1.0 + panel).prod() ** (ann / len(panel)) - 1.0
    mu_vec = mu.reindex(codes).fillna(0.0).to_numpy()

    n = len(codes)
    R = panel.to_numpy(dtype=float)
    rf = cons.rf

    def vol(w):
        return portfolio_volatility(w, Sigma)

    status, message = "ok", ""
    extra: List[dict] = []

    if objective == EQUAL_WEIGHT:
        w = np.ones(n) / n
        ok, message = (_feasible(w, codes, universe, cons),
                       "1/N (ยังไม่บังคับใช้เงื่อนไข)")
        if not ok:
            w, ok, message = _solve_slsqp(
                n, codes, universe, cons,
                lambda x: float(((x - 1.0 / n) ** 2).sum()))
            message = "ฉาย 1/N ลงบนเงื่อนไขความเหมาะสม"

    elif objective == INVERSE_VOL:
        sd = np.sqrt(np.diag(Sigma))
        raw = 1.0 / np.maximum(sd, 1e-9)
        w = raw / raw.sum()
        if not _feasible(w, codes, universe, cons):
            w, ok, message = _solve_slsqp(
                n, codes, universe, cons,
                lambda x, t=w: float(((x - t) ** 2).sum()))
            message = "ฉายน้ำหนักผกผันความผันผวนลงบนเงื่อนไขความเหมาะสม"
        else:
            message = "คำนวณจากสูตรปิด"

    elif objective == MIN_VOL:
        w, ok, message = _solve_slsqp(n, codes, universe, cons, lambda x: vol(x))
        status = "ok" if ok else "warning"

    elif objective == MAX_SHARPE:
        def neg_sharpe(x):
            v = vol(x)
            return -(float(x @ mu_vec) - rf) / v if v > 1e-12 else 1e6
        w, ok, message = _solve_slsqp(n, codes, universe, cons, neg_sharpe)
        status = "ok" if ok else "warning"

    elif objective == MAX_SORTINO:
        def neg_sortino(x):
            port = R @ x
            downside = port[port < 0]
            dd = float(np.sqrt((downside ** 2).mean()) * np.sqrt(ann)) if len(downside) else 0.0
            if dd <= 1e-12:
                return 1e6
            return -(float(x @ mu_vec) - rf) / dd
        w, ok, message = _solve_slsqp(n, codes, universe, cons, neg_sortino)
        status = "ok" if ok else "warning"

    elif objective == MAX_DIVERSIFICATION:
        sd = np.sqrt(np.diag(Sigma))
        def neg_dr(x):
            v = vol(x)
            return -(float(x @ sd) / v) if v > 1e-12 else 1e6
        w, ok, message = _solve_slsqp(n, codes, universe, cons, neg_dr)
        status = "ok" if ok else "warning"

    elif objective == RISK_PARITY:
        def erc(x):
            v = vol(x)
            if v <= 1e-12:
                return 1e6
            rc = x * (Sigma @ x) / v
            return float(((rc - rc.mean()) ** 2).sum()) * 1e4
        w, ok, message = _solve_slsqp(n, codes, universe, cons, erc, multistart=6)
        status = "ok" if ok else "warning"

    elif objective == TARGET_RETURN:
        if target is None:
            target = float(np.median(mu_vec))
        extra = [{"type": "ineq", "fun": (lambda x, t=target: float(x @ mu_vec - t))}]
        w, ok, message = _solve_slsqp(n, codes, universe, cons, lambda x: vol(x), extra)
        if not ok:
            # Target unreachable inside the bands — fall back to the maximum
            # attainable return and say so rather than returning nonsense.
            w2, _ok2, _msg2 = _solve_slsqp(n, codes, universe, cons,
                                           lambda x: -float(x @ mu_vec))
            if w2 is not None:
                w, status = w2, "warning"
                message = (f"ผลตอบแทน {target:.1%} ไม่สามารถทำได้ภายในกรอบสัดส่วน"
                           f"ความเหมาะสมของลูกค้ารายนี้ ค่าสูงสุดที่ทำได้คือ "
                           f"{float(w2 @ mu_vec):.1%} ซึ่งแสดงไว้ที่นี่")

    elif objective == TARGET_VOL:
        if target is None:
            target = float(np.sqrt(np.diag(Sigma)).mean())
        extra = [{"type": "ineq", "fun": (lambda x, t=target: float(t - vol(x)))}]
        w, ok, message = _solve_slsqp(n, codes, universe, cons,
                                      lambda x: -float(x @ mu_vec), extra)
        if not ok:
            w2, _ok2, _msg2 = _solve_slsqp(n, codes, universe, cons, lambda x: vol(x))
            if w2 is not None:
                w, status = w2, "warning"
                message = (f"เพดานความผันผวน {target:.1%} ไม่สามารถทำได้ภายใน"
                           f"กรอบสัดส่วน จึงแสดงพอร์ตที่ความผันผวนต่ำสุด "
                           f"({vol(w2):.1%}) แทน")

    elif objective == MIN_CVAR:
        w, ok, message = _min_cvar_lp(panel, codes, universe, cons)
        status = "ok" if ok else "warning"

    elif objective == MIN_DRAWDOWN:
        w, ok, message = _solve_slsqp(
            n, codes, universe, cons,
            lambda x: -_max_drawdown_of(x, R), multistart=7)
        status = "ok" if ok else "warning"
        if w is not None:
            message = f"Max Drawdown ที่เกิดขึ้นจริงในข้อมูล: {_max_drawdown_of(w, R):.1%}"

    else:
        return Solution({}, objective, 0.0, 0.0, 0.0, "error",
                        f"ไม่รู้จักวัตถุประสงค์ {objective!r}")

    # ---- no feasible point: say so instead of returning a plausible fake ---
    if w is None or not np.all(np.isfinite(w)):
        return Solution(
            {}, objective, 0.0, 0.0, 0.0, "error",
            "ไม่มีพอร์ตใดที่ผ่านเงื่อนไขความเหมาะสมของลูกค้ารายนี้ "
            "กรุณาเพิ่มเพดานน้ำหนักต่อกองทุน ขยาย Fund Universe ที่ลงทุนได้ "
            "หรืออนุญาตวงเงินสำหรับผลิตภัณฑ์ที่เสี่ยงสูงกว่าระดับของลูกค้า",
        )

    # ---- clean up: shrink to a portfolio an RM can actually present --------
    if prune:
        w = _prune(w, cons, codes, universe)
        shortlisted = _shrink_holdings(returns, codes, universe, objective, cons,
                                       mu, cov, target, w)
        if shortlisted is not None:
            w, codes = shortlisted
            Sigma = cov.loc[codes, codes].to_numpy()
            mu_vec = mu.reindex(codes).fillna(0.0).to_numpy()
            R = returns[codes].dropna(how="any").to_numpy(dtype=float)

            def vol(x, _S=Sigma):  # noqa: F811 - rebind to the reduced set
                return portfolio_volatility(x, _S)

        # Drop residual dust so the presented allocation does not carry
        # positions that round to 0.0% on the client's statement.
        w = _prune(w, Constraints(**{**cons.__dict__, "min_position": 0.005}),
                   codes, universe)

    weights = {c: float(x) for c, x in zip(codes, w) if x > 1e-6}
    exp_ret = float(w @ mu_vec)
    v = vol(w)
    sharpe = (exp_ret - rf) / v if v > 0 else 0.0

    diagnostics = {
        "n_positions": len(weights),
        "history_start": panel.index[0],
        "history_end": panel.index[-1],
        "history_periods": len(panel),
        "eligible_funds": n,
        "feasible": _feasible(w, codes, universe, cons),
        "max_drawdown_sample": _max_drawdown_of(w, R),
    }
    return Solution(weights, objective, exp_ret, v, sharpe, status, message, diagnostics)


def _bucket_seed_counts(cons: Constraints) -> Dict[str, int]:
    """How many funds per bucket a shortlist needs before 100% is reachable.

    A per-fund cap turns the allocation bands into a capacity problem. With a
    35% single-fund limit and profile 7's caps (10 / 40 / 40 / 20), one fund
    per bucket tops out at 10 + 35 + 35 + 20 = 100% only if all four buckets
    are present — drop the cash fund and the most any three-fund shortlist can
    invest is 90%, so no portfolio exists. This walks bucket capacity up until
    the shortlist can actually be fully invested.
    """
    caps = {b: cons.bands.get(b, (0.0, 1.0))[1] for b in SEC_BUCKETS}
    floors = {b: cons.bands.get(b, (0.0, 1.0))[0] for b in SEC_BUCKETS}
    cap_w = max(cons.max_weight, 1e-6)

    counts = {b: (int(np.ceil(floors[b] / cap_w)) if floors[b] > 0 else 0)
              for b in SEC_BUCKETS}

    def capacity(bucket: str) -> float:
        return min(caps[bucket], counts[bucket] * cap_w)

    for _ in range(64):
        if sum(capacity(b) for b in SEC_BUCKETS) >= 1.0 - 1e-9:
            break
        headroom = {b: caps[b] - capacity(b) for b in SEC_BUCKETS}
        best = max(SEC_BUCKETS, key=lambda b: headroom[b])
        if headroom[best] <= 1e-9:
            break
        counts[best] += 1
    return counts


def _shrink_holdings(
    returns: pd.DataFrame,
    codes: Sequence[str],
    universe: Mapping,
    objective: str,
    cons: Constraints,
    mu: pd.Series,
    cov: pd.DataFrame,
    target: Optional[float],
    w: np.ndarray,
):
    """Re-solve on the highest-conviction funds so the answer is presentable.

    Simply zeroing the small positions and re-normalising would push the
    portfolio outside the suitability bands, which is why the naive prune keeps
    getting rejected. Re-running the *same* objective on the shortlisted funds
    instead produces a genuinely optimal portfolio of that size. The shortlist
    grows until a feasible solution exists, so this can never make the result
    worse than not shrinking at all.
    """
    if cons.max_funds is None:
        return None
    held = int((w > 1e-6).sum())
    if held <= cons.max_funds:
        return None

    ranking = list(np.argsort(w)[::-1])

    # Taking the top k by weight alone tends to produce a shortlist that spans
    # only one or two buckets — an inverse-volatility solution's largest
    # positions are all bond funds — and no portfolio of those funds can satisfy
    # an allocation band that requires equity. Seed the shortlist with the best
    # representative of each bucket first so a k-fund answer stays reachable.
    B = cons.bucket_matrix(codes, universe)
    counts = _bucket_seed_counts(cons)
    rank_of = {idx: pos for pos, idx in enumerate(ranking)}
    seeds: List[int] = []
    for b_idx, bucket in enumerate(SEC_BUCKETS):
        needed = counts.get(bucket, 0)
        if needed <= 0:
            continue
        # Prefer core funds: seeding a 60% equity floor with satellites alone
        # would blow the satellite cap and make the shortlist unsolvable.
        candidates = sorted(
            (i for i in ranking if B[b_idx, i] > 0.5),
            key=lambda i: (universe[codes[i]].role == SATELLITE, rank_of[i]))
        for idx in candidates[:needed]:
            if idx not in seeds:
                seeds.append(idx)
    ordered = seeds + [i for i in ranking if i not in seeds]
    sub_cons = Constraints(**{**cons.__dict__, "max_funds": None})

    for k in range(cons.max_funds, min(held, len(codes)) + 1):
        shortlist = [codes[i] for i in ordered[:k]]
        sub = optimise(returns, shortlist, universe, objective, sub_cons,
                       mu.reindex(shortlist), cov.loc[shortlist, shortlist],
                       target=target, prune=False)
        if sub.weights and sub.status != "error" and sub.diagnostics.get("feasible"):
            vec = np.array([sub.weights.get(c, 0.0) for c in shortlist])
            if vec.sum() > 0:
                return vec / vec.sum(), shortlist

        # The diffuse objectives — equal weight, inverse volatility, risk
        # parity — spread across the whole universe by design, so their own
        # solver can fail on a short list. Projecting the full solution onto
        # the shortlist is a well-conditioned problem that does not, and keeps
        # the answer as close to the intended method as the bands allow.
        target_w = w[[i for i in ordered[:k]]]
        if target_w.sum() > 0:
            target_w = target_w / target_w.sum()
            projected, ok, _ = _solve_slsqp(
                k, shortlist, universe, sub_cons,
                lambda x, t=target_w: float(((x - t) ** 2).sum()))
            if projected is not None and _feasible(projected, shortlist,
                                                   universe, sub_cons):
                return projected, shortlist
    return None


def _prune(w: np.ndarray, cons: Constraints, codes, universe) -> np.ndarray:
    """Zero out dust positions, keeping the result inside the bands.

    The holding *count* is handled by :func:`_shrink_holdings`, which re-solves
    rather than re-normalising.
    """
    trial = w.copy()
    trial[trial < cons.min_position] = 0.0
    if trial.sum() <= 0:
        return w
    trial = trial / trial.sum()
    trial = np.minimum(trial, cons.max_weight)
    if trial.sum() <= 0:
        return w
    trial = trial / trial.sum()
    # Only accept the pruned version if it is still suitable.
    return trial if _feasible(trial, codes, universe, cons) else w


# --------------------------------------------------------------------------- #
# Efficient frontier and comparison helpers
# --------------------------------------------------------------------------- #
def efficient_frontier(
    returns: pd.DataFrame,
    codes: Sequence[str],
    universe: Mapping,
    cons: Constraints,
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 18,
) -> pd.DataFrame:
    """Volatility-minimising portfolios across the attainable return range."""
    lo_sol = optimise(returns, codes, universe, MIN_VOL, cons, mu, cov, prune=False)
    hi_sol = optimise(returns, codes, universe, TARGET_RETURN, cons, mu, cov,
                      target=float(mu.reindex(codes).max()), prune=False)
    if not lo_sol.weights or not hi_sol.weights:
        return pd.DataFrame()

    lo, hi = lo_sol.expected_return, hi_sol.expected_return
    if hi <= lo:
        return pd.DataFrame([{"volatility": lo_sol.volatility,
                              "expected_return": lo, "sharpe": lo_sol.sharpe}])

    rows = []
    for t in np.linspace(lo, hi, n_points):
        sol = optimise(returns, codes, universe, TARGET_RETURN, cons, mu, cov,
                       target=float(t), prune=False)
        if sol.weights and sol.status != "error":
            rows.append({"volatility": sol.volatility,
                         "expected_return": sol.expected_return,
                         "sharpe": sol.sharpe})
    frame = pd.DataFrame(rows).drop_duplicates().sort_values("volatility")
    # Keep only the efficient upper edge.
    return frame[frame["expected_return"].cummax() == frame["expected_return"]]


def core_satellite_split(weights: Mapping[str, float], universe: Mapping) -> Dict[str, float]:
    core = sum(w for c, w in weights.items()
               if c in universe and universe[c].role != SATELLITE)
    sat = sum(w for c, w in weights.items()
              if c in universe and universe[c].role == SATELLITE)
    return {"Core": core, "Satellite": sat}


def turnover(current: Mapping[str, float], proposed: Mapping[str, float]) -> float:
    """One-way turnover needed to move from one portfolio to the other.

    Bounded by construction at 1.0 (sell everything, buy everything else);
    clamped because summing floats can put two fully disjoint portfolios a
    rounding error past it and "101% turnover" is not a thing.
    """
    codes = set(current) | set(proposed)
    raw = 0.5 * sum(abs(proposed.get(c, 0.0) - current.get(c, 0.0)) for c in codes)
    return float(min(max(raw, 0.0), 1.0))


def trade_list(current: Mapping[str, float], proposed: Mapping[str, float],
               universe: Mapping, aum: float) -> pd.DataFrame:
    codes = sorted(set(current) | set(proposed))
    rows = []
    for c in codes:
        now = current.get(c, 0.0)
        new = proposed.get(c, 0.0)
        delta = new - now
        if abs(delta) < 1e-6:
            continue
        fund = universe.get(c)
        rows.append({
            "กองทุน": c,
            "ชื่อกองทุน": fund.name if fund else c,
            "Asset Class": fund.asset_class if fund else "",
            "ปัจจุบัน": now,
            "เสนอใหม่": new,
            "เปลี่ยนแปลง": delta,
            "รายการ": "ซื้อ" if delta > 0 else "ขาย",
            "มูลค่า (บาท)": delta * aum,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.reindex(frame["เปลี่ยนแปลง"].abs().sort_values(ascending=False).index)
