"""Monte Carlo engine for OFCI equipment availability.

For every open unit the simulated availability date is

    A = Compass forecast  +  re-date drift (fitted from Compass history)  +  risk-register events

following the PMI three-model structure (Barreras, 2011):
    Baseline        - forecast uncertainty only (re-date drift)
    Pre-mitigated   - + discrete risk events at their unmitigated probability/impact
    Post-mitigated  - + the same risk events after the planned mitigations
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .calibrate import POOLED, RedateModel
from .risks import Risk


@dataclass
class Scenario:
    name: str
    include_risks: bool
    mitigated: bool


SCENARIOS = [
    Scenario("Baseline", include_risks=False, mitigated=False),
    Scenario("Pre-mitigated", include_risks=True, mitigated=False),
    Scenario("Post-mitigated", include_risks=True, mitigated=True),
]


@dataclass
class OpenUnits:
    """Open (not yet received) units as aligned arrays; all dates are days after the data date."""
    frame: pd.DataFrame
    lead: np.ndarray          # Compass forecast, days after as-of (>= 7 for overdue units)
    roj: np.ndarray
    roj_complete: np.ndarray
    pick: np.ndarray
    compass_item: np.ndarray
    building: np.ndarray
    vendor: np.ndarray
    equipment: np.ndarray
    group: np.ndarray


def prepare_open_units(units: pd.DataFrame, item_map: Dict[str, str], as_of: pd.Timestamp) -> OpenUnits:
    u = units[units["is_open"]].copy().reset_index(drop=True)

    # Missing Compass dates: place the unit at its ROJ shifted by the equipment type's typical
    # forecast-vs-ROJ gap (or the type's median forecast if ROJ is also missing).
    gap = (u["avail_current"] - u["roj"]).dt.days
    med_gap = gap.groupby(u["equipment"]).median()
    med_date = u.groupby("equipment")["avail_current"].median()
    missing = u["avail_current"].isna()
    fill = u["roj"] + pd.to_timedelta(u["equipment"].map(med_gap), unit="D")
    fill = fill.fillna(u["equipment"].map(med_date))
    u["forecast"] = u["avail_current"].fillna(fill)
    u["forecast_imputed"] = missing

    lead = (u["forecast"] - as_of).dt.days.to_numpy(float)
    u["overdue_at_as_of"] = lead <= 0
    lead = np.maximum(lead, 7.0)   # due but not received: expect it within about a week
    u["forecast_days"] = lead

    def days(col):
        return (u[col] - as_of).dt.days.to_numpy(float)

    u["compass_item"] = u["equipment"].map(item_map).fillna(POOLED)
    return OpenUnits(
        frame=u, lead=lead, roj=days("roj"), roj_complete=days("roj_complete"), pick=days("pick"),
        compass_item=u["compass_item"].to_numpy(), building=u["bldg"].to_numpy(),
        vendor=u["vendor"].fillna("Unknown").to_numpy(), equipment=u["equipment"].to_numpy(),
        group=u["group_row"].to_numpy(),
    )


def simulate_redate(model: RedateModel, items: np.ndarray, bldgs: np.ndarray, lead: np.ndarray,
                    n: int, rng: np.random.Generator, window_weeks: Optional[int] = None):
    """Replay Compass re-date history over each unit's remaining lead time (block bootstrap).

    Each iteration chains random `block_weeks`-long stretches of a class's real weekly history
    (all buildings together). A unit is exposed for the weeks until its *current* Compass date
    (exposure does not grow as it slips - that feedback ran away for the most volatile classes).
    Same-vendor classes replay the same relative stretch of history with probability rho.

    Returns per-unit slip (days), event-week count and exposure weeks, each shaped (n, units).
    """
    combo = pd.DataFrame({"item": items, "bldg": bldgs, "lead": lead})
    keys = combo.drop_duplicates().reset_index(drop=True)
    unit_combo = combo.merge(keys.reset_index(), on=["item", "bldg", "lead"], how="left")["index"].to_numpy()

    cls = sorted(set(keys["item"]))
    params = [model.params(c) for c in cls]
    vendors = sorted({p.vendor or p.name for p in params})
    v_idx = np.array([vendors.index(p.vendor or p.name) for p in params])
    K, V, Lb = len(cls), len(vendors), model.block_weeks
    W = window_weeks if window_weeks is not None else int(math.ceil(lead.max() / 7))
    J = int(math.ceil(W / Lb))

    u_vendor = rng.random((n, V, J))[:, v_idx, :]
    same = rng.random((n, K, J)) < model.vendor_correlation
    u = np.where(same, u_vendor, rng.random((n, K, J)))
    u_pool = rng.random((n, K, J))
    own_draw = rng.random((n, K, J))
    week = np.arange(W)

    cum = np.empty((K, n, len(model.buildings), W), dtype=np.float32)
    cnt = np.empty_like(cum)
    for k, p in enumerate(params):
        pool = model.all_starts
        st = pool[(u_pool[:, k, :] * len(pool)).astype(int)]
        if len(p.starts):
            own = p.starts[(u[:, k, :] * len(p.starts)).astype(int)]
            st = np.where(own_draw[:, k, :] < p.pool_weight, own, st)
        idx = st[:, week // Lb] + week % Lb                              # (n, W) series columns
        path = model.series[:, idx].transpose(1, 0, 2)                    # (n, B, W)
        moved = path != 0
        cum[k] = np.cumsum(path, axis=2)
        cnt[k] = np.cumsum(moved, axis=2)

    exposure = np.minimum(np.ceil(keys["lead"].to_numpy() / 7).astype(int), W)
    k_of = np.array([cls.index(i) for i in keys["item"]])
    b_of = np.array([model.buildings.index(b) if b in model.buildings else 0 for b in keys["bldg"]])
    e_idx = np.maximum(exposure - 1, 0)
    combo_slip = np.where(exposure > 0, cum[k_of, :, b_of, e_idx].T, 0.0)
    combo_cnt = np.where(exposure > 0, cnt[k_of, :, b_of, e_idx].T, 0.0)
    combo_weeks = np.broadcast_to(exposure, (n, len(keys))).astype(float)
    return combo_slip[:, unit_combo], combo_cnt[:, unit_combo], combo_weeks[:, unit_combo]


def _lot_noise(model: RedateModel, events: np.ndarray, weeks: np.ndarray, lots: np.ndarray,
               rng: np.random.Generator):
    """Scatter around the class shift plus stand-alone re-dates, drawn once per schedule lot.

    History shows a lot's units move together: the gap between a lot's last unit and its median
    unit is small (days), while independent per-unit draws inflate it several-fold (weeks to
    months) and with it every "last unit" date. So the scatter is drawn once per lot, plus a small
    independent unit share (`unit_scatter_share`, fitted to the observed gap) so lots can spread.
    """
    keys, first, inv = np.unique(lots, return_index=True, return_inverse=True)
    k = events[:, first].astype(int)
    noise = np.zeros(k.shape, dtype=np.float32)
    for j in range(int(k.max()) if k.size else 0):
        draw = rng.choice(model.residuals, size=k.shape).astype(np.float32)
        noise += np.where(k > j, draw, 0.0)
    if model.idio_prob > 0 and len(model.idio_magnitudes):
        jn = rng.binomial(weeks[:, first].astype(int), model.idio_prob)
        for j in range(int(jn.max()) if jn.size else 0):
            draw = rng.choice(model.idio_magnitudes, size=jn.shape).astype(np.float32)
            noise += np.where(jn > j, draw, 0.0)
    noise = noise[:, inv]
    if model.unit_scatter_share > 0:
        ku = events.astype(int)
        for j in range(int(ku.max()) if ku.size else 0):
            draw = rng.choice(model.residuals, size=ku.shape).astype(np.float32)
            noise += model.unit_scatter_share * np.where(ku > j, draw, 0.0)
    return noise


def _apply_risks(A: np.ndarray, ou: OpenUnits, risks: List[Risk], mitigated: bool,
                 rng: np.random.Generator, as_of: pd.Timestamp, drivers: Dict[str, np.ndarray]):
    n, U = A.shape
    for r in risks:
        mask = r.applies(ou.equipment)
        if not mask.any():
            drivers[r.label] = np.zeros(n)
            continue
        cols = np.flatnonzero(mask)
        p = r.probability * (r.prob_mult_post if mitigated else 1.0)
        imp_mult = r.impact_mult_post if mitigated else 1.0
        added = np.zeros((n, len(cols)))
        if r.scope == "unit":
            pu = np.full((n, len(cols)), p)
            if r.seasonal_months:
                dates = np.datetime64(as_of.date()) + np.clip(A[:, cols], 0, 3000).astype("timedelta64[D]")
                months = dates.astype("datetime64[M]").astype(int) % 12 + 1
                in_season = np.isin(months, r.seasonal_months)
                pu = np.where(in_season, np.minimum(pu * r.seasonal_prob_mult, 1.0), pu)
            occ = rng.random((n, len(cols))) < pu
            added = np.where(occ, r.sample_impact(rng, (n, len(cols))) * imp_mult, 0.0)
        else:
            key = {"group": ou.group, "vendor": ou.vendor, "class": ou.equipment}[r.scope][cols]
            ents, ent_idx = np.unique(key, return_inverse=True)
            E = len(ents)
            impact = r.sample_impact(rng, (n, E)) * imp_mult
            if r.scope == "group":
                occ = rng.random((n, E)) < p
                added = np.where(occ[:, ent_idx], impact[:, ent_idx], 0.0)
            else:
                # Rate-based event: chance grows with how long the entity still has open units,
                # and only units not yet available at the onset date are delayed.
                horizon = np.zeros((n, E))
                np.maximum.at(horizon.T, ent_idx, A[:, cols].T)
                horizon = np.maximum(horizon, 1.0)
                p_e = 1.0 - (1.0 - p) ** (horizon / r.rate_basis_days)
                occ = rng.random((n, E)) < p_e
                onset = rng.random((n, E)) * horizon
                hit = occ[:, ent_idx] & (A[:, cols] > onset[:, ent_idx])
                added = np.where(hit, impact[:, ent_idx], 0.0)
        A[:, cols] += added
        drivers[r.label] = added.sum(axis=1)


def run_scenario(ou: OpenUnits, model: RedateModel, risks: List[Risk], scenario: Scenario,
                 settings: dict, as_of: pd.Timestamp, seed: int):
    """Returns availability days-after-as-of (iterations x units, int16) and tornado drivers."""
    n_total, chunk = settings["iterations"], settings["chunk_size"]
    # Common random numbers: every scenario sees the same re-date draws, so differences
    # between scenarios come from the risk events and mitigations only.
    rng_redate = np.random.default_rng(seed)
    rng_risk = np.random.default_rng(seed + 1)
    out = np.empty((n_total, len(ou.lead)), dtype=np.int16)
    drivers: Dict[str, List[np.ndarray]] = {}
    items = np.unique(ou.compass_item)
    for start in range(0, n_total, chunk):
        n = min(chunk, n_total - start)
        slip, events, weeks = simulate_redate(model, ou.compass_item, ou.building, ou.lead, n, rng_redate)
        slip = slip + _lot_noise(model, events, weeks, ou.group, rng_redate)
        A = np.maximum(ou.lead[None, :] + slip, 0.0)
        d: Dict[str, np.ndarray] = {}
        for it in items:
            m = ou.compass_item == it
            d[f"Compass re-date: {it}"] = (A[:, m] - ou.lead[m]).sum(axis=1)
        if scenario.include_risks:
            _apply_risks(A, ou, risks, scenario.mitigated, rng_risk, as_of, d)
        out[start:start + n] = np.clip(np.rint(A), -32000, 32000).astype(np.int16)
        for k, v in d.items():
            drivers.setdefault(k, []).append(v)
    return out, {k: np.concatenate(v) for k, v in drivers.items()}


def backtest(model: RedateModel, units: pd.DataFrame, item_map: Dict[str, str], settings: dict,
             seed: int) -> pd.DataFrame:
    """Holdout test: re-forecast the start-of-window Compass dates to the data date with a model
    fit only on history before the window, and score against what Compass says now (PIT)."""
    t0 = pd.Timestamp(settings["backtest_from"])
    as_of = pd.Timestamp(settings["as_of_resolved"])
    u = units.dropna(subset=["avail_0429", "avail_current"]).copy()
    u = u[u["avail_0429"] > t0]
    if u.empty:
        return pd.DataFrame()
    lead = (u["avail_0429"] - t0).dt.days.to_numpy(float)
    window = int(round((as_of - t0).days / 7))
    items = u["equipment"].map(item_map).fillna(POOLED).to_numpy()
    rng = np.random.default_rng(seed)
    n = 2000
    slip, events, weeks = simulate_redate(model, items, u["bldg"].to_numpy(), lead, n, rng,
                                          window_weeks=window)
    slip = slip + _lot_noise(model, events, weeks, u["group_row"].to_numpy(), rng)
    sim = np.maximum(lead[None, :] + slip, 0) - lead[None, :]
    obs = (u["avail_current"] - u["avail_0429"]).dt.days.to_numpy(float)
    below = (sim < obs[None, :]).mean(axis=0)
    equal = (sim == obs[None, :]).mean(axis=0)
    u["observed_slip_days"] = obs
    u["simulated_p10"] = np.percentile(sim, 10, axis=0)
    u["simulated_p50"] = np.percentile(sim, 50, axis=0)
    u["simulated_p90"] = np.percentile(sim, 90, axis=0)
    u["pit"] = below + 0.5 * equal
    u["inside_p10_p90"] = (obs >= u["simulated_p10"]) & (obs <= u["simulated_p90"])
    return u

