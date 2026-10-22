"""Fit the Compass "re-date" process from history.

What the history shows (see README, Calibration):
  * Compass re-dates an equipment class in lumps: in a given week a class/building is either
    re-dated almost entirely or not at all, and most buildings move together.
  * Big moves are often followed by partial reversals (a large pull-in reversed by a push-out a
    week later), so weekly moves are *not* independent.
  * Behaviour shifts over time (regimes): a model fit only on older snapshots was too optimistic
    for the following window (see the holdout test), so the most recent window is part of history.

History = weekly Compass snapshots (up to `history_end`) + the matrix's own start-of-window
availability column -> "Current Availability" window, appended as one net move per class/building.
The model keeps each class's real week-by-week history (one series per building) and replays it
with a moving-block bootstrap: random `block_weeks`-long stretches, all buildings together,
chained to cover each unit's remaining lead time. Classes with little history borrow blocks from
the pooled history of all classes (empirical Bayes).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

POOLED = "__pooled__"


@dataclass
class ClassParams:
    name: str
    weeks_observed: int
    events: int                # building-weeks with a weekly re-date event in the snapshots
    starts: np.ndarray         # valid block start columns in RedateModel.series
    pool_weight: float         # P(a block comes from this class rather than the pooled history)
    vendor: str = ""
    event_shifts: np.ndarray = field(default_factory=lambda: np.array([]))
    window_net_shift: float = np.nan   # median net re-date over the recent matrix window


@dataclass
class RedateModel:
    buildings: List[str]
    series: np.ndarray               # (buildings, total weeks) weekly class shifts, classes concatenated
    block_weeks: int
    classes: Dict[str, ClassParams]
    all_starts: np.ndarray
    residuals: np.ndarray            # unit scatter around an event shift (days)
    idio_prob: float                 # P(unit re-dated on its own in a given week)
    idio_magnitudes: np.ndarray
    vendor_correlation: float
    building_hit_probability: float
    window_weeks: int = 0
    unit_scatter_share: float = 0.0  # independent unit share of event scatter (lots mostly co-move)
    events: pd.DataFrame = field(repr=False, default=None)

    def params(self, name: str) -> ClassParams:
        return self.classes.get(name, self.classes[POOLED])

    def summary(self) -> pd.DataFrame:
        rows = []
        for c in self.classes.values():
            has = len(c.event_shifts) > 0
            rows.append({
                "compass_item": c.name, "vendor": c.vendor, "weeks_of_history": c.weeks_observed,
                "snapshot_building_week_events": c.events,
                "weekly_event_shift_p10_days": np.percentile(c.event_shifts, 10) if has else np.nan,
                "weekly_event_shift_median_days": np.median(c.event_shifts) if has else np.nan,
                "weekly_event_shift_p90_days": np.percentile(c.event_shifts, 90) if has else np.nan,
                f"net_redate_last_{self.window_weeks}wk_median_days": c.window_net_shift,
                "share_of_blocks_from_own_history": round(c.pool_weight, 3),
            })
        return pd.DataFrame(rows).sort_values("compass_item")


def revision_pairs(snaps: pd.DataFrame, max_valid_days: int) -> pd.DataFrame:
    """Match each unit between consecutive snapshots (same item, job tag and date rank)."""
    snaps = snaps.sort_values(["item", "tag", "date"]).copy()
    snaps["k"] = snaps.groupby(["snapshot", "item", "tag"]).cumcount()
    snaps["n"] = snaps.groupby(["snapshot", "item", "tag"])["date"].transform("size")
    dates = sorted(snaps["snapshot"].unique())
    out = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        a = snaps[snaps["snapshot"] == d0]
        b = snaps[snaps["snapshot"] == d1]
        m = a.merge(b, on=["item", "tag", "k"], suffixes=("0", "1"))
        m = m[m["n0"] == m["n1"]]   # only compare lots whose unit count did not change
        m = m.assign(
            week=d0,
            delta=(m["date1"] - m["date0"]).dt.days,
            lead=(m["date0"] - d0).dt.days,
            bldg=m["bldg0"],
        )
        out.append(m[["item", "tag", "bldg", "week", "lead", "delta"]])
    pairs = pd.concat(out, ignore_index=True)
    # Only units not yet available can be re-dated; drop data-entry typos (e.g. 100-year jumps).
    return pairs[(pairs["lead"] > 0) & (pairs["delta"].abs() <= max_valid_days)]


def matrix_window(units: pd.DataFrame, item_map: Dict[str, str], start: pd.Timestamp,
                  max_valid_days: int) -> pd.DataFrame:
    """Per-unit net re-date between the matrix's start-of-window and current Compass dates.

    Only units still in the future at the window start are used - the same exposure rule the
    simulation applies. Units Compass listed as already available and later pushed back out are a
    one-off reclassification - flag them for follow-up rather than replaying them as drift.
    """
    u = units.dropna(subset=["avail_0429", "avail_current"])
    u = u[u["avail_0429"] > start].copy()
    u["item"] = u["equipment"].map(item_map)
    u["slip"] = (u["avail_current"] - u["avail_0429"]).dt.days
    return u[u["item"].notna() & (u["slip"].abs() <= max_valid_days)][["item", "bldg", "slip"]]


def _fill_buildings(wide: pd.DataFrame) -> pd.DataFrame:
    # A building not observed in a week (lot not yet listed, count changed) moves with the
    # buildings that were observed that week.
    return wide.apply(lambda c: c.fillna(c.mean()), axis=0).fillna(0.0)


def fit_redate_model(snaps: pd.DataFrame, item_vendor: Dict[str, str], settings: dict,
                     history_end: Optional[pd.Timestamp] = None,
                     window: Optional[pd.DataFrame] = None, window_weeks: int = 0) -> RedateModel:
    pairs = revision_pairs(snaps, settings["max_valid_redate_days"])
    if history_end is not None:
        pairs = pairs[pairs["week"] < history_end]
    pairs["revised"] = pairs["delta"] != 0
    thr = settings["event_fraction_threshold"]
    Lb = settings["block_weeks"]

    bw = pairs.groupby(["item", "bldg", "week"]).agg(
        frac=("revised", "mean"),
        shift=("delta", lambda s: s[s != 0].median() if (s != 0).any() else 0.0),
    ).reset_index()
    bw["event"] = bw["frac"] >= thr
    bw["applied"] = np.where(bw["event"], bw["shift"], 0.0)
    events = bw[bw["event"]]

    in_event = pairs.merge(events[["item", "bldg", "week", "shift"]], on=["item", "bldg", "week"])
    residuals = (in_event["delta"] - in_event["shift"])[in_event["revised"]].to_numpy(float)
    not_event = pairs.merge(bw[~bw["event"]][["item", "bldg", "week"]], on=["item", "bldg", "week"])
    idio_prob = float(not_event["revised"].mean()) if len(not_event) else 0.0
    idio_mags = not_event.loc[not_event["revised"], "delta"].to_numpy(float)

    cw = bw.groupby(["item", "week"]).agg(any_event=("event", "any"), hit=("event", "mean"),
                                         n_bldg=("event", "size")).reset_index()
    multi = cw[cw["any_event"] & (cw["n_bldg"] > 1)]
    h_global = float(multi["hit"].mean()) if len(multi) else 0.75

    buildings = sorted(set(snaps["bldg"].dropna()) | {f"B{i}" for i in range(1, 6)})
    win = None
    if window is not None and len(window) and window_weeks > 0:
        win = window.groupby(["item", "bldg"])["slip"].median().unstack("bldg").reindex(columns=buildings)
    s_q = settings["prior_weeks_strength"]

    blocks, classes, col = [], {}, 0
    items = sorted(set(bw["item"]) | set(item_vendor) | (set(win.index) if win is not None else set()))
    for item in items:
        parts = []
        sub = bw[bw["item"] == item]
        if not sub.empty:
            wide = sub.pivot(index="bldg", columns="week", values="applied").reindex(buildings)
            parts.append(_fill_buildings(wide).to_numpy(float))
        net = np.nan
        if win is not None and item in win.index:
            row = win.loc[item]
            net = float(row.median())
            lump = row.fillna(row.mean()).fillna(0.0).to_numpy(float)
            seg = np.zeros((len(buildings), window_weeks))
            seg[:, window_weeks // 2] = lump        # timing inside the window is unknown: one lump
            parts.append(seg)
        shifts = events.loc[events["item"] == item, "shift"].to_numpy(float)
        T = sum(p.shape[1] for p in parts)
        n_starts = max(T - Lb + 1, 0)
        if T:
            blocks.append(np.concatenate(parts, axis=1))
        classes[item] = ClassParams(
            name=item, weeks_observed=T, events=len(shifts), starts=np.arange(col, col + n_starts),
            pool_weight=(T / (T + s_q)) if n_starts else 0.0,
            vendor=item_vendor.get(item, ""), event_shifts=shifts, window_net_shift=net,
        )
        col += T
    series = np.concatenate(blocks, axis=1)
    all_starts = np.concatenate([c.starts for c in classes.values()])
    classes[POOLED] = ClassParams(POOLED, int(sum(c.weeks_observed for c in classes.values())),
                                  len(events), all_starts, 0.0, "", events["shift"].to_numpy(float))

    rho = settings.get("vendor_event_correlation")
    if rho is None:
        rho = _estimate_vendor_correlation(cw, item_vendor)

    return RedateModel(buildings=buildings, series=series, block_weeks=Lb, classes=classes,
                       all_starts=all_starts, residuals=residuals, idio_prob=idio_prob,
                       idio_magnitudes=idio_mags, vendor_correlation=float(rho),
                       building_hit_probability=h_global, window_weeks=window_weeks,
                       unit_scatter_share=float(settings.get("unit_scatter_share", 0.0)), events=events)


def _estimate_vendor_correlation(cw: pd.DataFrame, item_vendor: Dict[str, str]) -> float:
    """Average phi coefficient of weekly event indicators between classes from the same vendor."""
    wide = cw.pivot(index="week", columns="item", values="any_event")
    phis: List[float] = []
    items = list(wide.columns)
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if not item_vendor.get(a) or item_vendor.get(a) != item_vendor.get(b):
                continue
            both = wide[[a, b]].dropna().astype(float)
            if len(both) >= 4 and both[a].std() > 0 and both[b].std() > 0:
                phis.append(float(np.corrcoef(both[a], both[b])[0, 1]))
    if not phis:
        return 0.3   # documented fallback when the history cannot support an estimate
    return float(np.clip(np.mean(phis), 0.0, 0.8))
