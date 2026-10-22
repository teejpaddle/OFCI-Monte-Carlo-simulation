"""Turn simulated availability dates into decision tables.

Need dates compared against each simulated availability date:
  ROJ           - contract Required-On-Job date (the date the matrix already tracks variance to)
  ROJ complete  - end of the ROJ install window; arriving after it means the window is missed
  Pick          - planned pick date from MEI, where the pick plan has one
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from .engine import OpenUnits


def _dates(as_of: pd.Timestamp, days) -> pd.Series:
    return pd.Series(as_of + pd.to_timedelta(np.asarray(days, float), unit="D"))


def _prob_by(A: np.ndarray, need: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        p = (A <= need[None, :]).mean(axis=0)
    return np.where(np.isnan(need), np.nan, p)


def _expected_late(A: np.ndarray, need: np.ndarray) -> np.ndarray:
    late = np.maximum(A - np.nan_to_num(need)[None, :], 0).mean(axis=0)
    return np.where(np.isnan(need), np.nan, late)


def rating(p_miss_window: float, p_late_roj: float, thresholds: Dict[str, float]) -> str:
    if np.isnan(p_miss_window):
        return "No ROJ"
    if p_miss_window >= thresholds["Critical"]:
        return "Critical"
    if p_miss_window >= thresholds["High"]:
        return "High"
    if p_miss_window >= thresholds["Medium"] or p_late_roj >= thresholds["High"]:
        return "Medium"
    return "Low"


def reduce_max(A: np.ndarray, keys: np.ndarray):
    """Per-iteration max of A over units sharing a key -> (unique keys, n x K array)."""
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    return sk[starts], np.maximum.reduceat(A[:, order], starts, axis=1), order, starts


def unit_table(ou: OpenUnits, A: np.ndarray, as_of, pcts: List[int]) -> pd.DataFrame:
    f = ou.frame
    t = f[["matrix_row", "sched", "bldg", "dh", "equipment", "vendor", "sub", "roj", "roj_complete",
           "pick", "forecast", "forecast_imputed", "overdue_at_as_of"]].copy()
    t.rename(columns={"forecast": "compass_forecast"}, inplace=True)
    Af = A.astype(np.float32)
    for p, v in zip(pcts, np.percentile(Af, pcts, axis=0)):
        t[f"P{p}_available"] = _dates(as_of, v).dt.normalize()
    t["p_by_roj"] = _prob_by(Af, ou.roj)
    t["p_by_roj_complete"] = _prob_by(Af, ou.roj_complete)
    t["p_by_pick"] = _prob_by(Af, ou.pick)
    t["expected_days_late_vs_roj"] = _expected_late(Af, ou.roj)
    t["deterministic_percentile"] = (Af <= ou.lead[None, :]).mean(axis=0)
    return t


def group_table(ou: OpenUnits, A: np.ndarray, as_of, pcts: List[int], settings: dict) -> pd.DataFrame:
    keys, G, order, starts = reduce_max(A.astype(np.float32), ou.group)
    f = ou.frame.iloc[order].iloc[starts]
    det = pd.Series(ou.lead).groupby(ou.group).max().reindex(keys).to_numpy()
    roj = (f["roj"] - as_of).dt.days.to_numpy(float)
    rojc = (f["roj_complete"] - as_of).dt.days.to_numpy(float)
    t = pd.DataFrame({
        "sched": f["sched"].to_numpy(), "bldg": f["bldg"].to_numpy(), "dh": f["dh"].to_numpy(),
        "equipment": f["equipment"].to_numpy(), "vendor": f["vendor"].to_numpy(), "sub": f["sub"].to_numpy(),
        "open_units": pd.Series(ou.group).value_counts().reindex(keys).to_numpy(),
        "roj": f["roj"].to_numpy(), "roj_complete": f["roj_complete"].to_numpy(),
        "compass_last_unit": _dates(as_of, det).to_numpy(),
    })
    pv = np.percentile(G, pcts, axis=0)
    for p, v in zip(pcts, pv):
        t[f"P{p}_last_unit"] = _dates(as_of, v).dt.normalize().to_numpy()
    t["p_all_by_roj"] = _prob_by(G, roj)
    t["p_all_by_roj_complete"] = _prob_by(G, rojc)
    t["deterministic_percentile"] = (G <= det[None, :]).mean(axis=0)
    cp, mr = settings["contingency_percentile"], settings["management_reserve_percentile"]
    t[f"contingency_days_P{cp}"] = np.percentile(G, cp, axis=0) - det
    t[f"reserve_days_P{mr}"] = np.percentile(G, mr, axis=0) - det
    t[f"float_to_roj_complete_at_P{cp}"] = rojc - np.percentile(G, cp, axis=0)
    t["risk_rating"] = [rating(1 - a, 1 - b, settings["risk_rating_thresholds"])
                        for a, b in zip(t["p_all_by_roj_complete"], t["p_all_by_roj"])]
    t["group_row"] = keys
    return t


def building_table(ou: OpenUnits, A: np.ndarray, as_of, pcts: List[int]):
    keys, Bm, _, _ = reduce_max(A.astype(np.float32), ou.building)
    rows = []
    for j, b in enumerate(keys):
        m = ou.building == b
        det = ou.lead[m].max()
        last_rojc = np.nanmax(ou.roj_complete[m]) if np.isfinite(ou.roj_complete[m]).any() else np.nan
        row = {"bldg": b, "open_units": int(m.sum()),
               "compass_all_available": _dates(as_of, [det])[0].normalize(),
               "latest_roj_complete": _dates(as_of, [last_rojc])[0].normalize() if np.isfinite(last_rojc) else pd.NaT}
        for p in pcts:
            row[f"P{p}_all_available"] = _dates(as_of, [np.percentile(Bm[:, j], p)])[0].normalize()
        q90 = np.percentile(A[:, m].astype(np.float32), 90, axis=1)   # date 90% of units are available
        row["compass_90pct_available"] = _dates(as_of, [np.percentile(ou.lead[m], 90)])[0].normalize()
        for p in (50, 80):
            row[f"P{p}_90pct_available"] = _dates(as_of, [np.percentile(q90, p)])[0].normalize()
        row["deterministic_percentile"] = float((Bm[:, j] <= det).mean())
        row["p_all_by_latest_roj_complete"] = float((Bm[:, j] <= last_rojc).mean()) if np.isfinite(last_rojc) else np.nan
        rows.append(row)
    return pd.DataFrame(rows), keys, Bm


def equipment_table(ou: OpenUnits, A: np.ndarray, as_of) -> pd.DataFrame:
    """Per equipment x building, including a criticality index: share of iterations in which
    this equipment type is the last OFCI item to become available in that building."""
    Af = A.astype(np.float32)
    combo = pd.Series(ou.equipment).astype(str) + "||" + pd.Series(ou.building).astype(str)
    ckeys, C, _, _ = reduce_max(Af, combo.to_numpy())
    bld = np.array([k.split("||")[1] for k in ckeys])
    crit = np.zeros(len(ckeys))
    for b in np.unique(bld):
        cols = np.flatnonzero(bld == b)
        winner = C[:, cols].argmax(axis=1)
        crit[cols] = np.bincount(winner, minlength=len(cols)) / len(winner)
    rows = []
    for j, k in enumerate(ckeys):
        eq, b = k.split("||")
        m = (ou.equipment == eq) & (ou.building == b)
        late = Af[:, m] > ou.roj[m][None, :]
        miss = Af[:, m] > ou.roj_complete[m][None, :]
        rows.append({
            "equipment": eq, "bldg": b, "vendor": ou.vendor[m][0], "open_units": int(m.sum()),
            "p_unit_by_roj": float(np.nanmean(_prob_by(Af[:, m], ou.roj[m]))) if np.isfinite(ou.roj[m]).any() else np.nan,
            "expected_units_late_vs_roj": float(late.sum(axis=1).mean()),
            "expected_units_missing_roj_window": float(miss.sum(axis=1).mean()),
            "compass_last_unit": _dates(as_of, [ou.lead[m].max()])[0].normalize(),
            "P50_last_unit": _dates(as_of, [np.percentile(C[:, j], 50)])[0].normalize(),
            "P80_last_unit": _dates(as_of, [np.percentile(C[:, j], 80)])[0].normalize(),
            "criticality_index": crit[j],
        })
    return pd.DataFrame(rows).sort_values(["bldg", "criticality_index"], ascending=[True, False])


def project_kpis(ou: OpenUnits, A: np.ndarray, pcts: List[int]) -> Dict[str, float]:
    Af = A.astype(np.float32)
    has_roj = np.isfinite(ou.roj)
    late = (Af[:, has_roj] > ou.roj[has_roj][None, :]).sum(axis=1)
    miss = (Af[:, has_roj] > ou.roj_complete[has_roj][None, :]).sum(axis=1)
    days = np.maximum(Af[:, has_roj] - ou.roj[has_roj][None, :], 0).sum(axis=1)
    det_late = int((ou.lead[has_roj] > ou.roj[has_roj]).sum())
    det_miss = int((ou.lead[has_roj] > ou.roj_complete[has_roj]).sum())
    out = {"open_units_with_roj": int(has_roj.sum()),
           "compass_units_late_vs_roj": det_late, "compass_units_missing_roj_window": det_miss,
           "compass_unit_days_late": float(np.maximum(ou.lead[has_roj] - ou.roj[has_roj], 0).sum())}
    for name, arr in (("units_late_vs_roj", late), ("units_missing_roj_window", miss), ("unit_days_late", days)):
        out[f"mean_{name}"] = float(arr.mean())
        for p in pcts:
            out[f"P{p}_{name}"] = float(np.percentile(arr, p))
    return out


def total_unit_days_late(ou: OpenUnits, A: np.ndarray) -> np.ndarray:
    has = np.isfinite(ou.roj)
    return np.maximum(A[:, has].astype(np.float32) - ou.roj[has][None, :], 0).sum(axis=1)


def tornado(drivers: Dict[str, np.ndarray], output: np.ndarray) -> pd.DataFrame:
    """Spearman rank correlation of each driver with the output (as in @Risk / Shinde 2020)."""
    out_rank = pd.Series(output).rank().to_numpy()
    rows = []
    for name, v in drivers.items():
        if np.std(v) == 0:
            continue
        rho = float(np.corrcoef(pd.Series(v).rank().to_numpy(), out_rank)[0, 1])
        rows.append({"driver": name, "spearman_rho": rho})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["contribution_to_variance"] = t["spearman_rho"] ** 2 / (t["spearman_rho"] ** 2).sum()
    t["mean_unit_days_added"] = [float(np.mean(drivers[d])) for d in t["driver"]]
    return t.sort_values("contribution_to_variance", ascending=False).reset_index(drop=True)
