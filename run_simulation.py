#!/usr/bin/env python3
"""OFCI procurement Monte Carlo simulation.

    python run_simulation.py --workbook "Cx + OFCI Matrix - LIVE.xlsx"

See README.md for the method, inputs and outputs.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from ofci_mc import analysis, report
from ofci_mc.calibrate import fit_redate_model, matrix_window
from ofci_mc.data import load_project
from ofci_mc.engine import SCENARIOS, backtest, prepare_open_units, run_scenario
from ofci_mc.risks import load_risk_register

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workbook", required=True, help="Path to the Cx + OFCI Matrix workbook (.xlsx)")
    p.add_argument("--settings", default=os.path.join(HERE, "config", "settings.json"))
    p.add_argument("--risk-register", default=os.path.join(HERE, "config", "risk_register.csv"))
    p.add_argument("--equipment-map", default=os.path.join(HERE, "config", "equipment_map.csv"))
    p.add_argument("--out", default=os.path.join(HERE, "outputs"))
    p.add_argument("--as-of", help="Data date (YYYY-MM-DD); default reads the matrix 'last updated' stamp")
    p.add_argument("--iterations", type=int)
    p.add_argument("--seed", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    with open(args.settings) as fh:
        settings = json.load(fh)
    if args.as_of:
        settings["as_of"] = args.as_of
    if args.iterations:
        settings["iterations"] = args.iterations
    if args.seed is not None:
        settings["seed"] = args.seed
    seed = settings["seed"]
    pcts = settings["percentiles"]
    t0 = time.time()

    print(f"Loading {args.workbook}")
    proj = load_project(args.workbook, settings)
    as_of = proj.as_of
    settings["as_of_resolved"] = str(as_of.date())
    units = proj.units
    item_map = pd.read_csv(args.equipment_map).set_index("matrix_equipment")["compass_item"].to_dict()
    unmapped = sorted(set(units.loc[units["is_open"], "equipment"]) - set(item_map))
    if unmapped:
        print(f"  ! equipment without a Compass mapping (pooled parameters used): {unmapped}")
    item_vendor = (units.assign(item=units["equipment"].map(item_map)).dropna(subset=["item"])
                   .groupby("item")["vendor"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "")
                   .to_dict())
    print(f"  data date {as_of.date()}: {len(units):,} units, {int(units['is_open'].sum()):,} open, "
          f"{proj.snapshots['snapshot'].nunique()} Compass snapshots")

    print("Calibrating Compass re-date process")
    win_start = pd.Timestamp(settings["backtest_from"])
    win_weeks = int(round((as_of - win_start).days / 7))
    window = matrix_window(units, item_map, win_start, settings["max_valid_redate_days"])
    # Holdout: history before the window only, scored on what actually happened in the window.
    holdout = fit_redate_model(proj.snapshots, item_vendor, settings, history_end=win_start)
    bt = backtest(holdout, units, item_map, settings, seed)
    # Production model: snapshot history up to the window + the window itself (most recent regime).
    model = fit_redate_model(proj.snapshots, item_vendor, settings, history_end=win_start,
                             window=window, window_weeks=win_weeks)
    pooled = model.classes["__pooled__"]
    print(f"  {pooled.weeks_observed} class-weeks of history ({win_weeks}-week matrix window from "
          f"{win_start.date()}), {model.block_weeks}-week blocks, building hit prob "
          f"{model.building_hit_probability:.2f}, vendor correlation {model.vendor_correlation:.2f}")
    if not bt.empty:
        print(f"  holdout (fit before {win_start.date()}, scored at {as_of.date()}): P10-P90 coverage "
              f"{bt['inside_p10_p90'].mean():.0%} (target 80%), mean PIT {bt['pit'].mean():.2f} (target 0.50)")

    risks = load_risk_register(args.risk_register)
    ou = prepare_open_units(units, item_map, as_of)

    kpis, buildings, bm, missing, groups, equipment, unit_tables, torn = {}, [], {}, {}, {}, {}, {}, None
    for sc in SCENARIOS:
        ts = time.time()
        A, drivers = run_scenario(ou, model, risks, sc, settings, as_of, seed)
        kpis[sc.name] = analysis.project_kpis(ou, A, pcts)
        b, bkeys, Bm = analysis.building_table(ou, A, as_of, pcts)
        buildings.append(b.assign(scenario=sc.name))
        bm[sc.name] = (bkeys, Bm)
        has = np.isfinite(ou.roj)
        missing[sc.name] = (A[:, has] > ou.roj_complete[has][None, :]).sum(axis=1)
        groups[sc.name] = analysis.group_table(ou, A, as_of, pcts, settings)
        if sc.name != "Baseline":
            equipment[sc.name] = analysis.equipment_table(ou, A, as_of)
            unit_tables[sc.name] = analysis.unit_table(ou, A, as_of, pcts)
        if sc.name == "Pre-mitigated":
            torn = analysis.tornado(drivers, analysis.total_unit_days_late(ou, A))
        del A
        print(f"  {sc.name:<15} {settings['iterations']:,} iterations in {time.time() - ts:.1f}s")

    # ---- assemble decision tables (Post-mitigated is the plan of record, per PMI) --------------
    cp = settings["contingency_percentile"]
    post, pre, base = groups["Post-mitigated"], groups["Pre-mitigated"], groups["Baseline"]
    g = post.copy()
    g.insert(g.columns.get_loc(f"P{cp}_last_unit") + 1, f"P{cp}_pre_mitigation", pre[f"P{cp}_last_unit"].to_numpy())
    g.insert(g.columns.get_loc(f"P{cp}_pre_mitigation") + 1, f"P{cp}_baseline", base[f"P{cp}_last_unit"].to_numpy())
    g["mitigation_value_days"] = (pre[f"P{cp}_last_unit"] - post[f"P{cp}_last_unit"]).dt.days.to_numpy()
    g["p_all_by_roj_complete_pre_mitigation"] = pre["p_all_by_roj_complete"].to_numpy()
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "No ROJ": 4}
    g = g.sort_values(["risk_rating", "p_all_by_roj_complete", f"P{cp}_last_unit"],
                      key=lambda s: s.map(order) if s.name == "risk_rating" else s).drop(columns="group_row")

    kpi = pd.DataFrame(kpis).T.reset_index().rename(columns={"index": "scenario"})
    bld = pd.concat(buildings, ignore_index=True)
    bld = bld[["scenario"] + [c for c in bld.columns if c != "scenario"]]

    out_dir = report.ensure_dir(args.out)
    charts = {
        "Building readiness S-curves": os.path.join(out_dir, "building_s_curves.png"),
        "Units missing ROJ window": os.path.join(out_dir, "units_missing_roj_window.png"),
        "Tornado - risk drivers": os.path.join(out_dir, "tornado.png"),
        "Equipment x building heatmap": os.path.join(out_dir, "equipment_building_heatmap.png"),
        "Calibration backtest": os.path.join(out_dir, "backtest_calibration.png"),
    }
    report.chart_building_s_curves(bm, bld, as_of, charts["Building readiness S-curves"])
    report.chart_units_missing(missing, kpis["Baseline"]["compass_units_missing_roj_window"],
                               charts["Units missing ROJ window"])
    report.chart_tornado(torn, charts["Tornado - risk drivers"])
    report.chart_heatmap(equipment["Post-mitigated"], charts["Equipment x building heatmap"])
    if not bt.empty:
        report.chart_backtest(bt, charts["Calibration backtest"])
    else:
        charts.pop("Calibration backtest")

    bt_by_class = pd.DataFrame()
    if not bt.empty:
        bt_by_class = bt.groupby("equipment").agg(
            units=("pit", "size"), observed_mean_slip_days=("observed_slip_days", "mean"),
            simulated_median_slip_days=("simulated_p50", "mean"),
            simulated_p90_slip_days=("simulated_p90", "mean"),
            share_inside_p10_p90=("inside_p10_p90", "mean"), mean_pit=("pit", "mean")).reset_index()
        bt_by_class.loc[len(bt_by_class)] = ["ALL", len(bt), bt["observed_slip_days"].mean(),
                                             bt["simulated_p50"].mean(), bt["simulated_p90"].mean(),
                                             bt["inside_p10_p90"].mean(), bt["pit"].mean()]

    rr = pd.read_csv(args.risk_register)
    meta = pd.DataFrame([
        ("Data date (as-of)", str(as_of.date())),
        ("Source workbook", os.path.basename(args.workbook)),
        ("Iterations per scenario", f"{settings['iterations']:,}"),
        ("Random seed", seed),
        ("Open units simulated", f"{len(ou.lead):,}"),
        ("Open units without a Compass date (imputed)", int(ou.frame["forecast_imputed"].sum())),
        ("Open units overdue at data date (Compass date passed, not received)", int(ou.frame["overdue_at_as_of"].sum())),
        ("Open units without a contract ROJ", int((~np.isfinite(ou.roj)).sum())),
        ("Compass history used (class-weeks / building-week re-date events)",
         f"{pooled.weeks_observed} / {pooled.events}"),
        ("Block bootstrap length (weeks)", model.block_weeks),
        ("Unit scatter share (lots co-move; fitted to Apr-Aug last-unit vs median-unit gap)",
         model.unit_scatter_share),
        ("Building hit probability given a class re-date", round(model.building_hit_probability, 3)),
        ("Vendor correlation of re-date timing (Gaussian copula)", round(model.vendor_correlation, 3)),
        ("Recent matrix window appended to history", f"{win_start.date()} to {as_of.date()} ({win_weeks} weeks, "
                                                     f"{len(window):,} units)"),
        ("Holdout test P10-P90 coverage (target 80%)",
         f"{bt['inside_p10_p90'].mean():.1%}" if not bt.empty else "n/a"),
        ("Holdout test mean PIT (target 0.50; above = Compass dates slipped more than predicted)",
         f"{bt['pit'].mean():.3f}" if not bt.empty else "n/a"),
        ("Contingency definition", f"P{cp} (Post-mitigated) minus Compass deterministic date, per schedule group (PMI)"),
        ("Management reserve definition", f"P{settings['management_reserve_percentile']} minus Compass deterministic date"),
        ("Risk rating", "Critical >= 80% / High >= 50% / Medium >= 20% chance the last unit misses the ROJ "
                        "window (ROJ complete); Medium also if >= 50% chance of missing ROJ start"),
        ("Risk register values", "ASSUMPTIONS - review probabilities, impacts and mitigation factors with risk owners"),
        ("Run time (s)", round(time.time() - t0, 1)),
    ], columns=["item", "value"])

    sheets = {
        "Summary": kpi,
        "Buildings": bld,
        "Schedule Groups": g,
        "Equipment x Building": equipment["Post-mitigated"],
        "Units (Post-mitigated)": unit_tables["Post-mitigated"].sort_values("p_by_roj_complete"),
        "Tornado": torn,
        "Risk Register": rr,
        "Calibration": model.summary(),
        "Backtest": bt_by_class,
        "Run Info": meta,
    }
    xlsx = os.path.join(out_dir, f"OFCI_Monte_Carlo_{as_of:%Y%m%d}.xlsx")
    report.write_workbook(xlsx, sheets, charts)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"as_of": str(as_of.date()), "kpis": kpis,
                   "buildings": json.loads(bld.to_json(orient="records", date_format="iso")),
                   "tornado": json.loads(torn.head(15).to_json(orient="records"))}, fh, indent=2)

    print(f"\nWrote {xlsx}")
    print(kpi.set_index("scenario")[[
        "compass_units_missing_roj_window", "mean_units_missing_roj_window",
        f"P{cp}_units_missing_roj_window", f"P{cp}_unit_days_late"]].round(0).to_string())
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
