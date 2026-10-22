import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ofci_mc.analysis import reduce_max, tornado  # noqa: E402
from ofci_mc.calibrate import POOLED, ClassParams, RedateModel  # noqa: E402
from ofci_mc.data import MATRIX_COLUMNS, load_matrix  # noqa: E402
from ofci_mc.engine import _apply_risks, OpenUnits, simulate_redate  # noqa: E402
from ofci_mc.risks import Risk, load_risk_register  # noqa: E402

BUILDINGS = ["B1", "B2", "B3", "B4", "B5"]


def make_model(weekly_shift: float, weeks: int = 12, block: int = 4) -> RedateModel:
    series = np.full((len(BUILDINGS), weeks), weekly_shift)
    starts = np.arange(weeks - block + 1)
    cls = ClassParams("X", weeks, 0, starts, pool_weight=1.0)
    return RedateModel(buildings=BUILDINGS, series=series, block_weeks=block,
                       classes={"X": cls, POOLED: ClassParams(POOLED, weeks, 0, starts, 0.0)},
                       all_starts=starts, residuals=np.zeros(1), idio_prob=0.0,
                       idio_magnitudes=np.zeros(0), vendor_correlation=0.0, building_hit_probability=1.0)


def test_redate_zero_history_gives_zero_slip():
    m = make_model(0.0)
    slip, cnt, weeks = simulate_redate(m, np.array(["X", "X"]), np.array(["B1", "B2"]),
                                       np.array([30.0, 90.0]), 50, np.random.default_rng(0))
    assert np.all(slip == 0) and np.all(cnt == 0)
    assert list(weeks[0]) == [5, 13]   # exposure = weeks until the original Compass date


def test_redate_slip_scales_with_original_lead_only():
    m = make_model(2.0)
    slip, cnt, weeks = simulate_redate(m, np.array(["X", "X"]), np.array(["B1", "B1"]),
                                       np.array([14.0, 70.0]), 20, np.random.default_rng(1))
    assert np.allclose(slip[:, 0], 4.0)     # 2 weeks x 2 days
    assert np.allclose(slip[:, 1], 20.0)    # 10 weeks x 2 days; no runaway as the date moves
    assert np.allclose(cnt[:, 1], 10)


def test_unknown_class_uses_pooled_history():
    m = make_model(1.0)
    slip, _, _ = simulate_redate(m, np.array([POOLED]), np.array(["B3"]), np.array([28.0]), 10,
                                 np.random.default_rng(2))
    assert np.allclose(slip, 4.0)


def risk(**kw):
    base = dict(risk_id="R", name="r", scope="unit", applies_to=None, probability=1.0,
                rate_basis_days=30.0, impact_min=5, impact_ml=5, impact_max=5, distribution="triangular",
                seasonal_months=[], seasonal_prob_mult=1.0, mitigation="", prob_mult_post=0.5,
                impact_mult_post=0.5)
    base.update(kw)
    return Risk(**base)


def open_units(n_units=4):
    return OpenUnits(frame=pd.DataFrame(), lead=np.full(n_units, 50.0), roj=np.full(n_units, 40.0),
                     roj_complete=np.full(n_units, 60.0), pick=np.full(n_units, np.nan),
                     compass_item=np.array(["X"] * n_units), building=np.array(["B1"] * n_units),
                     vendor=np.array(["V1", "V1", "V2", "V2"]), equipment=np.array(["PDU", "PDU", "UPS", "UPS"]),
                     group=np.array([1, 1, 2, 2]))


def test_certain_unit_risk_adds_impact_and_mitigation_scales_it():
    ou = open_units()
    for mitigated, expected in ((False, 5.0), (True, 2.5)):
        A = np.full((100, 4), 50.0)
        d = {}
        _apply_risks(A, ou, [risk(probability=1.0, prob_mult_post=1.0)], mitigated,
                     np.random.default_rng(3), pd.Timestamp("2026-08-18"), d)
        assert np.allclose(A, 50.0 + expected)


def test_group_risk_hits_whole_lot_together():
    ou = open_units()
    A = np.full((2000, 4), 50.0)
    _apply_risks(A, ou, [risk(scope="group", probability=0.5, impact_min=1, impact_ml=5, impact_max=9)],
                 False, np.random.default_rng(4), pd.Timestamp("2026-08-18"), {})
    assert np.allclose(A[:, 0], A[:, 1]) and np.allclose(A[:, 2], A[:, 3])
    assert 0.45 < (A[:, 0] > 50).mean() < 0.55


def test_applies_to_filters_equipment():
    ou = open_units()
    A = np.full((10, 4), 50.0)
    _apply_risks(A, ou, [risk(applies_to=["UPS"])], False, np.random.default_rng(5),
                 pd.Timestamp("2026-08-18"), {})
    assert np.allclose(A[:, :2], 50.0) and np.allclose(A[:, 2:], 55.0)


def test_pert_and_triangular_stay_in_bounds():
    rng = np.random.default_rng(6)
    for dist in ("triangular", "pert"):
        x = risk(impact_min=7, impact_ml=14, impact_max=35, distribution=dist).sample_impact(rng, 20000)
        assert x.min() >= 7 and x.max() <= 35
    pert = risk(impact_min=7, impact_ml=14, impact_max=35, distribution="pert").sample_impact(rng, 200000)
    assert abs(pert.mean() - (7 + 4 * 14 + 35) / 6) < 0.2   # PERT mean = (a + 4m + b) / 6


def test_reduce_max_per_group():
    A = np.array([[1, 5, 2, 9], [3, 1, 4, 0]], dtype=float)
    keys, M, _, _ = reduce_max(A, np.array([2, 1, 2, 1]))
    assert list(keys) == [1, 2]
    assert M.tolist() == [[9, 2], [1, 4]]


def test_tornado_ranks_the_real_driver_first():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    t = tornado({"a": a, "b": b}, 3 * a + 0.3 * b)
    assert t.iloc[0]["driver"] == "a" and t.iloc[0]["spearman_rho"] > 0.9


def test_risk_register_validation(tmp_path):
    p = tmp_path / "rr.csv"
    cols = "risk_id,name,scope,applies_to,probability,rate_basis_days,impact_min,impact_most_likely,impact_max,distribution\n"
    p.write_text(cols + "R1,x,unit,ALL,0.1,,10,5,20,triangular\n")
    with pytest.raises(ValueError, match="min <= most likely"):
        load_risk_register(str(p))
    p.write_text(cols + "R1,x,vendor,ALL,0.1,,1,2,3,triangular\n")
    with pytest.raises(ValueError, match="rate_basis_days"):
        load_risk_register(str(p))


def test_default_risk_register_loads():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    risks = load_risk_register(os.path.join(here, "config", "risk_register.csv"))
    assert {r.scope for r in risks} <= {"unit", "group", "vendor", "class"} and len(risks) >= 1


def test_matrix_units_attach_to_header_above_even_when_schedule_numbers_repeat(tmp_path):
    rows = [[None] * len(MATRIX_COLUMNS) for _ in range(3)]
    header = ["Schedule Number"] + MATRIX_COLUMNS[1:]
    def row(**kw):
        r = [None] * len(MATRIX_COLUMNS)
        for k, v in kw.items():
            r[MATRIX_COLUMNS.index(k)] = v
        return r
    data = [
        row(equipment="PDU"),
        row(sched="S-1", bldg="B1", equipment="PDU", vendor="Schneider", roj=pd.Timestamp("2026-09-01")),
        row(bldg="B1", avail_current=pd.Timestamp("2026-10-01"), qty_received=0),
        row(equipment="UPS"),
        row(sched="S-1", bldg="B1", equipment="UPS", vendor="Schneider", roj=pd.Timestamp("2026-12-01")),
        row(bldg="B1", avail_current=pd.Timestamp("2026-11-01"), qty_received=0),
        row(bldg="B1", avail_current=pd.Timestamp("2026-03-01"), qty_received=1),
    ]
    df = pd.DataFrame(rows + [header] + data)
    path = tmp_path / "m.xlsx"
    df.to_excel(path, sheet_name="2. OFCI Matrix", header=False, index=False)
    units, groups = load_matrix(str(path), "2. OFCI Matrix", pd.Timestamp("2026-08-18"))
    assert len(groups) == 2 and len(units) == 3
    assert units["equipment"].tolist() == ["PDU", "UPS", "UPS"]
    assert units["roj"].dt.month.tolist() == [9, 12, 12]
    assert units["is_open"].tolist() == [True, True, False]
