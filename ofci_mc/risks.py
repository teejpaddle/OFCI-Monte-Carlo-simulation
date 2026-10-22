"""Risk register: discrete delay events layered on top of Compass forecast uncertainty.

Scopes
  unit    - each unit independently (e.g. transit delay, damage on receipt)
  group   - once per schedule number / shipment lot, all its units together (e.g. failed FAT)
  vendor  - rate per `rate_basis_days` of exposure; hits every open unit of that vendor not yet
            available at a random onset date (e.g. capacity / material shortage)
  class   - same as vendor, per equipment type (e.g. design change on a released product)

Impacts are three-point estimates (days). Per PMI guidance, use triangular when only the three
points are known (more conservative) and PERT/Beta when the most-likely value is well supported.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

VALID_SCOPES = {"unit", "group", "vendor", "class"}


@dataclass
class Risk:
    risk_id: str
    name: str
    scope: str
    applies_to: Optional[List[str]]    # None = all equipment
    probability: float
    rate_basis_days: float
    impact_min: float
    impact_ml: float
    impact_max: float
    distribution: str
    seasonal_months: List[int]
    seasonal_prob_mult: float
    mitigation: str
    prob_mult_post: float
    impact_mult_post: float
    owner: str = ""
    source_note: str = ""

    @property
    def label(self) -> str:
        return f"{self.risk_id} {self.name}"

    def applies(self, equipment: np.ndarray) -> np.ndarray:
        if self.applies_to is None:
            return np.ones(len(equipment), dtype=bool)
        return np.isin(equipment, self.applies_to)

    def sample_impact(self, rng: np.random.Generator, size) -> np.ndarray:
        a, m, b = self.impact_min, self.impact_ml, self.impact_max
        if b <= a:
            return np.full(size, float(a))
        if self.distribution == "pert":
            alpha = 1 + 4 * (m - a) / (b - a)
            beta = 1 + 4 * (b - m) / (b - a)
            return a + rng.beta(alpha, beta, size=size) * (b - a)
        return rng.triangular(a, m, b, size=size)


def _split(v) -> List[str]:
    if pd.isna(v) or str(v).strip() == "":
        return []
    return [s.strip() for s in str(v).split(";") if s.strip()]


def load_risk_register(path: str) -> List[Risk]:
    df = pd.read_csv(path)
    risks = []
    for _, r in df.iterrows():
        scope = str(r["scope"]).strip().lower()
        if scope not in VALID_SCOPES:
            raise ValueError(f"{r['risk_id']}: scope must be one of {sorted(VALID_SCOPES)}")
        if not (r["impact_min"] <= r["impact_most_likely"] <= r["impact_max"]):
            raise ValueError(f"{r['risk_id']}: impacts must satisfy min <= most likely <= max")
        if not 0 <= r["probability"] <= 1:
            raise ValueError(f"{r['risk_id']}: probability must be between 0 and 1")
        applies = _split(r["applies_to"])
        rate = r.get("rate_basis_days")
        if scope in ("vendor", "class") and (pd.isna(rate) or rate <= 0):
            raise ValueError(f"{r['risk_id']}: vendor/class risks need rate_basis_days > 0")
        risks.append(Risk(
            risk_id=str(r["risk_id"]), name=str(r["name"]), scope=scope,
            applies_to=None if applies in ([], ["ALL"]) else applies,
            probability=float(r["probability"]),
            rate_basis_days=float(rate) if not pd.isna(rate) else 0.0,
            impact_min=float(r["impact_min"]), impact_ml=float(r["impact_most_likely"]),
            impact_max=float(r["impact_max"]),
            distribution=str(r.get("distribution", "triangular")).strip().lower(),
            seasonal_months=[int(m) for m in _split(r.get("seasonal_months"))],
            seasonal_prob_mult=float(r["seasonal_prob_mult"]) if not pd.isna(r.get("seasonal_prob_mult")) else 1.0,
            mitigation=str(r.get("mitigation", "")),
            prob_mult_post=float(r.get("prob_mult_post", 1.0)),
            impact_mult_post=float(r.get("impact_mult_post", 1.0)),
            owner=str(r.get("owner", "")), source_note=str(r.get("source_note", "")),
        ))
    return risks
