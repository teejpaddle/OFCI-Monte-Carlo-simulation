"""Load the Cx + OFCI Matrix workbook into tidy tables.

The "2. OFCI Matrix" sheet is hierarchical:
  * an equipment section row (only Equipment Type filled),
  * a schedule-group header row (Schedule Number, ROJ, Vendor, Sub, Qty Needed/Received),
  * one row per unit beneath it (Building, Data Hall, Compass availability, Pick date, Qty Received 0/1).
Units are attached to the header row directly above them by position, because Schedule
Numbers repeat across equipment types.
"""
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

MATRIX_COLUMNS = [
    "sched", "bldg", "dh", "equipment", "vendor", "sub", "roj", "roj_complete",
    "var_roj", "status", "avail_0429", "avail_current", "var_compass", "qty_needed",
    "qty_received", "pick", "pick_location", "var_pick", "historic", "notes",
    "mei_location", "receipt",
]


@dataclass
class ProjectData:
    as_of: pd.Timestamp
    units: pd.DataFrame       # every unit row, with an `is_open` flag
    groups: pd.DataFrame      # one row per schedule-group header
    snapshots: pd.DataFrame   # Compass weekly snapshots, long format


def _find_header_row(path: str, sheet: str) -> int:
    probe = pd.read_excel(path, sheet_name=sheet, header=None, nrows=15)
    for i, row in probe.iterrows():
        if row.astype(str).str.strip().eq("Schedule Number").any():
            return i
    raise ValueError(f"Could not find the 'Schedule Number' header in sheet {sheet!r}")


def detect_as_of(path: str, sheet: str) -> Optional[pd.Timestamp]:
    """Read the 'last updated YY MMDD' stamp above the Current Availability column."""
    probe = pd.read_excel(path, sheet_name=sheet, header=None, nrows=6)
    header_row = _find_header_row(path, sheet)
    col = None
    for j, v in probe.iloc[header_row].items():
        if isinstance(v, str) and "current availability" in v.lower():
            col = j
    if col is None:
        return None
    for i in range(header_row):
        v = probe.iat[i, col]
        m = re.search(r"(\d{2})\s?(\d{2})(\d{2})", str(v)) if isinstance(v, str) else None
        if m and "updated" in v.lower():
            return pd.Timestamp(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def load_matrix(path: str, sheet: str, as_of: pd.Timestamp):
    header_row = _find_header_row(path, sheet)
    df = pd.read_excel(path, sheet_name=sheet, header=header_row)
    if len(df.columns) < len(MATRIX_COLUMNS):
        raise ValueError(f"Expected {len(MATRIX_COLUMNS)} columns in {sheet!r}, found {len(df.columns)}")
    df = df.iloc[:, : len(MATRIX_COLUMNS)]
    df.columns = MATRIX_COLUMNS
    df["bldg"] = df["bldg"].astype(str).str.strip().replace("nan", np.nan)

    is_group = df["sched"].notna()
    df["group_row"] = pd.Series(np.where(is_group, df.index, np.nan)).ffill()

    groups = df[is_group].copy()
    groups["roj"] = pd.to_datetime(groups["roj"], errors="coerce")
    groups["roj_complete"] = pd.to_datetime(groups["roj_complete"], errors="coerce")
    # Vendor/Sub are sometimes left blank on later buildings; use the equipment type's usual value.
    for col in ("vendor", "sub"):
        mode = groups.groupby("equipment")[col].agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
        groups[col] = groups[col].fillna(groups["equipment"].map(mode))
    groups = groups[["sched", "bldg", "dh", "equipment", "vendor", "sub", "roj", "roj_complete",
                     "qty_needed", "qty_received", "notes"]]

    units = df[~is_group & df["bldg"].fillna("").str.match(r"^B\d$")].copy()
    for col in ("sched", "equipment", "vendor", "sub", "roj", "roj_complete"):
        units[col] = units["group_row"].map(groups[col])
    units["dh"] = units["dh"].astype(str).str.strip()
    for col in ("avail_0429", "avail_current", "pick"):
        units[col] = pd.to_datetime(units[col], errors="coerce")

    # Qty Received is 1/0 per unit; a blank means "not tracked" -> treat as open only if not yet due.
    qr = units["qty_received"]
    units["is_open"] = (qr == 0) | (qr.isna() & (units["avail_current"] > as_of))
    units = units[[
        "group_row", "sched", "bldg", "dh", "equipment", "vendor", "sub", "roj", "roj_complete",
        "avail_0429", "avail_current", "pick", "pick_location", "qty_received", "is_open",
    ]].reset_index().rename(columns={"index": "matrix_row"})
    units["matrix_row"] = units["matrix_row"] + header_row + 2  # 1-based Excel row number
    groups = groups.reset_index().rename(columns={"index": "group_row"})
    return units, groups


def load_compass_snapshots(path: str, sheet: str) -> pd.DataFrame:
    """Parse the side-by-side weekly Compass exports into (snapshot, item, job tag, date) rows."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    label_row = None
    for i in range(min(10, len(raw))):
        if raw.iloc[i].astype(str).str.fullmatch(r"\d{2} \d{4}").any():
            label_row = i
            break
    if label_row is None:
        raise ValueError(f"No 'YY MMDD' snapshot labels found in {sheet!r}")
    frames = []
    for c, lab in raw.iloc[label_row].items():
        if not (isinstance(lab, str) and re.fullmatch(r"\d{2} \d{4}", lab)):
            continue
        snap = pd.to_datetime("20" + lab.replace(" ", ""), format="%Y%m%d")
        block = raw.iloc[label_row + 3:, c:c + 3].copy()
        block.columns = ["item", "date", "tag"]
        block = block.dropna(subset=["item"])
        block["date"] = pd.to_datetime(block["date"], errors="coerce")
        block["snapshot"] = snap
        frames.append(block.dropna(subset=["date"]))
    snaps = pd.concat(frames, ignore_index=True)
    snaps["item"] = snaps["item"].astype(str).str.strip()
    snaps["bldg"] = "B" + snaps["tag"].astype(str).str.extract(r"Building (\d)")[0]
    return snaps


def load_project(path: str, settings: dict) -> ProjectData:
    as_of = settings.get("as_of")
    as_of = pd.Timestamp(as_of) if as_of else detect_as_of(path, settings["matrix_sheet"])
    if as_of is None:
        raise ValueError("Could not detect the data date; set 'as_of' in config/settings.json or pass --as-of")
    units, groups = load_matrix(path, settings["matrix_sheet"], as_of)
    snaps = load_compass_snapshots(path, settings["compass_sheet"])
    return ProjectData(as_of=as_of, units=units, groups=groups, snapshots=snaps)
