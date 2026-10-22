# OFCI Procurement Monte Carlo Simulation

Monte Carlo schedule-risk model for **Owner-Furnished, Contractor-Installed (OFCI)** equipment on a
multi-building project. It reads the live **Cx + OFCI Matrix** workbook, learns how the owner's
supply-chain availability dates actually move from their weekly history, layers a risk register on
top, and reports, for every open unit, schedule group and building, how likely the equipment is to
be available by its **ROJ**, **ROJ complete** and **pick** dates, with P10/P50/P80/P90 dates and
schedule contingency.

```
simulated availability = current Compass forecast + re-date drift (learned from history) + risk events
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_simulation.py --workbook "/path/to/Cx + OFCI Matrix - LIVE.xlsx"
```

The run takes about a minute for 10,000 iterations × 3 scenarios. Results go to `outputs/`
(git-ignored): an Excel workbook, PNG charts and `summary.json`.

| Option | Default | Purpose |
|---|---|---|
| `--as-of YYYY-MM-DD` | the matrix's *"last updated"* stamp above *Current Availability* | data date |
| `--iterations N` | 10,000 | iterations per scenario |
| `--risk-register PATH` | `config/risk_register.csv` | discrete risk events + mitigations |
| `--settings PATH` | `config/settings.json` | model settings |
| `--out DIR` | `outputs/` | output folder |

## Method

### 1. Inputs
* **`2. OFCI Matrix`**: hierarchical. Each schedule-group header row (Schedule Number, ROJ, ROJ
  Completes, Vendor, Sub) has one row per unit beneath it (Building, Data Hall, start-of-window
  availability, current Compass availability, Quantity Received, Pick date). Units are attached to
  the header **by position**, because Schedule Numbers repeat across equipment types. A unit is
  *open* when Quantity Received = 0.
* **`3.Compass Supply Chain_raw data`**: side-by-side weekly exports of Item / Available-For-Site
  Date / Job Tag. Consecutive snapshots are matched per item, job tag and date rank to measure
  week-by-week re-dates.
* `config/equipment_map.csv` maps matrix equipment types to Compass item names.

### 2. Forecast uncertainty: the Compass re-date process (fitted, not assumed)
The weekly history shows three things a textbook "independent triangular per item" model misses:

1. **Lumpy and correlated.** In a given week, an equipment class is re-dated across nearly all of
   its units, or not at all, and most buildings move together.
2. **Reversals.** Large pull-ins are often partly undone by push-outs a week later, so weekly moves
   are *not* independent. Sampling them independently badly inflates the long-run spread.
3. **Regimes.** A model fit only on older snapshots under-predicted the next window (see
   *Validation*). Recent behaviour has to be part of the history.

So the model keeps each class's **real weekly history** (one series per building) and replays it
with a **moving-block bootstrap**: random 4-week stretches of actual history, all buildings
together, chained to cover each unit's remaining lead time. History = weekly snapshots up to the
start of the matrix's own comparison window, plus that window (start-of-window availability →
current availability) appended as one net move per class and building.

* **Exposure** runs until the unit's *current* Compass date. It doesn't extend as a unit slips,
  because that feedback runs away for volatile classes.
* **Scatter** around a class move is drawn **once per schedule lot**, plus a small independent unit
  share (`unit_scatter_share`). History shows a lot's last unit lands only days after its median
  unit, and independent per-unit draws would inflate every "last unit" date.
* **Empirical-Bayes pooling.** A class with little history draws part of its blocks from the
  pooled history of all classes (`prior_weeks_strength`), so "no history" never means "no risk".
* **Vendor correlation.** Same-vendor classes replay the same stretch of history with probability
  ρ, estimated from co-occurring weekly re-dates.

### 3. Discrete risk events: the risk register
`config/risk_register.csv` holds named risks the history cannot show: logistics after availability,
damage on receipt, and low-frequency or high-impact vendor events. Each risk has:

* **Scope**:
  * `unit`: each unit independently.
  * `group`: once per schedule lot, so all of its units move together.
  * `vendor` / `class`: a rate per `rate_basis_days` of exposure, with a random onset date; it hits
    every open unit of that vendor or type not yet available at onset.
* **Impact**: a three-point estimate (min / most likely / max days), sampled **triangular** (PMI:
  the conservative choice when only three points are known) or **PERT/Beta** (PMI: when the
  most-likely value is well supported).
* **Seasonality**: optional months with a probability multiplier, e.g. winter logistics.
* **Mitigation**: `prob_mult_post` and `impact_mult_post` give the residual risk after the planned
  response.

> The shipped values are **placeholder assumptions** for the risk owners to validate. Monte Carlo is
> only as good as its inputs (PMI: *"garbage in, garbage out"*). The re-date history already
> contains whatever vendor events happened in it, so keep register probabilities to what history
> does **not** show, to avoid double counting.

### 4. Three models (PMI, Barreras 2011)
| Scenario | Contents | Use |
|---|---|---|
| **Baseline** | forecast uncertainty only | the spread of the current plan |
| **Pre-mitigated** | + risk events at full probability/impact | sizing exposure, ranking risks (tornado) |
| **Post-mitigated** | + risk events after mitigation | **plan of record**: contingency, ratings |

The scenarios share the same random draws (common random numbers), so differences between them
come only from the risks and mitigations. *Mitigation value* = Pre-P80 − Post-P80.

### 5. Outputs
For each open unit, schedule group (last unit of the lot) and building, the model reports:

* **Dates.** P10 / P50 / P80 / P90 availability dates.
* **Probabilities.** Being available by **ROJ**, by **ROJ complete** (missing this means the
  install window is lost), and by **pick date** where a pick is planned.
* **Contingency.** `P80 − Compass date` per schedule group, with a **management reserve** at
  `P90 − Compass date` (PMI uses P80 for the budget and P90 for reserve).
* **Deterministic percentile.** Where the Compass date sits in the simulated range. PMI's warning
  sign is a deterministic value below about P20, meaning the plan is aggressive.
* **Risk rating** (Post-mitigated), from the chance that the lot's last unit misses ROJ complete:
  Critical ≥ 80%, High ≥ 50%, Medium ≥ 20% (also Medium if ≥ 50% chance of missing ROJ start).
* **Criticality index.** The share of iterations in which an equipment type is the last item
  available in its building.
* **Building readiness.** Both the *last unit* date and the more robust *90 % of units* date. The
  last-unit date is a maximum over hundreds of units, so it sits in the tail by construction.
* **Tornado.** The Spearman rank correlation of each driver (re-date drift per class, each
  register risk) with total unit-days late vs ROJ, plus its share of variance, as in @Risk and
  Shinde (2020). It shows association, not causation; seasonal risks can show small negative
  values.

Workbook sheets: `Summary`, `Charts`, `Buildings`, `Schedule Groups`, `Equipment x Building`,
`Units (Post-mitigated)`, `Tornado`, `Risk Register`, `Calibration`, `Backtest`, `Run Info`.

### 6. Validation
**Holdout test.** A model fit only on history *before* the matrix window re-forecasts the
start-of-window dates over that window and is scored on what actually happened:

* **PIT** (where each actual landed in its simulated range): 0.5 is unbiased; above 0.5 means
  dates slipped more than predicted.
* **P10–P90 coverage**: the target is 80%.

The production model *includes* that window, so it cannot be scored on it. Re-run monthly and watch
whether new actuals land inside the bands (PMI: monitor and control).

## How the articles map to the code
| Source | Guidance | Where |
|---|---|---|
| PMI, *Monte Carlo simulation in cost estimating* (Barreras, 2011) | Baseline → pre-mitigated → post-mitigated models | `engine.SCENARIOS` |
| | Three-point inputs; triangular when little is known, Beta/PERT when ML is well supported | `risks.Risk.sample_impact` |
| | Correlations between items and between risks | block bootstrap (class/building), vendor ρ, group/vendor-scope risks |
| | ≥ 1,000 iterations; P80 for contingency, P90 for management reserve | `settings.json` |
| | Contingency = post-mitigated vs baseline at the chosen P-level; flag deterministic < P20 | `analysis.group_table` |
| | Tornado / sensitivity to prioritise risk responses | `analysis.tornado` |
| Project Management Academy, *Understanding the Monte Carlo analysis* | Define inputs and distributions → build → run → **validate against history and refine** | holdout test, calibration sheet |
| | Lognormal-like right skew for durations; independence assumptions can mislead | empirical right-skewed history; explicit correlation |
| Shinde (2020), *Monte Carlo simulations in construction project risk management* | Past data + expert input for distributions; 10,000 runs | Compass history + risk register; `iterations` |
| | Probability of on-time completion; delay bands; sensitivity (material shortage was the top driver) | P(by ROJ / ROJ complete / pick); `R2` vendor shortage; tornado |

## Configuration (`config/settings.json`)
| Key | Meaning |
|---|---|
| `iterations`, `seed`, `chunk_size` | run size and reproducibility; chunking bounds memory |
| `percentiles`, `contingency_percentile`, `management_reserve_percentile` | reporting levels |
| `block_weeks` | bootstrap block length (keeps reversals and co-movement intact) |
| `unit_scatter_share` | independent unit share of within-event scatter (lots mostly co-move) |
| `event_fraction_threshold` | share of a lot re-dated in a week for it to count as a class event |
| `prior_weeks_strength` | pooling strength for thin histories |
| `vendor_event_correlation` | `null` = estimate from data |
| `max_valid_redate_days` | drops data-entry typos (e.g. 100-year jumps) |
| `backtest_from` | start of the matrix comparison window (the start-of-window availability column) |

## Limitations
* **Short history.** A few months of snapshots can't contain every regime. Bands will widen or shift
  as history accumulates, so re-run as the matrix updates.
* **Persistence.** Replaying recent behaviour over long lead times assumes the recent re-date pace
  continues. Far-out units (and last-unit building dates) carry the most uncertainty.
* **Schedule only.** There is no cost dimension. A `$ / day late` per equipment type could be added
  to report cost contingency the PMI way.
* **Stale ROJ.** If the contract ROJ baseline is stale, most units will show as late against it.
  Then the P50/P80 dates, pick-date probabilities and contingency days are the decision-useful
  outputs.

## Tests
```bash
.venv/bin/python -m pytest -q tests
```

## Data handling
This repository is public. **Don't commit the matrix workbook or anything in `outputs/`.** They
contain project schedules, vendor and contact details. `.gitignore` excludes them.
