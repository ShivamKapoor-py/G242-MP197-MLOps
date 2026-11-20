"""
Stage 1 data audit for Group G242, problem MP197.
Run from the repository root:
    python src/data_audit.py --train data/raw/train.csv --later data/raw/later.csv --holdout data/raw/holdout_features.csv

This script NEVER modifies the input CSV files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET = "entries_per_hour"
NUMERIC_FEATURES = [
    "hour", "nearby_offices", "temp_c", "rain_mm",
    "feeder_buses", "ticket_price_rs",
]
EXPECTED_DAY_TYPES = {"weekday", "weekend", "holiday"}


def longest_constant_run(df: pd.DataFrame, column: str):
    ordered = df.sort_values("date", kind="stable").reset_index(drop=True)
    groups = ordered[column].ne(ordered[column].shift()).cumsum()
    runs = (
        ordered.assign(_run=groups)
        .groupby("_run", sort=False)
        .agg(count=(column, "size"), value=(column, "first"),
             start_date=("date", "first"), end_date=("date", "last"))
    )
    return runs.sort_values("count", ascending=False).iloc[0]


def audit(train_path: Path, later_path: Path, holdout_path: Path):
    # Read only; never write to the source datasets.
    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)

    results = {
        "rows": {
            "train": int(len(train)),
            "later": int(len(later)),
            "holdout": int(len(holdout)),
        },
        "data_issues": {},
    }

    # 1. Leakage: the problem's stated clue is a train column absent from holdout.
    missing_from_holdout = sorted(set(train.columns) - set(holdout.columns))
    leakage_candidates = [c for c in missing_from_holdout if c != TARGET]
    results["data_issues"]["leaking_column"] = {
        "present": bool(leakage_candidates),
        "columns_missing_from_holdout": missing_from_holdout,
        "leaking_columns": leakage_candidates,
    }

    # 2. Unit mix-up: compare numeric feature scales by the line grouping.
    # The expected issue would create a conspicuous scale difference for one line.
    group_stats = train.groupby("line")[NUMERIC_FEATURES].agg(["median", "mean", "min", "max"])
    scale_ratios = {}
    for col in NUMERIC_FEATURES:
        medians = train.groupby("line")[col].median()
        positive = medians[medians > 0]
        ratio = float(positive.max() / positive.min()) if len(positive) > 1 else 1.0
        scale_ratios[col] = ratio
    suspicious_unit_columns = [c for c, ratio in scale_ratios.items() if ratio >= 10]
    results["data_issues"]["unit_mixup"] = {
        "present": bool(suspicious_unit_columns),
        "scale_ratio_by_column": scale_ratios,
        "suspicious_columns": suspicious_unit_columns,
        "group_stats_by_line": {
            line: {
                col: {stat: float(group_stats.loc[line, (col, stat)])
                      for stat in ["median", "mean", "min", "max"]}
                for col in NUMERIC_FEATURES
            }
            for line in group_stats.index
        },
    }

    # 3. Sentinel: look for impossible placeholder numbers, not merely values
    # outside later.csv's range (valid training values can be outside that range).
    # Bounds are broad physical/business sanity limits for this problem.
    plausible_bounds = {
        "hour": (0, 23), "nearby_offices": (0, 10000),
        "temp_c": (-20, 60), "rain_mm": (0, 1000),
        "feeder_buses": (0, 1000), "ticket_price_rs": (0, 95),
        TARGET: (0, 100000),
    }
    sentinel_candidates = {}
    for col, (lo, hi) in plausible_bounds.items():
        out = train[(train[col] < lo) | (train[col] > hi)][col].value_counts()
        if len(out):
            sentinel_candidates[col] = {str(k): int(v) for k, v in out.items()}
    results["data_issues"]["sentinel_value"] = {
        "present": bool(sentinel_candidates),
        "outside_later_range": sentinel_candidates,
    }

    # 4. Exact duplicate rows; count extra copies, not all rows in duplicate groups.
    duplicate_count = int(train.duplicated().sum())
    results["data_issues"]["duplicate_rows"] = {
        "present": duplicate_count > 0,
        "count": duplicate_count,
    }

    # 5. Stuck sensor: longest uninterrupted same-value run after date sorting.
    stuck_candidates = {}
    for col in NUMERIC_FEATURES:
        run = longest_constant_run(train, col)
        if int(run["count"]) >= 20:
            stuck_candidates[col] = {
                "value": float(run["value"]),
                "count": int(run["count"]),
                "start_date": str(run["start_date"]),
                "end_date": str(run["end_date"]),
            }
    # Select the longest qualifying run, if any.
    stuck_col = max(stuck_candidates, key=lambda c: stuck_candidates[c]["count"]) if stuck_candidates else None
    results["data_issues"]["stuck_sensor"] = {
        "present": stuck_col is not None,
        "column": stuck_col,
        **(stuck_candidates[stuck_col] if stuck_col else {}),
        "all_candidates": stuck_candidates,
    }

    # 6. Target typo: a target typed 10x too large should be far outside the
    # target's ordinary scale. Compare against the later clean target range.
    target_hi = float(later[TARGET].max())
    typo_rows = train[train[TARGET] > target_hi * 1.5]
    results["data_issues"]["target_typos"] = {
        "present": len(typo_rows) > 0,
        "count": int(len(typo_rows)),
        "candidate_values": [int(x) for x in typo_rows[TARGET].tolist()],
        "clean_later_max": target_hi,
    }

    # 7. Spelling variants: normalize whitespace/case and compare with later's
    # clean category vocabulary.
    spelling_results = {}
    for col in ["day_type", "line"]:
        raw_levels = sorted(train[col].dropna().astype(str).unique().tolist())
        normalized = train[col].astype(str).str.strip().str.lower()
        later_levels = set(later[col].dropna().astype(str).str.strip().str.lower())
        if len(raw_levels) > len(later_levels):
            spelling_results[col] = {
                "present": True,
                "raw_levels": raw_levels,
                "true_levels": int(len(later_levels)),
                "normalized_levels": sorted(normalized.unique().tolist()),
            }
    spelling_col = next(iter(spelling_results), None)
    results["data_issues"]["spelling_variants"] = {
        "present": spelling_col is not None,
        "column": spelling_col,
        "true_levels": spelling_results[spelling_col]["true_levels"] if spelling_col else None,
        "details": spelling_results,
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--later", type=Path, default=Path("data/raw/later.csv"))
    parser.add_argument("--holdout", type=Path, default=Path("data/raw/holdout_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/stage1_audit.json"))
    args = parser.parse_args()

    report = audit(args.train, args.later, args.holdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\\nAudit report saved to: {args.output}")


if __name__ == "__main__":
    main()
