"""Score every dst_run config's running_log and aggregate JGA into a table/CSV.

For each dst_run config under the config root, score the matching
outputs/.../running_log.json (if present) against that config's test set, then
group runs by config directory (mean over split_v* seeds). Configs whose
running_log is absent are listed as missing; runs that fail to score as errors."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from combisearch.config.paths import (
    canonical_output_path,
    iter_experiment_configs,
    resolve_canonical_path,
    resolve_data_path,
)
from combisearch.evaluation.error_analysis import read_running_log_file
from combisearch.evaluation.score_run import score_running_log


def score_dst_runs(config_root: str | Path = "configs", table: str | None = None):
    """Return (scored, missing, errors). `scored` has one row per dst_run config
    whose running_log exists; `table` restricts to config paths containing it."""
    root = Path(config_root).resolve()
    scored: list[dict[str, Any]] = []
    missing: list[str] = []
    errors: list[str] = []
    for path, obj, stage in iter_experiment_configs(root):
        if stage != "dst_run":
            continue
        rel = path.relative_to(root).with_suffix("").as_posix()
        if table and table not in rel:
            continue
        running_log_path = resolve_canonical_path(
            canonical_output_path(path, stage, root)
        ) / "running_log.json"
        if not running_log_path.exists():
            missing.append(rel)
            continue
        try:
            test_set = json.loads(resolve_data_path(obj["test_fn"]).read_text())
            running_log = read_running_log_file(str(running_log_path))
            result = score_running_log(running_log, test_set, verbose=False)
        except Exception as exc:
            errors.append(f"{rel}: {exc}")
            continue
        group, _, split = rel.rpartition("/")
        scored.append({
            "group": group or rel,
            "split": split,
            "jga": result["jga"],
            "slot_acc": result["slot_acc"],
            "joint_f1": result["joint_f1"],
            "n_total": result["n_total"],
        })
    return scored, missing, errors


def aggregate(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean JGA / slot-acc / F1 per config group (over its split_v* runs)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[row["group"]].append(row)
    out = []
    for group, rows in sorted(groups.items()):
        jgas = [r["jga"] for r in rows]
        out.append({
            "group": group,
            "n_splits": len(rows),
            "jga_mean": sum(jgas) / len(jgas),
            "jga_std": statistics.pstdev(jgas) if len(jgas) > 1 else 0.0,
            "slot_acc_mean": sum(r["slot_acc"] for r in rows) / len(rows),
            "joint_f1_mean": sum(r["joint_f1"] for r in rows) / len(rows),
            "splits": ",".join(sorted(r["split"] for r in rows)),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", default="configs",
                        help="config tree to scan (default: configs)")
    parser.add_argument("--table", default=None,
                        help="only score configs whose path contains this (e.g. table_1)")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the per-group CSV here")
    parser.add_argument("--per-run", action="store_true",
                        help="also print each run, not just per-group means")
    args = parser.parse_args(argv)

    scored, missing, errors = score_dst_runs(args.config_root, args.table)
    agg = aggregate(scored)

    header = f"{'group':<48} {'n':>2} {'JGA':>7} {'std':>6} {'SlotAcc':>8} {'JointF1':>8}"
    print(header)
    print("-" * len(header))
    for r in agg:
        print(f"{r['group']:<48} {r['n_splits']:>2} {r['jga_mean']*100:>7.2f} "
              f"{r['jga_std']*100:>6.2f} {r['slot_acc_mean']*100:>8.2f} {r['joint_f1_mean']*100:>8.2f}")

    if args.per_run:
        print("\nper run:")
        for r in sorted(scored, key=lambda x: (x["group"], x["split"])):
            print(f"  {r['group']}/{r['split']:<10} JGA {r['jga']*100:6.2f}  (n={r['n_total']})")

    if errors:
        print(f"\n{len(errors)} run(s) failed to score:")
        for e in errors:
            print(f"  - {e}")
    if missing:
        print(f"\n{len(missing)} dst_run config(s) have no running_log yet:")
        for m in missing:
            print(f"  - {m}")

    if args.out:
        fields = ["group", "n_splits", "jga_mean", "jga_std",
                  "slot_acc_mean", "joint_f1_mean", "splits"]
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(agg)
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
