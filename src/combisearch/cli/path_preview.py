"""CLI: preview the canonical ``output_dir`` derived from each experiment config.

Wraps the ``combisearch.config.paths`` preview helpers. Each config under
``configs/`` is classified into a stage and its output path is computed;
reports collisions (two configs pointing at the same output) and a
stage-by-stage count. Returns exit code 1 if any collision is found.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from combisearch.cli.runner import apply_runtime_seed, configure_default_env
from combisearch.config.paths import (
    DEFAULT_CONFIG_ROOT,
    experiment_output_records,
    output_path_collisions,
)


def main(argv: list[str] | None = None) -> int:
    configure_default_env()
    apply_runtime_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--json", dest="json_path", help="optional path to dump the path registry")
    args = parser.parse_args(argv)

    config_root = Path(args.config_root).resolve()
    records = experiment_output_records(config_root)
    collisions = output_path_collisions(records)
    by_stage = Counter(r["stage"] for r in records)

    if args.json_path:
        target = Path(args.json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")

    print(f"planned_configs={len(records)} collisions={len(collisions)}")
    for stage, count in sorted(by_stage.items()):
        print(f"{stage}={count}")
    if collisions:
        print("collision paths:")
        for item in collisions:
            print(f"- {item['canonical_path']}")
    return 1 if collisions else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
