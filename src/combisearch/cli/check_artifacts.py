"""CLI: preflight-check config-declared input artifacts.

Wraps :func:`combisearch.config.preflight.scan_config_artifacts`. Walks the
configs/ tree and reports declared input paths that do not exist on disk.
Read-only — never modifies configs or path values.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

from combisearch.cli.runner import apply_runtime_seed, configure_default_env
from combisearch.config.preflight import (
    CATEGORY_ORDER,
    producer_hint,
    scan_config_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    configure_default_env()
    apply_runtime_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-root",
        default=os.environ.get("COMBISEARCH_CONFIG_ROOT", "configs"),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print configs referencing each missing artifact",
    )
    args = parser.parse_args(argv)

    config_root = Path(args.config_root).resolve()
    if not config_root.exists():
        print(f"config root not found: {config_root}", file=sys.stderr)
        return 2

    scan = scan_config_artifacts(config_root)
    print(f"Scanned {scan.n_configs} experiment configs from {config_root}\n")

    by_category: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for (cat, target), used_by in scan.missing.items():
        by_category[cat].append((target, used_by))

    for category in CATEGORY_ORDER:
        rows = sorted(by_category.get(category, []))
        print(f"### Missing {category} artifacts: {len(rows)}")
        for target, used_by in rows:
            print(f"  - {target}")
            print(f"      producer: {producer_hint(category, Path(target))}")
            if args.verbose:
                for cfg in used_by[:5]:
                    print(f"      used by: {cfg}")
                if len(used_by) > 5:
                    print(f"      ... and {len(used_by) - 5} more")
            else:
                print(f"      used by {len(used_by)} config(s)")
        print()

    if scan.semantic_errors:
        print(f"### Semantic path errors: {len(scan.semantic_errors)}")
        for cfg, message in scan.semantic_errors:
            print(f"  - {cfg}: {message}")
        print()

    print(f"Total missing artifact references: {scan.n_missing}")
    return 0 if scan.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
