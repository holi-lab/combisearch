#!/usr/bin/env python3
"""Build shared dense search indexes used by CombiSearch configs."""

from __future__ import annotations

import argparse
import sys

from combisearch.cli.runner import apply_runtime_seed, configure_default_env
from combisearch.indexing.build import build_one_index
from combisearch.indexing.encoder import (
    DEFAULT_INDEX_EMBEDDING_MODEL,
    load_encoder,
    resolve_encoder_path,
)
from combisearch.indexing.specs import select_specs
from combisearch.config.paths import embedding_model_tag


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="rebuild existing indexes")
    p.add_argument("--dry-run", action="store_true", help="print planned actions only")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--index-embedding-model-name",
        default=DEFAULT_INDEX_EMBEDDING_MODEL,
        help="encoder used to create index embeddings",
    )
    p.add_argument(
        "--only", action="append", default=[],
        help="only build specs matching sub_dir (or a prefix); can be repeated",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    configure_default_env()
    apply_runtime_seed()
    args = parse_args(argv)
    selected = select_specs(args.only)
    if not selected:
        print(f"no SPECS matched --only filters: {args.only}", file=sys.stderr)
        return 2

    encoder_path = resolve_encoder_path(
        args.index_embedding_model_name,
        prepare_if_missing=not args.dry_run,
    )
    encoder_tag = embedding_model_tag(str(encoder_path))
    print(f"== encoder ==\n  {encoder_path}\n")

    model = None if args.dry_run else load_encoder(encoder_path)

    print(f"== indexes ({len(selected)} target(s)) ==")
    built = sum(
        build_one_index(
            spec, model=model, encoder_tag=encoder_tag,
            force=args.force, dry_run=args.dry_run, batch_size=args.batch_size,
        )
        for spec in selected
    )
    print(f"\n{'planned' if args.dry_run else 'built'}: {built}/{len(selected)} indexes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
