"""Build dense search indexes from :class:`IndexSpec` declarations.

Shared by the ``build_indexes`` CLI and by DST inference (which lazily
builds a missing canonical pretrained index).
"""

from __future__ import annotations

from combisearch.indexing.datasets import domain_filter_for, load_turn_dataset
from combisearch.indexing.embed import save_embeddings
from combisearch.indexing.specs import IndexSpec


def build_one_index(
    spec: IndexSpec,
    *,
    model,
    encoder_tag: str,
    force: bool,
    dry_run: bool,
    batch_size: int,
) -> bool:
    """Build the dense index for a single spec; return True if it was (re)built."""
    out_file = spec.output_file(encoder_tag)
    if out_file.exists() and not force:
        print(f"  [skip]  {out_file}")
        return False

    print(f"  [build] {spec.train_fn} -> {out_file}")
    if dry_run:
        return True

    dataset = load_turn_dataset(
        spec.train_fn,
        input_type=spec.input_type,
        gt_type=spec.gt_type,
        domain_filter=domain_filter_for(spec.dataset),
    )
    print(f"          {len(dataset)} turns")
    save_embeddings(model, dataset, out_file, batch_size=batch_size)
    return True
