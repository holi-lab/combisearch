"""
File copied from the WoodScene/LDST and modified to convert SGD into MultiWOZ-style turn-level DST JSON files.:

@inproceedings{feng-etal-2023-ldst,
    title = "Towards LLM-driven Dialogue State Tracking",
    author = "Feng, Yujie  and
      Lu, Zexin  and
      Liu, Bo  and
      Zhan, Liming and
      Wu, Xiaoming",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
}
"""
import argparse
import json
import random
from pathlib import Path


DEFAULT_SUBSETS = {
    "train_5p": ("train", "train_5p_ids.json"),
    "test_30p": ("test", "test_30p_ids.json"),
}

OUTPUT_FILENAMES = {
    "train_5p": ["sgd_5p_train.json"],
    "test_30p": ["sgd_30p_test.json"],
    "dev_5p": ["sgd_5p_dev.json"],
}

# The dev set has no curated id list: it is a deterministic (seeded) random sample of
# train dialogues, disjoint from train_5p. It is only used for retriever checkpoint
# selection during fine-tuning, so the exact composition does not matter.
DEV_SUBSET = {"split": "train", "ratio": 0.05, "seed": 42}


def dialogue_sort_key(path):
    return int(path.stem.split("_")[-1])


def dialogue_uses_any_service(dialogue, services):
    return any(service in services for service in dialogue["services"])


def preprocess_dialogue_file(dialogue_path, included_services, selected_dialogue_ids):
    with open(dialogue_path, encoding="utf-8") as f:
        dialogues = json.load(f)

    processed_turns = []
    for dialogue in dialogues:
        if dialogue["dialogue_id"] not in selected_dialogue_ids:
            continue
        if not dialogue_uses_any_service(dialogue, included_services):
            continue

        dialogue_id = dialogue["dialogue_id"]
        dialogue_services = dialogue["services"]
        system_utterances = [""]
        user_utterances = []
        slot_values = {}
        last_slot_values = {}
        turn_id = 0

        for idx, turn in enumerate(dialogue["turns"]):
            if turn["speaker"] == "SYSTEM":
                system_utterances.append(turn["utterance"])
                continue

            user_utterances.append(turn["utterance"])
            assert turn_id == idx // 2

            turn_item = {}
            turn_slot_values = {}
            for frame in turn["frames"]:
                for slot_name, values in frame["state"]["slot_values"].items():
                    slot_values[f"{frame['service']}-{slot_name}"] = values[0]

                for slot_name, value in slot_values.items():
                    if slot_name not in last_slot_values:
                        turn_slot_values[slot_name] = value
                    elif slot_values[slot_name] != last_slot_values[slot_name]:
                        turn_slot_values[slot_name] = value

                for slot_name in last_slot_values:
                    if slot_name not in slot_values:
                        turn_slot_values[slot_name] = "[DELETE]"

                turn_item = {
                    "ID": dialogue_id,
                    "turn_id": turn_id,
                    "domains": dialogue_services,
                    "dialog": {
                        "sys": system_utterances.copy(),
                        "usr": user_utterances.copy(),
                    },
                    "slot_values": slot_values.copy(),
                    "turn_slot_values": turn_slot_values.copy(),
                    "last_slot_values": last_slot_values.copy(),
                }
                last_slot_values = slot_values.copy()

            turn_id += 1
            processed_turns.append(turn_item)

    return processed_turns


def find_same_services(sgd_dir):
    with open(sgd_dir / "train" / "schema.json", encoding="utf-8") as f:
        train_schema = json.load(f)
    with open(sgd_dir / "test" / "schema.json", encoding="utf-8") as f:
        test_schema = json.load(f)

    train_services = {domain["service_name"] for domain in train_schema}
    return [
        domain["service_name"]
        for domain in test_schema
        if domain["service_name"] in train_services
    ]


def load_dialogue_ids(path):
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))


def write_subset_outputs(output_dirs, subset_name, split_data):
    filenames = OUTPUT_FILENAMES[subset_name]
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            output_path = output_dir / filename
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2)
        print(f"Wrote {len(split_data)} turns to {output_dir} as {', '.join(filenames)}")


def convert_sgd_to_multiwoz(sgd_dir, output_dir, ids_dir, extra_output_dirs=None):
    sgd_dir = Path(sgd_dir)
    output_dirs = [Path(output_dir)]
    output_dirs.extend(Path(path) for path in (extra_output_dirs or []))
    ids_dir = Path(ids_dir)

    included_services = find_same_services(sgd_dir)
    print(f"Included services: {included_services}")
    print(f"Output directories: {output_dirs}")

    for subset_name, (split, ids_filename) in DEFAULT_SUBSETS.items():
        selected_dialogue_ids = load_dialogue_ids(ids_dir / ids_filename)
        print(
            f"Preprocessing {subset_name} from {split} "
            f"using {len(selected_dialogue_ids)} dialogue IDs"
        )
        dialogue_paths = [
            path
            for path in (sgd_dir / split).glob("*.json")
            if path.name != "schema.json"
        ]
        dialogue_paths = sorted(dialogue_paths, key=dialogue_sort_key)

        split_data = []
        for dialogue_path in dialogue_paths:
            split_data.extend(
                preprocess_dialogue_file(
                    dialogue_path,
                    included_services,
                    selected_dialogue_ids,
                )
            )

        write_subset_outputs(output_dirs, subset_name, split_data)

    _write_seeded_dev(sgd_dir, output_dirs, ids_dir, included_services)


def _write_seeded_dev(sgd_dir, output_dirs, ids_dir, included_services):
    """Write sgd_5p_dev.json: a deterministic (seeded) random sample of train dialogues,
    disjoint from train_5p. Used only for retriever checkpoint selection."""
    train_paths = sorted(
        [p for p in (sgd_dir / DEV_SUBSET["split"]).glob("*.json") if p.name != "schema.json"],
        key=dialogue_sort_key,
    )
    all_train_ids = []
    for path in train_paths:
        with open(path, encoding="utf-8") as f:
            all_train_ids.extend(dialogue["dialogue_id"] for dialogue in json.load(f))

    train_5p_ids = load_dialogue_ids(ids_dir / DEFAULT_SUBSETS["train_5p"][1])
    candidate_ids = [dialogue_id for dialogue_id in all_train_ids if dialogue_id not in train_5p_ids]
    n_dev = int(DEV_SUBSET["ratio"] * len(all_train_ids))
    dev_ids = set(random.Random(DEV_SUBSET["seed"]).sample(candidate_ids, n_dev))
    print(
        f"Preprocessing dev_5p from {DEV_SUBSET['split']} using {len(dev_ids)} seeded dialogue IDs "
        f"(seed={DEV_SUBSET['seed']}, ratio={DEV_SUBSET['ratio']}, disjoint from train_5p)"
    )

    dev_data = []
    for path in train_paths:
        dev_data.extend(preprocess_dialogue_file(path, included_services, dev_ids))
    write_subset_outputs(output_dirs, "dev_5p", dev_data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SGD into MultiWOZ-style turn-level DST JSON files."
    )
    # This file lives at data/code/; data assets are one level up under data/.
    data_dir = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--sgd-dir",
        default=str(data_dir / "raw" / "sgd" / "SGD"),
        help="Directory containing SGD train/dev/test/test_20p subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(data_dir),
        help="Directory where MultiWOZ-style SGD files will be written.",
    )
    parser.add_argument(
        "--ids-dir",
        default=str(data_dir / "sgd_ids"),
        help="Directory containing train_5p_ids.json and test_30p_ids.json.",
    )
    parser.add_argument(
        "--extra-output-dir",
        action="append",
        default=[],
        help="Additional output directory. Can be passed multiple times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_sgd_to_multiwoz(
        args.sgd_dir,
        args.output_dir,
        args.ids_dir,
        args.extra_output_dir,
    )


if __name__ == "__main__":
    main()
