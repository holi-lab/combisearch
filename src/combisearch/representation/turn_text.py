from typing import Callable, List, Union

from combisearch.representation.types import MultiWOZDict, Turn
from combisearch.prompting.parsing.python.python_serialization import (
    SLOT_NAME_REVERSE_REPLACEMENTS,
    get_state_reference,
    normalize_to_domains_and_slots,
)

DOMAINS = ["hotel", "restaurant", "attraction", "taxi", "train"]


def important_value_to_string(slot, value):
    if value in ["none", "dontcare"]:
        return f"{slot}{value}"
    return f"{slot}-{value}"


StateTransformationFunction = Callable[[Turn], Union[List[str], MultiWOZDict]]


def default_state_transform(turn: Turn) -> List[str]:
    return [
        important_value_to_string(slot, value)
        for slot, value in turn["turn_slot_values"].items()
        if slot.split("-")[0] in DOMAINS
    ]


def reference_aware_state_transform(turn: Turn) -> List[str]:
    return _reference_aware_state_transform(turn, reference_prefix="state.", reference_separator=".")


def reference_aware_nl_state_transform(turn: Turn) -> List[str]:
    return _reference_aware_state_transform(turn, reference_prefix="", reference_separator="_")


def _reference_aware_state_transform(
    turn: Turn,
    *,
    reference_prefix: str,
    reference_separator: str,
) -> List[str]:
    context_slot_values: MultiWOZDict = {
        slot: value.split("|")[0] for slot, value in turn["last_slot_values"].items()
    }
    context_domains_to_slot_pairs = normalize_to_domains_and_slots(context_slot_values)
    turn_domains_to_slot_pairs = normalize_to_domains_and_slots(turn["turn_slot_values"])
    user_str = turn["dialog"]["usr"][-1]
    sys_str = turn["dialog"]["sys"][-1]
    sys_str = sys_str if sys_str and sys_str != "none" else ""

    new_dict: MultiWOZDict = {}
    for domain, slot_pairs in turn_domains_to_slot_pairs.items():
        for norm_slot_name, norm_slot_value in slot_pairs.items():
            mwoz_slot_name = SLOT_NAME_REVERSE_REPLACEMENTS.get(
                norm_slot_name, norm_slot_name.replace("_", " ")
            )
            mwoz_slot_value = str(norm_slot_value)
            reference = get_state_reference(
                context_domains_to_slot_pairs,
                domain,
                norm_slot_name,
                norm_slot_value,
                turn_strings=[sys_str, user_str],
            )
            if reference is not None:
                referred_domain, referred_slot = reference
                reference_value = f"{reference_prefix}{referred_domain}{reference_separator}{referred_slot}"
                new_dict[f"{domain}-{mwoz_slot_name}"] = reference_value
            else:
                new_dict[f"{domain}-{mwoz_slot_name}"] = mwoz_slot_value

    return [
        important_value_to_string(slot, value)
        for slot, value in new_dict.items()
        if slot.split("-")[0] in DOMAINS
    ]


_STATE_TRANSFORMATIONS: dict[str, StateTransformationFunction] = {
    "default": default_state_transform,
    "ref_aware": reference_aware_state_transform,
    "ref_aware_nl": reference_aware_nl_state_transform,
}


def get_state_transformation_by_type(transformation_type: str) -> StateTransformationFunction:
    fn = _STATE_TRANSFORMATIONS.get(transformation_type or "default")
    if fn is None:
        raise ValueError(f"Unsupported state transformation type: {transformation_type}")
    return fn


def get_string_transformation_by_type(transformation_type: str):
    if not transformation_type or transformation_type == "default":
        return data_item_to_string
    raise ValueError(f"Unsupported string transformation type: {transformation_type}")


def _slot_dict_to_nl(slot_value_dict, prefix: str) -> str:
    output = prefix
    for slot, value in slot_value_dict.items():
        output += f"{' '.join(slot.split('-'))}: {value.split('|')[0]}, "
    return output


def state_to_nl(slot_value_dict):
    return _slot_dict_to_nl(slot_value_dict, "[CONTEXT] ")


def gt_to_nl(slot_value_dict):
    return _slot_dict_to_nl(slot_value_dict, " [BS] ")


def input_to_string(context_dict, sys_utt, usr_utt):
    history = state_to_nl(context_dict)
    sys_utt = "" if sys_utt == "none" else sys_utt
    usr_utt = "" if usr_utt == "none" else usr_utt
    return f"{history} [SYS] {sys_utt} [USER] {usr_utt}"


_GROUND_TRUTH_BUILDERS = {
    "gt_full_bs":    lambda d: gt_to_nl(d["slot_values"]),
    "gt_full_slot":  lambda d: " [BS] " + ", ".join(list(d["slot_values"].keys())),
    "gt_delta_bs":   lambda d: gt_to_nl(d["turn_slot_values"]),
    "gt_delta_slot": lambda d: " [BS] " + ", ".join(list(d["turn_slot_values"].keys())),
}


def get_ground_truth_by_type(data_item, gt_type: str | None):
    if gt_type is None:
        return ""
    builder = _GROUND_TRUTH_BUILDERS.get(gt_type)
    if builder is None:
        raise ValueError(f"Unsupported ground truth type: {gt_type}")
    return builder(data_item)


def data_item_to_string(
    data_item: Turn,
    string_transformation=input_to_string,
    full_history: bool = False,
    **kwargs,
) -> str:
    input_type = kwargs.get("input_type", "dialog_context")
    only_slot = kwargs.get("only_slot", False)
    gt_type = kwargs.get("gt_type", None)

    if input_type == "dialog_context":
        if full_history:
            history = ""
            for sys_utt, usr_utt in zip(data_item["dialog"]["sys"], data_item["dialog"]["usr"]):
                history += string_transformation({}, sys_utt, usr_utt)
            return history + get_ground_truth_by_type(data_item, gt_type)

        if only_slot:
            context = list(data_item["last_slot_values"].keys())
            sys_utt = data_item["dialog"]["sys"][-1]
            usr_utt = data_item["dialog"]["usr"][-1]
            sys_utt = "" if sys_utt == "none" else sys_utt
            usr_utt = "" if usr_utt == "none" else usr_utt
            history = "[CONTEXT] " + ", ".join(context)
            return f"{history} [SYS] {sys_utt} [USER] {usr_utt}" + get_ground_truth_by_type(data_item, gt_type)

        context = data_item["last_slot_values"]
        sys_utt = data_item["dialog"]["sys"][-1]
        usr_utt = data_item["dialog"]["usr"][-1]
        return string_transformation(context, sys_utt, usr_utt) + get_ground_truth_by_type(data_item, gt_type)

    if input_type == "context":
        if full_history:
            if len(data_item["dialog"]["sys"]) == 1:
                return "[CONTEXT]  [SYS]  [USER] "
            history = ""
            for sys_utt, usr_utt in zip(data_item["dialog"]["sys"][:-1], data_item["dialog"]["usr"][:-1]):
                history += string_transformation({}, sys_utt, usr_utt)
            return history + get_ground_truth_by_type(data_item, gt_type)

        if only_slot:
            context = list(data_item["last_slot_values"].keys())
            history = "[CONTEXT] " + ", ".join(context)
            return history + " [SYS]  [USER] " + get_ground_truth_by_type(data_item, gt_type)

        context = data_item["last_slot_values"]
        return string_transformation(context, "", "") + get_ground_truth_by_type(data_item, gt_type)

    if input_type == "dialog":
        sys_utt = data_item["dialog"]["sys"][-1]
        usr_utt = data_item["dialog"]["usr"][-1]
        return string_transformation({}, sys_utt, usr_utt) + get_ground_truth_by_type(data_item, gt_type)

    if "gt" in input_type:
        return get_ground_truth_by_type(data_item, input_type)

    raise ValueError(f"Unsupported input_type: {input_type}")
