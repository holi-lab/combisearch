"""Dialogue-state-tracking evaluation: joint goal accuracy, slot accuracy, and slot F1."""
import enum
from collections import namedtuple, Counter
from typing import List, Dict

from combisearch.representation.types import MultiWOZDict

EvalResult = namedtuple('EvalResult', ['jga', 'acc', 'f1'])
PRFResult = namedtuple('PRFResult', ['f1', 'precision', 'recall'])

# counts of informable slots for each domain
INFORMABLE_SLOTS_BY_DOMAIN: Dict[str, int] = {
    "attraction": 3,
    "hotel": 10,
    "restaurant": 7,
    "taxi": 4,
    "train": 6
}

# inheriting from str + enum.Enum allows for painless JSON serialization
class F1Class(str, enum.Enum):
    TP = 'tp'
    FP = 'fp'
    FN = 'fn'


def calc_prf(counter: Counter[F1Class]) -> PRFResult:
    precision_denom: int = (counter[F1Class.TP] + counter[F1Class.FP])
    precision: float = counter[F1Class.TP] / precision_denom if precision_denom else 0
    recall_denom: int = (counter[F1Class.TP] + counter[F1Class.FN])
    recall: float = counter[F1Class.TP] / recall_denom if recall_denom else 0
    f1: float = 0
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    return PRFResult(f1, precision, recall)


def compute_prf(gold: MultiWOZDict, pred: MultiWOZDict) -> PRFResult:
    """f1, precision, recall for a turn-level prediction over normalized MultiWOZ slots."""
    scores: Counter[F1Class] = Counter()
    for gold_slot, gold_value in gold.items():
        if gold_slot in pred and pred[gold_slot] == gold_value:
            scores[F1Class.TP] += 1
        else:
            scores[F1Class.FN] += 1
    for pred_slot, pred_value in pred.items():
        if pred_slot not in gold:
            scores[F1Class.FP] += 1
    return calc_prf(counter=scores)


def compute_acc(gold: MultiWOZDict, pred: MultiWOZDict, number_of_informable_slots: int = 30) -> float:
    """
    Slot accuracy for a turn-level prediction over normalized MultiWOZ slots. The total number of informable
    slots must be known so that correctly-absent entries in both prediction and gold count as accurate
    (number_of_informable_slots is 30 in IC-DST experiments).
    """
    false_negatives: int = 0
    missed_slots: List[str] = []
    for domain_and_slot_name in gold:
        if domain_and_slot_name not in pred or gold[domain_and_slot_name] != pred[domain_and_slot_name]:
            false_negatives += 1
            missed_slots.append(domain_and_slot_name.rsplit("-", 1)[0])
    false_positives = 0
    for domain_and_slot_name in pred:
        if domain_and_slot_name not in gold:
            false_positives += 1
    acc: float = (number_of_informable_slots - false_negatives - false_positives) / number_of_informable_slots
    return acc


def evaluate(prediction: MultiWOZDict, gold: MultiWOZDict) -> EvalResult:
    """Evaluate a single prediction against a gold reference (standard MultiWOZ format); returns (jga, acc, f1)."""

    for key in gold.keys():
        # if the gold value supports multiple ground truth values, and we predicted one, set the single-gold value to
        # the one we predicted.
        if '|' in gold[key]:
            gold_values = gold[key].split('|')
            if key in prediction and prediction[key] in gold_values:
                gold[key] = prediction[key]

    jga: int = 1 if prediction == gold else 0

    acc = compute_acc(gold, prediction)
    f1 = compute_prf(gold, prediction)[0]
    return jga, acc, f1


def evaluate_on_domain(prediction: MultiWOZDict, gold: MultiWOZDict, domain: str) -> EvalResult:
    """
    Evaluate a single prediction against a gold reference on one MultiWOZ domain (attraction, hotel, taxi,
    train, restaurant). No need to pre-filter slots from other domains. Returns (jga, acc, f1) for that domain.
    """
    prediction = {k: v for k, v in prediction.items() if k.split("-")[0] == domain}
    gold = {k: v for k, v in gold.items() if k.split("-")[0] == domain}

    for key in gold.keys():
        # if the gold value supports multiple ground truth values, and we predicted one, set the single-gold value to
        # the one we predicted.
        if '|' in gold[key]:
            gold_values = gold[key].split('|')
            if key in prediction and prediction[key] in gold_values:
                gold[key] = prediction[key]

    jga: int = 1 if prediction == gold else 0

    acc = compute_acc(gold, prediction, number_of_informable_slots=INFORMABLE_SLOTS_BY_DOMAIN[domain])
    f1 = compute_prf(gold, prediction, )[0]
    return jga, acc, f1