from collections import defaultdict, Counter
from typing import List, Dict

from combisearch.representation.types import Turn, SlotName, MultiWOZDict, SlotValue
from tqdm import tqdm

from combisearch.domain.normalization import AbstractNormalizer
from combisearch.domain.multiwoz.ontology import Ontology
from combisearch.runtime.io import read_json
from combisearch.runtime.io import read_json_resource


class DataOntologyNormalizer(AbstractNormalizer):

    # S" -> C
    ontology: Ontology
    supervised_set: List[Turn]
    # C -> S'
    canonical_to_surface: Dict[SlotName, str]

    def __init__(self, ontology: Ontology,
                 supervised_set: List[Turn] = None,
                 counts_from_ontology_file: str = None,
                 per_occurrence_in_ontology_file: int = 10) -> None:
        """
        Combines two components:
        1) ontology: maps a surface form S" to a canonical DB/schema form C (S" -> C).
        2) counts of gold-label surface forms, used to map each C back to its most likely annotated surface
           form S'. Needed for MultiWOZ-style eval where JGA is exact string match per slot. Sources:
             - supervised_set: training data for this run; counts surface forms in its labels.
             - counts_from_ontology_file: a slot-name -> surface-forms list (commonly named ontology.json in
               other works); each string is counted K=per_occurrence_in_ontology_file times (default 10) with
               no direct dialogue observation. Scoping these sources keeps normalization authentically few-shot.
        """
        super().__init__()
        self.ontology = ontology
        self.canonical_to_surface: Dict[SlotValue, Counter[SlotValue]] = defaultdict(lambda: Counter())
        if supervised_set:
            for turn in tqdm(supervised_set, desc="mapping supervised_set surface forms..."):
                for i, (slot, values) in enumerate(turn['slot_values'].items()):
                    for value in values.split("|"):
                        canonical = ontology.get_canonical(slot, value)
                        if canonical is not None:
                            self.canonical_to_surface[canonical][value] += 1
        if counts_from_ontology_file:
            self.counts_from_ontology_file(counts_from_ontology_file, per_occurence=per_occurrence_in_ontology_file)

    def get_most_common_surface_form(self, slot_name: SlotName, slot_value: SlotValue, keep_counting: bool = False) -> \
            SlotValue:
        """
        Return the most common surface form for the value referenced by slot_value, as determined by labels in
        the 'supervised_set' (in practice train & dev, not test).
        """
        canonical_form: SlotValue = self.ontology.get_canonical(slot_name, slot_value)
        if canonical_form is None:
            return None
        elif canonical_form not in self.canonical_to_surface:
            # in the training set, this surface form was never found. Return as 'most common' and count if permitted
            if keep_counting:
                self.canonical_to_surface[canonical_form][slot_value] += 1
            return slot_value
        else:
            # we have seen this canonical form before, get most common observed surface form for it
            # canonical_form -> Counter -> [(form, count)] -> (form, count) -> form
            return self.canonical_to_surface[canonical_form].most_common(1)[0][0]

    def normalize(self, raw_parse: MultiWOZDict, **kwargs) -> MultiWOZDict:
        """
        Normalize slot values in raw_parse to best match the surface forms expected in the evaluation set
        (determined by the supervised data given at construction). May omit existing slots.
        """
        normalized: MultiWOZDict = {}
        for slot_name, slot_value in raw_parse.items():
            if not self.ontology.is_valid_slot(slot_name):
                continue
            if type(slot_value) == str:
                slot_value = slot_value.split("|")[0]
            normalized_form: SlotValue = self.get_most_common_surface_form(slot_name, slot_value)
            if normalized_form:
                normalized[slot_name] = normalized_form
            elif not self.ontology.is_categorical(slot_name) and not self.ontology.is_name(slot_name):
                # preserve prediction values for non-categorical slots, even if they are wrong
                normalized[slot_name] = slot_value
        return normalized

    def counts_from_ontology_file(self, ontology_file: str, per_occurence: int = 10) -> None:
        """
        Read surface forms from an ontology.json (e.g. MultiWOZ 2.4 data repo), counting each per_occurence times.
        """
        try:
            ontology_data: Dict[SlotName, List[str]] = read_json_resource(ontology_file)
        except BaseException as e:
            ontology_data: Dict[SlotName, List[str]] = read_json(ontology_file)

        for slot_name, slot_value_strings in tqdm(ontology_data.items(),
                                                  desc=f"reading surface forms from ontology.json"):
            for slot_value_string in slot_value_strings:
                for slot_value in slot_value_string.split("|"):
                    canonical = self.ontology.get_canonical(slot_name, slot_value)
                    if canonical is not None:
                        self.canonical_to_surface[canonical][slot_value] += per_occurence
        return
