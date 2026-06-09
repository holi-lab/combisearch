import logging
import pprint

from combisearch.representation.types import MultiWOZDict, SlotName


def parse_nl_completion(nl_completion: str, state = None,
                            exceptions_are_empty: bool = True, **kwargs) -> MultiWOZDict:
    """
    Parse a natural language completion into a structured dialogue state dictionary, y_t.
    
    This function takes text like: 
        "{'restaurant-name': 'pizza hut', 'restaurant-time': '7pm'}" or "The value of slot book time of restaurant is 11:15."
    and converts it into a proper Python dictionary for MultiWOZ.
    
    :param nl_completion: The natural language text containing slot-value pairs
    :param state: The existing dialogue state (y_{t-1}, not used here, but kept for API consistency)
    :param exceptions_are_empty: If True, parsing errors result in empty dict instead of raising exception
    :param kwargs: Additional arguments (not used, but kept for compatibility)
    :return: Dictionary mapping slot names to their values y_t

    Example input:
        "{'attraction-name': 'cineworld cinema', 'train-book people': '6', 'train-destination': 'cambridge'}"
    Example output:
        {
            'attraction-name': 'cineworld cinema',
            'train-book people': '6',
            'train-destination': 'cambridge',
        }
    """
    bs_dict = {}
    try:
        full_statement = nl_completion.strip()
        full_statement = full_statement.replace('{', '').replace('}', '')

        for d_s_v in full_statement.split(','):
            d_s = eval(d_s_v.split(":")[0])
            v = d_s_v.split(":")[1:]  # Handle values that might contain colons
            v = str(eval(":".join(v)))

            try:
                assert f"{d_s}" in SlotName.__args__
            except AssertionError:
                print(f"{d_s} not found in SlotName")
                continue
            bs_dict[d_s] = v

        return bs_dict
    except Exception as e:
        print(f"got exception when parsing: {pprint.pformat(e)}")
        logging.warning(e)
        if not exceptions_are_empty:
            raise e
        return bs_dict


def parse_sgd_nl_completion(nl_completion: str, state = None,
                            exceptions_are_empty: bool = True, **kwargs) -> MultiWOZDict:
    """
    Parse a natural language completion for SGD (Schema-Guided Dialogue) dataset.
    
    Similar to parse_nl_completion but designed for SGD's different slot naming convention
    (e.g., "Hotels_2-check_in_date" instead of "hotel-check in date")
    
    :param nl_completion: The natural language text containing slot-value pairs
    :param state: The existing dialogue state (not used here, but kept for API consistency)
    :param exceptions_are_empty: If True, parsing errors result in empty dict instead of raising exception
    :param kwargs: Additional arguments (not used, but kept for compatibility)
    :return: Dictionary mapping SGD slot names to their values
    """
    bs_dict = {}
    try:
        full_statement = nl_completion.strip()
        full_statement = full_statement.replace('{', '').replace('}', '')

        for d_s_v in full_statement.split(','):
            d_s = eval(d_s_v.split(":")[0])
            v = d_s_v.split(":")[1:]
            v = str(eval(":".join(v)))
            bs_dict[d_s] = v

        return bs_dict
    except Exception:
        return bs_dict

