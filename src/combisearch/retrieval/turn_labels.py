from combisearch.representation.types import Turn

TurnLabel = str


def turn_label(turn: Turn) -> TurnLabel:
    return f"{turn['ID']}_turn_{turn['turn_id']}"


def label_to_data_item(data_items: list[Turn], label: TurnLabel) -> Turn:
    dialog_id, turn_id = label.split("_turn_")
    turn_id = int(turn_id)
    for data_item in data_items:
        if data_item["ID"] == dialog_id and data_item["turn_id"] == turn_id:
            return data_item
    raise ValueError(f"label {label} not found. check data items input")
