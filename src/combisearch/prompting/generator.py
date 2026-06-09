"""
CombiSearch Prompting Module

This module handles the generation of prompts for dialogue state tracking experiments.
It supports Python-based and natural language prompt formats for MultiWOZ and SGD datasets.
"""

import random
from typing import List, Dict

from combisearch.representation.types import CompletionParser

import combisearch.prompting.parsing.python.python_serialization as python_fmt
from combisearch.runtime.io import read_package_text
from combisearch.prompting.parsing.python.completion_parser import parse_python_completion

from combisearch.prompting.parsing.natural_language.nl_completion_parser import parse_nl_completion, parse_sgd_nl_completion

PYTHON_PROMPT: str = "python-prompt"
PYTHON_PROMPT_NULL: str = "python-prompt-NULL"

STOP_SEQUENCES: Dict[str, List[str]] = {
    PYTHON_PROMPT: ["\n\n", "#", "print("],
}

SGD_PROMPT_DICTS= {
    "Alarm_1": " **Alarm_1**\n  - **Description**: Manage alarms by getting and setting them easily\n  - **Slots**:\n      - alarm_time, new_alarm_time (time)\n      - alarm_name, new_alarm_name (text)\n",
    "Buses_3": " **Buses_3**\n  - **Description**: Affordable and comfortable bus travel across the country\n  - **Slots**:\n      - departure_date (date)\n      - departure_time (time)\n      - price (numerical)\n      - from_city, to_city, from_station, to_station, additional_luggage, num_passengers, category (text)\n",
    "Events_3": " **Events_3**\n  - **Description**: Find and book tickets to any cultural events in your area\n  - **Slots**:\n      - date (date)\n      - time (time)\n      - price_per_ticket (numerical)\n      - event_type, event_name, number_of_tickets, city, venue, venue_address (text)\n",
    "Flights_4": " **Flights_4**\n  - **Description**: Find cheap flights in seconds and book flights\n  - **Slots**:\n      - departure_date, return_date (date)\n      - outbound_departure_time, outbound_arrival_time, inbound_arrival_time, inbound_departure_time (time)\n      - price (numerical)\n      - number_of_tickets, seating_class, origin_airport, destination_airport, is_nonstop, airlines (text)\n",
    "Homes_2": " **Homes_2**\n  - **Description**: Service for finding properties to buy and rent\n  - **Slots**:\n      - visit_date (date)\n      - price (numerical)\n      - phone_number (phone)\n      - intent, area, address, property_name, has_garage, in_unit_laundry, number_of_beds, number_of_baths (text)\n",
    "Hotels_2": " **Hotels_2**\n  - **Description**: A popular service for searching and booking houses for short term stay\n  - **Slots**:\n      - check_in_date, check_out_date (date)\n      - rating, total_price (numerical)\n      - phone_number (phone)\n      - where_to, number_of_adults, address, has_laundry_service (text)\n",
    "Hotels_4": " **Hotels_4**\n  - **Description**: Accommodation searching and booking portal\n  - **Slots**:\n      - check_in_date (date)\n      - price_per_night (numerical)\n      - phone_number (phone)\n      - location, number_of_rooms, stay_length, star_rating, place_name, street_address, smoking_allowed (text)\n",
    "Media_3": " **Media_3**\n  - **Description**: Enjoy instant and unlimited access to best shows, movies, comedy, sports, documentaries and more.\n  - **Slots**:\n      - title, genre, subtitle_language, starring (text)\n",
    "Messaging_1": " **Messaging_1**\n  - **Description**: Connect and share locations with your contacts\n  - **Slots**:\n      - location, contact_name (text)\n",
    "Movies_1": " **Movies_1**\n  - **Description**: A go-to provider for finding movies, searching for show times and booking tickets\n  - **Slots**:\n      - show_date (date)\n      - show_time (time)\n      - price (numerical)\n      - number_of_tickets, show_type, theater_name, genre, street_address, location, movie_name (text)\n",
    "Movies_3": " **Movies_3**\n  - **Description**: A review-aggregation website for movies and television\n  - **Slots**:\n      - percent_rating (numerical)\n      - movie_title, genre, cast, directed_by (text)\n",
    "Music_3": " **Music_3**\n  - **Description**: A free, personalized platform that plays music you'll love. Discover new music and enjoy old favorites.\n  - **Slots**:\n      - track, artist, album, genre, year, device (text)\n",
    "Payment_1": " **Payment_1**\n  - **Description**: The fast, simple way to pay in apps, on the web, and in millions of stores\n  - **Slots**:\n      - receiver (count)\n      - amount (numerical)\n      - payment_method, private_visibility (text)\n",
    "RentalCars_3": " **RentalCars_3**\n  - **Description**: A leading global provider of car rental solutions\n  - **Slots**:\n      - start_date, end_date (date)\n      - pickup_time (time)\n      - price_per_day (numerical)\n      - car_type, car_name, pickup_location, city, add_insurance (text)\n",
    "Restaurants_2": " **Restaurants_2**\n  - **Description**: A popular restaurant search and reservation service\n  - **Slots**:\n      - date (date)\n      - time (time)\n      - rating (numerical)\n      - phone_number (phone)\n      - restaurant_name, has_seating_outdoors, has_vegetarian_options, address, number_of_seats, price_range, location, category (text)\n",
    "RideSharing_2": " **RideSharing_2**\n  - **Description**: App to book a cab to any destination\n  - **Slots**:\n      - wait_time (time)\n      - ride_fare (numerical)\n      - destination, ride_type, number_of_seats (text)\n",
    "Services_1": " **Services_1**\n  - **Description**: A widely used service for finding and reserving the hair stylist of your choice\n  - **Slots**:\n      - appointment_date (date)\n      - appointment_time (time)\n      - average_rating (numerical)\n      - phone_number (phone)\n      - stylist_name, is_unisex, street_address, city (text)\n",
    "Services_4": " **Services_4**\n  - **Description**: Discover the right therapist for you and make reservations easily\n  - **Slots**:\n      - appointment_date (date)\n      - appointment_time (time)\n      - phone_number (phone)\n      - therapist_name, address, city, type (text)\n",
    "Trains_1": " **Trains_1**\n  - **Description**: Service to find and reserve train journeys between cities\n  - **Slots**:\n      - date_of_journey (date)\n      - journey_start_time (time)\n      - total (numerical)\n      - from, to, from_station, to_station, number_of_adults, class, trip_protection (text)\n",
    "Travel_1": " **Travel_1**\n  - **Description**: The biggest database of tourist attractions and points of interest\n  - **Slots**:\n      - phone_number (phone)\n      - location, attraction_name, category, free_entry, good_for_kids (text)\n",
    "Weather_1": " **Weather_1**\n  - **Description**: Check the weather for any place and any date\n  - **Slots**:\n      - date (date)\n      - precipitation, humidity, wind, temperature, city (text)\n"
}


def get_completion_parser(prompt_format: str) -> CompletionParser:
    """
    Factory function to get the appropriate parser for a given prompt format.
    
    This helps us select the right parsing function based on how we formatted the prompt.
    Different prompt formats need different parsers to extract the dialogue state correctly.
    
    :param prompt_format: The format of the prompt (e.g., "python-prompt", "sgd", etc.)
    :return: The appropriate parsing function for that format
    """
    if prompt_format in (PYTHON_PROMPT, PYTHON_PROMPT_NULL):
        return parse_python_completion
    if 'sgd' in prompt_format.lower():
        return parse_sgd_nl_completion
    return parse_nl_completion


class CombiSearchPromptGenerator:
    """
    A friendly helper class for creating prompts for dialogue state tracking experiments!
    
    This class handles the complex task of formatting dialogue data into prompts that
    language models can understand. It supports multiple prompt styles:
    - Python: Code-based format (e.g., "agent.state.restaurant_name = 'pizza hut'")
    - Natural Language: Plain text descriptions of the dialogue state

    The class helps with both training (creating examples) and inference (testing on new data).
    """
    preambles: Dict[str, str]

    def __init__(self):
        """Initialize the prompt generator by loading preambles for different formats."""
        self.preambles = {
            PYTHON_PROMPT: read_package_text("prompting/formats/python/python_classes.py"),
            PYTHON_PROMPT_NULL: read_package_text("prompting/formats/python/python_classes.py"),
        }

    def get_prompt(
        self, 
        data_item, examples, given_context=None, n_examples=None, 
        prompt_format: str = None, chat_format: bool=False, add_guidelines: bool = True, add_instruction_every_turn: bool = False):
        """
        The main entry point for generating prompts! This function orchestrates everything.
        
        :param data_item: The current dialogue turn we want to make a prediction for
        :param examples: List of example dialogues to use for in-context learning (few-shot)
        :param given_context: Optional custom context (dialogue state) to use instead of the one in data_item
        :param n_examples: How many examples to include (None = use all provided examples)
        :param prompt_format: Which format to use (PYTHON_PROMPT or natural language)
        :param chat_format: If True, format as a chat conversation; if False, use completion format
        :param add_guidelines: If True, include helpful schema/guideline information
        :param add_instruction_every_turn: If True, repeat instructions for each turn (useful for clarity)
        :return: A formatted prompt string or list of chat messages
        """
        if prompt_format in (PYTHON_PROMPT, PYTHON_PROMPT_NULL):
            if prompt_format == PYTHON_PROMPT_NULL:
                reverse_x_and_y, use_null_data_item = True, True
            else:
                reverse_x_and_y, use_null_data_item = False, False
            detailed_state_string: bool = prompt_format == PYTHON_PROMPT
            
            final_prompt = self.get_python_chat_prompt(
                data_item, examples, given_context=given_context, n_examples=n_examples,
                reverse_x_and_y=reverse_x_and_y, use_null_data_item=use_null_data_item, 
                detailed_state_string=detailed_state_string, add_guidelines=add_guidelines,
            )
        
        else:
            if not chat_format:
                raise ValueError("Prompt with Plain_text only support chat template. Set chat_format=True")
            elif  "sgd" in prompt_format.lower():
                final_prompt = self.get_sgd_chat_prompt(
                    data_item, examples, given_context=given_context, n_examples=n_examples,
                    reverse_x_and_y=False, use_null_data_item=False, detailed_state_string=True, 
                    add_guidelines=add_guidelines, add_instruction_every_turn=add_instruction_every_turn
                )
            else:
                final_prompt = self.get_nl_chat_prompt(
                    data_item, examples, given_context=given_context, n_examples=n_examples,
                    reverse_x_and_y=False, use_null_data_item=False, detailed_state_string=True, 
                    add_guidelines=add_guidelines, add_instruction_every_turn=add_instruction_every_turn
                )
        return final_prompt

    def get_python_chat_prompt(
        self, data_item, examples, given_context=None, n_examples: int = None,
        reverse_x_and_y: bool = False, use_null_data_item: bool = False,
        detailed_state_string: bool = True, add_guidelines:bool = True) -> str:
    
        msg = [{"role": "system", "content": "You are an expert in Dialogue State Tracking(DST) and python coding.\n"}]
        max_n_examples: int = n_examples is not None and n_examples or len(examples)

        if max_n_examples > 0:
            for example_id, example in enumerate(examples[-max_n_examples:]):
                prefix_msg = f"\n    #### Example {example_id + 1} ####\n"
                
                # remove multiple choice in last slot values
                last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}

                last_sys_utt = example['dialog']['sys'][-1]
                if last_sys_utt == 'none':
                    last_sys_utt = ''
                user_string = python_fmt.get_user_string(example['dialog']['usr'][-1])
                state_string, update_string = python_fmt.get_python_statements(last_slot_values, example['slot_values'],
                                                                                turn_strings=[last_sys_utt, user_string],
                                                                                detailed_state_string=detailed_state_string)
                turn_msg = "    " + state_string + "\n"
                if last_sys_utt:
                    turn_msg += "    " + python_fmt.get_system_string(last_sys_utt) + "\n"
                turn_msg += "    " + user_string + "\n"
                
                bs_msg = ''
                for s in update_string.split("\n"):
                    bs_msg += "    " + s.strip() + "\n"
                if not reverse_x_and_y:
                    msg.append({"role": "user","content": prefix_msg+turn_msg})
                    msg.append({"role": "assistant","content": bs_msg})
                else:
                    msg.append({"role": "user", "content": prefix_msg+bs_msg})
                    msg.append({"role": "assistant","content": turn_msg})

        prefix_msg = f"\n    #### Example {max_n_examples + 1} ####\n"
        if given_context is None:
            last_slot_values = {s: v.split('|')[0] for s, v in data_item['last_slot_values'].items()}
        else:
            last_slot_values = given_context
        last_sys_utt = data_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        user_string = python_fmt.get_user_string(data_item['dialog']['usr'][-1])
        state_string, _ = python_fmt.get_python_statements(last_slot_values, {},
                                                            turn_strings=[last_sys_utt, user_string],
                                                            detailed_state_string=detailed_state_string)
        turn_msg = ''
        if not use_null_data_item:
            turn_msg += "    " + state_string + "\n"
            if last_sys_utt:
                turn_msg += "    " + python_fmt.get_system_string(last_sys_utt) + "\n"
            turn_msg += "    " + user_string + "\n"
        else:
            pass  # default adds our null input at end
        msg.append({"role": "user","content": prefix_msg+turn_msg})
        if add_guidelines:
            msg[1]['content'] = self.preambles[PYTHON_PROMPT] + "    \n" + msg[1]['content']
        return msg
        
    def get_nl_chat_prompt(
            self, data_item, examples, given_context=None, n_examples: int = None,
            reverse_x_and_y: bool = False, use_null_data_item: bool = False,
            detailed_state_string: bool = True, add_guidelines:bool = True, add_instruction_every_turn: bool = True
    ) -> str:
        system_msg = "**Task:** You are an expert in Dialogue State Tracking (DST) focused on managing and updating the dialogue state change based on system-user interactions. "
        system_msg += "The dialogue state represents the user's preferences and booking details across different domains: Hotel, Train, Attraction, Restaurant, and Taxi.\n\n"
        msg = [{"role": "system", "content": system_msg}]
        max_n_examples: int = n_examples is not None and n_examples or len(examples)

        if max_n_examples > 0:
            for example_id, example in enumerate(examples[-max_n_examples:]):
                prefix_msg = f"\n**Example {example_id + 1} of Dialogue State Change Update Task:**\n"
                
                # remove multiple choice in last slot values
                last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}
                turn_slot_values = {s: v.split('|')[0] for s, v in example['turn_slot_values'].items()}

                last_sys_utt = example['dialog']['sys'][-1]
                if last_sys_utt == 'none':
                    last_sys_utt = ''
                
                state_string = '    **Previous Belief State (Before the Latest User Interaction):** \n'
                state_string += f"        {last_slot_values}\n\n"
                
                turn_msg = state_string + "    **Latest Conversation Between System and User:** \n"
                if last_sys_utt:
                    turn_msg += '        **System:** "' + last_sys_utt + '"\n'
                
                turn_msg += '        **User:** "' + example['dialog']['usr'][-1] + '"\n\n'
                if add_instruction_every_turn:
                    turn_msg += '    **Instructions:**\n'
                    turn_msg += '        - Based on the user\'s latest input, update the belief state by correctly identifying and filling in the relevant domain(s), slot(s) and value(s).\n'
                    turn_msg += '        - Provide your output strictly in the Required Output Format below.\n\n'
                    turn_msg += "    **Required Output Format:**\n"
                    turn_msg += "        **Dialogue state change after Latest Conversation Between System and User:** \n"
                    turn_msg += '            { "DOMAIN_1-SLOT_1": "VALUE_1", "DOMAIN_2-SLOT_2": "VALUE_2", ... }\n'
                bs_msg = f"            {turn_slot_values}\n"
                if not reverse_x_and_y:
                    msg.append({"role": "user","content": prefix_msg+turn_msg})
                    msg.append({"role": "assistant","content": bs_msg})
                else:
                    msg.append({"role": "user", "content": prefix_msg+bs_msg})
                    msg.append({"role": "assistant","content": turn_msg})

        prefix_msg = f"\n**Example {max_n_examples + 1} of Dialogue State Change Update Task:**\n"
        if given_context is None:
            last_slot_values = {s: v.split('|')[0] for s, v in data_item['last_slot_values'].items()}
        else:
            last_slot_values = given_context
        
        last_sys_utt = data_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        
        state_string = '    **Previous Belief State (Before the Latest User Interaction):** \n'
        state_string += f"        {last_slot_values}\n\n"

        turn_msg = ''
        if not use_null_data_item:
            turn_msg = state_string + "    **Latest Conversation Between System and User:** \n"
            if last_sys_utt:
                    turn_msg += '        **System:** "' + last_sys_utt + '"\n'
            turn_msg += '        **User:** "' + data_item['dialog']['usr'][-1] + '"\n\n'
            if add_instruction_every_turn:
                turn_msg += '    **Instructions:**\n'
                turn_msg += '        - Based on the user\'s latest input, update the belief state by correctly identifying and filling in the relevant domain(s), slot(s) and value(s).\n'
                turn_msg += '        - Provide your output strictly in the Required Output Format below.\n\n'
                turn_msg += "    **Required Output Format:**\n"
                turn_msg += "        **Dialogue state change after Latest Conversation Between System and User:** \n"
                turn_msg += '            { "DOMAIN_1-SLOT_1": "VALUE_1", "DOMAIN_2-SLOT_2": "VALUE_2", ... }\n'
        else:
            pass  # default adds our null input at end
        msg.append({"role": "user","content": prefix_msg+turn_msg})
        if add_guidelines:
            tmp_msg_content = msg[1]['content']
            msg[1]['content'] = '**Guidelines:**\n\n'
            msg[1]['content'] +='    1. **Hotel:**\n'
            msg[1]['content'] +='        - **Slots:** name (string), pricerange (PriceRange), type (HotelType), parking (Option), book stay (integer), book day (DayOfWeek), book people (integer), area (Area), stars (integer between 0 and 5 or "dontcare"), internet (Option).\n'
            msg[1]['content'] +='        - **Valid Values:**\n'
            msg[1]['content'] +='            - PriceRange: "dontcare", "cheap", "moderate", "expensive".\n'
            msg[1]['content'] +='            - HotelType: "hotel", "guest house", "dontcare".\n'
            msg[1]['content'] +='            - Option: "yes", "no", "dontcare".\n'
            msg[1]['content'] +='            - DayOfWeek: "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday".\n'
            msg[1]['content'] +='            - Area: "dontcare", "centre", "east", "north", "south", "west".\n\n'
            msg[1]['content'] +='    2. **Train:**\n'
            msg[1]['content'] +='        - **Slots:** destination (string), departure (string), day (DayOfWeek), book people (integer), leaveat (hh:mm or "dontcare"), arriveby (hh:mm or "dontcare").\n\n'
            msg[1]['content'] +='    3. **Attraction:**\n'
            msg[1]['content'] +='        - **Slots:** name (string), area (Area), type (AttractionType).\n'
            msg[1]['content'] +='        - **Valid Values:**\n'
            msg[1]['content'] +='            - AttractionType: "architecture", "boat", "church", "cinema", "college", "concert hall", "entertainment", "hotspot", "multiple sports", "museum", "nightclub", "park", "special", "swimming pool", "theatre", "dontcare".\n\n'
            msg[1]['content'] +='    4. **Restaurant:**\n'
            msg[1]['content'] +='        - **Slots:** name (string), food (string), pricerange (PriceRange), area (Area), book time (hh:mm or "dontcare"), book day (DayOfWeek), book people (integer).\n\n'
            msg[1]['content'] +='    5. **Taxi:**\n'
            msg[1]['content'] +='        - **Slots:** destination (string), departure (string), leaveat (hh:mm or "dontcare"), arriveby (hh:mm or "dontcare").\n\n' 
            msg[1]["content"] += tmp_msg_content
        return msg
    
    def get_sgd_chat_prompt(
        self, data_item, examples, given_context=None, n_examples: int = None,
        reverse_x_and_y: bool = False, use_null_data_item: bool = False,
        detailed_state_string: bool = True, add_guidelines:bool = True, add_instruction_every_turn: bool = True
    ) -> str:
        system_msg ="You are an expert dialogue state tracker assisting users across multiple services such as Bank, Bus, Calender, Event, Flight, Home, Hotel, Media, Movie, Music, RentalCars, Restaurant, RideSharing, Service, Travel, Weather. Your primary tasks are:\n"
        system_msg += "1. **Understand the User's Request**: Accurately identify the service and intent based on the user's input.\n"
        system_msg += "2. **Extract Required Information**: Fill in the necessary slots using information provided by the user or apply default values when appropriate.\n"
        system_msg += "3. **Ensure Data Accuracy**: Validate that all slot values match the expected data types and predefined options.\n"
        system_msg += "4. **Provide a Structured Output**: Present the updated dialogue state strictly in the specified JSON format without any additional text.\n\n---\n\n"


        msg = [{"role": "system", "content": system_msg}]
        max_n_examples: int = n_examples is not None and n_examples or len(examples)

        if max_n_examples > 0:
            for example_id, example in enumerate(examples[-max_n_examples:]):
                prefix_msg = f"\n**Example {example_id + 1} of Dialogue State Change Update Task:**\n"
                
                # remove multiple choice in last slot values
                last_slot_values = {s: v.split('|')[0] for s, v in example['last_slot_values'].items()}
                turn_slot_values = {s: v.split('|')[0] for s, v in example['turn_slot_values'].items()}

                last_sys_utt = example['dialog']['sys'][-1]
                if last_sys_utt == 'none':
                    last_sys_utt = ''
                
                state_string = '    **Previous Belief State (Before the Latest User Interaction):** \n'
                state_string += f"        {last_slot_values}\n\n"
                
                turn_msg = state_string + "    **Latest Conversation Between System and User:** \n"
                if last_sys_utt:
                    turn_msg += '        **System:** "' + last_sys_utt + '"\n'
                
                turn_msg += '        **User:** "' + example['dialog']['usr'][-1] + '"\n\n'
                if add_instruction_every_turn:
                    turn_msg += '    **Task Instructions:**\n'
                    turn_msg += '    1. **Identify Service and Intent**: Match user input to the relevant service and intent.\n'
                    turn_msg += '    2. **Extract Slots**: Fill required slots from the input; apply defaults as needed.\n'
                    turn_msg += '    3. **Format Output**: Provide the updated dialogue state in JSON format:\n'
                    turn_msg += '        ```json\n'
                    turn_msg += '        { "DOMAIN-SLOT": "VALUE", ... }\n'
                    turn_msg += '        ```\n'

                bs_msg = f"            {turn_slot_values}\n"
                if not reverse_x_and_y:
                    msg.append({"role": "user","content": prefix_msg+turn_msg})
                    msg.append({"role": "assistant","content": bs_msg})
                else:
                    msg.append({"role": "user", "content": prefix_msg+bs_msg})
                    msg.append({"role": "assistant","content": turn_msg})

        prefix_msg = f"\n**Example {max_n_examples + 1} of Dialogue State Change Update Task:**\n"
        if given_context is None:
            last_slot_values = {s: v.split('|')[0] for s, v in data_item['last_slot_values'].items()}
        else:
            last_slot_values = given_context
        
        last_sys_utt = data_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        
        state_string = '    **Previous Belief State (Before the Latest User Interaction):** \n'
        state_string += f"        {last_slot_values}\n\n"

        turn_msg = ''
        if not use_null_data_item:
            turn_msg = state_string + "    **Latest Conversation Between System and User:** \n"
            if last_sys_utt:
                    turn_msg += '        **System:** "' + last_sys_utt + '"\n'
            turn_msg += '        **User:** "' + data_item['dialog']['usr'][-1] + '"\n\n'
            turn_msg += '    **Task Instructions:**\n'
            turn_msg += '    1. **Identify Service and Intent**: Match user input to the relevant service and intent.\n'
            turn_msg += '    2. **Extract Slots**: Fill required slots from the input; apply defaults as needed.\n'
            turn_msg += '    3. **Format Output**: Provide the updated dialogue state in JSON format:\n'
            turn_msg += '        ```json\n'
            turn_msg += '        { "DOMAIN-SLOT": "VALUE", ... }\n'
            turn_msg += '        ```\n'


        else:
            pass  # default adds our null input at end
        msg.append({"role": "user","content": prefix_msg+turn_msg})

        guidelines = ''
        idx = 0
        visited = []
        for domain in data_item['domains']:
            if domain in SGD_PROMPT_DICTS:
                idx += 1
                guidelines += f"  {idx}. {SGD_PROMPT_DICTS[domain]}"

        remain_keys = [k for k in list(SGD_PROMPT_DICTS.keys()) if k not in visited]
        random_seed = [0,1,2,3]
        while idx < 4:
            random.seed(random_seed[idx])
            domain = random.choice(remain_keys)
            if domain in visited:
                continue
            visited.append(domain)
            idx += 1
            guidelines += f"  {idx}. {SGD_PROMPT_DICTS[domain]}"
        
        if add_guidelines:
            tmp_msg_content = msg[1]['content']
            msg[1]['content'] = '**Guidelines:**\n\n'
            msg[1]['content'] += guidelines
            msg[1]["content"] += tmp_msg_content
        return msg
