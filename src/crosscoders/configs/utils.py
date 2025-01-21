



from pathlib import Path
from typing import Any, Dict
import yaml




def get_config(fp: Path | str) -> Dict[str, Any]:

    with open(fp, 'r') as infl:
        cfg = yaml.safe_load(infl)

    return cfg



from dataclasses import is_dataclass, fields
from typing import Any, Dict

def update_dataclass(config: Any, updates: Dict[str, Any]) -> None:
    """
    Recursively updates a dataclass instance with values from a dictionary.

    :param config: The dataclass instance to update.
    :param updates: A dictionary containing updates.
    """

    assert isinstance(updates, dict)

    print(config, updates)
    for field_name, new_value in updates.items():

        try:
            current_value = getattr(config, field_name)

            if is_dataclass(current_value):
                update_dataclass(current_value, new_value)

            else:
                setattr(config, field_name, new_value)

        except:
            raise ValueError()


from dataclasses import is_dataclass, fields
from typing import Any, Type, TypeVar, Dict, List

T = TypeVar('T')

def from_dict(data_class: Type[T], data: Dict[str, Any]) -> T:
    """
    Recursively converts a dictionary to a dataclass instance.

    :param data_class: The dataclass type to instantiate.
    :param data: The dictionary containing the data.
    :return: An instance of data_class populated with data.
    """
    if not is_dataclass(data_class):
        raise ValueError(f"{data_class} is not a dataclass.")

    field_set = {f.name for f in fields(data_class)}
    init_kwargs = {}

    for field in fields(data_class):
        field_name = field.name
        field_type = field.type
        if field_name in data:
            value = data[field_name]
            if is_dataclass(field_type):
                init_kwargs[field_name] = from_dict(field_type, value)
            elif hasattr(field_type, '__origin__') and field_type.__origin__ == list:
                # Handle List types
                list_item_type = field_type.__args__[0]
                if is_dataclass(list_item_type):
                    init_kwargs[field_name] = [from_dict(list_item_type, item) for item in value]
                else:
                    init_kwargs[field_name] = value
            else:
                init_kwargs[field_name] = value
        # else:
        #     init_kwargs[field_name] = None  # or set a default if needed

    return data_class(**init_kwargs)
