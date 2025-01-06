from typing import Any

# meta
TEXT_TOKEN = -1
IGNORE_TOKEN = -2

def get_attr_by_name(obj: Any, name: str) -> Any:
    """
    Get an attribute from an object using a dot notation string.
    e.g., get_attr_by_name(model, "layers.0.self_attn.q_proj") will return model.layers[0].self_attn.q_proj
    """
    levels = name.split('.')
    current = obj
    for level in levels:
        if level.isdigit():
            current = current[int(level)]
        else:
            current = getattr(current, level)
    return current