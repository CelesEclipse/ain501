import torch.nn as nn
import re

def model_to_dict(model: nn.Module) -> dict:
    model_dict = {
        "name": model.__class__.__name__,
        "children": {}
    }

    for name, child in model.named_children():
        # Recurse sub-layers
        if list(child.children()):
            model_dict["children"][name] = model_to_dict(child)
        else:
            layer_str = str(child)
            params = re.sub(r'^[^(]+\((.*)\)$', r'\1', layer_str)
            model_dict["children"][name] = {
                "type": child.__class__.__name__,
                "params": params if params != layer_str else ""
            }
    return model_dict
