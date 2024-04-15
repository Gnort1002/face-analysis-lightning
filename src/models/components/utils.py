from inspect import isfunction


def get_named_function(module):
    """Get the class member in module."""
    return {k: v for k, v in module.__dict__.items() if isfunction(v)}


def load_pretrained(state_dict, pretrained_state_dict):
    for name, pretrained_param in pretrained_state_dict.items():
        if name in state_dict.keys():
            param = state_dict[name].data
            pretrained_param = pretrained_param.data
            if param.shape == pretrained_param.shape:
                param.copy_(pretrained_param)