from typing import Any, Dict


def replace_substring_in_state_dict_if_present(
    state_dict: Dict[str, Any], substring: str, replace: str
):
    keys = sorted(state_dict.keys())
    for key in keys:
        if substring in key:
            newkey = key.replace(substring, replace)
            state_dict[newkey] = state_dict.pop(key)
    
    if "_metadata" in state_dict:
        metadata = state_dict['_metadata']
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove
            # 'module': for the actual model
            # 'module.xx.xx': for the rest
            if len(key) == 0:
                continue
            newkey = key.replace(substring, replace)
            metadata[newkey] = metadata.pop(key)
