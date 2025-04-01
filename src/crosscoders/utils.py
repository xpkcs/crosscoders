



import os
from pathlib import Path
from typing import Any, Dict
import torch
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

    # print(config, updates)
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



from dataclasses import dataclass, is_dataclass, fields

def dataclass_repr(dc, indent=0):
    """Recursively produce a multi-line repr for dataclass instances."""
    if not is_dataclass(dc):
        return repr(dc)
    spacer = ' ' * indent
    lines = [f"{spacer}{dc.__class__.__name__}("]
    for f in fields(dc):
        value = getattr(dc, f.name)
        # Recursively format nested dataclasses.
        value_repr = dataclass_repr(value, indent + 4) if is_dataclass(value) else repr(value)
        lines.append(f"{spacer}    {f.name} = {value_repr.lstrip()},")
    lines.append(f"{spacer})")
    return "\n".join(lines)


def dataclass_to_dict(dc):

    out = {}
    for f in fields(dc):
        value = getattr(dc, f.name)
        out[f.name] = value if not is_dataclass(value) else dataclass_to_dict(value)


    return out

def flatten_dict(nested_dict, parent_key='', sep='.'):
    items = []
    for k, v in nested_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



import hydra
from omegaconf import OmegaConf

# def load_dataset_object():

#     print('> loading dataset:')
#     print(OmegaConf.to_yaml(cfg.runner.dataset, resolve=True), end='\n\n')
#     ds = hydra.utils.instantiate(cfg.runner.dataset)



def instantiate(cfg, resolve: bool = False, recursive: bool = False):

    print('> instantiating class:')
    print(OmegaConf.to_yaml(cfg, resolve=resolve), end='\n\n')
    _ = hydra.utils.instantiate(cfg, recursive=recursive)


    return _




def check_required_env_vars(env_vars = [
    'CONFIG_NAME',
]):
    for ev in env_vars:
        assert ev in os.environ





# def convert_activation_dict(data):
#     # Original tokens tensor with shape [B, T]
#     tokens = data['tokens']  # e.g. shape: [10, 213]
#     B, T = tokens.shape

#     # Define the activation keys in the order you want them to appear.
#     act_keys = [
#         'tiny-stories-33M.resid_mid',
#         'tiny-stories-33M.ln2.normalized',
#         'tiny-stories-33M.mlp_out',
#         'tiny-stories-33M.resid_post'
#     ]

#     activations_list = []
#     activation_types = []  # This will store an integer label for each activation type.
#     layer_ids = []         # This will store the layer index (from the third dimension).

#     for act_type, key in enumerate(act_keys):
#         # Each tensor x has shape: [B, T, L, D]
#         x = data[key]
#         B_, T_, L, D = x.shape  # B_ should equal B, and T_ equals T.

#         # Permute to bring the L dimension next to B so that we can flatten them together.
#         # Option 1: transpose tokens and layers: from [B, T, L, D] -> [B, L, T, D]
#         x = x.transpose(1, 2)  # Now shape is [B, L, T, D]
#         # Then flatten the first two dimensions: [B * L, T, D]
#         x_reshaped = x.reshape(B * L, T, D)
#         activations_list.append(x_reshaped)

#         # Create activation type vector: each row in x_reshaped gets the current act_type label.
#         activation_types.append(torch.full((B * L,), act_type, dtype=torch.long))

#         # Create layer indices for each sample.
#         # For each sample in B, we have layers 0...L-1.
#         layer_ids.append(torch.arange(L).unsqueeze(0).repeat(B, 1).reshape(-1))

#     # Concatenate along the flattened batch dimension.
#     activations = torch.cat(activations_list, dim=0)      # shape: [B * (# act keys) * L, T, D]
#     activation_type = torch.cat(activation_types, dim=0)    # shape: [B * (# act keys) * L]
#     layer = torch.cat(layer_ids, dim=0)                     # shape: [B * (# act keys) * L]

#     # For tokens, replicate each token sequence for each corresponding activation.
#     # tokens: [B, T] -> [B, (# act keys) * L, T] then reshape to [B * (# act keys) * L, T]
#     tokens_repeated = tokens.unsqueeze(1).repeat(1, len(act_keys) * L, 1).reshape(B * len(act_keys) * L, T)

#     return {
#         'tokens': tokens_repeated,          # shape: [4 * 4 * 10, 213] i.e. [160, 213]
#         'activations': activations,         # shape: [160, 213, 768]
#         'activation_type': activation_type, # shape: [160]
#         'layer': layer                      # shape: [160]
#     }

import numpy as np
def convert_activation_dict(data):
    # Original tokens array with shape [B, T]
    tokens = data['tokens']  # e.g. shape: [10, 213]
    B, T = tokens.shape

    # Define the activation keys in the order you want them to appear.
    act_keys = [
        'resid_mid',
        'ln2.normalized',
        'mlp_out',
        'resid_post'
    ]

    activations_list = []
    activation_types = []  # This will store an integer label for each activation type.
    layer_ids = []         # This will store the layer index (from the third dimension).

    for act_type, key in enumerate(act_keys):
        # Each array x has shape: [B, T, L, D]
        x = data[key]
        B_, T_, L, D = x.shape  # B_ should equal B, and T_ equals T.

        # Permute to bring the L dimension next to B so that we can flatten them together.
        # Option 1: transpose tokens and layers: from [B, T, L, D] -> [B, L, T, D]
        x = np.transpose(x, (0, 2, 1, 3))  # Now shape is [B, L, T, D]
        # Then flatten the first two dimensions: [B * L, T, D]
        x_reshaped = x.reshape(B * L, T, D)
        activations_list.append(x_reshaped)

        # Create activation type vector: each row in x_reshaped gets the current act_type label.
        # activation_types.append(np.full((B * L,), act_type, dtype=np.int64))
        activation_types.append(np.full((B * L,), key, dtype=object))

        # Create layer indices for each sample.
        # For each sample in B, we have layers 0...L-1.
        layer_ids.append(np.tile(np.arange(L).reshape(1, -1), (B, 1)).reshape(-1))

    # Concatenate along the flattened batch dimension.
    activations = np.concatenate(activations_list, axis=0)      # shape: [B * (# act keys) * L, T, D]
    activation_type = np.concatenate(activation_types, axis=0)    # shape: [B * (# act keys) * L]
    layer = np.concatenate(layer_ids, axis=0)                     # shape: [B * (# act keys) * L]

    # For tokens, replicate each token sequence for each corresponding activation.
    # tokens: [B, T] -> [B, (# act keys) * L, T] then reshape to [B * (# act keys) * L, T]
    tokens_repeated = np.tile(tokens[:, np.newaxis, :], (1, len(act_keys) * L, 1)).reshape(B * len(act_keys) * L, T)

    return {
        'tokens': tokens_repeated,          # shape: [4 * 4 * 10, 213] i.e. [160, 213]
        'activations': activations,         # shape: [160, 213, 768]
        'activation_type': activation_type, # shape: [160]
        'layer': layer                      # shape: [160]
    }



import boto3
def delete_files_in_s3(bucket_name, prefix, dry_run=False):
    s3 = boto3.client('s3')

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')

    if 'Contents' in response:
        files = [{'Key': obj['Key']} for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
        print(f'deleting {len(files)} files')
        # print(files)
        if not dry_run:
            s3.delete_objects(Bucket=bucket_name, Delete={'Objects': files, 'Quiet': False})

        # for obj in response['Contents']:
        #     if obj['Key'].endswith('.parquet'):
        #         print(f"Deleting: {obj['Key']}")
        #         if not dry_run:
        #             s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
    else:
        print("No Parquet files found in the specified path.")

# _ = delete_files_in_s3('crosscoders', 'data/tiny-stories-v1/language_model=tiny-stories-33M/slice=train/tag=tiny-stories-33M-1B/activations/')
# _
