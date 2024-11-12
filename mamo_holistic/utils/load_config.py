import yaml
import sys
import os

# Parse base YAML config and mrge override YAML config
# Return a dictionary
def load_config(base_file, override_file=None):
    with open(base_file, 'r') as f:
        base_config = yaml.safe_load(f)

    # Merge base config with override config
    def merge_configs(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and key in base:
                base[key] = merge_configs(base.get(key, {}), value)
            else:
                base[key] = value
        return base

    if override_file is not None:
        with open(override_file, 'r') as f:
            override_config = yaml.safe_load(f)
        return merge_configs(base_config, override_config)
    else:
        return base_config

def get_parameter(config, path, mode='default', default=None):
    """
    Retrieve a parameter from a nested dictionary.

    Args:
        config (dict): The configuration dictionary.
        path (list): The list of keys representing the path to the parameter.
        mode (str): The behavior if the parameter is not found ('warn' or 'exit').

    Returns:
        The value of the parameter if found, or None if not found and mode is 'warn'.

    Raises:
        SystemExit: If the parameter is not found and mode is 'exit'.
    """
    try:
        # Traverse the nested dictionary using the given path
        for key in path:
            config = config[key]
        return config
    except KeyError as e:
        # Handle the case where a key is not found
        if mode == 'default':
            return default
        elif mode == 'warn':
            print(f"Warning: Parameter {'->'.join(path)} not found. Default None is returned.")
            return None
        elif mode == 'exit':
            print(f"Error: Parameter {'->'.join(path)} not found. Exiting the program.")
            sys.exit(1)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Use 'warn' or 'exit'.")

def show_config(base_file, override_file):

    # Check if the base config YAML exists
    if not os.path.exists(base_file):
        print(f"Error: Base config YAML file '{base_file}' not found.")
        return
    else:
        print(f"Base config YAML file {base_file}")

    # Check if the override YAML exists
    if override_file is None:
        print(f"Warning: Override config YAML file not provided.")
    elif not os.path.exists(override_file):
        print(f"Warning: Override config YAML file '{override_file}' not found.")
        override_file = None
    else:
        print(f"Override config YAML file {override_file}")

    # Load and merge configurations
    config = load_config(base_file, override_file)

    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
