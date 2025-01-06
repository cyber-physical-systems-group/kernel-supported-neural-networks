from itertools import chain
from typing import Any


def left_dict_join(main: dict, other: dict) -> dict:
    """Merges two dictionaries into single one, where keys from main are added when duplicate is found"""
    return dict(chain(other.items(), main.items()))


def prepare_config_for_sweep(config: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Prepares W&B config for running sweep, based on two distinct configs

    :param config: general sweep config with values such as name, method or metric
                   for more details see: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
    :param parameters: parameters to sweep over, each given as list

    :return: configuration dictionary ready for sweep
    """
    parameters = {
        key: {"values": values if isinstance(values, list) else [values]} for key, values in parameters.items()
    }
    config.update({"parameters": parameters})

    return config
