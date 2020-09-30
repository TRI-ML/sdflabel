import torch
from torch import nn


def read_cfg_string(cfgp, section, key, default):
    """
    Read string from a config file
    Args:
        cfgp: Config parser
        section: [section] of the config file
        key: Key to be read
        default: Value if couldn't be read

    Returns: Resulting string

    """
    if cfgp.has_option(section, key):
        return cfgp.get(section, key)
    else:
        return default


def read_cfg_int(cfgp, section, key, default):
    """
    Read int from a config file
    Args:
        cfgp: Config parser
        section: [section] of the config file
        key: Key to be read
        default: Value if couldn't be read

    Returns: Resulting int

    """
    if cfgp.has_option(section, key):
        return cfgp.getint(section, key)
    else:
        return default


def read_cfg_precision(cfgp, section, key, default):
    """
    Read float precision from a config file
    Args:
        cfgp: Config parser
        section: [section] of the config file
        key: Key to be read
        default: Value if couldn't be read

    Returns: Resulting torch.dtype

    """
    if cfgp.has_option(section, key):
        str = cfgp.get(section, key)
        if str == 'float32':
            return torch.float32
        elif str == 'float16':
            return torch.float16
    else:
        return default


def read_cfg_float(cfgp, section, key, default):
    """
    Read float from a config file
    Args:
        cfgp: Config parser
        section: [section] of the config file
        key: Key to be read
        default: Value if couldn't be read

    Returns: Resulting float

    """
    if cfgp.has_option(section, key):
        return cfgp.getfloat(section, key)
    else:
        return default


def read_cfg_bool(cfgp, section, key, default):
    """
    Read bool from a config file
    Args:
        cfgp: Config parser
        section: [section] of the config file
        key: Key to be read
        default: Value if couldn't be read

    Returns: resulting bool

    """
    if cfgp.has_option(section, key):
        return cfgp.get(section, key) in ['True', 'true']
    else:
        return default


def convert_to_precision(network, precision):
    """
    Convert network layer to precision
    Args:
        network: Network model
        precision: Desired precision
    """
    network.to(dtype=precision)
    for layer in network.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
