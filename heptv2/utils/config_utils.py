"""Small YAML override helpers for command-line experiments."""
import ast
import re

import yaml


_SCIENTIFIC_FLOAT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+$")


def _parse_value(raw):
    try:
        value = yaml.safe_load(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw
    if isinstance(value, str) and _SCIENTIFIC_FLOAT.match(value):
        return float(value)
    return value


def apply_overrides(cfg, overrides):
    """Apply dotted overrides like ``optimizer_kwargs.lr=1e-5`` in-place."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        parts = key.split(".")
        cur = cfg
        for part in parts[:-1]:
            if part not in cur or cur[part] is None:
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = _parse_value(raw_value)
    return cfg
