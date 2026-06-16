import math
from typing import Tuple


def resolve_block_sizes(config) -> Tuple[int, int]:
    legacy = config.get("block_size")
    enc = config.get("encoder_block_size", legacy)
    dec = config.get("decoder_block_size", legacy)
    if enc is None and dec is None:
        raise KeyError("Expected encoder_block_size/decoder_block_size in model_kwargs.")
    if enc is None:
        enc = dec
    if dec is None:
        dec = enc
    enc, dec = int(enc), int(dec)
    if enc <= 0 or dec <= 0:
        raise ValueError(f"block sizes must be positive, got encoder={enc}, decoder={dec}")
    return enc, dec


def round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def get_tracking_padded_lengths(fixed_len, encoder_block_size, decoder_block_size, num_sub_events) -> Tuple[int, int]:
    full_padded_len = round_up_to_multiple(fixed_len, encoder_block_size)
    nominal_decoder_len = math.ceil(full_padded_len / num_sub_events)
    if decoder_block_size == encoder_block_size:
        decoder_sub_event_len = max(1, nominal_decoder_len)
    else:
        decoder_sub_event_len = round_up_to_multiple(max(1, nominal_decoder_len), decoder_block_size)
    return full_padded_len, decoder_sub_event_len
