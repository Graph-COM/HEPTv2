import math
from typing import Any, Mapping, Tuple


def resolve_block_sizes(config: Mapping[str, Any]) -> Tuple[int, int]:
    legacy_block_size = config.get("block_size")
    encoder_block_size = config.get("encoder_block_size", legacy_block_size)
    decoder_block_size = config.get("decoder_block_size", legacy_block_size)

    if encoder_block_size is None and decoder_block_size is None:
        raise KeyError("Expected encoder_block_size/decoder_block_size or legacy block_size in model_kwargs.")
    if encoder_block_size is None:
        encoder_block_size = decoder_block_size
    if decoder_block_size is None:
        decoder_block_size = encoder_block_size

    encoder_block_size = int(encoder_block_size)
    decoder_block_size = int(decoder_block_size)
    if encoder_block_size <= 0:
        raise ValueError(f"encoder_block_size must be positive, got {encoder_block_size}.")
    if decoder_block_size <= 0:
        raise ValueError(f"decoder_block_size must be positive, got {decoder_block_size}.")
    return encoder_block_size, decoder_block_size


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}.")
    return ((value + multiple - 1) // multiple) * multiple


def get_tracking_padded_lengths(
    fixed_len: int,
    encoder_block_size: int,
    decoder_block_size: int,
    num_sub_events: int,
) -> Tuple[int, int]:
    full_padded_len = round_up_to_multiple(fixed_len, encoder_block_size)
    nominal_decoder_len = math.ceil(full_padded_len / num_sub_events)
    if decoder_block_size == encoder_block_size:
        decoder_sub_event_len = max(1, nominal_decoder_len)
    else:
        decoder_sub_event_len = round_up_to_multiple(max(1, nominal_decoder_len), decoder_block_size)
    return full_padded_len, decoder_sub_event_len
