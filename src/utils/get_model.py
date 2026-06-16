import torch
import functools
from models.baselines import Transformer
from fvcore.nn import FlopCountAnalysis, flop_count_table


def _patch_dynamo_nvcc_probe():
    """
    Guard torch._dynamo debug repro generation when `nvcc --version`
    is not executable in the current environment.
    """
    try:
        from torch._dynamo import debug_utils as dynamo_debug_utils  # type: ignore
    except Exception:
        return

    original_probe = getattr(dynamo_debug_utils, "_cuda_system_info_comment", None)
    if original_probe is None or getattr(original_probe, "__hept_nvcc_safe__", False):
        return

    @functools.cache
    def _safe_cuda_system_info_comment():
        try:
            return original_probe()
        except PermissionError as exc:
            return f"# nvcc is not executable in this environment ({exc})\n"

    _safe_cuda_system_info_comment.__hept_nvcc_safe__ = True
    dynamo_debug_utils._cuda_system_info_comment = _safe_cuda_system_info_comment


def _should_compile_model(model_name, model_kwargs):
    return hasattr(torch, "compile")


def get_model(model_name, model_kwargs, dataset, test_N=10000, test_k=100):
    model_type = model_name.split("_")[0]
    if model_type == "trans":
        model = Transformer(
            attn_type=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            coords_dim=dataset.coords_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    else:
        raise NotImplementedError(f"Only transformer models are supported, got {model_type}")

    if _should_compile_model(model_name, model_kwargs):
        _patch_dynamo_nvcc_probe()
        model.encode = torch.compile(model.encode, dynamic=False)
        model.decode = torch.compile(model.decode, dynamic=False)
        model = torch.compile(model, dynamic=False)

    model.model_name = model_name
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    # count_flops_and_params(model, dataset, test_N, test_k)
    return model


@torch.no_grad()
def count_flops_and_params(model, dataset, N, k):
    E = k * N
    x = torch.randn((N, dataset.x_dim))
    edge_index = torch.randint(0, N, (2, E))
    coords = torch.randn((N, dataset.coords_dim))
    pos = coords[..., :2]
    batch = torch.zeros(N, dtype=torch.long)
    edge_weight = torch.randn((E, 1))

    if dataset.dataset_name == "pileup":
        x[..., -2:] = 0.0

    data = {"x": x, "edge_index": edge_index, "coords": coords, "pos": pos, "batch": batch, "edge_weight": edge_weight}
    print(flop_count_table(FlopCountAnalysis(model, data), max_depth=1))
