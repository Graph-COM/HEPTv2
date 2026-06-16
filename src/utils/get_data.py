from torch_geometric.loader import DataLoader
from datasets import Tracking, TrackingTransform
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def get_data_loader(dataset, idx_split, batch_size, num_workers=4, prefetch_factor=2, ddp=False):
    train_sampler = DistributedSampler(dataset[idx_split["train"]]) if ddp else None
    train_loader = DataLoader(
        dataset[idx_split["train"]],
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        sampler=train_sampler,
    )
    valid_sampler = DistributedSampler(dataset[idx_split["valid"]], shuffle=False) if ddp else None
    valid_loader = DataLoader(
        dataset[idx_split["valid"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        sampler=valid_sampler,
    )
    test_sampler = DistributedSampler(dataset[idx_split["test"]], shuffle=False) if ddp else None
    test_loader = DataLoader(
        dataset[idx_split["test"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        sampler=test_sampler,
    )
    loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}
    if ddp:
        samplers = {"train": train_sampler, "valid": valid_sampler, "test": test_sampler}
        return loaders, samplers
    return loaders


def get_dataset(dataset_name, data_dir):
    if "track" in dataset_name:
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                dataset = Tracking(data_dir, transform=TrackingTransform(), dataset_name=dataset_name)
            dist.barrier()
            if dist.get_rank() != 0:
                dataset = Tracking(data_dir, transform=TrackingTransform(), dataset_name=dataset_name)
        else:
            dataset = Tracking(data_dir, transform=TrackingTransform(), dataset_name=dataset_name)
    else:
        raise NotImplementedError(f"Only tracking dataset is supported, got {dataset_name}")
    dataset.dataset_name = dataset_name
    return dataset
