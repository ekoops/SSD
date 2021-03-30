import torch

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from ssd.data import samplers
from ssd.data.datasets import build_dataset
from ssd.data.transforms import build_transforms, build_target_transform
from ssd.structures.container import Container


class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            targets = Container(
                {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
            )
        else:
            targets = None
        return images, targets, img_ids


def get_datasets(cfg, phase):
    if phase == "train":
        data_transform = build_transforms(cfg, phase="train")
        target_transform = build_target_transform(cfg)
        dataset_list = cfg.DATASETS.TRAIN
    elif phase == "style":
        data_transform = build_transforms(cfg, phase="style")
        target_transform = None
        dataset_list = cfg.ADAIN.DATASETS.STYLE
    elif phase == "test":
        data_transform = build_transforms(cfg, phase="test")
        target_transform = None
        dataset_list = cfg.DATASETS.TEST
    else:
        raise RuntimeError("You shouldn't be here")
    return build_dataset(
        dataset_list,
        transform=data_transform,
        target_transform=target_transform,
        is_train=phase != "test"
    )


def make_data_loader(cfg, phase, distributed=False, max_iter=None, start_iter=0):
    datasets = get_datasets(cfg, phase)
    is_train = phase != "test"
    shuffle = is_train

    data_loaders = []

    for dataset in datasets:
        if distributed:
            sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
        if max_iter is not None:
            batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter,
                                                                start_iter=start_iter)

        data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(phase == "train"))
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
