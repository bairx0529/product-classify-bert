from enum import Enum

from datasets import load_from_disk
from torch.utils.data import DataLoader

from configuration import config


class DatasetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


def get_dataset(data_type=DatasetType.TRAIN):
    path = str(config.PROCESSED_DATA_DIR / data_type.value)
    dataset = load_from_disk(path)
    return dataset


def get_dataloader(data_type=DatasetType.TRAIN):
    dataset = get_dataset(data_type)
    dataset.set_format(type='torch')
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
