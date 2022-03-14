import os
from glob import glob

from parquet_dataset.pytorch import BaseParquetDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_CACHED_PARQUET = 32
NUM_WORKERS = 4
REFRESH_FREQUENCY = 4
BATCH_SIZE = 16
EPOCHS = 10


class Dataset(BaseParquetDataset):
    def __getitem__(self, idx: int):
        pd_raw = super().__getitem__(idx)
        index, label = pd_raw["index"], pd_raw["label"]
        return index, label


def main():
    paths = glob(os.path.join("./DATASET", "*.parquet"))
    dataset = Dataset(
        paths,
        columns=["index", "label"],
        index_column="index",
        num_cached_parquet=NUM_CACHED_PARQUET,
        num_workers=NUM_WORKERS,
        refresh_frequency=REFRESH_FREQUENCY,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    for epoch in range(EPOCHS):
        for index, label in tqdm(dataloader):
            # let's training!
            pass


if __name__ == "__main__":
    main()
