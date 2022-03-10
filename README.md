# Parquet Dataset

base source code from [Naver D2 : Data Loader, Better, Faster, Stronger](https://d2.naver.com/helloworld/3773258)

# Install

```bash
$ pip install parquet-dataset
```

## How to use

```bash
├── DATASET
│   ├── 000.parquet
│   ├── 001.parquet
│   ├── 002.parquet
│   ├── 003.parquet
│   ├── 004.parquet
│   ├── 00N.parquet
└── example_dataset.py
```

```python
from glob import glob
from parquet_dataset.pytorch import BaseParquetDataset
class Dataset(BaseParquetDataset):
    def __getitem__(self, idx: int):
        pd_raw = super().__getitem__(idx)
        index, label = pd_raw["index"], pd_raw["label"]
        return index, label

...


paths = glob(os.path.join("./DATASET", "*.parquet"))
dataset = Dataset(
    paths,
    columns=["index", "label"],
    index_column="index",
    num_cached_parquet=32,
    num_workers=4,
    refresh_frequency=4,
)
```
