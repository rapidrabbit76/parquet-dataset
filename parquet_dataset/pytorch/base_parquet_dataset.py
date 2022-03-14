import time
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch.utils import data


class BaseParquetDataset(data.Dataset):
    def __init__(
        self,
        parquet_paths: List[str],
        columns: List[str] = [],
        index_column: Union[List[str], str] = [],
        transforms: Callable = None,
        num_cached_parquet: int = 1,
        num_workers: int = 0,
        refresh_frequency: int = 0,
    ):
        assert num_workers >= 0, "num_workers cannot be less than 0"
        assert refresh_frequency >= 0, "refresh_frequency cannot be less than 0"
        assert len(columns) > 0, "select more than one column"

        if num_workers > 0 and refresh_frequency == 0:
            raise "check refresh_frequency"

        self.columns = columns
        self.transforms = transforms
        self.num_workers = num_workers
        self.refresh_frequency = refresh_frequency
        self.num_cached_parquet = num_cached_parquet
        self.index = (
            [index_column] if isinstance(index_column, str) else index_column
        )
        self.parquet_paths = parquet_paths

        self.steps_cache = int(
            np.ceil(len(self.parquet_paths) / self.num_cached_parquet)
        )

        self.current_parquet_idx = 0
        self.current_pd_parquets = None  # cached parquets(DataFrame)
        self.current_indices_in_cache = []  # data index in cached parquet
        self.total_len = self._get_total_length()
        self._cache_setting()

    def __len__(self):
        return self.total_len

    def _get_total_length(self) -> int:
        """read total parquet list and calcurate number of data row"""
        fdf = pq.ParquetDataset(self.parquet_paths)
        return len(fdf.read(columns=self.index))

    def _cache_setting(self):
        """parquet data load from storage to memory"""
        cur_pd, cur_indices = self._caching(self.current_parquet_idx)
        self.current_pd_parquets = cur_pd
        self.current_indices_in_cache = cur_indices

    def _caching(self, idx: int) -> Tuple[pd.DataFrame, List[int]]:
        """parquet data caching"""
        next_idx = (idx + 1) * self.num_cached_parquet
        next_idx = None if next_idx > len(self.parquet_paths) else next_idx

        parquet_cache_list = self.parquet_paths[
            idx * self.num_cached_parquet : next_idx
        ]
        fparquet = pq.ParquetDataset(parquet_cache_list)
        df_data = fparquet.read(columns=self.columns).to_pandas()

        random_generator = self._build_random_generator()
        list_indices = random_generator.permutation(len(df_data)).tolist()

        return df_data, list_indices

    @staticmethod
    def _build_random_generator():
        now = time.time()
        seed = int((now - int(now)) * 100000)
        random_generator = np.random.RandomState(seed=seed)
        return random_generator

    def _get_index(self) -> int:
        if self.num_workers == 0:
            return self.current_indices_in_cache.pop()

        random_generator = self._build_random_generator()
        rand_idx = random_generator.randint(len(self.current_indices_in_cache))
        pd_idx = self.current_indices_in_cache[rand_idx]
        del self.current_indices_in_cache[rand_idx]
        return pd_idx

    def _recache_check(self, refresh_idx: int) -> bool:
        if len(self.current_indices_in_cache) >= refresh_idx:
            return False

        self.current_parquet_idx += 1

        if self.current_parquet_idx >= self.steps_cache:
            self.current_parquet_idx = 0

        if self.num_workers > 0:
            now = time.time()
            seed = int((now - int(now)) * 100000)
            rng = np.random.RandomState(seed=seed)
            rng.shuffle(self.parquet_paths)

        return True

    def __getitem__(self, idx=None):
        refresh_idx = 1
        if self.num_workers > 0:
            refresh_idx = len(self.current_pd_parquets) - len(
                self.current_pd_parquets
            ) // (self.refresh_frequency * self.num_workers)

        if self._recache_check(refresh_idx):
            # parquet cache loading
            self._cache_setting()

        pd_idx = self._get_index()
        pd_raw = self.current_pd_parquets.iloc[pd_idx]
        return pd_raw
