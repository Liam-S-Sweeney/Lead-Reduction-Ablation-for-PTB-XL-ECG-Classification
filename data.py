import logging
import wfdb
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from functools import lru_cache
from utility.startup import load_metadata, device_check
from utility.paths import CACHE_PATH, FULL_DATASET_PATH, STATS_PATH


@lru_cache(maxsize=1)
def prepare_labels():
    logger = logging.getLogger(__name__)
    db, scp = load_metadata()

    # Dx Mask Selection
    diagnostic_mask = scp['diagnostic_class'].notna()
    diagnostic_df = scp[diagnostic_mask]
    code_to_class = diagnostic_df['diagnostic_class']
    mapping_dict = code_to_class.to_dict()

    # print(f"Superclass:\n{code_to_class.values}\n\nCodes:\n{code_to_class.keys}\n\n= = = = = =\n\n")

    # Aggregation
    def to_superclass(codes):
        return set(mapping_dict[key] for key in codes.keys() if key in mapping_dict)
    db['superclasses'] = db['scp_codes'].apply(to_superclass)

    # print(db['superclasses'].explode().value_counts())
    # print((db['superclasses'].apply(len) == 0).sum())

    # MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(db['superclasses'])
    db = pd.concat([db, pd.DataFrame(Y, columns=mlb.classes_, index=db.index)], axis=1)

    # Updated Masks
    masks = {
        "train": db['strat_fold'] < 9,
        "val":   db['strat_fold'] == 9,
        "test":  db['strat_fold'] == 10
        }

    logger.info(f"{len(mlb.classes_)} superclasses, {(db['superclasses'].apply(len) == 0).sum()} unlabeled records")
    
    return db, Y, list(mlb.classes_), masks


@lru_cache(maxsize=1)
def npy_generation():
    logger = logging.getLogger(__name__)

    if not CACHE_PATH.exists():
        db, *_ = prepare_labels()
        
        n = len(db)
        
        X = np.zeros((n, 12, 1000), dtype=np.float32)

        missing_files = []

        for i, filename in enumerate(tqdm(db['filename_lr'])):
            full_path = FULL_DATASET_PATH / filename
            try:
                signal, _ = wfdb.rdsamp(str(full_path))
                X[i] = np.asarray(signal).transpose()
            except (FileNotFoundError, OSError) as e:
                logger.error(f"Error loading index {i}: {e}")
                missing_files.append(str(filename))

        if not missing_files:
            logger.info("Cache Loop was successful. No missing files.")
        else:
            logger.warning(f"Cache Loop unsuccessful. Missing records: [{len(missing_files)}]")
            raise RuntimeError(f"{len(missing_files)} records missing; not saving cache")

        check_i = 12345
        sig, _ = wfdb.rdsamp(str(FULL_DATASET_PATH / db.iloc[check_i]['filename_lr']))
        assert np.allclose(X[check_i], np.asarray(sig).transpose()), "cache misaligned"
        assert not np.isnan(X).any(), "NaNs in cache"

        np.save(CACHE_PATH, X)
        del X
        logger.info(f"Cache written to {CACHE_PATH}")

    return np.load(CACHE_PATH, mmap_mode='r')


class ECGDataset(Dataset):
    def __init__(self, cache_path, Y, mask, mean, std, leads=None):
        self.cache_path = cache_path
        self.indices = np.where(mask)[0]
        self.Y = Y
        self.mean = mean
        self.std = std
        self.leads = np.arange(12) if leads is None else np.asarray(leads)
        self._X = None

    @property
    def X(self):
        if self._X is None:
            self._X = np.load(self.cache_path, mmap_mode='r')
        return self._X

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        global_i = self.indices[index]

        signal = self.X[global_i]   # (12, 1000)
        signal = (signal - self.mean) / self.std
        signal = signal[self.leads]

        label = self.Y[global_i]    # (5,)

        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return signal_tensor, label_tensor


@lru_cache(maxsize=1)
def get_norm_stats():                                       # Add an assert check to verify
    if STATS_PATH.exists():
        d = np.load(STATS_PATH)
        return d['mean'], d['std']
    
    X = npy_generation()
    _, _, _, masks = prepare_labels()

    train_indices = np.where(masks['train'])[0]

    sums = np.zeros(12, dtype=np.float64)
    sumsq = np.zeros(12, dtype=np.float64)
    count = 0

    for start in range(0, len(train_indices), 512):
        chunk = np.asarray(X[train_indices[start:start+512]], dtype=np.float64)
        sums += chunk.sum(axis=(0, 2))
        sumsq += (chunk ** 2).sum(axis=(0, 2))
        count += chunk.shape[0] * chunk.shape[2]

    mean = (sums / count).astype(np.float32)[:, None]
    std = (np.sqrt(sumsq / count - (sums / count) ** 2) + 1e-8).astype(np.float32)[:, None]

    np.savez(STATS_PATH, mean=mean, std=std)

    return mean, std


def generate_loaders(leads=None, workers=4, bs=64, device=None):         # leads is unhashable, no lru_cache with generate_loaders()
    _, Y, _, masks = prepare_labels()
    X = npy_generation()
    mean, std = get_norm_stats()
    device = device_check() 

    train_ds = ECGDataset(CACHE_PATH, Y, masks['train'], mean, std, leads)
    val_ds   = ECGDataset(CACHE_PATH, Y, masks['val'],   mean, std, leads)
    test_ds  = ECGDataset(CACHE_PATH, Y, masks['test'],  mean, std, leads)

    use_cuda = device is not None and device.type == "cuda"
    loader_kwargs = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0) if device.type == "cuda" else {}

    loaders ={
        'train': DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, **loader_kwargs),
        'val': DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs),
        'test': DataLoader(test_ds, batch_size=bs, shuffle=False, **loader_kwargs),
    }

    return loaders
