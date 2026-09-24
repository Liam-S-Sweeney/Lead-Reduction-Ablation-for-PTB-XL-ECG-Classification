"""
This file is used to prpeare data for CNN model
----
Contains:
    - F: prepare_labels()
    - F: npy_generation()
    - C: ECGDataset(Dataset)
    - F: get_norm_stats()
    - F: generate_loaders(leads=None, workers=4, bs=64, device=None)
"""

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


@lru_cache(maxsize=1)   # Keeps only single most recent function call + resutls in memory
def prepare_labels():
    """
    1.  Takes the rows of SCP Statements that have a value in the "Diagnostic Class" col
    (only diagnostic statements do; form and rhythm statements are blank there, so they're dropped),
    then uses those rows to make a code-to-superclass mapping dictionary.

    2.  A new col, "superclasses", is created out of sets of corresponding superclass vals from rows 
        in the "scp_codes" col in the PTB-XL csv.

    3.  The "superclasses" col is converted from multi-label sets to binary column features composed 
        of every unique superclass label as new cols then appends them to an updated PTB-XL dataframe.

    4.  From this new db, following recommendations, the data is split into train, val, and test masks.
    
    Returns:
        db:                 updated PTB-XL DataFrame containing "superclasses" col 
                            and one 0/1 col per class added
        Y:                  (21799, 5) array containing 0/1 labels, row-aligned with db
        classes:            list of classes in column order (alphabetical)
        masks:              dict of boolean Series splits for 'train', 'val', and 'test'
    """
    logger = logging.getLogger(__name__)
    db, scp = load_metadata()

    # Dx Mask Selection
    diagnostic_mask = scp['diagnostic_class'].notna()
    diagnostic_df = scp[diagnostic_mask]
    code_to_class = diagnostic_df['diagnostic_class']
    mapping_dict = code_to_class.to_dict()


    # Aggregation
    def to_superclass(codes: dict):
        """
        Loops through each key in the input codes dict, checks if that key exists in mapping_dict, 
        looks up its corresponding superclass value, and groups those values into a set to remove duplicates.

        """
        return set(mapping_dict[key] for key in codes.keys() if key in mapping_dict)

    db['superclasses'] = db['scp_codes'].apply(to_superclass) 

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
    """
    If the cache doesn't exist yet:
    1.  Using the db from prepare_labels(), creates a float32 array of zeros with shape
        (n records, 12 leads, 1000 samples = 10 s at 100 Hz).
    2.  Fills each record's slice with its waveform, transposed from wfdb's
        (time, leads) layout to the (leads, time) layout Conv1d expects.
    3.  Before saving, checks that no records are missing, that no values are
        NaN or Inf, and that a spot-checked row matches its source file.
    Then memory-maps the cache read-only, so records are read from disk only when indexed.

    Returns:
        cache: memory-mapped (21799, 12, 1000) float32 array
    """
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

        if not np.isfinite(X).all():
            bad = np.unique(np.where(~np.isfinite(X))[0])
            raise ValueError(
                f"Non-finite values in {len(bad)} record(s), "
                f"e.g. rows {bad[:5].tolist()}; not saving cache")

        check_i = 12345
        sig, _ = wfdb.rdsamp(str(FULL_DATASET_PATH / db.iloc[check_i]['filename_lr']))
        if not np.allclose(X[check_i], np.asarray(sig).T):
            raise RuntimeError(
                f"Cache misaligned at row {check_i}: stored signal does not match "
                f"{db.iloc[check_i]['filename_lr']}; not saving cache")

        np.save(CACHE_PATH, X)
        del X
        logger.info(f"Cache written to {CACHE_PATH}")

    return np.load(CACHE_PATH, mmap_mode='r')


class ECGDataset(Dataset):
    """
    Prepares ecg data for a specified split for training, validation, and testing
    """
    def __init__(self, cache_path, Y, mask, mean, std, leads=None):
        self.cache_path = cache_path
        self.indices = np.where(mask)[0]    # Find indices for rows marked "True" by the split's mask, becomes array/vector
        self.Y = Y
        self.mean = mean
        self.std = std
        self.leads = np.arange(12) if leads is None else np.asarray(leads)  # Selects leads for trainin (all 12 default)
        self._X = None

    @property   # Keep self._X = None when initialized to keep lightweight
    def X(self):
        if self._X is None:
            self._X = npy_generation()
        return self._X

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Why PyTorch requests and iteam at a relative index:
            1.  Maps to global db index
            2.  Retrieves the single 12*1000 matrix at that position
            3.  Normalizes the raw voltages for more efficient learning (z-score)
            4.  Selects only rows for prespecified leads
            5.  Takes those rows' corresponding superclass labels (0/1 * 5) and signal 
                and converts these into two float32 tensors

        Returns:
            signal_tensor: (n_leads, 1000) float32, normalized
            label_tensor:  (5,) float32 of 0/1 labels
        """
        global_i = self.indices[index]  # Pairs local index with global index to access Cache's actual index properly 

        signal = self.X[global_i]   # (12, 1000)
        signal = (signal - self.mean) / self.std
        signal = signal[self.leads]

        label = self.Y[global_i]    # (5,)

        signal_tensor = torch.tensor(signal, dtype=torch.float32)   # must match the model's float32 weights
        label_tensor = torch.tensor(label, dtype=torch.float32)     # BCEWithLogitsLoss needs float targets

        return signal_tensor, label_tensor


def _validate_norm_stats(mean, std):
    """Raise if normalization stats have the wrong shape or contain NaN/Inf."""
    if mean.shape != (12, 1) or std.shape != (12, 1):
        raise ValueError(f"Shape mismatch: mean={mean.shape}, std={std.shape}")
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError("NaN or Inf in normalization statistics")


@lru_cache(maxsize=1)
def get_norm_stats():
    """
    If norm_stats.npz exists, loads it and checks that its stored n_train matches the
    current training split and that mean/std have the right shape and no NaN/Inf.
    A mismatch raises an error (delete the file to recompute) rather than recomputing silently.
    If the file doesn't exist:
    1.  Creating 1D NumPy arrays containing 12 initilizaed 0s for sums and sums squared, plus count = 0.

    2.  Then, in chunks (records, leads, time) of up to 512 training records read from the cache:
        - Add the chunk's values, summed across records [0] and time [2], to a running per-lead total [1]
        - Add the chunk's squared values, summed the same way, to a running per-lead sum of squares
        - Add the number of samples per lead in the chunk (records * timepoints) to the running count
    
    3.  Finally, mean, var, and std are calculated using sums, sumsq, and count with mean, std, 
        and the length of the training indices being saved as STATS_PATH

    Returns:
        mean: (12, 1) float32, per-lead mean over all training samples
        std:  (12, 1) float32, per-lead standard deviation (plus 1e-8)
    The (12, 1) shape broadcasts against a (12, 1000) signal in __getitem__.
    """
    _, _, _, masks = prepare_labels()
    train_indices = np.where(masks['train'])[0]
    n_train = len(train_indices)

    if STATS_PATH.exists():
        d = np.load(STATS_PATH)
        if int(d['n_train']) != n_train:
            raise ValueError(
                f"{STATS_PATH.name} was computed on {int(d['n_train'])} training records "
                f"but the current split has {n_train}; delete it to recompute")
        _validate_norm_stats(d['mean'], d['std'])
        return d['mean'], d['std']

    X = npy_generation()
    sums = np.zeros(12, dtype=np.float64)
    sumsq = np.zeros(12, dtype=np.float64)
    count = 0

    for start in range(0, n_train, 512):
        chunk = np.asarray(X[train_indices[start:start + 512]], dtype=np.float64)
        sums += chunk.sum(axis=(0, 2))
        sumsq += (chunk ** 2).sum(axis=(0, 2))
        count += chunk.shape[0] * chunk.shape[2]

    mu = sums / count
    var = np.maximum(sumsq / count - mu ** 2, 0.0)  # floating-point cancellation can make var slightly negative; 
                                                    # clamp to 0
    raw_std = np.sqrt(var)

    if np.any(raw_std < 1e-3):                        # mV; below any real ECG signal
        flat = np.where(raw_std < 1e-3)[0].tolist()
        raise ValueError(f"Near-zero variance in lead(s) {flat}")

    mean = mu.astype(np.float32)[:, None]
    std = (raw_std + 1e-8).astype(np.float32)[:, None]
    _validate_norm_stats(mean, std)

    np.savez(STATS_PATH, mean=mean, std=std, n_train=n_train)
    return mean, std


def generate_loaders(leads=None, workers=4, bs=64, device=None):
    """
    Build train/val/test DataLoaders over the cached PTB-XL signals.

    All three splits share the training-fold normalization stats and the
    same lead subset. Worker processes and pinned memory are used only on CUDA.
    """
    npy_generation()    # ensure cache exists
    _, Y, _, masks = prepare_labels()
    mean, std = get_norm_stats()

    ds = {split: ECGDataset(CACHE_PATH, Y, masks[split], mean, std, leads)
          for split in ('train', 'val', 'test')}

    use_cuda = device is not None and device.type == "cuda"
    loader_kwargs = (dict(num_workers=workers,  # specifies how many subprocess used for data loading
                          pin_memory=True,  # allocates your loaded data in "pinned" (page-locked) CPU memory
                                            # significantly faster data transfer from CPU -> GPU
                          persistent_workers=workers > 0)   # keeps data loader worker processes alive between 
                                                            # epochs reducing overhead + speed of transitions
                     if use_cuda else {})

    return {
        'train': DataLoader(ds['train'], batch_size=bs, shuffle=True,
                            drop_last=True, **loader_kwargs),   # drop partial final batch
        'val':   DataLoader(ds['val'],   batch_size=bs, shuffle=False, **loader_kwargs),
        'test':  DataLoader(ds['test'],  batch_size=bs, shuffle=False, **loader_kwargs),
    }
