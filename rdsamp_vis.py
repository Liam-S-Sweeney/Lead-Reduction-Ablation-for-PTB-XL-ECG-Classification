import pandas as pd
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
from data import prepare_labels


db, *_ = prepare_labels()

signals, fields = wfdb.rdsamp(str(FULL_DATASET_PATH / db.iloc[0]['filename_lr']))
# print(fields)

leads_dict = {i: [lead] for i, lead in enumerate(fields['sig_name'])}
print(pd.DataFrame(leads_dict))

wfdb.plot_items(
    signal=signals, 
    fs=fields['fs'], 
    title='ecg vis test'
)