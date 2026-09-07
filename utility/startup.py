import logging
from functools import lru_cache
import pandas as pd
import torch
import ast
from utility.paths import LOG_DIR, PTBXL_DB_PATH, SCP_STATEMENTS_PATH


def initialize_logger():
    logger = logging.getLogger()          
    if logger.handlers:                   
        return logger
    logger.setLevel(logging.DEBUG)

    LOG_DIR.mkdir(exist_ok=True)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(LOG_DIR / "details.log")
    console_handler.setLevel(logging.WARNING)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def device_check():
    logger = logging.getLogger(__name__)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():   # Apple Silicon
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"device: {device}")
    return device


@lru_cache(maxsize=1)
def load_metadata():
    logger = logging.getLogger(__name__)
    db = pd.read_csv(PTBXL_DB_PATH, index_col='ecg_id')
    db['scp_codes'] = db['scp_codes'].apply(ast.literal_eval)
    scp = pd.read_csv(SCP_STATEMENTS_PATH, index_col=0)

    logger.info("Metadata loaded")

    return db, scp