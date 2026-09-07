import pandas as pd
import logging
from utility.startup import load_metadata

def data_overview():
    """
    **Used to get a brief descriptive overview of a database:**\n
    - For more information on these variables and what they mean refer to: https://physionet.org/content/ptb-xl/1.0.3/
    ---
    **Expected values:**
    - 21799 clinical 12-lead ECG records
    - 18869 uniuqe patients
    - 10s length per ECG
    - 71 unique ECG statements
    - if pt age > 89yo, they appear as 300yo
    - Recommended 10-fold train-test splits *(1-8 training, 9 validation, 10 test)*
    """
    logger = logging.getLogger(__name__)

    db, _ = load_metadata()
    assert len(db) == 21799, f"expected 21799 records, got {len(db)}"
    assert db["patient_id"].nunique() == 18869
    col_names = [col for col in db.columns]

    for col in col_names:
        n_missing = db[col].isna().sum()
        if n_missing > 0.5 * len(db):
            logger.warning(f"{col} is more than half missing")

        if pd.api.types.is_numeric_dtype(db[col]):
            count = db[col].count()                     # expected 21799 ecg record
            val_count = db[col].value_counts().head()
            nunique = db[col].nunique(dropna=True)      # expected 18869 uniuqe patients
            mean = db[col].mean()
            median = db[col].median()
            mode = db[col].mode().tolist()
            std = db[col].std()
            variance = db[col].var()
            min = db[col].min()
            max = db[col].max()

            print(f"{col}:\n"
                f"    count = {count}\n"
                f"    val_count = {val_count}\n"
                f"    nunique = {nunique}\n"
                f"    mean = {mean:.2f}\n"
                f"    median = {median}\n"
                f"    mode = {mode}\n"
                f"    std = {std:.2f}\n"
                f"    variance = {variance:.2f}\n"
                f"    min = {min}\n"
                f"    max = {max}\n"
                f"    missing = {n_missing}\n"
                )
        else:
            print(f"{col}'s dtype  {db[col].dtype} (NaN: {n_missing})")


if __name__ == "__main__": data_overview()
