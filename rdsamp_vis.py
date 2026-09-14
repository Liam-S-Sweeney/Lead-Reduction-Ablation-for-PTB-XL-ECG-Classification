import pandas as pd
import wfdb
import pandas as pd
from utility.paths import FULL_DATASET_PATH
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