import argparse
import numpy as np
import ruptures as rpt
import pandas as pd
import dataforemater

parser = argparse.ArgumentParser(description="data segmenter")
parser.add_argument("--exp_name", type=str, default="exchange")
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

col_def = {"traffic": {'id': 'id', 'target': 'values', 'covariates': []},
           "electricity": {'id': 'id', 'target': 'power_usage', 'covariates': []},
           "solar": {'id': 'id', 'target': 'Power(MW)', 'covariates': []},
           "air_quality": {'id': 'id', 'target':'NO2', 'covariates': ['CO', 'TEMP']},
           "watershed": {'id': 'id', 'target': 'Conductivity', 'covariates': ['Q', 'temp']},
           "exchange": {'id': 'id', 'target': 'OT', 'covariates': ['0', '1', '2', '3', '4', '5']}}

data_formatter = dataforemater.DataFormatter(column_definition=col_def[args.exp_name])

data = pd.read_csv(args.data_path, dtype={'date': str})
data.sort_values(by=["id", "hours_from_start"], inplace=True)
data = data_formatter.transform_data(data)
data = data[data_formatter.real_inputs].values

print(data.shape)

print("start segmenting...")
algo = rpt.BottomUp().fit(data)
change_indices = algo.predict(pen=0.5)
change_indices[-1] = change_indices[-1] - 1

change_indices_array = np.zeros(len(data))
change_indices_array[change_indices] = 1
result_array = np.zeros((len(data), data.shape[1]+1))
result_array[:, data.shape[1]] = change_indices_array
np.save("{}.npy".format(args.exp_name), result_array)

