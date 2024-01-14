import argparse
import os

import numpy as np
import ruptures as rpt
import pandas as pd
import dataforemater

parser = argparse.ArgumentParser(description="data segmenter")
parser.add_argument("--exp_name", type=str, default="exchange")
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

data_formatter = dataforemater.DataFormatter(args.exp_name)

df = pd.read_csv(args.data_path, dtype={'date': str})
df.sort_values(by=["id", "hours_from_start"], inplace=True)
data = data_formatter.transform_data(df)
data_real = data[data_formatter.real_inputs]


print("start segmenting...")

cp_array = np.zeros((len(data_real)))
pr_len = 0

for id, df in data.groupby("id"):

    df_real = df[data_formatter.real_inputs].values
    algo = rpt.Pelt().fit(df_real)
    change_indices = algo.predict(pen=3)
    change_indices = [ind+pr_len for ind in change_indices]
    change_indices[-1] = change_indices[-1] - 1
    cp_array[change_indices] = 1
    pr_len += len(df)


result_array = np.zeros((len(data_real), data_real.shape[1] + 1))

result_array[:, :data_real.shape[1]] = data_real
result_array[:, data_real.shape[1]] = cp_array

real_inputs = data_formatter.real_inputs
real_inputs.remove(data_formatter.target_column)

if len(real_inputs) > 0:
    columns = [data_formatter.target_column,
               real_inputs, 'CP']
else:
    columns = [data_formatter.target_column, 'CP']

result_df = pd.DataFrame(result_array, columns=columns)

result_df["id"] = data["id"].values

path = "data_CP"

if not os.path.exists(path):

    os.makedirs(path)

result_df.to_csv(os.path.join(path, "{}.csv".format(args.exp_name)))
