import argparse
import numpy as np
import ruptures as rpt
import pandas as pd
from data_formatters import solar, traffic, watershed, exchange, air_quality, electricity


parser = argparse.ArgumentParser(description="data segmenter")
parser.add_argument("--exp_name", type=str, default="exchange")
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

data_formatter = {"traffic": traffic.TrafficFormatter,
                  "electricity": electricity.ElectricityFormatter,
                  "solar": solar.SolarFormatter,
                  "air_quality": air_quality.AirQualityFormatter,
                  "watershed": watershed.WatershedFormatter,
                  "exchange": exchange.ExchangeFormatter}

data = pd.read_csv(args.data_path, dtype={'date': str})
data.sort_values(by=["id", "hours_from_start"], inplace=True)
data_format = data_formatter[args.exp_name]()
data = data_format.transform_data(data)
data = data[data_format.real_inputs].values

print(data.shape)

print("start segmenting...")
algo = rpt.BottomUp().fit(data)
change_indices = algo.predict(pen=0.5)
change_indices[-1] = change_indices[-1] - 1

change_indices_array = np.zeros(len(data))
change_indices_array[change_indices] = 1
result_array = np.zeros((len(data), data.shape[1]+1))
result_array[:, len(data.shape)] = change_indices_array
np.save("{}.npy".format(args.exp_name), result_array)

