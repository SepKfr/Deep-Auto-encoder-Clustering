# time-step, x acceleration, y acceleration, z acceleration
import os
import sklearn.preprocessing
import pandas as pd

data_path = "User_Identification_From_Walking_Activity"
list_of_dfs = []

for file in os.listdir(data_path):
    if file != "README":
        user_id = file.split(".")[0]
        df = pd.read_csv(os.path.join(data_path, file), header=None, encoding="utf-8")

        # Assign column names
        column_names = ["time", "x", "y", "z"]
        df.rename(columns=dict(zip(df.columns, column_names)), inplace=True)
        df["id"] = user_id
        list_of_dfs.append(df)

combined_df = pd.concat(list_of_dfs, ignore_index=True)
df_sorted = combined_df.sort_values(by='time')
df_sorted = df_sorted.dropna()
df_real = df_sorted[["time", "x", "y", "z"]]

standardscaler = {"id": sklearn.preprocessing.StandardScaler().fit(df_real)}

df_sorted[["time", "x", "y", "z"]] = standardscaler["id"].transform(df_real)

df_sorted.to_csv("User_id.csv")




