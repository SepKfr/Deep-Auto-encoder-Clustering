import pandas as pd


#nurseCharting = pd.read_csv("nurseCharting.csv")
lab = pd.read_csv("lab.csv")


df_list = []

MAP_ranges = [(0, 49, 4), (50, 69, 2), (110, 129, 2), (130, 159, 3), (160, 1000, 4)]
temperature_ranges = [(0, 29.9, 4), (30, 31.9, 3), (32, 33.9, 2),
                      (34, 35.9, 1), (38.5, 38.9, 1), (39, 40.9, 3),
                      (41, 1000, 4)]
respiratory_ranges = [(0, 5, 4), (6, 9, 2), (10, 11, 1), (25, 34, 1), (35, 49, 3), (50, 1000, 4)]
HCO3_ranges = [(0, 15, 4), (15, 17.9, 3), (18, 21.9, 2), (32, 40.9, 1), (41, 51.9, 3), (52, 1000, 4)]
sodium_ranges = [(0, 110, 4), (111, 119, 3), (120, 129, 2), (150, 154, 1), (155, 159, 2),
                 (160, 179, 3), (180, 1000, 4)]
potassium_ranges = [(0, 2.5, 4), (2.5, 2.9, 2), (3, 3.4, 1), (3.5, 5.4, 0),
                    (5, 5.9, 1), (6, 6.9, 3), (7, 1000, 4)]
creatinine_ranges = [(0, 0.6, 2), (1.5, 1.9, 2), (2, 3.4, 3), (3.5, 1000, 4)]

# df["nursingchartcelltypevallabel"] == "MAP (mmHg)"
#     df["nursingchartcelltypevallabel"] == "Temperature"
#     df["nursingchartcelltypevallabel"] == "Heart Rate"
#     df["nursingchartcelltypevallabel"] == "Heart Rate"

for id, df in lab.groupby("patientunitstayid"):

    print(id)
    df = df.sort_values(by='labresultrevisedoffset')


