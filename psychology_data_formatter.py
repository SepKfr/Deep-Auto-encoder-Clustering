import numpy as np
import pandas as pd
from tqdm import tqdm

nurseCharting = pd.read_csv("nurseCharting.csv", nrows=1000000)
lab = pd.read_csv("lab.csv", nrows=1000000)
vitalPeriodic = pd.read_csv("vitalPeriodic.csv", nrows=1000000)

df_list = []

scoring_criteria = {'map': [(0, 49, 4), (50, 69, 2), (110, 129, 2), (130, 159, 3), (160, 1000, 4)],
                    'temp': [(0, 29.9, 4), (30, 31.9, 3), (32, 33.9, 2),
                      (34, 35.9, 1), (38.5, 38.9, 1), (39, 40.9, 3),
                      (41, 1000, 4)],
                    'respiratory': [(0, 5, 4), (6, 9, 2), (10, 11, 1), (25, 34, 1), (35, 49, 3), (50, 1000, 4)],
                    'hco3': [(0, 15, 4), (15, 17.9, 3), (18, 21.9, 2), (32, 40.9, 1), (41, 51.9, 3), (52, 1000, 4)],
                    'sodium': [(0, 110, 4), (111, 119, 3), (120, 129, 2), (150, 154, 1), (155, 159, 2),
                 (160, 179, 3), (180, 1000, 4)],
                    'potassium': [(0, 2.5, 4), (2.5, 2.9, 2), (3, 3.4, 1), (3.5, 5.4, 0),
                    (5, 5.9, 1), (6, 6.9, 3), (7, 1000, 4)],
                    'creatinine': [(0, 0.6, 2), (1.5, 1.9, 2), (2, 3.4, 3), (3.5, 1000, 4)]}


time_factor = 1.0/60.0

list_patients_6 = []
list_patients_12 = []
list_patients_24 = []

for id, df in tqdm(lab.groupby("patientunitstayid"), desc='eICU processing'):

    df_lab = df.sort_values(by='labresultrevisedoffset')
    df_nurse = nurseCharting[nurseCharting["patientunitstayid"] == id].sort_values(by='nursingchartoffset')

    vitalPeriodic = vitalPeriodic[vitalPeriodic["patientunitstayid"] == id].sort_values(by='observationoffset')

    lab_time = df_lab["labresultrevisedoffset"]
    nurse_time = df_nurse["nursingchartoffset"]
    vitalPeriodic_time = vitalPeriodic["observationoffset"]

    min_lab_time = lab_time.min()
    min_nurse_time = nurse_time.min()
    min_vital_time = vitalPeriodic_time.min()

    lab_time += abs(min_lab_time)
    nurse_time += abs(min_nurse_time)
    vitalPeriodic_time += abs(min_vital_time)

    df_lab['time'] = lab_time
    df_nurse['time'] = nurse_time
    vitalPeriodic['time'] = vitalPeriodic_time

    df_lab = df_lab.groupby("time").first()
    if len(df_nurse) > 0:
        df_nurse = df_nurse.groupby("time").first()
    if len(vitalPeriodic) > 0:
        vitalPeriodic = vitalPeriodic.groupby("time").first()

    df_lab["t"] = df_lab.index
    df_nurse["t"] = df_nurse.index
    vitalPeriodic["t"] = vitalPeriodic.index

    hco3 = df_lab[df_lab["labname"] == "HCO3"][["labresult", "t"]]
    creatinine = df_lab[df_lab["labname"] == "creatinine"][["labresult", "t"]]
    potassium = df_lab[df_lab["labname"] == "potassium"][["labresult", "t"]]
    sodium = df_lab[df_lab["labname"] == "sodium"][["labresult", "t"]]
    temp = df_nurse[df_nurse["nursingchartcelltypevallabel"] == "Temperature"][["nursingchartvalue", "t"]]
    map = df_nurse[df_nurse["nursingchartcelltypevallabel"] == "MAP (mmHg)"][["nursingchartvalue", "t"]]
    respiratory = df_nurse[df_nurse["nursingchartcelltypevallabel"] == "Respiratory Rate"][["nursingchartvalue", "t"]]
    heart_rate = vitalPeriodic[["heartrate", "t"]]

    variables = {'hco3': hco3,
                 'creatinine': creatinine,
                 'potassium': potassium,
                 'sodium': sodium,
                 'temp': temp,
                 'map': map,
                 'respiratory': respiratory}

    # df_new = pd.DataFrame(variables)
    # df_new.index = np.arange(len(df_lab))

    hours = [6, 12, 24]

    apache_len = max(len(df_lab), len(df_nurse), len(vitalPeriodic))
    apache_score = np.zeros(apache_len+1)

    for variable, df in variables.items():

        if variable != "id":
            df = df.sort_values(by="t")
            df_val = df[df.columns[~df.columns.isin(['t'])]].values.reshape(-1)
            indices = np.arange(len(df_val))
            values = df_val

            for ind, val in zip(indices, values):
                try:
                    for range_min, range_max, score in scoring_criteria[variable]:
                        if range_min <= val <= range_max:
                            apache_score[ind] += score
                except TypeError:
                    pass

    variables["heart_rate"] = heart_rate

    df_list = []

    for variable, df in variables.items():

        df = df.sort_values(by='t')
        df_e_time = df[df.columns[~df.columns.isin(['t'])]]
        df_e_time.index = np.arange(len(df_e_time))
        df_list.append(df_e_time)

    df_new = pd.concat(df_list, axis=1)
    df_new.columns = variables.keys()

    df_new = df_new.sort_index()
    six_indexes = 6 * 60
    twelve_indexes = 12 * 60
    twenty_four_indexes = 24 * 60

    six_hours = df_new.loc[:six_indexes, :]
    twelve_hours = df_new.loc[:twelve_indexes, :]
    twenty_four_hours = df_new.loc[:twenty_four_indexes, :]

    if len(six_hours) > 0:

        apache_6 = apache_score[:six_indexes]
        apache_12 = apache_score[:twelve_indexes]
        apache_24 = apache_score[:twenty_four_indexes]

        a_score_6 = max(apache_6)
        a_score_12 = max(apache_12)
        a_score_24 = max(apache_24)

        six_hours = six_hours.copy()
        six_hours["apache"] = a_score_6
        six_hours["id"] = id

        twelve_hours = twelve_hours.copy()
        twelve_hours["apache"] = a_score_12
        twelve_hours["id"] = id

        twenty_four_hours = twenty_four_hours.copy()
        twenty_four_hours["apache"] = a_score_24
        twenty_four_hours["id"] = id

        list_patients_6.append(six_hours)
        list_patients_12.append(twelve_hours)
        list_patients_24.append(twenty_four_hours)


patients_6 = pd.concat(list_patients_6, ignore_index=True)
patients_12 = pd.concat(list_patients_12, ignore_index=True)
patients_24 = pd.concat(list_patients_24, ignore_index=True)

patients_6 = patients_6.sort_index()
patients_6 = patients_6.interpolate(method='linear')
patients_6 = patients_6.fillna(0.0)

numeric_columns = patients_6.columns
patients_6[numeric_columns] = patients_6[numeric_columns].apply(pd.to_numeric, errors='coerce')
patients_6 = patients_6.dropna()


patients_12 = patients_12.sort_index()
patients_12 = patients_12.interpolate(method='linear')

patients_12 = patients_12.fillna(0.0)

numeric_columns = patients_12.columns
patients_12[numeric_columns] = patients_12[numeric_columns].apply(pd.to_numeric, errors='coerce')
patients_12 = patients_12.dropna()

patients_24 = patients_24.sort_index()
patients_24 = patients_24.interpolate(method='linear')

patients_24 = patients_24.fillna(0.0)

numeric_columns = patients_24.columns
patients_24[numeric_columns] = patients_24[numeric_columns].apply(pd.to_numeric, errors='coerce')
patients_24 = patients_24.dropna()

patients_6.to_csv("patients_6.csv")
patients_12.to_csv("patients_12.csv")
patients_24.to_csv("patients_24.csv")
