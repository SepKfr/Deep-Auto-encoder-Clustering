import numpy as np
import pandas as pd


nurseCharting = pd.read_csv("nurseCharting.csv")
lab = pd.read_csv("lab.csv")
vitalPeriodic = pd.read_csv("vitalPeriodic.csv")


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

for id, df in lab.groupby("patientunitstayid"):

    df_lab = df.sort_values(by='labresultrevisedoffset')
    df_nurse = nurseCharting[nurseCharting["patientunitstayid"] == id].sort_values(by='nursingchartoffset')

    vitalPeriodic = vitalPeriodic[vitalPeriodic["patientunitstayid"] == id].sort_values(by='observationoffset')

    lab_time = df_lab["labresultrevisedoffset"] * time_factor
    nurse_time = df_nurse["nursingchartoffset"] * time_factor
    vitalPeriodic_time = vitalPeriodic["observationoffset"] * time_factor

    df_lab['time'] = lab_time
    df_nurse['time'] = nurse_time
    vitalPeriodic['time'] = vitalPeriodic_time

    hco3 = df_lab[df_lab["labname"] == "HCO3"][["labresult", "time"]]
    creatinine = df_lab[df_lab["labname"] == "creatinine"][["labresult", "time"]]
    potassium = df_lab[df_lab["labname"] == "potassium"][["labresult", "time"]]
    sodium = df_lab[df_lab["labname"] == "sodium"][["labresult", "time"]]
    temp = df_nurse[df_nurse["nursingchartcelltypevallabel"] == "Temperature"][["nursingchartvalue", "time"]]
    map = df_nurse[df_nurse["nursingchartcelltypevallabel"] == "MAP (mmHg)"][["nursingchartvalue", "time"]]
    respiratory = df_nurse[df_nurse["nursingchartcelltypevallabel"] == "Respiratory Rate"][["nursingchartvalue", "time"]]
    heart_rate = vitalPeriodic[["heartrate", "time"]]

    variables = {'hco3': hco3,
                 'creatinine': creatinine,
                 'potassium': potassium,
                 'sodium': sodium,
                 'temp': temp,
                 'map': map,
                 'respiratory': respiratory,
                 'time': lab_time,
                 'id': df_lab["patientunitstayid"]}

    # df_new = pd.DataFrame(variables)
    # df_new.index = np.arange(len(df_lab))

    hours = [6, 12, 24]

    apache_score = np.zeros(max(len(df_lab), len(df_nurse)))

    for variable, df in variables.items():
        if variable != "time" and variable != "id":
            df_val = df[df.columns[~df.columns.isin(['time'])]].values.reshape(-1)
            df_val = pd.to_numeric(df_val, errors="ignore", downcast="float")
            for i, val in enumerate(df_val):
                for range_min, range_max, score in scoring_criteria[variable]:
                    if range_min <= val < range_max:
                        apache_score[i] += score

    min_time = lab_time.min()
    max_time = lab_time.max()
    time = df_lab[df_lab.columns[df_lab.columns.isin(['time'])]]

    variables["heart_rate"] = heart_rate

    df_list = []

    for variable, df in variables.items():

        if variable != "time" and variable != "id":
            df = df.sort_values(by='time')
            df = df.loc[(df['time'] >= min_time) & (df['time'] <= max_time)]
            df_e_time = df[df.columns[~df.columns.isin(['time'])]].values
            df_new = pd.DataFrame(df_e_time, columns=[variable])
            df_list.append(df_new)

    df_list.append(time)

    df_new = pd.concat(df_list, axis=1)
    df_new = df_new.sort_values(by='time')

    six_hours = df_new.loc[(df_new['time'] >= min_time) & (df_new['time'] <= 6+min_time)]
    twelve_hours = df_new.loc[(df_new['time'] >= min_time) & (df_new['time'] <= 12+min_time)]
    twenty_four_hours = df_new.loc[(df_new['time'] >= min_time) & (df_new['time'] <= 24+min_time)]

    six_hours.index = np.arange(len(six_hours))
    twelve_hours.index = np.arange(len(twelve_hours))
    twenty_four_hours.index = np.arange(len(twenty_four_hours))
    apache_6 = apache_score[six_hours.index]
    apache_12 = apache_score[twelve_hours.index]
    apache_24 = apache_score[twenty_four_hours.index]

    a_score_6 = max(apache_6)
    a_score_12 = max(apache_12)
    a_score_24 = max(apache_24)

    six_hours["apache"] = a_score_6
    twelve_hours["apache"] = a_score_12
    twenty_four_hours["apache"] = apache_24

    list_patients_6.append(six_hours)
    list_patients_12.append(twelve_hours)
    list_patients_24.append(twenty_four_hours)


patients_6 = pd.concat(list_patients_6, ignore_index=True)
patients_12 = pd.concat(list_patients_12, ignore_index=True)
patients_24 = pd.concat(list_patients_24, ignore_index=True)

patients_6 = patients_6.sort_values(by='time')
patients_6 = patients_6.interpolate(method='linear')

patients_12 = patients_12.sort_values(by='time')
patients_12 = patients_12.interpolate(method='linear')

patients_24 = patients_24.sort_values(by='time')
patients_24 = patients_24.interpolate(method='linear')

patients_6.to_csv("patients_6.csv")
patients_12.to_csv("patients_12.csv")
patients_24.to_csv("patients_24.csv")










