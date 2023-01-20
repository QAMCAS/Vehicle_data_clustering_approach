import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import json


def main(location):
    # adapt file path to the Original_data folder.
    file_path = os.path.join('data/Original_data/bus_signals/', location, 'bus_signals.json')
    df = pd.read_json(file_path)
    print(df)

    # plotting each sensor separately
    fig, axes = plt.subplots(22, 1, figsize=(15, 60))
    for (col, ax) in zip(df.columns, axes):
        att = np.array(df[col]["values"])
        ax.plot(att[:, 0], att[:, 1])
        ax.title.set_text(col)
    # plt.show()

    acc_x = np.array(
        df["acceleration_x"]["values"])  # create an array containing the values of the first sensor acceleration

    # search for the minimum Timestamp
    TS_min = acc_x[:, 0][0]
    col_min = "acceleration_x"
    for (col, ax) in zip(df.columns, axes):
        att = np.array(df[col]["values"])
        x = att[:, 0]
        for idx in range(len(x)):
            if x[idx] < TS_min:
                TS_min = x[idx]
                col_min = col

    print("Minimal TS ", TS_min, " / ", col_min)

    # search for the maximum Timestamp
    TS_max = acc_x[:, 0][0]
    for (col, ax) in zip(df.columns, axes):
        att = np.array(df[col]["values"])
        x = att[:, 0]
        for idx in range(len(x)):
            if x[idx] > TS_max:
                TS = x[idx]
                TS_max = TS
                col_max = col

    print("Final TS ", TS_max, " / ", col_max)

    # look for the minimum timestamp difference
    min_sensor = np.array(df[col_min]["values"])
    min_TS_diff = min_sensor[1][0] - min_sensor[0][0]
    Min_column = np.array(df.columns[0])
    for (col, ax) in zip(df.columns, axes):
        att1 = np.array(df[col]["values"])
        for idx in range(len(att1)):
            x_diff = att1[idx][0] - att1[idx - 1][0]
            if 0 < x_diff < min_TS_diff:
                min_TS_diff = x_diff
                Min_column = col
    print("Minimum time difference ", min_TS_diff)

    # search for the sensor having the min nbr of values
    Min_S = np.array(df[df.columns[0]]["values"])
    Sensor_MIN = df.columns[0]
    for (col, ax) in zip(df.columns, axes):
        S = np.array(df[col]["values"])
        if len(S) < len(Min_S):
            Min_S = S
            Sensor_MIN = col

    print("Sensor with minimum number of values", Sensor_MIN, " ", len(Min_S))

    # search for the sensor having the max nbr of values
    Max_S = np.array(df[df.columns[0]]["values"])
    Sensor_MAX = df.columns[0]
    for (col, ax) in zip(df.columns, axes):
        S = np.array(df[col]["values"])
        if len(S) > len(Max_S):
            Max_S = S
            Sensor_MAX = col

    print("Sensor with maximum number of values", Sensor_MAX, " ", len(Max_S))

    # create a new timeline
    timeline = [TS_min]
    i = 0
    while timeline[i] < TS_max:
        timeline.append(timeline[i] + min_TS_diff)
        i = i + 1

    timeline[len(timeline) - 1] = TS_max

    # writing interpolated data  in a json file
    new_data_dict = dict()
    with open('data/Interpolated_data/data_' + location + '.json', 'w') as f:
        for (col, ax) in zip(df.columns, axes):
            data_list = list()
            att = np.array(df[col]["values"])
            x = att[:, 0]
            y = att[:, 1]
            cs = CubicSpline(x, y)
            y_new = cs(timeline)
            for i in range(len(y_new)):
                data_list.append([int(timeline[i]), y_new[i]])
            new_data_dict[col] = ({'unit': df[col]["unit"], 'values': data_list})

        json.dump(new_data_dict, f, sort_keys=True, indent=4)


"""
 Uncomment the following code  to run this script separately, 
 Please Change the right location name,
 the available locations are "Gaimersheim", "Munich" and "Ingolstadt"

"""
if __name__ == "__main__":
    location = 'Gaimersheim'
    main(location)
