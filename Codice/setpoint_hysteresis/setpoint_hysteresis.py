import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "/Users/albertozaupa/Desktop/Projects/Melinda/datasets/Cella_13.csv"
OUTPUT_PATH = "/Users/albertozaupa/Desktop/Projects/Melinda/datasets/Cella_13_SH.csv"
TEMP_COLUMN = "TemperaturaCelle"
PUMP_COLUMN = "PompaGlicoleMarcia"


def get_setpoint_hysteresis(pump, temp, n):
    window_size = 10

    output_setpoint = []
    output_hysteresis = []

    looking_for_1 = True

    for i in range(window_size, n):
        if looking_for_1 and pump[i] == 1:
            window = temp[i - window_size : min(i + window_size + 1, n)]
            window_argmax = window.argmax() + i - window_size
            window_max = temp[window_argmax]
            output_hysteresis.append((window_argmax, window_max))
            looking_for_1 = False
        elif not looking_for_1 and pump[i] == 0:
            window = temp[i - window_size: min(i + window_size + 1, n)]
            window_argmin = window.argmin() + i - window_size
            window_min = temp[window_argmin]
            output_setpoint.append((window_argmin, window_min))
            looking_for_1 = True

    return output_setpoint, output_hysteresis


def extend(inputs, n):
    output = np.zeros((n,))

    for i in range(len(inputs)):
        idx = inputs[i][0]
        value = inputs[i][1]
        if i == 0:
            output[:idx] = value

        if i == len(inputs) - 1:
            output[idx:] = value
        else:
            next_idx = inputs[i + 1][0]
            next_value = inputs[i + 1][1]
            if value == next_value:
                output[idx:next_idx] = value
            else:
                midway = int((idx + next_idx) / 2) + 1
                output[idx:midway] = value
                output[midway:next_idx] = next_value

    return output


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    temp = df[TEMP_COLUMN]
    temp.replace({',': '.'}, regex=True).astype(float)
    pump = df[PUMP_COLUMN]

    setpoints, hysteresises = get_setpoint_hysteresis(pump, temp, df.shape[0])
    final_setpoints = extend(setpoints, df.shape[0])
    final_hysteresises = extend(hysteresises, df.shape[0])

    # display_setpoints = np.zeros((df.shape[0]), )
    # display_hysteresises = np.zeros((df.shape[0]), )
    # for el in setpoints:
    #     display_setpoints[el[0]] = el[1]
    # for el in hysteresises:
    #     display_hysteresises[el[0]] = el[1]
    #
    # iterations = None
    # batch_size = 5000
    # if df.shape[0] % batch_size == 0:
    #     iterations = df.shape[0] / batch_size
    # else:
    #     iterations = int(df.shape[0] / batch_size) + 1
    #
    # for i in range(iterations):
    #     if i == iterations - 1:
    #         temps = temp[i*batch_size:]
    #         pumps = pump[i*batch_size]
    #         sets = display_setpoints[i*batch_size:]
    #         hysts = display_hysteresises[i*batch_size:]
    #         x_axis = range(df.shape[0] - i*batch_size)
    #     else:
    #         temps = temp[i*batch_size : (i + 1)*batch_size]
    #         pumps = pump[i*batch_size : (i + 1)*batch_size]
    #         sets = display_setpoints[i * batch_size : (i + 1)*batch_size]
    #         hysts = display_hysteresises[i * batch_size : (i + 1)*batch_size]
    #         x_axis = range(batch_size)
    #     plt.plot(x_axis, temps, color="black")
    #     plt.plot(x_axis, pumps, color="pink")
    #     plt.scatter(x_axis, sets, color="green")
    #     plt.scatter(x_axis, hysts, color="blue")
    #     plt.show()
    #     input()

    plt.plot(final_setpoints, label="setpoint")
    plt.plot(final_hysteresises, label="hysteresis")
    plt.legend()
    plt.show()

    if input("Satisfied? ") == "Y":
        new_df = pd.DataFrame(np.transpose(np.array([final_setpoints, final_hysteresises])))
        new_df.to_csv(OUTPUT_PATH)
