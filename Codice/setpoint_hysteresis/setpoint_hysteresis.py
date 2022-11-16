import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "/Users/albertozaupa/Desktop/Projects/Melinda/datasets/Cella_13.csv"
OUTPUT_PATH = "/Users/albertozaupa/Desktop/Projects/Melinda/datasets/Cella_13_SH.csv"
TARGET_COLUMN = "TemperaturaCelle"


def get_setpoint_hysteresis_running_maxmin(A, n, max_window_size, min_window_size):
    assert max_window_size < n
    assert min_window_size < n

    output_setpoint = []
    output_hysteresis = []

    # Prima vengono cercati i massimi
    i = 1
    prev_argmax = A[:max_window_size].argmax()
    prev_max = A[prev_argmax]
    while i < n - max_window_size:
        argmax = A[i:i + max_window_size].argmax() + i
        max = A[argmax]
        if max < prev_max:
            output_hysteresis.append((prev_argmax, prev_max))
            prev_argmax = -1
            prev_max = -10
            i = i + max_window_size
        else:
            prev_argmax = argmax
            prev_max = max
            i = i + 1

    # E poi i minimi
    i = 1
    prev_argmin = A[:min_window_size].argmin()
    prev_min = A[prev_argmin]
    while i < n - min_window_size:
        argmin = A[i:i + min_window_size].argmin() + i
        min = A[argmin]
        if min > prev_min:
            output_setpoint.append((prev_argmin, prev_min))
            prev_argmin = -1
            prev_min = 10
            i = i + min_window_size
        else:
            prev_argmin = argmin
            prev_min = min
            i = i + 1

    return output_setpoint, output_hysteresis


def get_setpoint_hysteresis_filtering(A, n, window_size, alpha, beta):
    assert window_size < n

    output_setpoint = []
    output_hysteresis = []
    dAdt = convolve(A, n, high_pass(201), 201) * 10

    i = 0
    while i < n - window_size:
        dAdt_min = dAdt[i:i + window_size].min()
        dAdt_max = dAdt[i:i + window_size].max()
        if dAdt_min < alpha and dAdt_max > beta:
            argmin = A[i:i + window_size].argmin() + i
            argmax = A[i:i + window_size].argmax() + i
            output_setpoint.append((argmin, A[argmin]))
            output_hysteresis.append((argmax, A[argmax]))
            i = i + 2 * window_size
        else:
            i = i + int(window_size / 2)

    while i < n:
        i = i + int(window_size / 2)

    return output_setpoint, output_hysteresis


def convolve(A, n, H, k):
    output = np.zeros((n,))
    k_2 = int(k / 2)

    for i in range(k_2, n - k_2):
        output[i] = np.mean(A[i - k_2:i + k_2 + 1] * H)

    return np.array(output)


def high_pass(k):
    assert k % 2 == 1
    _filter = np.zeros((k,))
    for i in range(int(k / 2)):
        _filter[i] = -1
        _filter[k - 1 - i] = 1
    return _filter


def low_pass(k):
    assert k % 2 == 1
    return np.ones((k,)) / k


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)[TARGET_COLUMN].astype(float)
    df.replace({',': '.'}, regex=True)
    default_max_window_size = max_window_size = 200
    default_min_window_size = min_window_size = 400
    default_window_size = window_size = 300
    alpha = -1
    beta = 1

    iterations = None
    batch_size = 5000
    if df.shape[0] % batch_size == 0:
        iterations = df.shape[0] / batch_size
    else:
        iterations = int(df.shape[0] / batch_size) + 1

    output_setpoint = np.zeros((df.shape[0],))
    output_hysteresis = np.zeros((df.shape[0],))
    setpoints = []
    hysteresises = []

    for i in range(iterations):
        df_values = df[i * batch_size: (i + 1) * batch_size].values
        x_axis = range(df_values.shape[0])
        user_input = "Y"

        while user_input == "Y":
            batch_setpoints, batch_hysteresises = get_setpoint_hysteresis_running_maxmin(df_values, df_values.shape[0],
                                                                                         max_window_size,
                                                                                         min_window_size)
            output_setpoint_slice = np.zeros((df_values.shape[0],))
            output_hysteresis_slice = np.zeros((df_values.shape[0],))
            for el in batch_setpoints:
                output_setpoint_slice[el[0]] = el[1]
            for el in batch_hysteresises:
                output_hysteresis_slice[el[0]] = el[1]

            plt.plot(x_axis, df_values, color="orange")
            plt.scatter(x_axis, output_setpoint_slice, color="blue")
            plt.scatter(x_axis, output_hysteresis_slice, color="green")
            plt.show()

            user_input = input("Retry? ")
            if user_input == "Y":
                max_window_size = int(input("Max window size: "))
                min_window_size = int(input("Min window size: "))
            else:
                for el in batch_setpoints:
                    setpoints.append((el[0] + i*batch_size, el[1]))
                for el in batch_hysteresises:
                    hysteresises.append((el[0] + i*batch_size, el[1]))
                max_window_size = default_max_window_size
                min_window_size = default_min_window_size

    for i in range(len(setpoints)):
        setpoint_index = setpoints[i][0]
        setpoint = setpoints[i][1]
        if i == 0:
            output_setpoint[:setpoint_index] = setpoint

        if i == len(setpoints) - 1:
            output_setpoint[setpoint_index:] = setpoint
        else:
            next_setpoint_index = setpoints[i + 1][0]
            next_setpoint = setpoints[i + 1][1]
            if setpoint == next_setpoint:
                output_setpoint[setpoint_index:next_setpoint_index] = setpoint
            else:
                midway = int((setpoint_index + next_setpoint_index) / 2) + 1
                output_setpoint[setpoint_index:midway] = setpoint
                output_setpoint[midway:next_setpoint_index] = next_setpoint

    for i in range(len(hysteresises)):
        hysteresis_index = hysteresises[i][0]
        hysteresis = hysteresises[i][1]
        if i == 0:
            output_hysteresis[:hysteresis_index] = hysteresis

        if i == len(hysteresises) - 1:
            output_hysteresis[hysteresis_index:] = hysteresis
        else:
            next_hysteresis_index = hysteresises[i + 1][0]
            next_hysteresis = hysteresises[i + 1][1]
            if hysteresis == next_hysteresis:
                output_hysteresis[hysteresis_index:next_hysteresis_index] = hysteresis
            else:
                midway = int((hysteresis_index + next_hysteresis_index) / 2) + 1
                output_hysteresis[hysteresis_index:midway] = hysteresis
                output_hysteresis[midway:next_hysteresis_index] = next_hysteresis

    plt.title("Setpoints")
    plt.plot(output_setpoint, color="blue")
    plt.show()
    plt.title("Hysteresises")
    plt.plot(output_hysteresis, color="orange")
    plt.show()

    if input("Satisfied? ") == "Y":
        new_df = pd.DataFrame(np.transpose(np.array([output_setpoint, output_hysteresis])))
        new_df.to_csv(OUTPUT_PATH)
