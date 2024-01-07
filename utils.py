import torch
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


def load_data(path: str) -> pd.DataFrame:
    headers = ['T_Out_AVG', 'T_1ST_AVG', 'T_MBR_AVG',
               'T_ALT_AVG', 'T_Plenum_AVG', 'RH_Out_AVG',
               'RH_1ST_AVG', 'RH_MBR_AVG', 'RH_ALT_AVG', 'KWH_TOTAL_TOT']

    # Filter the data so that only the selected columns remain
    df = pd.read_csv('House_12.csv', usecols=headers)

    # Filter the NaNs
    df = df.dropna()

    return df


def create_dataset(df: pd.DataFrame, n_in: int = 1, n_out: int = 1) -> tuple:
    n_vars = df.shape[1]
    cols, names = list(), list()

    ## Normalize the data
    # Scaler = MinMaxScaler(feature_range=(0, 1))
    # Scaled_data = scaler.fit_transform(df.values)

    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        # names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        names += [f"var{j+1}(t-{i})" for j in range(n_vars)]

    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            # names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            names += [f"var{j+1}(t)" for j in range(n_vars)]
        else:
            # names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
            names += [f"var{j+1}(t+{i})" for j in range(n_vars)]

    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    agg.dropna(inplace=True)

    n_split = int(0.3 * len(data))

    train_X = Variable(torch.from_numpy(agg[n_split:, :-1]), requires_grad=False)
    train_y = Variable(torch.from_numpy(agg[n_split:, 1:]), requires_grad=False)
    test_X = Variable(torch.from_numpy(agg[:n_split, :-1]), requires_grad=False)
    test_y = Variable(torch.from_numpy(agg[:n_split, 1:]), requires_grad=False)

    return train_X, train_y, test_X, test_y


def filter_outliers(df: pd.DataFrame, z_thresh: float) -> pd.DataFrame:
    df = df[(np.abs(zscore(df)) < z_thresh).all(axis=1)]

    # Reset the index
    df = df.reset_index(drop=True)

    return df

def inject_outliers(data: np.ndarray, prob: float, outlier_threshold: float, limits: tuple) -> np.ndarray:
    """Simulate sensor faults by injecting outliers into a column. The probability
    of an outlier being injected is given by `prob`.

    Args:
        df (np.ndarray): The dataframe to inject outliers into.
        prob (float): The probability of an outlier being injected.
        outlier_threshold (float): The threshold for determining outliers.
        limits (tuple): The lower and upper limits of the column.

    Returns:
        pd.DataFrame: The dataframe with outliers injected.

    """

    # Create a copy of the dataframe
    data = data.copy()

    # Compute the mean and standard deviation of the column
    mean, std = data.mean(), data.std()

    # Compute the lower and upper bounds for outliers
    lower, upper = mean - outlier_threshold * std, mean + outlier_threshold * std

    # Generate a random outlier value for each row (upper outlier or lower outlier)
    for i in range(1, len(data)-1):
        if np.random.random() < prob:
            print(f"Outlier injected at row {i}")
            if np.random.random() < 0.5:
                data[i] = np.random.uniform(limits[0], lower)
            else:
                data[i] = np.random.uniform(upper, limits[1])

    return data

def inject_offset(data: np.ndarray, offset: float, failure_rate: float) -> np.ndarray:
    """Inject an offset into a column. The offset is introduced at a random row
    based on the exponential distribution with the given failure rate. After the offset
    is introduced, it is added to each subsequent row.

    Args:
        df (np.ndarray): The dataframe to inject an offset into.
        offset (float): The offset to inject.
        failure_rate (float): The average number of rows between each offset injection.

    Returns:
        np.ndarray: The dataframe with the offset injected.

    """

    # Create a copy of the dataframe
    data = data.copy()

    # Compute the row, where the offset is introduced
    row = np.random.exponential(failure_rate)

    # Inject the offset into the dataframe
    data[int(row):] += offset

    return data

def inject_drift(data: np.ndarray, drift_slope: float, failure_rate: float) -> np.ndarray:
    """Inject a drift into a column. The drift is introduced at a random row
    based on the exponential distribution with the given failure rate. After the drift
    is introduced, it is added to each subsequent row.

    Args:
        df (np.ndarray): The dataframe to inject a drift into.
        drift_slope (float): The slope of the drift.
        failure_rate (float): The average number of rows between each drift injection.

    Returns:
        np.ndarray: The dataframe with the drift injected.

    """

    # Create a copy of the dataframe
    data = data.copy()

    # Compute the row, where the drift is introduced
    row = int(np.random.exponential(failure_rate))

    # Compute the linear function of the drift error for each subsequent row
    x = np.arange(len(data) - row) + 1
    drift = x * drift_slope

    # Inject the drift into the dataframe
    data[int(row):] += drift

    return data

def detect_anomalies(measurements: np.ndarray, predictions: np.ndarray, error_threshold: float, outlier_threshold: float) -> np.ndarray:
    """Detect drift in a time series based on the prediction error. The drift is
    detected by computing the integral of the prediction error and comparing it to the quadratic
    function of the number of measurements.

    Args:
        measurements (np.ndarray): The measurements of the time series.
        predictions (np.ndarray): The predictions of the time series.
        error_threshold (float): The threshold for determining drift.
        outlier_threshold (float): The threshold for determining outliers.

    Returns:
        np.ndarray: The index of the drift.
    """

    # Compute the prediction error
    error = measurements - predictions
    abs_error = abs(error)
    time = np.arange(len(error))

    # Detect the outliers
    outlier_indices = np.where(zscore(abs_error) > outlier_threshold)[0]
    outliers = error[outlier_indices]
    not_outliers = np.where(zscore(abs_error) <= outlier_threshold)[0]

    plt.plot(time, error, label="Error")

    # Filter the outliers
    error = error[not_outliers]
    abs_error = abs_error[not_outliers]
    time = time[not_outliers]

    plt.scatter(outlier_indices, outliers, label="Outliers", color="red")
    plt.show()

    plt.plot(time, error, label="Error")
    plt.show()

    # Keep only the error values that are above the threshold
    # valid_indices = np.where(abs_error > error_threshold)[0]
    # error = error[valid_indices]
    # time = time[valid_indices]

    plt.plot(time, error, label="Error")
    plt.show()

    # Compute the integral of the prediction error
    integral = np.cumsum(error)

    # Interpolate the quadratic function of the number of measurements
    fit, residuals, rank, singular_values, rcond = np.polyfit(time, integral, 1, full=True)
    model = np.poly1d(fit)
    linear = model(time)

    # Plot the integral of the prediction error and the quadratic function of the number of measurements
    plt.plot(time, integral)
    plt.plot(time, linear)
    plt.show()
    print(f"Model: {model}")
    print(f"Outliers: {outliers}")



if __name__ == "__main__":
    data = load_data('House_12.csv')

    # Visualize the column "T_1ST_AVG"
    data['T_1ST_AVG'].plot()
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.show()

    # Filter the outliers
    data = filter_outliers(data, 3)

    # Visualize the column "T_1ST_AVG" after filtering the outliers
    data['T_1ST_AVG'].plot()
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.show()

    data_old = data.copy()

    # # Inject outliers into the column "T_1ST_AVG"
    # data['T_1ST_AVG'] = inject_outliers(data['T_1ST_AVG'], 0.001, 3, (0, 100))
    #
    # # Visualize the column "T_1ST_AVG" after injecting outliers
    # data['T_1ST_AVG'].plot()
    # plt.xlabel('Time')
    # plt.ylabel('Temperature')
    # plt.show()
    #
    # print(detect_outliers(data['T_1ST_AVG'], data_old['T_1ST_AVG'], 2))

    # # Inject an offset into the column "T_1ST_AVG"
    # data['T_1ST_AVG'] = inject_offset(data['T_1ST_AVG'], 10, 1000)
    #
    # # Visualize the column "T_1ST_AVG" after injecting an offset
    # data['T_1ST_AVG'].plot()
    # plt.xlabel('Time')
    # plt.ylabel('Temperature')
    # plt.show()
    #
    # detect_offset(data['T_1ST_AVG'].array, data_old['T_1ST_AVG'].array, 0.1)

    # Inject a drift into the column "T_1ST_AVG"
    data['T_1ST_AVG'] = inject_drift(data['T_1ST_AVG'], 0.1, 1000)

    # Visualize the column "T_1ST_AVG" after injecting a drift
    data['T_1ST_AVG'].plot()
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.show()

    # detect_drift(data['T_1ST_AVG'].array, data_old['T_1ST_AVG'].array, 0.1)