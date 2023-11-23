import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from utils import load_data, filter_outliers, create_dataset


def main():
    data = load_data('House_12.csv')
    data = filter_outliers(data, 3)
    dataset = create_dataset(data, 100, 1)
    print(dataset)


if __name__ == "__main__":
    main()
