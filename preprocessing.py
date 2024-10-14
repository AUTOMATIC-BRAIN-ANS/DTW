"""
@author: Radoslaw Plawecki
Sources:
https://medium.com/@nirajan.acharya777/understanding-outlier-removal-using-interquartile-range-iqr-b55b9726363e
"""

from DTW.common import use_latex, values_in_order
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path


class PreprocessData:
    def __init__(self, filepath, column):
        if not path.exists(filepath):
            raise FileNotFoundError("File not found!")
        if not path.isfile(filepath):
            raise IsADirectoryError("The path exists but is not a file!")
        if path.splitext(filepath)[1] != '.csv':
            raise ValueError("File must be a CSV file!")
        data = pd.read_csv(filepath, delimiter=';')
        df = pd.DataFrame(data)
        if not list(df.columns).__contains__(column):
            raise KeyError("Column doesn't exist in a file!")
        self.signal = df[column]

    def interpolate_signal(self, method=None, order=None):
        signal = self.signal
        start, stop = 0, len(signal)
        nan_indices = signal.index[signal.isna()].tolist()
        ordered_nans = values_in_order(nan_indices)
        for count, loc in ordered_nans:
            if count > 5:
                if abs(start - loc) < abs(loc - stop):
                    start = loc + count
                else:
                    stop = loc
        signal = signal[start:stop]
        return signal.interpolate(method=method, order=order)

    def remove_outliers(self, threshold=2.5):
        signal = self.interpolate_signal(method='cubic')
        q1 = np.percentile(signal, 25)
        q3 = np.percentile(signal, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        new_signal = signal[(signal >= lower_bound) & (signal <= upper_bound)]
        return new_signal

    def preprocess_signal(self):
        return self.remove_outliers()

    def plot_signals(self, filename=None):
        use_latex()
        start = 0
        x, y = self.signal, self.preprocess_signal()
        tx_stop = tx_num = len(x)
        ty_stop = ty_num = len(y)
        tx = np.linspace(start=start, stop=tx_stop, num=tx_num)
        ty = np.linspace(start=start, stop=ty_stop, num=ty_num)
        fig, ax = plt.subplots(2)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.55)
        ax[0].plot(tx, x)
        ax[1].plot(ty, y)
        # add a title and labels to the 1st plot
        ax[0].set_xlabel("Czas [s]")
        ax[0].set_ylabel("Raw signal")
        ax[0].set_title(f"Zależność sygnału od czasu")
        ax[0].set_xlim(xmin=0, xmax=tx_stop)
        ax[0].grid()
        # add a title and labels to the 2nd plot
        ax[1].set_xlabel("Czas [s]")
        ax[1].set_ylabel("Preprocessed signal")
        ax[1].set_title("Zależność sygnału od czasu")
        ax[1].set_xlim(xmin=0, xmax=ty_stop)
        ax[1].grid()
        if filename is not None:
            plt.savefig(f"{filename}.pdf", format='pdf')
        plt.show()
