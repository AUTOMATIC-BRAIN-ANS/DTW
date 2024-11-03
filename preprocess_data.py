"""
@author: Radoslaw Plawecki
Sources:
https://medium.com/@nirajan.acharya777/understanding-outlier-removal-using-interquartile-range-iqr-b55b9726363e
"""

from DTW.common import use_latex, values_in_order
from DTW.normalization import NormalizeData
from DTW.nan_handler import NaNHandler as nanh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path


class PreprocessData:
    def __init__(self, filepath, first_column, second_column):
        """
        Method to initialize params of a class.
        :param filepath: path to a file with data.
        :param first_column: name of a first column to analyse.
        :param second_column: name of a second column to analyse.
        :raise FileNotFoundError: if a file is not found.
        :raise IsADirectoryError: if a path exists but leads to, for example, directory instead of a file.
        :raise ValueError: if a file is not a CSV file.
        """
        if not path.exists(filepath):
            raise FileNotFoundError("File not found!")
        if not path.isfile(filepath):
            raise IsADirectoryError("The path exists but is not a file!")
        if path.splitext(filepath)[1] != '.csv':
            raise ValueError("File must be a CSV file!")
        data = pd.read_csv(filepath, delimiter=';')
        df = pd.DataFrame(data)
        self.__check_column_existence(df=df, col=first_column)
        self.__check_column_existence(df=df, col=second_column)
        s1, s2 = df[first_column], df[second_column]
        self.first_signal, self.second_signal = (nanh.replace_zeros_with_nans(s1),
                                                 nanh.replace_zeros_with_nans(s2))

    def get_first_signal(self):
        """
        Getter to get a first, initialized signal.
        :return:
        """
        return self.first_signal

    def get_second_signal(self):
        """
        Getter to get a second, initialized signal.
        :return: a second signal.
        """
        return self.second_signal

    @staticmethod
    def __check_column_existence(df, col):
        """
        Method to check if a column with the given name exists in a file.
        :param df: data as DataFrame object.
        :param col: name of a column to look for.
        :raise KeyError: if a column doesn't exist in a file.
        """
        if not list(df.columns).__contains__(col):
            raise KeyError(f"Column '{col}' doesn't exist in a file!")

    @staticmethod
    def iqr_outlier_removal(s, threshold):
        """
        Method to detect and remove outliers using the IQR method.
        :param s: signal.
        :param threshold: threshold after which the outliers will be removed.
        :return: signal with removed outliers.
        """
        q1 = np.nanpercentile(s, 25)
        q3 = np.nanpercentile(s, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        # copy signal to avoid modifying the original
        s = np.array(s.copy())
        for i in range(1, len(s) - 1, 1):
            if s[i] < lower_bound or s[i] > upper_bound:
                s[i] = np.nan
        return s

    def remove_outliers(self, threshold=1.5):
        s1, s2 = self.get_first_signal(), self.get_second_signal()
        return self.iqr_outlier_removal(s1, threshold), self.iqr_outlier_removal(s2, threshold)

    def get_first_signal_outliers_removed(self):
        return self.remove_outliers()[0]

    def get_second_signal_interpolated(self):
        return self.remove_outliers()[1]

    @staticmethod
    def trim_signals(s1, s2):
        nan_in_s1, nan_in_s2 = nanh.get_nan_number(s1), nanh.get_nan_number(s2)
        if nan_in_s1 >= nan_in_s2:
            s = pd.Series(s1)
        else:
            s = pd.Series(s2)
        start, stop = 0, len(s)
        nan_indices = s.index[s.isna()].tolist()
        ordered_nans = values_in_order(nan_indices)
        for count, loc in ordered_nans:
            if count > 300:
                if abs(start - loc) < abs(loc - stop):
                    start = loc + count
                else:
                    stop = loc
        return start, stop

    @staticmethod
    def __fill_nans(s1, s2, method=None, order=None):
        ppd = PreprocessData
        start, stop = ppd.trim_signals(s1, s2)
        s1, s2 = s1[start:stop], s2[start:stop]
        s1, s2 = s1.interpolate(method=method, order=order), s2.interpolate(method=method, order=order)
        if len(s1) < 600 or len(s2) < 600:
            raise ValueError("Signal has too many gaps!")
        else:
            return s1, s2

    def interpolate_signal(self, method=None, order=None):
        s1, s2 = self.first_signal, self.second_signal
        try:
            return self.__fill_nans(s1, s2, method=method, order=order)
        except ValueError as e:
            print(f"Error occurred: {e}")

    def get_first_signal_interpolated(self):
        return self.interpolate_signal(method='quadratic')[0]

    def get_second_signal_interpolated(self):
        return self.interpolate_signal(method='quadratic')[1]

    def normalize_signal(self, method, outlier_removal=True):
        if outlier_removal is True:
            nd = NormalizeData(self.remove_outliers())
        else:
            nd = NormalizeData(self.get_signal_interpolated())
        return nd.normalize(method=method, min_value=-1, max_value=1)

    def get_preprocessed_signal(self):
        return self.normalize_signal()

    @staticmethod
    def get_time(signal):
        return np.linspace(0, len(signal), len(signal))

    def get_data_for_plot(self):
        signals = [self.signal, self.get_signal_interpolated(),
                   self.remove_outliers(), self.normalize_signal()]
        timeseries = [self.get_time(signals[0]), self.get_time(signals[1]),
                      self.get_time(signals[2]), self.get_time(signals[3])]
        titles = ["Sygnał nieprzetworzony", "Sygnał interpolowany",
                  "Sygnał po usunięciu wartości odstających", "Sygnał znormalizowany"]
        return timeseries, signals, titles

    def plot_signals(self, filename=None):
        use_latex()
        timeseries, signals, titles = self.get_data_for_plot()
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.65)
        k = 0
        label_pad = 12
        label_fontsize = 12
        for i in range(2):
            for j in range(2):
                ax[i, j].plot(timeseries[k], signals[k])
                ax[i, j].set_title(titles[k], pad=label_pad, fontsize=label_fontsize + 2)
                ax[i, j].set_xlabel("Czas [min]", labelpad=label_pad, fontsize=label_fontsize)
                ax[i, j].set_ylabel("Amplituda [a.u.]", labelpad=label_pad, fontsize=label_fontsize)
                ax[i, j].set_xlim(xmin=0, xmax=len(timeseries[k]))
                ax[i, j].grid()
                k += 1
        plt.show()
        if filename is not None:
            plt.savefig(f"{filename}.pdf", format='pdf')
        plt.show()
