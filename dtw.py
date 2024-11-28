"""
@author: Radoslaw Plawecki
Sources:
[1] Kamper, H. (2021). Dynamic time warping 2: Algorithm [Video]. YouTube.
Available on: https://www.youtube.com/watch?v=X6phfLqN5pY&list=PLmZlBIcArwhMJoGk5zpiRlkaHUqy5dLzL&index=3.
Access: 29.10.2024.
[2] Kamper, H. (2021). Dynamic time warping (DTW) tutorial notebook. GitHub. Available on:
https://github.com/kamperh/lecture_dtw_notebook/blob/main/dtw.ipynb. Access: 29.10.2024.
"""

from DTW.common import use_latex
import numpy as np
import matplotlib.pyplot as plt


class DTW:
    __cost_matrix = []
    __matches, __insertions, __deletions = 0, 0, 0
    __path = []

    def __init__(self, x, y):
        """
        Method to initialize params of a class.
        :param x: first signal.
        :param y: second signal.
        """
        self.x = np.array(x)
        self.y = np.array(y)

    def __initialize_matrix(self):
        """
        Method to initialize a matrix.
        :return: initialized matrix.
        """
        x, y = self.x, self.y
        rows, cols = len(x) + 1, len(y) + 1
        matrix = np.zeros([rows, cols])
        matrix[0, 1:], matrix[1:, 0], matrix[0, 0] = np.inf, np.inf, 0
        return matrix

    def fill_matrix(self):
        """
        Method to fill a matrix.
        :return: filled matrix.
        """
        x, y = self.x, self.y
        matrix = self.__initialize_matrix()
        rows, cols = np.shape(matrix)
        for i in range(1, rows):
            for j in range(1, cols):
                distance = abs(x[i - 1] - y[j - 1])
                component = np.min([matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i][j - 1]])
                matrix[i][j] = distance + component
        return matrix

    def __init_global_variables(self, i, j):
        """
        Method to initialize global variables.
        :param i: number of rows in a matrix.
        :param j: number of columns in a matrix.
        :return: initialized global variables.
        """
        self.__cost_matrix = self.fill_matrix()
        self.__matches, self.__insertions, self.__deletions = 0, 0, 0
        self.__path = [(i - 1, j - 1)]

    def traceback(self):
        """
        Method get traceback matrix.
        :return: traceback matrix.
        """
        x, y = self.x, self.y
        i, j = rows, cols = len(x), len(y)
        self.__init_global_variables(i, j)
        traceback_matrix = np.zeros([rows + 1, cols + 1])
        cost_matrix = self.__cost_matrix
        while i > 0 and j > 0:
            score = cost_matrix[i][j]
            distance = abs(x[i - 1] - y[j - 1])
            match, insertion, deletion = [cost_matrix[i - 1][j - 1],
                                          cost_matrix[i - 1][j],
                                          cost_matrix[i][j - 1]]
            if score == distance + match:
                traceback_matrix[i][j] = 1
                self.__matches += 1
                i -= 1
                j -= 1
            elif score == distance + insertion:
                traceback_matrix[i][j] = 1
                self.__insertions += 1
                i -= 1
            else:
                traceback_matrix[i][j] = 1
                self.__deletions += 1
                j -= 1
            self.__path.append((i - 1, j - 1))
        return traceback_matrix

    def get_statistics(self):
        """
        Method to get statistics of DTW.
        :return: number of matches, insertions and deletions.
        """
        self.traceback()
        return self.__matches, self.__insertions, self.__deletions

    def calc_alignment_cost(self, method):
        """
        Method to calculate alignment cost using a given method.
        :param method: method to calculate alignment cost.
        :return: alignment cost.
        """
        if method == "d-method":
            return self.__use_distance_method()
        elif method == "td-method":
            return self.__use_time_distance_method()
        elif method == "c-method":
            return self.__use_cost_method()
        else:
            raise ValueError(f"Allowed methods to calculate alignment cost are: 'd-method', 'td-method' and "
                             f"'c-method'. Got '{method}' instead.")

    def __use_distance_method(self):
        """
        Method to calculate alignment cost using the distance method.
        :return: alignment cost.
        """
        matrix = self.fill_matrix()[1:, 1:]
        n, m = np.shape(matrix)
        return matrix[n - 1][m - 1] / (n + m)

    def __use_time_distance_method(self):
        """
        Method to calculate alignment cost using the time-distance method.
        :return: alignment cost.
        """
        self.traceback()
        matrix = self.__cost_matrix[1:, 1:]
        cost = sum(matrix[n, m] for n, m in self.__path[:-1])
        len_traceback = len(self.__path[:-1])
        return cost / len_traceback

    def __use_cost_method(self):
        """
        Method to calculate alignment cost using the cost method.
        :return: alignment cost.
        """
        matches, insertions, deletions = self.get_statistics()
        len_traceback = matches + insertions + deletions
        return matches / len_traceback

    def __sliding_window_dtw(self, window_size, step):
        if window_size < 5:
            raise ValueError("Window is not big enough!")
        if step <= 0:
            raise ValueError("Step must have a positive value!")
        x, y = self.x, self.y
        windows = []
        alignment_costs = []
        for i in range(0, max(len(x), len(y)) - window_size + 1, step):
            window = [i, window_size + i]
            alignment_cost = DTW(x[window[0]:window[1]], y[window[0]:window[1]]).fill_matrix()[-1, -1]
            windows.append(window)
            alignment_costs.append(alignment_cost)
        return alignment_costs, windows

    def __perform_dtw_window(self, windows, pos):
        dtw = DTW(self.x[windows[pos][0]:windows[pos][1]], self.y[windows[pos][0]:windows[pos][1]])
        dtw.traceback()
        dtw.plot_signals()
        dtw.plot_alignment()
        dtw.plot_cost_matrix()

    def __get_min_alignment_cost(self, window_size, step):
        alignment_costs, windows = self.__sliding_window_dtw(window_size, step)
        min_alignment_cost = np.min(alignment_costs)
        min_positions = np.where(alignment_costs == min_alignment_cost)[0]
        for position in min_positions:
            self.__perform_dtw_window(windows, position)
        return min_alignment_cost

    def __get_max_alignment_cost(self, window_size, step):
        alignment_costs, windows = self.__sliding_window_dtw(window_size, step)
        max_alignment_cost = np.max(alignment_costs)
        max_positions = np.where(alignment_costs == max_alignment_cost)[0]
        for position in max_positions:
            self.__perform_dtw_window(windows, position)
        return max_alignment_cost

    def find_min_max_alignment_cost(self, look_for=None, window_size=10, step=1):
        if look_for != "MIN" and look_for != "MAX":
            raise ValueError("Type 'MIN' or 'MAX' to look for the window with the smallest and the biggest alignment "
                             "cost, relatively.")
        if look_for == "MIN":
            return self.__get_min_alignment_cost(window_size, step)
        else:
            return self.__get_max_alignment_cost(window_size, step)

    def plot_signals(self, x_signal=None, y_signal=None, filename=None):
        use_latex()
        x, y = self.x, self.y
        start = 0
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
        ax[0].set_ylabel(f"{x_signal}")
        ax[0].set_title(f"Zależność {x_signal} od czasu")
        ax[0].set_xlim(xmin=0, xmax=tx_stop)
        ax[0].grid()
        # add a title and labels to the 2nd plot
        ax[1].set_xlabel("Czas [s]")
        ax[1].set_ylabel(f"{y_signal}")
        ax[1].set_title(f"Zależność {y_signal} od czasu")
        ax[1].set_xlim(xmin=0, xmax=ty_stop)
        ax[1].grid()
        if filename is not None:
            plt.savefig(f"{filename}.pdf", format='pdf')
        plt.show()

    def plot_cost_matrix(self, x_signal=None, y_signal=None, filename=None):
        use_latex()
        plt.figure(figsize=(6, 4))
        c = plt.imshow(self.fill_matrix()[1:, 1:], cmap=plt.get_cmap("Blues"), interpolation="nearest", origin="upper")
        plt.colorbar(c)
        x_path, y_path = zip(*self.__path[:-1])
        plt.plot(y_path, x_path, color="#003A7D", linewidth=1.5)
        plt.title("Macierz kosztów")
        plt.xlabel(f"{x_signal}")
        plt.ylabel(f"{y_signal}")
        plt.legend(['Dopasowanie'])
        if filename is not None:
            plt.savefig(f"{filename}.pdf", format='pdf')
        plt.show()

    def plot_alignment(self, x_signal=None, y_signal=None, filename=None):
        use_latex()
        x, y = self.x, self.y
        plt.figure(figsize=(6, 4))
        for x_i, y_j in self.__path[:-1]:
            plt.plot([x_i, y_j], [x[x_i] + 1.5, y[y_j] - 1.5], c="C7")
        plt.plot(np.arange(x.shape[0]), x + 1.5, "-o", c="C3")
        plt.plot(np.arange(y.shape[0]), y - 1.5, "-o", c="C0")
        plt.xlabel(f"{x_signal}")
        plt.ylabel(f"{y_signal}")
        plt.axis("off")
        if filename is not None:
            plt.savefig(f"{filename}.pdf", format='pdf')
        plt.show()
