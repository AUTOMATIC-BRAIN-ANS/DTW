"""
@author: Radoslaw Plawecki
"""

import matplotlib.pyplot as plt
import numpy as np


def use_latex():
    """
    Function to use LaTeX formatting for plots.
    """
    # use LaTeX for text rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({
        'text.latex.preamble': r'\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc}'
    })


def check_column_existence(df, col):
    """
    Function to check if a column with the given name exists in a file.
    :param df: data as DataFrame object.
    :param col: name of a column to look for.
    :raise KeyError: if a column doesn't exist in a file.
    """
    if not list(df.columns).__contains__(col):
        raise KeyError(f"Column '{col}' doesn't exist in a file!")


def values_in_order(index_list):
    """
    Function to replace the list of rowed indices into an array of tuples (A, B), where A is the number in sequence
    occurring indices, and B is the first element of the given sequence.
    :param index_list: list of rowed indices.
    :return: array of tuples.
    """
    count = 1
    summary = []
    for i in range(len(index_list) - 1, -1, -1):
        if index_list[i] == index_list[i - 1] + 1:
            count += 1
        else:
            if count >= 1:
                last_index = index_list[i]
                summary.append((count, last_index))
            count = 1
    return list(reversed(summary))


def make_blocks(s):
    """
    Function to make divide a signal into blocks.
    :param s: signal.
    :return: signal divided into blocks.
    """
    # one_day = 24 * 60 * 60 // 10
    one_day = 4
    s = [s[0:one_day], s[one_day:one_day * 2], s[one_day * 2:]]
    return s


def filter_toxa(df, col_sto2, col_filtered, min_value=20, max_value=100):
    """
    Function to clear the TOXA signal from artefacts (when STO2 < 20 or STO2 > 100).
    :param df: data in the DataFrame format.
    :param col_sto2: column with values of the STO2 signal.
    :param col_filtered: column to be filtered with values of the STO2/TOXA signal.
    :param min_value: minimum value of a range.
    :param max_value: maximum value of a range.
    :return: cleared signal.
    :raise ValueError: if signals have different length.
    """
    if min_value >= max_value:
        raise ValueError("Minimum value cannot be greater or equal than maximum value!")
    sto2, toxa = df[col_sto2].copy(), df[col_filtered].copy()
    if len(sto2) != len(toxa):
        raise ValueError("Signals have different lengths!")
    length = len(toxa)
    for i in range(1, length):
        if sto2[i] > max_value or sto2[i] < min_value:
            sto2[i], toxa[i] = np.nan, np.nan
    if col_filtered == 'Toxa':
        return toxa
    else:
        return sto2
