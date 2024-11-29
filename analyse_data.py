"""
@author: Radoslaw Plawecki
"""
import os

from DTW.common import use_latex, check_column_existence, make_blocks
from DTW.dtw import DTW
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path, listdir


class AnalyseData:
    def __init__(self, directory, filename, x, y):
        filepath = f"patients/mockup_data/{directory}/{filename}.csv"
        if not path.exists(filepath):
            raise FileNotFoundError("File not found!")
        if not path.isfile(filepath):
            raise IsADirectoryError("The path exists but is not a file!")
        if path.splitext(filepath)[1] != '.csv':
            raise ValueError("File must be a CSV file!")
        self.directory = directory
        self.filename = filename
        data = pd.read_csv(filepath, delimiter=';')
        df = pd.DataFrame(data)
        check_column_existence(df=df, col=x)
        check_column_existence(df=df, col=y)
        self.x, self.y = make_blocks(df[x]), make_blocks(df[y])

    def dtw_analyse(self, method):
        x, y = self.x, self.y
        results = []
        for i in range(len(x)):
            dtw = DTW(x[i], y[i])
            results.append(dtw.calc_alignment_cost(method))
        return results

    def export_results(self, directory, method):
        results = self.dtw_analyse(method=method)
        files, first_block, second_block, third_block = "Plik", "I. blok", "II. blok", "III. blok"
        filenames = listdir(f"patients/mockup_data/{directory}")
        print(filenames)
        data = {
            files: filenames,
            first_block: results[0],
            second_block: results[1],
            third_block: results[2]
        }
