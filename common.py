"""
@author: Radoslaw Plawecki
"""

import matplotlib.pyplot as plt


def use_latex():
    # use LaTeX for text rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({
        'text.latex.preamble': r'\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc}'
    })

def values_in_order(index_list):
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
