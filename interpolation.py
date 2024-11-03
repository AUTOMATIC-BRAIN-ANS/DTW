"""
@author: Radoslaw Plawecki
Sources:
[1] Esquef, P. A. A., Välimäki, V., Roth, K., & Kauppinen, I. (2003). Interpolation of long gaps in audio signals using
the warped Burg’s method. In Proceedings of the 6th International Conference on Digital Audio Effects (DAFx-03).
London, UK. Available on:
https://aaltodoc.aalto.fi/server/api/core/bitstreams/d74aa35d-ac3b-4e7c-83dc-da7ad590f3bb/content. Access: 03.11.2024.
"""

import numpy as np


class WarpedBasedInterpolation:
    def __init__(self, signal, order=20, warp_factor=0.8):
        if order < 1:
            raise ValueError(f"Order must be greater than 1! Got {order} instead.")
        if warp_factor > 1 or warp_factor < 0:
            raise ValueError("Warping factor takes values from 0 to 1!")
        self.signal = np.array(signal)
        self.order = order
        self.warp_factor = warp_factor
        self.reflection_coefficients = np.zeros(order)
        self.ar_coefficients = None

    def calc_bwd_pred_error(self):
        b = self.signal
        m = len(b)  # stage
        n = len(b)  # element
        b_alt = np.zeros([m, n])
        b_alt[0] = b
        for i in range(1, m):
            for j in range(i, n):
                b_alt[i][j] = b_alt[i - 1][j - 1] - self.warp_factor * (b_alt[i - 1][j] - b_alt[i][j - 1])
        return b_alt
