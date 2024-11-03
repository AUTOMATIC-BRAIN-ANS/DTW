"""
@author: Radoslaw Plawecki
Sources:
[1] Cao, X. H., Stojkovic, I., & Obradovic, Z. (2016). A robust data scaling algorithm to improve classification
accuracies in biomedical data. BMC Bioinformatics, 17, 359. Available on: https://doi.org/10.1186/s12859-016-1236-x.
Access: 30.10.2024.
"""

import numpy as np
from scipy.optimize import minimize


class GeneralizedLogistic:
    def __init__(self, data):
        self.data = data
        self.ecdf_values = np.arange(1, len(data) + 1) / len(data)

    @staticmethod
    def generalized_logistic(x, Q, B, M, nu):
        return 1 / (1 + Q * np.exp(-B * (x - M))) ** (1 / nu)

    def objective(self, params):
        Q, B, M, nu = params
        gl_values = self.generalized_logistic(self.data, Q, B, M, nu)
        return np.sum((gl_values - self.ecdf_values) ** 2)

    def initialize_parameters(self):
        x_min, x_med, x_max = np.min(self.data), np.median(self.data), np.max(self.data)
        M0 = x_med
        Q0 = 2 * (x_max - x_min) / (x_max - x_med) - 1
        B0 = 4 / (x_max - x_min)
        nu0 = 1
        return Q0, B0, M0, nu0

    def fit_gl_to_ecdf(self):
        initial_params = self.initialize_parameters()
        result = minimize(self.objective, initial_params, method='L-BFGS-B', bounds=[(0.1, 10), (0.1, 10),
                                                                                     (None, None), (0.1, 10)])
        if result.success:
            optimized_params = result.x
        else:
            raise RuntimeError("Optimization failed: " + result.message)
        return optimized_params

    def __call__(self):
        Q_opt, B_opt, M_opt, nu_opt = self.fit_gl_to_ecdf()
        return self.generalized_logistic(self.data, Q_opt, B_opt, M_opt, nu_opt)
