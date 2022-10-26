import sys
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


class EquationSolver(object):
    def __init__(self):
        self._rows: int = 13
        self._cols: int = 14
        self._mat: np.ndarray = np.zeros(shape=(self._rows, self._rows), dtype=float)
        for i in range(self._rows):
            for j in range(self._rows):
                self._mat[i][j] = 1 / (i + j + 1)
        self._b: np.ndarray = np.array(
            [1889 / 594, 1441 / 640, 3511 / 1931, 1812 / 1171, 1363 / 1005, 1379 / 1138, 1797 / 1637,
             1029 / 1024, 1548 / 1669, 1990 / 2309, 1059 / 1315, 1088 / 1439, 397 / 557],
            dtype=float)
        self._max_iterations: int = 10000
        self._threshold: float = 1e-3
        # The estimated solution is all zero
        self._estimated_solution = np.array(np.zeros(self._rows))

    def _print_solution(self, x: np.ndarray):
        for i in range(self._rows):
            print(f"x_{i} = {x[i]}", end="\n")
        print()

    @staticmethod
    def _show_norm_line_chart(norms: list[float], label: str, x_axis: float = -1):
        iterations: np.ndarray = np.arange(len(norms))
        plt.plot(iterations, norms, color='r', marker='.',
                 markeredgecolor='r', markersize='1', markeredgewidth=1, label=label)
        if x_axis > 0:
            x_major_locator = MultipleLocator(x_axis)
            ax = plt.gca()
            ax.xaxis.set_major_locator(x_major_locator)
        plt.xlabel("iterations")
        plt.ylabel("norm")
        plt.legend(loc="best")
        plt.show()

    @abstractmethod
    def solve(self):
        pass


class GaussianEliminationSolver(EquationSolver):
    def __init__(self):
        super().__init__()
        self._mat: np.ndarray = np.zeros(shape=(self._rows, self._cols), dtype=float)
        for i in range(self._rows):
            for j in range(self._rows):
                self._mat[i][j] = 1 / (i + j + 1)
        self._mat[:, self._cols - 1] = self._b.T

    @staticmethod
    def __swag(a: np.ndarray, b: np.ndarray):
        for i in range(0, len(a)):
            t = a[i]
            a[i] = b[i]
            b[i] = t

    def __select(self, i, col):
        if 0.00 in set(self._mat[i]) and len(set(self._mat[i])) == 1:
            return
        for k in range(0, i):
            temp = self._mat[i][k] / self._mat[k][k]
            if temp == 0:
                continue
            for j in range(0, col):
                self._mat[i][j] = self._mat[i][j] - self._mat[k][j] * temp

    def __check(self, i, row, col):
        if 0.00 in set(self._mat[i]) and len(set(self._mat[i])) == 1:
            for j in range(row - 1, i, -1):
                if not (0.00 in set(self._mat[j]) and len(set(self._mat[j])) == 1):
                    self.__swag(self._mat[i], self._mat[j])
                    self.__select(i, col)
                    break

    def __get_unique_solution(self):
        for i in range(self._rows):
            temp = self._mat[i][i]
            for j in range(i, self._cols):
                self._mat[i][j] = self._mat[i][j] / temp
        for i in range(self._rows - 1):
            for j in range(i + 1, self._cols - 1):
                temp = self._mat[i][j]
                for k in range(j, self._cols):
                    self._mat[i][k] = self._mat[i][k] - self._mat[j][k] * temp

    def solve(self):
        for i in range(self._rows):
            if self._mat[i][i] == 0:
                for j in range(i + 1, self._rows):
                    if self._mat[j][i] != 0:
                        self.__swag(self._mat[i], self._mat[j])
                        break
            self.__select(i, self._cols)
            self.__check(i, self._rows, self._cols)
        # By default, the equation has a solution
        self.__get_unique_solution()
        print("Gaussian Elimination Solution:")
        self._print_solution(self._mat[:, self._cols - 1].T)


class GaussSeidelSolver(EquationSolver):
    def solve(self):
        x0: np.ndarray = self._estimated_solution.copy()
        x: np.ndarray = x0.copy()
        k: int = 0
        temp_x: np.ndarray = x0.copy()
        norms: list[float] = []
        while k < self._max_iterations:
            for i in range(self._rows):
                temp = 0
                temp_x = x0.copy()
                for j in range(self._rows):
                    if i != j:
                        temp += x[j] * self._mat[i][j]
                x[i] = (self._b[i] - temp) / self._mat[i][i]
                x0[i] = x[i].copy()
            norm: float = np.linalg.norm(x - temp_x)
            norms.append(norm)
            k += 1
            if norm < self._threshold:
                break
            else:
                x0 = x.copy()
        print("Gauss Seidel Solution:")
        print(f"Iterations: {k}")
        self._print_solution(x)
        self._show_norm_line_chart(norms=norms, label="Gauss Seidel Norm")


class GradientDescentSolver(EquationSolver):
    def solve(self):
        x0: np.ndarray = self._estimated_solution.copy()
        x: np.ndarray = x0.copy()
        k = 0
        norms: list[float] = []
        while k < self._max_iterations:
            x0 = x.copy()
            k += 1
            r = self._b - np.dot(self._mat, x0)
            a = np.dot(r.T, r) / np.dot(r.T, np.dot(self._mat, r))
            x = x0 + a * r
            norm: float = np.linalg.norm(x0 - x)
            norms.append(norm)
            if norm < self._threshold:
                break
        print("Gradient Descent Solution:")
        print(f"Iterations: {k}")
        self._print_solution(x)
        self._show_norm_line_chart(norms=norms, label="Gradient Descent Norm")


class ConjugateGradientSolver(EquationSolver):
    def solve(self):
        k: int = 0
        x: np.ndarray = self._estimated_solution.copy()
        r: np.ndarray = self._b - np.dot(self._mat, x)
        p = r.copy()
        norms: list[float] = []
        while True:
            k += 1
            r1 = r
            a = np.dot(r.T, r) / np.dot(p.T, np.dot(self._mat, p))
            x = x + a * p
            r = r - a * np.dot(self._mat, p)
            norm = np.linalg.norm(np.dot(self._mat, x) - self._b) / np.linalg.norm(self._b)
            norms.append(norm)
            if norm < self._threshold:
                break
            else:
                beta = np.linalg.norm(r) ** 2 / np.linalg.norm(r1) ** 2
                p = r + beta * p
        print("Conjugate Gradient Solution:")
        print(f"Iterations: {k}")
        self._print_solution(x)
        self._show_norm_line_chart(norms=norms, label="Conjugate Gradient Norm", x_axis=1)


def run(solvers: dict[str, bool]):
    if not solvers:
        return
    for solver_name in solvers.keys():
        if solvers[solver_name]:
            getattr(sys.modules[__name__], solver_name)().solve()


if __name__ == '__main__':
    run({
        "GaussianEliminationSolver": True,
        "GaussSeidelSolver": True,
        "GradientDescentSolver": True,
        "ConjugateGradientSolver": True
    })
