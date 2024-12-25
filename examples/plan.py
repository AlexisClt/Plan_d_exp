import sys
from pathlib import Path

import numpy as np
from numpy.linalg import lstsq

Plan_d_exp_dir = (Path.cwd() / Path("../")).resolve().as_posix()

sys.path.append(Plan_d_exp_dir)

from Plan_d_exp.src.equations import Equations

E2 = Equations(("1", "2"), 2)

matrix = list(E2.generate_line({"1": 1.0, "2": 1.0}))
matrix.extend(E2.generate_line({"1": 1.0, "2": 1.0}))
matrix.extend(E2.generate_line({"1": 1.0, "2": -1.0}))
matrix.extend(E2.generate_line({"1": -1.0, "2": -1.0}))

M = np.array(matrix).reshape(4, len(E2.col_names))

b = np.ones((4, 1))

a, residuals, rank, eign = lstsq(M, b)

print("Error interpolating example with 2 factors at second order :")
print(np.sqrt(np.sum(np.pow((M @ a - b), 2))))
print(f"Matrix rank is : {rank}")
print(f"First eigen value : {eign[0]}")
print(f"Last non-zero eigen value : {eign[rank-1]}")
print(50 * "-")


E3 = Equations(("1", "2", "3"), 3)

matrix = list(E3.generate_line({"1": 1.0, "2": 1.0, "3": -1.0}))
matrix.extend(E3.generate_line({"1": 1.0, "2": 1.0, "3": 1.0}))
matrix.extend(E3.generate_line({"1": 1.0, "2": -1.0, "3": 1.0}))
matrix.extend(E3.generate_line({"1": -1.0, "2": -1.0, "3": -1.0}))

M = np.array(matrix).reshape(4, len(E3.col_names))

b = np.ones((4, 1))

a, residuals, rank, eign = lstsq(M, b)

print("Error interpolating example with 3 factors at third order :")
print(np.sqrt(np.sum(np.pow((M @ a - b), 2))))
print(f"Matrix rank is : {rank}")
print(f"First eigen value : {eign[0]}")
print(f"Last non-zero eigen value : {eign[rank-1]}")
print(50 * "-")
