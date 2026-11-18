"""
this example must be run in example directory
"""
import sys
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

Plan_d_exp_dir = (Path.cwd() / Path("../")).resolve().as_posix()

sys.path.append(Plan_d_exp_dir)

from Plan_d_exp.src.equations import Equations, Plan

nb_var = 12  # in this example it can't be more
variables = [f'X_{i}' for i in range(nb_var)] # !!! first var is 1, last is nb_var-1
m = len(variables)
print(f'{m} variables created')
len_coef = int(1 + 3*m/2 + m**2/2)
print(f'{len_coef:d} coefficients needed')

E = Equations(variables, 2)

print("E.col_names")
print(E.col_names)
print("end of E.col_names")

# création des coefficients  (first coef is the mean, here = 1)
y = np.linspace(-1., -4., num=(len_coef-1), endpoint=True)
Coef = np.insert(np.power(10., y), 0, 1.) # first coef must 1.

# fonction F
def F(data: Mapping[str, float]) -> float:
    return np.sum(Coef*np.array(E.generate_line(data)))

P = Plan(variables)
P.E=Equations(variables, 2)

# generating definitive screening plan (P is updated)
# this function is limitated to 12 variables
P.generate_def_scr({}, variables, 'Exp')

# calculating vector F value of each experiment
b_vector=np.array([F(a) for a in P.E.generate_def_scr({}, variables)])

# calculating best Coef with theses experiments (first coef is the mean)
best_coef, max_diff, min_diff, rank = P.solve(2, b_vector)

# plotting
fig, ax = plt.subplots(figsize=(6,4))
x = np.linspace(1, len_coef, num=len_coef, endpoint=True)
plt.scatter(x, Coef, label="Coeficients", c='blue')
plt.scatter(x, best_coef, label="Best Coeficients", c='red')
ax.set_yscale('log', base=10)
plt.ylabel("Coefficient")
plt.xlabel("Var Number")
plt.grid(True)
plt.legend()
plt.show()
