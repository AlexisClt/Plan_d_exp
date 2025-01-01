import sys
from pathlib import Path

import numpy as np

Plan_d_exp_dir = (Path.cwd() / Path("../")).resolve().as_posix()

sys.path.append(Plan_d_exp_dir)

from Plan_d_exp.src.equations import Equations, Plan

P = Plan(("1", "2"))

P.add({"1": 0.0, "2": 0.0}, "Eq1")
P.add({"1": 1.0, "2": 0.0}, "Eq2")
P.add({"1": 0.5, "2": 0.866}, "Eq3")
P.add({"1": -1.0, "2": 0.0}, "Eq4")
P.add({"1": -0.5, "2": -0.866}, "Eq5")
P.add({"1": 0.5, "2": -0.866}, "Eq6")
P.add({"1": -0.5, "2": 0.866}, "Eq7")

Doehlert_2_var = """N 	X1 	X2
1 	+0.0000 	+0.0000
2 	+1.0000 	+0.0000
3 	+0.5000 	+0.8660
4 	-1.0000 	+0.0000
5 	-0.5000 	-0.8660
6 	+0.5000 	-0.8660
7 	-0.5000 	+0.8660"""

prec, rank, emax, emin, M = P.precision(2)

print(f"precision : {prec}\nrank : {rank}\nratio eigen value : {emax/emin}")
