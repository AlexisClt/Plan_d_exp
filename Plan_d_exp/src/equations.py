# main class
import logging
from collections import Counter
from collections.abc import Mapping, MutableSequence, Sequence
from functools import cached_property, reduce
from itertools import combinations_with_replacement as cwr
from itertools import product
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numpy.linalg import lstsq

logger = logging.getLogger(__name__)


class Equations:
    """
    Manage type of equations in order to speed up the generation of the equation
    """

    def __init__(self, indexes: Sequence[str], order: int) -> None:
        """
        Init of class
        indexes are the indexes of the X_i s
        order is the maximum order of the equation
        """
        if len(indexes) == 0:
            raise ValueError("indexes is empty")
        if order < 0:
            raise ValueError(f"{order} : Wrong value for order")
        if order == 0:
            logger.warning("order is 0")
        if order > len(indexes):
            raise ValueError(
                f"""order : {order}
index : {indexes}
length of indexes : {len(indexes)}
order should be less or equal to length of indexes"""
            )

        msg0 = "(" + ", ".join([f'"{a}"' for a in indexes]) + ")"
        msg = "\n".join(
            [
                f'Index "{a[1]}" appears {a[0]} times in {msg0}'
                for a in sorted(
                    [(b[1], b[0]) for b in list(Counter(indexes).items()) if b[1] != 1],
                    reverse=True,
                )
            ]
        )

        if len(msg) != 0:
            raise ValueError(msg)

        self.indexes = indexes
        self.set_indexes = set(indexes)
        self.order = order
        self.ind_indexes = dict((i[1], i[0]) for i in enumerate(indexes))

    @cached_property
    def col_names(self) -> Sequence[str]:
        col_names = [
            "mean",
        ]

        for i in range(self.order + 1):
            col_names.extend(
                [
                    reduce(lambda x, y: x + "." + y, a)
                    for a in cwr(self.indexes, i)
                    if len(a) != 0
                ]
            )
        return col_names

    def generate_line(self, datas: Mapping[str, float]) -> Sequence[float]:
        """
        Generate the coefficients of the first equation
        """

        set_datas = set(datas.keys())
        msg0 = ", ".join([f'"{a}"' for a in datas.keys()])

        msg = ""
        msg += "\n".join(
            [
                f'Index "{a}" is missing in : {msg0}'
                for a in sorted(self.set_indexes - set_datas)
            ]
        )

        msg += "".join(
            [
                f'\nIndex "{a}" is not a valid index'
                for a in sorted(set_datas - self.set_indexes)
            ]
        )

        if len(msg) != 0:
            raise ValueError(msg)
        ret = [
            1.0,
        ]

        for i in range(self.order + 1):
            ret.extend(
                [
                    reduce(lambda x, y: x * y, a, 1.0)
                    for a in cwr((datas[b] for b in self.indexes), i)
                    if len(a) != 0
                ]
            )

        return ret

    def check_datas(
        self, data1: Mapping[str, float], data2: Mapping[str, float]
    ) -> bool:

        s1 = set(data1.keys())
        s2 = set(data2.keys())
        s3 = s1 & s2

        msg0 = ", ".join([f"'{a}'" for a in sorted(s3)])
        if len(msg0) != 0:
            raise ValueError(
                f"""Index {msg0} is in both arguments :
{data1}
and
{data2}"""
            )

        s4 = self.set_indexes - s1 - s2
        msg0 = ", ".join([f"'{a}'" for a in sorted(s4)])
        if len(msg0) != 0:
            raise ValueError(
                f"""Index {msg0} is missing in either :
{data1}
or
{data2}"""
            )

        s5 = s1 - self.set_indexes
        s6 = s2 - self.set_indexes

        msg0 = ", ".join([f"'{a}'" for a in sorted(s5)])
        if len(msg0) != 0:
            raise ValueError(
                f"""Index {msg0} in :
{data1}
but not in :
{self.indexes}"""
            )

        msg0 = ", ".join([f"'{a}'" for a in sorted(s6)])
        if len(msg0) != 0:
            raise ValueError(
                f"""Index {msg0} in :
{data2}
but not in :
{self.indexes}"""
            )

        return True

    def generate_circular(
        self, data1: Mapping[str, float], data2: Mapping[str, float]
    ) -> MutableSequence[Mapping[str, float]]:

        self.check_datas(data1, data2)

        res: MutableSequence[Mapping[str, float]] = []

        constant = list(data1.items())
        variable = [
            (a[1], a[2])
            for a in sorted(
                [(self.ind_indexes[b[0]], b[0], b[1]) for b in data2.items()],
                key=lambda x: x[0],
            )
        ]

        k1 = list(map(lambda x: x[0], variable))
        v1 = list(map(lambda x: x[1], variable))

        res.append(dict(constant + variable))
        for i in range(len(k1) - 1):
            ele = v1.pop(-1)
            v1 = [
                ele,
            ] + v1
            res.append(
                dict(
                    constant
                    + list(
                        zip(
                            k1,
                            v1,
                        )
                    )
                )
            )

        return res

    def generate_product(
        self, data1: Mapping[str, float], data2: Mapping[str, Sequence[float]]
    ) -> MutableSequence[Mapping[str, float]]:

        msg0 = ", ".join([f"'{k}'" for k, v in data2.items() if len(v) == 0])

        if len(msg0) != 0:
            raise ValueError(
                f"""List of index : {msg0}
are empty in :
{data2}"""
            )

        self.check_datas(data1, dict([(k, v[0]) for k, v in data2.items()]))

        res: MutableSequence[Mapping[str, float]] = []

        constant = list(data1.items())
        variable = [
            (a[1], a[2])
            for a in sorted(
                [(self.ind_indexes[b[0]], b[0], b[1]) for b in data2.items()],
                key=lambda x: x[0],
            )
        ]

        k1 = list(map(lambda x: x[0], variable))
        v1 = list(product(*map(lambda x: x[1], variable)))

        for i in v1:
            res.append(
                dict(
                    constant
                    + list(
                        zip(
                            k1,
                            i,
                        )
                    )
                )
            )

        return res


class Plan:
    """
    Manage and solve several equations
    """

    def __init__(self, indexes: Sequence[str]) -> None:
        self.Equations_table: MutableSequence[Mapping[str, float]] = []
        self.Equations_name: MutableSequence[str] = []
        self.E = Equations(indexes, 1)

    def to_csv(self, sep: str = ";") -> str:
        """Return the content of the object in csv format"""
        ret = (
            sep.join(
                [
                    "Name",
                ]
                + list(self.E.indexes)
            )
            + "\n"
        )
        ret += "\n".join(
            (
                sep.join(
                    [
                        a,
                    ]
                    + [str(b[c]) for c in self.E.indexes]
                )
                for a, b in zip(self.Equations_name, self.Equations_table)
            )
        )
        return ret

    def add(self, datas: Mapping[str, float], name: str) -> int:
        """Add an experiment in Plan"""

        self.E.generate_line(datas)
        self.Equations_table.append(datas)
        self.Equations_name.append(name)

        return len(self.Equations_table)

    def solve(
        self, order: int, b: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], float, float, int]:
        """Solve and find the coefficients
        return:
        the solution a of M @ a = b
        the maximum difference between M @ a and b
        the minimum difference between M @ a and b
        the rank"""

        E1 = Equations(self.E.indexes, order)
        M: MutableSequence[float] = []

        for e in self.Equations_table:
            M.extend(E1.generate_line(e))

        M1 = np.array(M).reshape(len(self.Equations_table), len(E1.col_names))
        a, residuals, rank, eign = lstsq(M1, b)
        max_diff = np.max(M1 @ a - b)
        min_diff = np.min(M1 @ a - b)

        return (a, max_diff, min_diff, rank)

    def precision(
        self, order: int
    ) -> Tuple[float, int, float, int, npt.NDArray[np.float64]]:
        """Find the precision
        return:
        maximum difference between M @ a and b (b=1 so it is the maximum percentage)
        the rank
        first (highest) eigen value
        last non-zero (lowest) eigen value
        the associated matrix of the design experiment
        """

        E1 = Equations(self.E.indexes, order)
        M: MutableSequence[float] = []

        for e in self.Equations_table:
            M.extend(E1.generate_line(e))

        M1 = np.array(M).reshape(len(self.Equations_table), len(E1.col_names))
        b = np.ones((len(self.Equations_table), 1))
        a, residuals, rank, eign = lstsq(M1, b)
        max_diff = np.max(np.abs(M1 @ a - b))

        return (max_diff, rank, eign[0], eign[rank - 1], M1)

    def generate_circular(
        self, data1: Mapping[str, float], data2: Mapping[str, float], template_name: str
    ) -> None:
        """Add new equations
        Using data1 and data2, new equations are generated using data1 as constant value associated to its indexes,
        data2 will be used to generate values associated to its indexes using circular permutation.
        The keys of data1 + data2 must be equal to self.indexes.
        Each new Equation is named using incremental name starting at 1 based on template_name
        """
        gen_cir = self.E.generate_circular(data1, data2)
        self.Equations_name.extend(
            [(template_name + f"_{i}") for i in range(1, len(gen_cir) + 1)]
        )
        self.Equations_table.extend(gen_cir)

    def generate_product(
        self,
        data1: Mapping[str, float],
        data2: Mapping[str, Sequence[float]],
        template_name: str,
    ) -> None:
        """Add new equations
        Using data, new equations are generated my combining values in lists elements
        Each new Equation is named using incremental name starting at 1 based on template_name
        """
        gen_prod = self.E.generate_product(data1, data2)
        self.Equations_name.extend(
            [(template_name + f"_{i}") for i in range(1, len(gen_prod) + 1)]
        )
        self.Equations_table.extend(gen_prod)
