# main class
import logging
from collections import Counter
from collections.abc import Mapping, MutableSequence, Sequence
from functools import cached_property, reduce
from itertools import combinations_with_replacement as cwr
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


class Plan:
    """
    Manage and solve several equations
    """

    def __init__(self, indexes: Sequence[str]) -> None:
        self.Equations_table: MutableSequence[Mapping[str, float]] = []
        self.E = Equations(indexes, 1)

    def add(self, datas: Mapping[str, float]) -> int:
        """Add an experiment in Plan"""

        self.E.generate_line(datas)
        self.Equations_table.append(datas)

        return len(self.Equations_table)

    def solve(self, order: int) -> Tuple[npt.NDArray[np.float64], float, int]:
        """Find the best coefficients for the equations of the plan"""

        E1 = Equations(self.E.indexes, order)
        M: MutableSequence[float] = []

        for e in self.Equations_table:
            M.extend(E1.generate_line(e))

        M1 = np.array(M).reshape(len(self.Equations_table), len(E1.col_names))
        b = np.ones((len(self.Equations_table), 1))
        a, residuals, rank, eign = lstsq(M1, b)
        max_diff = np.max(np.abs(M @ a - b))

        return (a, max_diff, rank)
