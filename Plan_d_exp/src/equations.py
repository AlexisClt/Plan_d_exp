# main class
import logging
from collections.abc import Mapping, Sequence
from functools import reduce
from itertools import combinations_with_replacement as cwr

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

        self.indexes = indexes
        self.set_indexes = set(indexes)
        self.order = order
        self.col_names = [
            "mean",
        ]

        for i in range(self.order + 1):
            self.col_names.extend(
                [
                    reduce(lambda x, y: x + "." + y, a)
                    for a in cwr(self.indexes, i)
                    if len(a) != 0
                ]
            )
