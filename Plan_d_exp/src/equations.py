# main class
import logging
from collections.abc import Sequence

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
