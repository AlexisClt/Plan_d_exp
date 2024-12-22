import logging
from typing import Any

import pytest

from Plan_d_exp.src.equations import Equations


def test_Equations_1() -> None:
    with pytest.raises(ValueError, match="indexes is empty"):
        E = Equations((), 0)


def test_Equations_2() -> None:
    with pytest.raises(ValueError, match="-1 : Wrong value for order"):
        E = Equations(("a",), -1)


def test_Equations_3(caplog: Any) -> None:
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        E = Equations(("1",), 0)
        assert caplog.record_tuples == [
            ("Plan_d_exp.src.equations", logging.WARNING, "order is 0")
        ]


def test_Equations_4() -> None:
    with pytest.raises(
        ValueError,
        match=r"""order : 3
index : \('1', 'a'\)
length of indexes : 2
order should be less or equal to length of indexes""",
    ):
        E = Equations(("1", "a"), 3)
