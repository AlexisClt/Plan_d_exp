import logging
from math import fabs
from string import ascii_uppercase
from typing import Any

import numpy as np
import pytest
from Plan_d_exp.src.equations import Equations, Plan


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


def test_Equations_4_1() -> None:
    with pytest.raises(
        ValueError,
        match=r"""Index "a" appears 2 times in \("1", "a", "a"\)""",
    ):
        E = Equations(("1", "a", "a"), 3)


def test_Equations_4_2() -> None:
    with pytest.raises(
        ValueError,
        match=r"""Index "b" appears 3 times in \("b", "1", "b", "a", "a", "b"\)
Index "a" appears 2 times in \("b", "1", "b", "a", "a", "b"\)""",
    ):
        E = Equations(("b", "1", "b", "a", "a", "b"), 3)


def test_Equations_5() -> None:
    E = Equations(("1",), 0)
    assert E.col_names == [
        "mean",
    ]


def test_Equations_6() -> None:
    E = Equations(("1",), 1)
    assert E.col_names == [
        "mean",
        "1",
    ]


def test_Equations_7() -> None:
    E = Equations(("1", "toto"), 2)
    assert E.col_names == [
        "mean",
        "1",
        "toto",
        "1**2",
        "1.toto",
        "toto**2",
    ]


def test_Equations_8() -> None:
    E = Equations(("1", "toto", "b"), 3)
    assert E.col_names == [
        "mean",
        "1",
        "toto",
        "b",
        "1**2",
        "1.toto",
        "1.b",
        "toto**2",
        "toto.b",
        "b**2",
        "1**3",
        "1**2.toto",
        "1**2.b",
        "1.toto**2",
        "1.toto.b",
        "1.b**2",
        "toto**3",
        "toto**2.b",
        "toto.b**2",
        "b**3",
    ]


def test_Plan_6() -> None:
    P = Plan(("1", "b", "2", "a"))
    P.generate_circular({}, {"1": 2.0, "2": 4.0, "a": -2.0, "b": 6.0}, "E")
    P.generate_product({"a": 1.0, "b": 2.0}, {"1": [3.0, 4.0], "2": [5.0, 6.0]}, "P")
    ma, r, f1, f2, M = P.precision(2)
    assert fabs(ma) < 9e-16
    assert r == 8
    assert 98.096 < f1 < 98.097
    assert 0.19001 < f2 < 0.19002
    assert M.shape == (8, 15)
