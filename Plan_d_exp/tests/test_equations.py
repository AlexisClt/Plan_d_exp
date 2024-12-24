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
        "1.1",
        "1.toto",
        "toto.toto",
    ]


def test_Equations_8() -> None:
    E = Equations(("1", "toto", "b"), 3)
    assert E.col_names == [
        "mean",
        "1",
        "toto",
        "b",
        "1.1",
        "1.toto",
        "1.b",
        "toto.toto",
        "toto.b",
        "b.b",
        "1.1.1",
        "1.1.toto",
        "1.1.b",
        "1.toto.toto",
        "1.toto.b",
        "1.b.b",
        "toto.toto.toto",
        "toto.toto.b",
        "toto.b.b",
        "b.b.b",
    ]


def test_generate_line_1() -> None:
    E = Equations(("1", "a"), 0)
    assert E.generate_line({"1": 1.0, "a": 1.0}) == [
        1.0,
    ]


def test_generate_line_2() -> None:
    E = Equations(("1", "a"), 1)
    assert E.generate_line({"1": 1.0, "a": 1.0}) == [1.0, 1.0, 1.0]


def test_generate_line_3() -> None:
    with pytest.raises(
        ValueError,
        match=r"""Index "a" is missing in : "1", "b"
Index "b" is not a valid index""",
    ):
        E = Equations(("1", "a"), 0)
        E.generate_line({"1": 1.0, "b": 1.0})


def test_generate_line_4() -> None:
    E = Equations(("1", "a"), 2)
    assert E.generate_line({"1": 1.0, "a": 3.0}) == [1.0, 1.0, 3.0, 1.0, 3.0, 9.0]
