import logging
from math import fabs
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from Plan_d_exp.src.equations import (
    Equations,
    Equations_tri,
    Plan,
    def_scr,
    genere_def_scr_int,
    limi_1,
    limi_2,
    limi_3,
    vers_int,
)


@pytest.fixture
def setup_res(tmp_path):
    tmp_res = tmp_path / Path("result.csv")
    tmp_res.write_text("Nom;val")
    return tmp_res


def test_vers_chaine() -> None:
    assert all((i == 0 for i in vers_int("000")))
    assert all((i == 1 for i in vers_int(9 * "+")))
    assert all((i == -1 for i in vers_int(9 * "-")))
    assert tuple(vers_int("0+-")) == (0, 1, -1)
    with pytest.raises(ValueError) as excinfo:
        a = vers_int("0+-a")
    assert str(excinfo.value) == "le caractere 'a' n'est pas dans '-+0'"
    for k, v in def_scr.items():
        for vv in v:
            assert len(vv) == k


def test_genere_def_scr_int() -> None:
    a = genere_def_scr_int(
        {
            3: ("0+-", "-0-", "--0"),
            4: ("0+--", "-0-+", "--0-", "-++0"),
        }
    )
    assert len(a) == 2
    assert 3 in a.keys()
    assert 4 in a.keys()
    assert len(a[3]) == 7
    assert a[3][0] == [0, +1, -1]
    assert a[3][1] == [0, -1, +1]
    assert a[3][2] == [-1, 0, -1]
    assert a[3][3] == [+1, 0, +1]
    assert a[3][4] == [-1, -1, 0]
    assert a[3][5] == [+1, +1, 0]
    assert a[3][6] == [0, 0, 0]
    assert len(a[4]) == 9
    assert a[4][8] == [0, 0, 0, 0]
    assert a[4] == [
        [0, 1, -1, -1],
        [0, -1, 1, 1],
        [-1, 0, -1, 1],
        [1, 0, 1, -1],
        [-1, -1, 0, -1],
        [1, 1, 0, 1],
        [-1, 1, 1, 0],
        [1, -1, -1, 0],
        [0, 0, 0, 0],
    ]


def test_Equations_1() -> None:
    with pytest.raises(ValueError, match="indexes is empty"):
        E = Equations((), 0)


def test_Equations_1_2() -> None:
    with pytest.raises(ValueError, match="indexes number 2 has length equal zero"):
        E = Equations(("a", "", "b"), 1)


def test_Equations_1_3() -> None:
    with pytest.raises(ValueError, match="indexes number 2, 4 has length equal zero"):
        E = Equations(("a", "", "b", ""), 1)


def test_Equations_2() -> None:
    with pytest.raises(ValueError, match="-1 : Wrong value for order"):
        E = Equations(("a",), -1)


def test_Equations_3(caplog: Any) -> None:
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        E = Equations(("1",), 0)
        assert caplog.record_tuples == [
            ("Plan_d_exp.src.equations", logging.WARNING, "order is 0")
        ]
        assert len(E.dict_cwr_ind) == 0
        assert len(E.dict_cwr_str) == 0


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
    assert E.dict_cwr_str == {
        1: (("1",),),
    }
    assert E.dict_cwr_ind == {
        1: [
            ((0, 1),),
        ],
    }


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
    assert E.dict_cwr_str == {
        1: (("1",), ("toto",)),
        2: (("1", "1"), ("1", "toto"), ("toto", "toto")),
    }
    assert E.dict_cwr_ind == {
        1: [((0, 1),), ((1, 1),)],
        2: [((0, 2),), ((0, 1), (1, 1)), ((1, 2),)],
    }


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
    assert E.dict_cwr_str == {
        1: (("1",), ("toto",), ("b",)),
        2: (
            ("1", "1"),
            ("1", "toto"),
            ("1", "b"),
            ("toto", "toto"),
            ("toto", "b"),
            ("b", "b"),
        ),
        3: (
            ("1", "1", "1"),
            ("1", "1", "toto"),
            ("1", "1", "b"),
            ("1", "toto", "toto"),
            ("1", "toto", "b"),
            ("1", "b", "b"),
            ("toto", "toto", "toto"),
            ("toto", "toto", "b"),
            ("toto", "b", "b"),
            ("b", "b", "b"),
        ),
    }
    assert E.dict_cwr_ind == {
        1: [
            ((0, 1),),
            ((1, 1),),
            ((2, 1),),
        ],
        2: [
            ((0, 2),),
            ((0, 1), (1, 1)),
            ((0, 1), (2, 1)),
            ((1, 2),),
            ((1, 1), (2, 1)),
            ((2, 2),),
        ],
        3: [
            ((0, 3),),
            ((0, 2), (1, 1)),
            ((0, 2), (2, 1)),
            ((0, 1), (1, 2)),
            ((0, 1), (1, 1), (2, 1)),
            ((0, 1), (2, 2)),
            ((1, 3),),
            ((1, 2), (2, 1)),
            ((1, 1), (2, 2)),
            ((2, 3),),
        ],
    }


def test_Equations_9() -> None:
    E = Equations(("1", "b", "2", "a"), 2)
    assert E.generate_circular({"1": 2.0, "2": 4.0}, {"a": -2.0, "b": 6.0}) == [
        {"1": 2.0, "b": 6.0, "2": 4.0, "a": -2.0},
        {"1": 2.0, "b": -2.0, "2": 4.0, "a": 6.0},
    ]


def test_Equations_10() -> None:
    E = Equations(("1", "b", "2", "a"), 2)
    assert E.generate_circular({}, {"1": 2.0, "2": 4.0, "a": -2.0, "b": 6.0}) == [
        {"1": 2.0, "b": 6.0, "2": 4.0, "a": -2.0},
        {"1": -2.0, "b": 2.0, "2": 6.0, "a": 4.0},
        {"1": 4.0, "b": -2.0, "2": 2.0, "a": 6.0},
        {"1": 6.0, "b": 4.0, "2": -2.0, "a": 2.0},
    ]


def test_Equations_11() -> None:
    E = Equations(("1", "b", "2", "a"), 2)
    assert E.generate_circular({"1": 2.0, "2": 4.0, "a": -2.0, "b": 6.0}, {}) == [
        {"1": 2.0, "b": 6.0, "2": 4.0, "a": -2.0},
    ]


def test_Equations_12() -> None:
    with pytest.raises(
        ValueError,
        match=r"""Index '1' is missing in either :
{'2': 4.0}
or
{'a': -2.0, 'b': 6.0}""",
    ):
        E = Equations(("1", "b", "2", "a"), 2)
        E.generate_circular({"2": 4.0}, {"a": -2.0, "b": 6.0})


def test_Equations_13() -> None:
    with pytest.raises(
        ValueError,
        match=r"""Index '1' is in both arguments :
{'1': 2.0, '2': 4.0}
and
{'1': 2.0, 'a': -2.0, 'b': 6.0}""",
    ):
        E = Equations(("1", "b", "2", "a"), 2)
        E.generate_circular({"1": 2.0, "2": 4.0}, {"1": 2.0, "a": -2.0, "b": 6.0})


def test_Equations_14() -> None:
    with pytest.raises(
        ValueError,
        match=r"""Index 'c' in :
{'c': 2.0, 'a': -2.0, 'b': 6.0}
but not in :
\('1', 'b', '2', 'a'\)""",
    ):
        E = Equations(("1", "b", "2", "a"), 2)
        E.generate_circular({"1": 2.0, "2": 4.0}, {"c": 2.0, "a": -2.0, "b": 6.0})


def test_Equations_15() -> None:
    with pytest.raises(
        ValueError,
        match=r"""Index 'c' in :
{'c': 3.0, '1': 2.0, '2': 4.0}
but not in :
\('1', 'b', '2', 'a'\)""",
    ):
        E = Equations(("1", "b", "2", "a"), 2)
        E.generate_circular({"c": 3.0, "1": 2.0, "2": 4.0}, {"a": -2.0, "b": 6.0})


def test_Equations_16() -> None:
    with pytest.raises(
        ValueError,
        match=r"""List of index : 'a'
are empty in :
{'a': \[\], 'b': \[6.0, 3.0, 4.0\]}""",
    ):
        E = Equations(("1", "b", "2", "a"), 2)
        E.generate_product({"1": 2.0, "2": 4.0}, {"a": [], "b": [6.0, 3.0, 4.0]})


def test_Equations_17() -> None:
    E = Equations(("1", "b", "2", "a"), 2)
    assert E.generate_product(
        {"1": 2.0, "2": 4.0}, {"a": [1.0, 2.0], "b": [6.0, 3.0, 4.0]}
    ) == [
        {"1": 2.0, "b": 6.0, "2": 4.0, "a": 1.0},
        {"1": 2.0, "b": 6.0, "2": 4.0, "a": 2.0},
        {"1": 2.0, "b": 3.0, "2": 4.0, "a": 1.0},
        {"1": 2.0, "b": 3.0, "2": 4.0, "a": 2.0},
        {"1": 2.0, "b": 4.0, "2": 4.0, "a": 1.0},
        {"1": 2.0, "b": 4.0, "2": 4.0, "a": 2.0},
    ]


def test_Equations_18() -> None:
    E = Equations(("1", "b", "2", "a"), 2)
    assert E.excel_columns == list(ascii_uppercase[:17])
    assert (
        E.to_excel_formula(np.arange(1.0, 18.0, 1.0).reshape(1, 17))
        == "=1.0+2.0*B2+3.0*C2+4.0*D2+5.0*E2+6.0*F2+7.0*G2+8.0*H2+9.0*I2+10.0*J2+11.0*K2+12.0*L2+13.0*M2+14.0*N2+15.0*O2+16.0*P2+17.0*Q2"
    )


def test_Equations_19() -> None:
    E = Equations(("1", "b", "2", "a", "c"), 3)
    assert E.excel_columns == list(ascii_uppercase) + [
        f"A{i}" for i in ascii_uppercase
    ] + [f"B{i}" for i in ascii_uppercase[:6]]


def test_Equations_20() -> None:
    E = Equations(("C:1", "C:2", "C:3", "C:4", "V:1", "V:2", "V:3"), 2)
    assert E.generate_def_scr(
        {"C:4": 0, "V:1": 1, "V:3": -1}, ["C:1", "C:2", "C:3", "V:2"]
    ) == [
        {"C:1": 0, "C:2": 1, "C:3": -1, "C:4": 0, "V:1": 1, "V:2": -1, "V:3": -1},
        {"C:1": 0, "C:2": -1, "C:3": 1, "C:4": 0, "V:1": 1, "V:2": 1, "V:3": -1},
        {"C:1": -1, "C:2": 0, "C:3": -1, "C:4": 0, "V:1": 1, "V:2": 1, "V:3": -1},
        {"C:1": 1, "C:2": 0, "C:3": 1, "C:4": 0, "V:1": 1, "V:2": -1, "V:3": -1},
        {"C:1": -1, "C:2": -1, "C:3": 0, "C:4": 0, "V:1": 1, "V:2": -1, "V:3": -1},
        {"C:1": 1, "C:2": 1, "C:3": 0, "C:4": 0, "V:1": 1, "V:2": 1, "V:3": -1},
        {"C:1": -1, "C:2": 1, "C:3": 1, "C:4": 0, "V:1": 1, "V:2": 0, "V:3": -1},
        {"C:1": 1, "C:2": -1, "C:3": -1, "C:4": 0, "V:1": 1, "V:2": 0, "V:3": -1},
        {"C:1": 0, "C:2": 0, "C:3": 0, "C:4": 0, "V:1": 1, "V:2": 0, "V:3": -1},
    ]


def test_Equations_21() -> None:
    E = Equations_tri(("C:1", "C:2", "C:3", "C:4", "V:1", "V:2", "V:3"), 1)
    assert E.comb(1) == [
        ("C:1",),
        ("C:2",),
        ("C:3",),
        ("C:4",),
        ("V:1",),
        ("V:2",),
        ("V:3",),
    ]
    assert E.comb(2) == [
        ("C:1", "C:1"),
        ("C:1", "C:2"),
        ("C:1", "C:3"),
        ("C:1", "C:4"),
        ("C:1", "V:1"),
        ("C:1", "V:2"),
        ("C:1", "V:3"),
        ("C:2", "C:2"),
        ("C:2", "C:3"),
        ("C:2", "C:4"),
        ("C:2", "V:1"),
        ("C:2", "V:2"),
        ("C:2", "V:3"),
        ("C:3", "C:3"),
        ("C:3", "C:4"),
        ("C:3", "V:1"),
        ("C:3", "V:2"),
        ("C:3", "V:3"),
        ("C:4", "C:4"),
        ("C:4", "V:1"),
        ("C:4", "V:2"),
        ("C:4", "V:3"),
        ("V:1", "V:1"),
        ("V:1", "V:2"),
        ("V:1", "V:3"),
        ("V:2", "V:2"),
        ("V:2", "V:3"),
        ("V:3", "V:3"),
    ]
    assert E.comb(3) == [
        ("C:1", "C:1", "C:2"),
        ("C:1", "C:1", "C:3"),
        ("C:1", "C:1", "C:4"),
        ("C:1", "C:1", "V:1"),
        ("C:1", "C:1", "V:2"),
        ("C:1", "C:1", "V:3"),
        ("C:1", "C:2", "C:2"),
        ("C:1", "C:2", "C:3"),
        ("C:1", "C:2", "C:4"),
        ("C:1", "C:2", "V:1"),
        ("C:1", "C:2", "V:2"),
        ("C:1", "C:2", "V:3"),
        ("C:1", "C:3", "C:3"),
        ("C:1", "C:3", "C:4"),
        ("C:1", "C:3", "V:1"),
        ("C:1", "C:3", "V:2"),
        ("C:1", "C:3", "V:3"),
        ("C:1", "C:4", "C:4"),
        ("C:1", "C:4", "V:1"),
        ("C:1", "C:4", "V:2"),
        ("C:1", "C:4", "V:3"),
        ("C:1", "V:1", "V:1"),
        ("C:1", "V:1", "V:2"),
        ("C:1", "V:1", "V:3"),
        ("C:1", "V:2", "V:2"),
        ("C:1", "V:2", "V:3"),
        ("C:1", "V:3", "V:3"),
        ("C:2", "C:2", "C:3"),
        ("C:2", "C:2", "C:4"),
        ("C:2", "C:2", "V:1"),
        ("C:2", "C:2", "V:2"),
        ("C:2", "C:2", "V:3"),
        ("C:2", "C:3", "C:3"),
        ("C:2", "C:3", "C:4"),
        ("C:2", "C:3", "V:1"),
        ("C:2", "C:3", "V:2"),
        ("C:2", "C:3", "V:3"),
        ("C:2", "C:4", "C:4"),
        ("C:2", "C:4", "V:1"),
        ("C:2", "C:4", "V:2"),
        ("C:2", "C:4", "V:3"),
        ("C:2", "V:1", "V:1"),
        ("C:2", "V:1", "V:2"),
        ("C:2", "V:1", "V:3"),
        ("C:2", "V:2", "V:2"),
        ("C:2", "V:2", "V:3"),
        ("C:2", "V:3", "V:3"),
        ("C:3", "C:3", "C:4"),
        ("C:3", "C:3", "V:1"),
        ("C:3", "C:3", "V:2"),
        ("C:3", "C:3", "V:3"),
        ("C:3", "C:4", "C:4"),
        ("C:3", "C:4", "V:1"),
        ("C:3", "C:4", "V:2"),
        ("C:3", "C:4", "V:3"),
        ("C:3", "V:1", "V:1"),
        ("C:3", "V:1", "V:2"),
        ("C:3", "V:1", "V:3"),
        ("C:3", "V:2", "V:2"),
        ("C:3", "V:2", "V:3"),
        ("C:3", "V:3", "V:3"),
        ("C:4", "C:4", "V:1"),
        ("C:4", "C:4", "V:2"),
        ("C:4", "C:4", "V:3"),
        ("C:4", "V:1", "V:1"),
        ("C:4", "V:1", "V:2"),
        ("C:4", "V:1", "V:3"),
        ("C:4", "V:2", "V:2"),
        ("C:4", "V:2", "V:3"),
        ("C:4", "V:3", "V:3"),
        ("V:1", "V:1", "V:2"),
        ("V:1", "V:1", "V:3"),
        ("V:1", "V:2", "V:2"),
        ("V:1", "V:2", "V:3"),
        ("V:1", "V:3", "V:3"),
        ("V:2", "V:2", "V:3"),
        ("V:2", "V:3", "V:3"),
    ]
    assert E.combi(1) == [
        ((0, 1),),
        ((1, 1),),
        ((2, 1),),
        ((3, 1),),
        ((4, 1),),
        ((5, 1),),
        ((6, 1),),
    ]
    assert E.combi(2) == [
        ((0, 2),),
        ((0, 1), (1, 1)),
        ((0, 1), (2, 1)),
        ((0, 1), (3, 1)),
        ((0, 1), (4, 1)),
        ((0, 1), (5, 1)),
        ((0, 1), (6, 1)),
        ((1, 2),),
        ((1, 1), (2, 1)),
        ((1, 1), (3, 1)),
        ((1, 1), (4, 1)),
        ((1, 1), (5, 1)),
        ((1, 1), (6, 1)),
        ((2, 2),),
        ((2, 1), (3, 1)),
        ((2, 1), (4, 1)),
        ((2, 1), (5, 1)),
        ((2, 1), (6, 1)),
        ((3, 2),),
        ((3, 1), (4, 1)),
        ((3, 1), (5, 1)),
        ((3, 1), (6, 1)),
        ((4, 2),),
        ((4, 1), (5, 1)),
        ((4, 1), (6, 1)),
        ((5, 2),),
        ((5, 1), (6, 1)),
        ((6, 2),),
    ]
    assert E.combi(3) == [
        ((0, 2), (1, 1)),
        ((0, 2), (2, 1)),
        ((0, 2), (3, 1)),
        ((0, 2), (4, 1)),
        ((0, 2), (5, 1)),
        ((0, 2), (6, 1)),
        ((0, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1)),
        ((0, 1), (1, 1), (3, 1)),
        ((0, 1), (1, 1), (4, 1)),
        ((0, 1), (1, 1), (5, 1)),
        ((0, 1), (1, 1), (6, 1)),
        ((0, 1), (2, 2)),
        ((0, 1), (2, 1), (3, 1)),
        ((0, 1), (2, 1), (4, 1)),
        ((0, 1), (2, 1), (5, 1)),
        ((0, 1), (2, 1), (6, 1)),
        ((0, 1), (3, 2)),
        ((0, 1), (3, 1), (4, 1)),
        ((0, 1), (3, 1), (5, 1)),
        ((0, 1), (3, 1), (6, 1)),
        ((0, 1), (4, 2)),
        ((0, 1), (4, 1), (5, 1)),
        ((0, 1), (4, 1), (6, 1)),
        ((0, 1), (5, 2)),
        ((0, 1), (5, 1), (6, 1)),
        ((0, 1), (6, 2)),
        ((1, 2), (2, 1)),
        ((1, 2), (3, 1)),
        ((1, 2), (4, 1)),
        ((1, 2), (5, 1)),
        ((1, 2), (6, 1)),
        ((1, 1), (2, 2)),
        ((1, 1), (2, 1), (3, 1)),
        ((1, 1), (2, 1), (4, 1)),
        ((1, 1), (2, 1), (5, 1)),
        ((1, 1), (2, 1), (6, 1)),
        ((1, 1), (3, 2)),
        ((1, 1), (3, 1), (4, 1)),
        ((1, 1), (3, 1), (5, 1)),
        ((1, 1), (3, 1), (6, 1)),
        ((1, 1), (4, 2)),
        ((1, 1), (4, 1), (5, 1)),
        ((1, 1), (4, 1), (6, 1)),
        ((1, 1), (5, 2)),
        ((1, 1), (5, 1), (6, 1)),
        ((1, 1), (6, 2)),
        ((2, 2), (3, 1)),
        ((2, 2), (4, 1)),
        ((2, 2), (5, 1)),
        ((2, 2), (6, 1)),
        ((2, 1), (3, 2)),
        ((2, 1), (3, 1), (4, 1)),
        ((2, 1), (3, 1), (5, 1)),
        ((2, 1), (3, 1), (6, 1)),
        ((2, 1), (4, 2)),
        ((2, 1), (4, 1), (5, 1)),
        ((2, 1), (4, 1), (6, 1)),
        ((2, 1), (5, 2)),
        ((2, 1), (5, 1), (6, 1)),
        ((2, 1), (6, 2)),
        ((3, 2), (4, 1)),
        ((3, 2), (5, 1)),
        ((3, 2), (6, 1)),
        ((3, 1), (4, 2)),
        ((3, 1), (4, 1), (5, 1)),
        ((3, 1), (4, 1), (6, 1)),
        ((3, 1), (5, 2)),
        ((3, 1), (5, 1), (6, 1)),
        ((3, 1), (6, 2)),
        ((4, 2), (5, 1)),
        ((4, 2), (6, 1)),
        ((4, 1), (5, 2)),
        ((4, 1), (5, 1), (6, 1)),
        ((4, 1), (6, 2)),
        ((5, 2), (6, 1)),
        ((5, 1), (6, 2)),
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


def test_limi_3() -> None:
    assert limi_3(2) == {
        1: [((0, 1),), ((1, 1),)],
        2: [((0, 2),), ((0, 1), (1, 1)), ((1, 2),)],
        3: [((0, 2), (1, 1)), ((0, 1), (1, 2))],
        4: [
            ((0, 2), (1, 2)),
        ],
    }
    assert limi_3(3) == {
        1: [((0, 1),), ((1, 1),), ((2, 1),)],
        2: [
            ((0, 2),),
            ((0, 1), (1, 1)),
            ((0, 1), (2, 1)),
            ((1, 2),),
            ((1, 1), (2, 1)),
            ((2, 2),),
        ],
        3: [
            ((0, 2), (1, 1)),
            ((0, 2), (2, 1)),
            ((0, 1), (1, 2)),
            ((0, 1), (1, 1), (2, 1)),
            ((0, 1), (2, 2)),
            ((1, 2), (2, 1)),
            ((1, 1), (2, 2)),
        ],
        4: [
            ((0, 2), (1, 2)),
            ((0, 2), (1, 1), (2, 1)),
            ((0, 2), (2, 2)),
            ((0, 1), (1, 2), (2, 1)),
            ((0, 1), (1, 1), (2, 2)),
            ((1, 2), (2, 2)),
        ],
        5: [
            ((0, 2), (1, 2), (2, 1)),
            ((0, 2), (1, 1), (2, 2)),
            ((0, 1), (1, 2), (2, 2)),
        ],
        6: [((0, 2), (1, 2), (2, 2))],
    }


def test_limi_2() -> None:
    assert limi_2(["a", "b", "c", "d"]) == {
        1: [("a",), ("b",), ("c",), ("d",)],
        2: [
            ("a", "a"),
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("b", "b"),
            ("b", "c"),
            ("b", "d"),
            ("c", "c"),
            ("c", "d"),
            ("d", "d"),
        ],
        3: [
            ("a", "a", "b"),
            ("a", "a", "c"),
            ("a", "a", "d"),
            ("a", "b", "b"),
            ("a", "b", "c"),
            ("a", "b", "d"),
            ("a", "c", "c"),
            ("a", "c", "d"),
            ("a", "d", "d"),
            ("b", "b", "c"),
            ("b", "b", "d"),
            ("b", "c", "c"),
            ("b", "c", "d"),
            ("b", "d", "d"),
            ("c", "c", "d"),
            ("c", "d", "d"),
        ],
        4: [
            ("a", "a", "b", "b"),
            ("a", "a", "b", "c"),
            ("a", "a", "b", "d"),
            ("a", "a", "c", "c"),
            ("a", "a", "c", "d"),
            ("a", "a", "d", "d"),
            ("a", "b", "b", "c"),
            ("a", "b", "b", "d"),
            ("a", "b", "c", "c"),
            ("a", "b", "c", "d"),
            ("a", "b", "d", "d"),
            ("a", "c", "c", "d"),
            ("a", "c", "d", "d"),
            ("b", "b", "c", "c"),
            ("b", "b", "c", "d"),
            ("b", "b", "d", "d"),
            ("b", "c", "c", "d"),
            ("b", "c", "d", "d"),
            ("c", "c", "d", "d"),
        ],
        5: [
            ("a", "a", "b", "b", "c"),
            ("a", "a", "b", "b", "d"),
            ("a", "a", "b", "c", "c"),
            ("a", "a", "b", "c", "d"),
            ("a", "a", "b", "d", "d"),
            ("a", "a", "c", "c", "d"),
            ("a", "a", "c", "d", "d"),
            ("a", "b", "b", "c", "c"),
            ("a", "b", "b", "c", "d"),
            ("a", "b", "b", "d", "d"),
            ("a", "b", "c", "c", "d"),
            ("a", "b", "c", "d", "d"),
            ("a", "c", "c", "d", "d"),
            ("b", "b", "c", "c", "d"),
            ("b", "b", "c", "d", "d"),
            ("b", "c", "c", "d", "d"),
        ],
        6: [
            ("a", "a", "b", "b", "c", "c"),
            ("a", "a", "b", "b", "c", "d"),
            ("a", "a", "b", "b", "d", "d"),
            ("a", "a", "b", "c", "c", "d"),
            ("a", "a", "b", "c", "d", "d"),
            ("a", "a", "c", "c", "d", "d"),
            ("a", "b", "b", "c", "c", "d"),
            ("a", "b", "b", "c", "d", "d"),
            ("a", "b", "c", "c", "d", "d"),
            ("b", "b", "c", "c", "d", "d"),
        ],
        7: [
            ("a", "a", "b", "b", "c", "c", "d"),
            ("a", "a", "b", "b", "c", "d", "d"),
            ("a", "a", "b", "c", "c", "d", "d"),
            ("a", "b", "b", "c", "c", "d", "d"),
        ],
        8: [
            ("a", "a", "b", "b", "c", "c", "d", "d"),
        ],
    }


def test_limi_1() -> None:
    assert limi_1(4, 3) == {
        1: [((0, 1),), ((1, 1),), ((2, 1),), ((3, 1),)],
        2: [
            ((0, 2),),
            ((0, 1), (1, 1)),
            ((0, 1), (2, 1)),
            ((0, 1), (3, 1)),
            ((1, 2),),
            ((1, 1), (2, 1)),
            ((1, 1), (3, 1)),
            ((2, 2),),
            ((2, 1), (3, 1)),
            ((3, 2),),
        ],
        3: [
            ((0, 3),),
            ((0, 2), (1, 1)),
            ((0, 2), (2, 1)),
            ((0, 2), (3, 1)),
            ((0, 1), (1, 2)),
            ((0, 1), (1, 1), (2, 1)),
            ((0, 1), (1, 1), (3, 1)),
            ((0, 1), (2, 2)),
            ((0, 1), (2, 1), (3, 1)),
            ((0, 1), (3, 2)),
            ((1, 3),),
            ((1, 2), (2, 1)),
            ((1, 2), (3, 1)),
            ((1, 1), (2, 2)),
            ((1, 1), (2, 1), (3, 1)),
            ((1, 1), (3, 2)),
            ((2, 3),),
            ((2, 2), (3, 1)),
            ((2, 1), (3, 2)),
            ((3, 3),),
        ],
    }


def test_generate_array() -> None:
    E = Equations(("1", "a"), 2)
    assert_almost_equal(
        E.generate_array(np.array([1.0, 3.0])), np.array([1.0, 1.0, 3.0, 1.0, 3.0, 9.0])
    )


def test_generate_array_1() -> None:
    E = Equations(("1", "a"), 2)
    assert_almost_equal(
        E.generate_array(np.array([[1.0, 3.0], [1.5, 2.5], [2.0, 3.5]])),
        np.array(
            [
                [1.0, 1.0, 3.0, 1.0, 3.0, 9.0],
                [1.0, 1.5, 2.5, 2.25, 3.75, 6.25],
                [1.0, 2.0, 3.5, 4.0, 7.0, 12.25],
            ]
        ),
    )


def test_generate_array_1_2(caplog) -> None:
    E = Equations(("1", "a", "2", "b"), 2)
    assert E.combi(1) == [((0, 1),), ((1, 1),), ((2, 1),), ((3, 1),)]
    assert E.combi(2) == [
        ((0, 2),),
        ((0, 1), (1, 1)),
        ((0, 1), (2, 1)),
        ((0, 1), (3, 1)),
        ((1, 2),),
        ((1, 1), (2, 1)),
        ((1, 1), (3, 1)),
        ((2, 2),),
        ((2, 1), (3, 1)),
        ((3, 2),),
    ]
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert_almost_equal(
            E.generate_array(
                np.array(
                    [
                        [1.0, 3.0, 2.0, -1.0],
                        [1.5, 2.5, 2.0, -1.0],
                        [2.0, 3.5, -1.0, 3.0],
                    ]
                )
            ),
            np.array(
                [
                    [
                        1.0,
                        1.0,
                        3.0,
                        2.0,
                        -1.0,
                        1.0,
                        3.0,
                        2.0,
                        -1.0,
                        9.0,
                        6.0,
                        -3.0,
                        4.0,
                        -2.0,
                        1.0,
                    ],
                    [
                        1.0,
                        1.5,
                        2.5,
                        2.0,
                        -1.0,
                        2.25,
                        3.75,
                        3.0,
                        -1.5,
                        6.25,
                        5.0,
                        -2.5,
                        4.0,
                        -2.0,
                        1.0,
                    ],
                    [
                        1.0,
                        2.0,
                        3.5,
                        -1.0,
                        3.0,
                        4.0,
                        7.0,
                        -2.0,
                        6.0,
                        12.25,
                        -3.5,
                        10.5,
                        1.0,
                        -3.0,
                        9.0,
                    ],
                ]
            ),
        )
    assert caplog.record_tuples == []


def test_generate_array_2() -> None:
    E = Equations(("1", "a"), 2)
    with pytest.raises(
        ValueError,
        match=r"""la taille de l'array passée à generate_array n'est
pas acceptée: [(]3, 4[)][.] La deuxième dimension attendue est 2""",
    ):
        E.generate_array(
            np.array([[1.0, 3.0, 4.0, 5.0], [1.5, 2.5, 4.0, 6.0], [2.0, 3.5, 4.0, 7.0]])
        )


def test_generate_array_3() -> None:
    E = Equations(("1", "a"), 2)
    with pytest.raises(
        ValueError,
        match=r"""le nombre de dimensions de l'array passée à generate_array n'est
pas accepté: 3 est strictement supérieur à 2""",
    ):
        E.generate_array(
            np.array(
                [
                    [[1.0, 3.0], [4.0, 5.0]],
                    [[1.5, 2.5], [4.0, 6.0]],
                    [[2.0, 3.5], [4.0, 7.0]],
                ]
            )
        )


def test_Plan_1() -> None:
    P = Plan(("1", "2", "3"))
    assert P.Equations_name == []
    assert P.add({"1": 1.0, "2": 1.0, "3": 1.0}, "eq1") == 1
    assert P.add({"1": 1.0, "2": -1.0, "3": 1.0}, "eq2") == 2
    assert P.add({"1": -1.0, "2": -1.0, "3": -1.0}, "eq3") == 3
    assert P.add({"1": -1.0, "2": -1.0, "3": -1.0}, "eq4") == 4
    assert P.Equations_name == [
        "eq1",
        "eq2",
        "eq3",
        "eq4",
    ]


def test_Plan_2() -> None:
    with pytest.raises(
        ValueError,
        match=r"""Index "a" is missing in : "1", "b"
Index "b" is not a valid index""",
    ):
        P = Plan(("1", "a"))
        P.add({"1": 1.0, "b": 1.0}, "eq5")


def test_Plan_3() -> None:
    P = Plan(("1", "b", "2", "a"))
    P.generate_circular({}, {"1": 2.0, "2": 4.0, "a": -2.0, "b": 6.0}, "E")

    assert P.Equations_table == [
        {"1": 2.0, "b": 6.0, "2": 4.0, "a": -2.0},
        {"1": -2.0, "b": 2.0, "2": 6.0, "a": 4.0},
        {"1": 4.0, "b": -2.0, "2": 2.0, "a": 6.0},
        {"1": 6.0, "b": 4.0, "2": -2.0, "a": 2.0},
    ]
    assert P.Equations_name == [
        "E_1",
        "E_2",
        "E_3",
        "E_4",
    ]


def test_Plan_4() -> None:
    P = Plan(("1", "b", "2", "a"))
    P.generate_product({"a": 1.0, "b": 2.0}, {"1": [3.0, 4.0], "2": [5.0, 6.0]}, "P")
    assert P.Equations_table == [
        {"1": 3.0, "b": 2.0, "2": 5.0, "a": 1.0},
        {"1": 3.0, "b": 2.0, "2": 6.0, "a": 1.0},
        {"1": 4.0, "b": 2.0, "2": 5.0, "a": 1.0},
        {"1": 4.0, "b": 2.0, "2": 6.0, "a": 1.0},
    ]
    assert P.Equations_name == [
        "P_1",
        "P_2",
        "P_3",
        "P_4",
    ]


def test_Plan_5() -> None:
    P = Plan(("1", "b", "2", "a"))
    P.generate_circular({}, {"1": 2.0, "2": 4.0, "a": -2.0, "b": 6.0}, "E")
    P.generate_product({"a": 1.0, "b": 2.0}, {"1": [3.0, 4.0], "2": [5.0, 6.0]}, "P")
    assert (
        P.to_csv()
        == """Name;1;b;2;a
E_1;2.0;6.0;4.0;-2.0
E_2;-2.0;2.0;6.0;4.0
E_3;4.0;-2.0;2.0;6.0
E_4;6.0;4.0;-2.0;2.0
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
    )


def test_Plan_6() -> None:
    P = Plan(("1", "b", "2", "a"))
    P.generate_circular({}, {"1": 2.0, "2": 4.0, "a": -2.0, "b": 6.0}, "E")
    P.generate_product({"a": 1.0, "b": 2.0}, {"1": [3.0, 4.0], "2": [5.0, 6.0]}, "P")
    ma, r, f1, f2, M = P.precision(2)
    assert fabs(ma) < 2e-15
    assert r == 8
    assert 98.0 < f1 < 98.1
    assert 0.19 < f2 < 0.191
    assert M.shape == (8, 15)


def test_Plan_7() -> None:
    P = Plan(("C:1", "C:2", "C:3", "C:4", "V:1", "V:2", "V:3"))
    P.generate_def_scr(
        {"C:4": 0, "V:1": 1, "V:3": -1}, ["C:1", "C:2", "C:3", "V:2"], "D"
    )
    assert P.Equations_table == [
        {"C:1": 0, "C:2": 1, "C:3": -1, "C:4": 0, "V:1": 1, "V:2": -1, "V:3": -1},
        {"C:1": 0, "C:2": -1, "C:3": 1, "C:4": 0, "V:1": 1, "V:2": 1, "V:3": -1},
        {"C:1": -1, "C:2": 0, "C:3": -1, "C:4": 0, "V:1": 1, "V:2": 1, "V:3": -1},
        {"C:1": 1, "C:2": 0, "C:3": 1, "C:4": 0, "V:1": 1, "V:2": -1, "V:3": -1},
        {"C:1": -1, "C:2": -1, "C:3": 0, "C:4": 0, "V:1": 1, "V:2": -1, "V:3": -1},
        {"C:1": 1, "C:2": 1, "C:3": 0, "C:4": 0, "V:1": 1, "V:2": 1, "V:3": -1},
        {"C:1": -1, "C:2": 1, "C:3": 1, "C:4": 0, "V:1": 1, "V:2": 0, "V:3": -1},
        {"C:1": 1, "C:2": -1, "C:3": -1, "C:4": 0, "V:1": 1, "V:2": 0, "V:3": -1},
        {"C:1": 0, "C:2": 0, "C:3": 0, "C:4": 0, "V:1": 1, "V:2": 0, "V:3": -1},
    ]
    assert P.Equations_name == [
        "D_1",
        "D_2",
        "D_3",
        "D_4",
        "D_5",
        "D_6",
        "D_7",
        "D_8",
        "D_9",
    ]


def test_Plan_8(caplog: Any) -> None:
    P = Plan()
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a
E_1;2.0;6.0;4.0;-2.0
E_2;-2.0;2.0;6.0;4.0
E_3;4.0;-2.0;2.0;6.0
E_4;6.0;4.0;-2.0;2.0
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
        )
        assert caplog.record_tuples == [
            ("Plan_d_exp.src.equations", logging.INFO, "lecture de 8 expériences")
        ]
    assert P.E.indexes == ("1", "b", "2", "a")
    assert P.Equations_name == [f"E_{i}" for i in range(1, 5)] + [
        f"P_{i}" for i in range(1, 5)
    ]
    assert P.Equations_table == [
        {"1": 2.0, "b": 6.0, "2": 4.0, "a": -2.0},
        {"1": -2.0, "b": 2.0, "2": 6.0, "a": 4.0},
        {"1": 4.0, "b": -2.0, "2": 2.0, "a": 6.0},
        {"1": 6.0, "b": 4.0, "2": -2.0, "a": 2.0},
        {"1": 3.0, "b": 2.0, "2": 5.0, "a": 1.0},
        {"1": 3.0, "b": 2.0, "2": 6.0, "a": 1.0},
        {"1": 4.0, "b": 2.0, "2": 5.0, "a": 1.0},
        {"1": 4.0, "b": 2.0, "2": 6.0, "a": 1.0},
    ]


def test_Plan_9(caplog: Any) -> None:
    P = Plan()
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert not P.from_csv(
            b"""Name;1;b;2;a
E_1;2.0;6.0;4.0;-2.I
E_2;-2.0;2.0;6.0;4.0
E_3;4.0;-2.0;2.0;6.0
E_4;6.0;4.0;-2.0;2.0
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
        )
        assert caplog.record_tuples == [
            (
                "Plan_d_exp.src.equations",
                logging.ERROR,
                "à la ligne 2 colonne 5: impossible de convertir '-2.I'",
            ),
            ("Plan_d_exp.src.equations", logging.INFO, "lecture de 7 expériences"),
        ]
    assert P.E.indexes == ("1", "b", "2", "a")
    assert P.Equations_name == [f"E_{i}" for i in range(2, 5)] + [
        f"P_{i}" for i in range(1, 5)
    ]
    assert P.Equations_table == [
        {"1": -2.0, "b": 2.0, "2": 6.0, "a": 4.0},
        {"1": 4.0, "b": -2.0, "2": 2.0, "a": 6.0},
        {"1": 6.0, "b": 4.0, "2": -2.0, "a": 2.0},
        {"1": 3.0, "b": 2.0, "2": 5.0, "a": 1.0},
        {"1": 3.0, "b": 2.0, "2": 6.0, "a": 1.0},
        {"1": 4.0, "b": 2.0, "2": 5.0, "a": 1.0},
        {"1": 4.0, "b": 2.0, "2": 6.0, "a": 1.0},
    ]


def test_Plan_10(caplog: Any) -> None:
    P = Plan()
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert not P.from_csv(
            b"""Name;1;b;2;a
E_1;2.0;6.0;4.0;-2.0
E_2;-2.0;2.0;6.0
E_3;4.0;-2.0;2.0;6.0
E_4;6.0;4.0;-2.0;2.0
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
        )
        assert caplog.record_tuples == [
            (
                "Plan_d_exp.src.equations",
                logging.ERROR,
                """la ligne 3:
'E_2;-2.0;2.0;6.0'
ne contient pas suffisamment de colonnes: 3 au lieu de 4""",
            ),
            ("Plan_d_exp.src.equations", logging.INFO, "lecture de 7 expériences"),
        ]
    assert P.E.indexes == ("1", "b", "2", "a")
    assert P.Equations_name == [f"E_{i}" for i in (1, 3, 4)] + [
        f"P_{i}" for i in range(1, 5)
    ]
    assert P.Equations_table == [
        {"1": 2.0, "b": 6.0, "2": 4.0, "a": -2.0},
        {"1": 4.0, "b": -2.0, "2": 2.0, "a": 6.0},
        {"1": 6.0, "b": 4.0, "2": -2.0, "a": 2.0},
        {"1": 3.0, "b": 2.0, "2": 5.0, "a": 1.0},
        {"1": 3.0, "b": 2.0, "2": 6.0, "a": 1.0},
        {"1": 4.0, "b": 2.0, "2": 5.0, "a": 1.0},
        {"1": 4.0, "b": 2.0, "2": 6.0, "a": 1.0},
    ]


def test_Plan_11(caplog: Any) -> None:
    P = Plan()
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a;;
E_1;2.0;6.0;4.0;-2.0

E_2;-2.0;2.0;6.0;4.0
E_3;4.0;-2.0;2.0;6.0;
E_4;6.0;4.0;-2.0;2.0;;;;;;;;;
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
        )
        assert caplog.record_tuples == [
            ("Plan_d_exp.src.equations", logging.WARNING, "la ligne 3 est vide"),
            ("Plan_d_exp.src.equations", logging.INFO, "lecture de 8 expériences"),
        ]
    assert P.E.indexes == ("1", "b", "2", "a")
    assert P.Equations_name == [f"E_{i}" for i in range(1, 5)] + [
        f"P_{i}" for i in range(1, 5)
    ]
    assert P.Equations_table == [
        {"1": 2.0, "b": 6.0, "2": 4.0, "a": -2.0},
        {"1": -2.0, "b": 2.0, "2": 6.0, "a": 4.0},
        {"1": 4.0, "b": -2.0, "2": 2.0, "a": 6.0},
        {"1": 6.0, "b": 4.0, "2": -2.0, "a": 2.0},
        {"1": 3.0, "b": 2.0, "2": 5.0, "a": 1.0},
        {"1": 3.0, "b": 2.0, "2": 6.0, "a": 1.0},
        {"1": 4.0, "b": 2.0, "2": 5.0, "a": 1.0},
        {"1": 4.0, "b": 2.0, "2": 6.0, "a": 1.0},
    ]


def test_Plan_12(caplog: Any) -> None:
    P = Plan()
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert not P.from_csv(
            b"""Name;1;;2;a
E_1;2.0;6.0;4.0;-2.0
E_2;-2.0;2.0;6.0;4.0
E_3;4.0;-2.0;2.0;6.0
E_4;6.0;4.0;-2.0;2.0
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
        )
        assert caplog.record_tuples == [
            (
                "Plan_d_exp.src.equations",
                logging.ERROR,
                """le fichier csv contient des noms de variables vides:
indexes number 2 has length equal zero""",
            )
        ]
    assert P.E.indexes == ("a", "b")
    assert len(P.Equations_name) == 0
    assert len(P.Equations_table) == 0


def test_Plan_13(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.read_results_from_csv(setup_res, 1) == {}
        assert caplog.record_tuples == [
            (
                "Plan_d_exp.src.equations",
                logging.ERROR,
                "col_no = 1 or il doit être suppérieur ou égal à 2",
            )
        ]


def test_Plan_14(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.read_results_from_csv(setup_res, 2) == {}
        assert caplog.record_tuples == [
            (
                "Plan_d_exp.src.equations",
                logging.INFO,
                f"le fichier {setup_res} ne contient pas plus d'une ligne",
            )
        ]


def test_Plan_15(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    setup_res.write_text(
        """Nom;val

P_1;0.17"""
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a
E_1;2.0;6.0;4.0;-2.0
E_2;-2.0;2.0;6.0;4.0
E_3;4.0;-2.0;2.0;6.0
E_4;6.0;4.0;-2.0;2.0
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
        )
        assert P.read_results_from_csv(setup_res, 2) == {"P_1": 0.17}
        assert caplog.record_tuples == [
            ("Plan_d_exp.src.equations", logging.INFO, "lecture de 8 expériences"),
            (
                "Plan_d_exp.src.equations",
                logging.INFO,
                f"le fichier {setup_res} contient 1 ligne(s) vide(s) numérotée(s): 2",
            ),
            (
                "Plan_d_exp.src.equations",
                logging.INFO,
                f"1 lignes lues dans le fichier {setup_res}",
            ),
        ]


def test_Plan_16(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    setup_res.write_text(
        """Nom;val
E_1
P_1;0.17"""
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a
E_1;2.0;6.0;4.0;-2.0
E_2;-2.0;2.0;6.0;4.0
E_3;4.0;-2.0;2.0;6.0
E_4;6.0;4.0;-2.0;2.0
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
        )
        assert P.read_results_from_csv(setup_res, 2) == {"P_1": 0.17}
        assert caplog.record_tuples == [
            ("Plan_d_exp.src.equations", logging.INFO, "lecture de 8 expériences"),
            (
                "Plan_d_exp.src.equations",
                logging.WARNING,
                f"le fichier {setup_res} contient 1 ligne(s) sans colonne no 2:\n"
                "ligne no: 2 qui contient 'E_1'",
            ),
            (
                "Plan_d_exp.src.equations",
                logging.INFO,
                f"1 lignes lues dans le fichier {setup_res}",
            ),
        ]


def test_Plan_17(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    setup_res.write_text(
        """Nom;val
E_1;0.18
P_0;0.17"""
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a
E_1;2.0;6.0;4.0;-2.0
E_2;-2.0;2.0;6.0;4.0
E_3;4.0;-2.0;2.0;6.0
E_4;6.0;4.0;-2.0;2.0
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
        )
        assert P.read_results_from_csv(setup_res, 2) == {"P_0": 0.17, "E_1": 0.18}
        assert caplog.record_tuples == [
            ("Plan_d_exp.src.equations", logging.INFO, "lecture de 8 expériences"),
            (
                "Plan_d_exp.src.equations",
                logging.WARNING,
                f"le fichier {setup_res} contient 1 ligne(s) dont le nom n'est pas reconnu:\n"
                "ligne no: 3 qui contient 'P_0'",
            ),
            (
                "Plan_d_exp.src.equations",
                logging.INFO,
                f"2 lignes lues dans le fichier {setup_res}",
            ),
        ]


def test_Plan_18(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    setup_res.write_text(
        """Nom;val
E_1;0.18
P_1;0.1t"""
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a
E_1;2.0;6.0;4.0;-2.0
E_2;-2.0;2.0;6.0;4.0
E_3;4.0;-2.0;2.0;6.0
E_4;6.0;4.0;-2.0;2.0
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
        )
        assert P.read_results_from_csv(setup_res, 2) == {"E_1": 0.18}
        assert caplog.record_tuples == [
            ("Plan_d_exp.src.equations", logging.INFO, "lecture de 8 expériences"),
            (
                "Plan_d_exp.src.equations",
                logging.WARNING,
                f"le fichier {setup_res} contient 1 ligne(s) dont la valeur n'est pas convertissable en réel:\n"
                "ligne no: 3 qui contient '0.1t'",
            ),
            (
                "Plan_d_exp.src.equations",
                logging.INFO,
                f"1 lignes lues dans le fichier {setup_res}",
            ),
        ]


def test_Plan_19(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    setup_res.write_text(
        """Nom;val
E_1;0;0.18
E_2;1;0.08
E_3;2;0.12
E_4;3;0.28
P_1;4;0.37
P_2;4;0.27
P_3;4;0.17
P_4;5;0.07
"""
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a
E_1;2.0;6.0;4.0;-2.0
E_2;-2.0;2.0;6.0;4.0
E_3;4.0;-2.0;2.0;6.0
E_4;6.0;4.0;-2.0;2.0
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
        )
        val = P.read_results_from_csv(setup_res, 3)
        assert val == {
            "E_1": 0.18,
            "E_2": 0.08,
            "E_3": 0.12,
            "E_4": 0.28,
            "P_1": 0.37,
            "P_2": 0.27,
            "P_3": 0.17,
            "P_4": 0.07,
        }
        R, O, E, constants, f1, f2, r = P.solve2(2, val)
        assert constants == []
        assert O == [
            "E_1",
            "E_2",
            "E_3",
            "E_4",
            "P_1",
            "P_2",
            "P_3",
            "P_4",
        ]

    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "lecture de 8 expériences"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"8 lignes lues dans le fichier {setup_res}",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"après la résolution l'écart maximal est: {f1}\n"
            f"l'écart minimal est: {f2}\n"
            f"le rang de la matrice est: {r}",
        ),
    ]


def test_Plan_20(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    setup_res.write_text(
        """Nom;val
E_1;0.18
E_2;0.08
E_3;0.12
P_1;0.37
P_2;0.27
P_3;0.17
P_4;0.07
"""
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a
E_1;2.0;6.0;4.0;-2.0
E_2;-2.0;2.0;6.0;4.0
E_3;4.0;-2.0;2.0;6.0
E_4;6.0;4.0;-2.0;2.0
P_1;3.0;2.0;5.0;1.0
P_2;3.0;2.0;6.0;1.0
P_3;4.0;2.0;5.0;1.0
P_4;4.0;2.0;6.0;1.0"""
        )
        val = P.read_results_from_csv(setup_res, 2)
        assert val == {
            "E_1": 0.18,
            "E_2": 0.08,
            "E_3": 0.12,
            "P_1": 0.37,
            "P_2": 0.27,
            "P_3": 0.17,
            "P_4": 0.07,
        }
        R, O, E, constants, f1, f2, r = P.solve2(2, val)
        assert constants == []
        assert O == [
            "E_1",
            "E_2",
            "E_3",
            "P_1",
            "P_2",
            "P_3",
            "P_4",
        ]

    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "lecture de 8 expériences"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"7 lignes lues dans le fichier {setup_res}",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"après la résolution l'écart maximal est: {f1}\n"
            f"l'écart minimal est: {f2}\n"
            f"le rang de la matrice est: {r}",
        ),
    ]


def test_Plan_21(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    setup_res.write_text(
        """Nom;val
E_1;1.
E_2;0.1
E_3;-1.0
E_4;-0.1
"""
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a
E_1;1.0;0.0;0.0;0.0
E_2;0.0;1.0;0.0;0.0
E_3;0.0;0.0;1.0;0.0
E_4;0.0;0.0;0.0;1.0"""
        )
        val = P.read_results_from_csv(setup_res, 2)
        R, O, E, constants, f1, f2, r = P.solve2(1, val)
        assert constants == []
        assert O == [
            "E_1",
            "E_2",
            "E_3",
            "E_4",
        ]
        assert r == 4
        assert fabs(f1) < 1e-15
        assert fabs(f2) < 1e-15
        assert P.E.col_names == [
            "mean",
            "1",
            "b",
            "2",
            "a",
        ]

    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "lecture de 4 expériences"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"4 lignes lues dans le fichier {setup_res}",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"après la résolution l'écart maximal est: {f1}\n"
            f"l'écart minimal est: {f2}\n"
            f"le rang de la matrice est: {r}",
        ),
    ]
    assert_almost_equal(R, np.array([0.0, 1.0, 0.1, -1, -0.1]))


def test_Plan_22(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    setup_res.write_text(
        """Nom;val
E_1;1.
E_2;0.1
E_3;-1.0
E_4;-0.1
"""
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a
E_1;1.0;0.0;0.0;0.0
E_2;0.0;1.0;0.0;0.0
E_3;0.0;0.0;1.0;0.0
E_4;0.0;0.0;0.0;1.0"""
        )
        val = P.read_results_from_csv(setup_res, 2)
        R, O, E, constants, f1, f2, r = P.solve2(2, val)
        assert constants == []
        assert O == [
            "E_1",
            "E_2",
            "E_3",
            "E_4",
        ]
        assert r == 4
        assert fabs(f1) < 1e-15
        assert fabs(f2) < 1e-15
        assert E.col_names == [
            "mean",
            "1",
            "b",
            "2",
            "a",
            "1.1",
            "1.b",
            "1.2",
            "1.a",
            "b.b",
            "b.2",
            "b.a",
            "2.2",
            "2.a",
            "a.a",
        ]

    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "lecture de 4 expériences"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"4 lignes lues dans le fichier {setup_res}",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"après la résolution l'écart maximal est: {f1}\n"
            f"l'écart minimal est: {f2}\n"
            f"le rang de la matrice est: {r}",
        ),
    ]
    assert_almost_equal(
        R,
        np.array(
            [
                0.0,
                0.5,
                0.05,
                -0.5,
                -0.05,
                0.5,
                0.0,
                0.0,
                0.0,
                0.05,
                0.0,
                0.0,
                -0.5,
                0.0,
                -0.05,
            ]
        ),
    )


def test_Plan_23(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a", "c"))
    setup_res.write_text(
        """Nom;val
E_1;1.
E_2;0.1
E_3;-1.0
E_4;-0.1
"""
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a;c
E_1;1.0;0.0;0.0;0.0;1.0
E_2;0.0;1.0;0.0;0.0;1.0
E_3;0.0;0.0;1.0;0.0;1.0
E_4;0.0;0.0;0.0;1.0;1.0"""
        )
        val = P.read_results_from_csv(setup_res, 2)
        R, O, E, constants, f1, f2, r = P.solve2(2, val)
        assert constants == [
            ("c", 1.0),
        ]
        assert O == [
            "E_1",
            "E_2",
            "E_3",
            "E_4",
        ]
        assert r == 4
        assert fabs(f1) < 1e-15
        assert fabs(f2) < 1e-15
        assert E.col_names == [
            "mean",
            "1",
            "b",
            "2",
            "a",
            "1.1",
            "1.b",
            "1.2",
            "1.a",
            "b.b",
            "b.2",
            "b.a",
            "2.2",
            "2.a",
            "a.a",
        ]

    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "lecture de 4 expériences"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"4 lignes lues dans le fichier {setup_res}",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "les variables constantes dans les équations sont:\nc: 1.0",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"après la résolution l'écart maximal est: {f1}\n"
            f"l'écart minimal est: {f2}\n"
            f"le rang de la matrice est: {r}",
        ),
    ]
    assert_almost_equal(
        R,
        np.array(
            [
                0.0,
                0.5,
                0.05,
                -0.5,
                -0.05,
                0.5,
                0.0,
                0.0,
                0.0,
                0.05,
                0.0,
                0.0,
                -0.5,
                0.0,
                -0.05,
            ]
        ),
    )


def test_Plan_24(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a;c
E_1;1.0;0.0;0.0;0.0;1.0
E_2;0.0;1.0;0.0;0.0;1.0
E_3;0.0;0.0;1.0;0.0;1.0
E_4;0.0;0.0;0.0;1.0;1.0"""
        )
        assert P.search_constant_var() == [
            ("c", 1.0),
        ]
    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "lecture de 4 expériences"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "les variables constantes dans les équations sont:\nc: 1.0",
        ),
    ]


def test_Plan_25(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a"))
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a;c
E_1;1.0;0.0;0.0;0.0;1.0
E_2;0.0;1.0;0.0;0.0;1.0
E_3;0.0;0.0;0.0;0.0;1.0
E_4;0.0;0.0;0.0;1.0;1.0"""
        )
        assert P.search_constant_var() == [
            ("2", 0.0),
            ("c", 1.0),
        ]
    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "lecture de 4 expériences"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "les variables constantes dans les équations sont:\n2: 0.0\nc: 1.0",
        ),
    ]


def test_Plan_26(setup_res: Any, caplog: Any) -> None:
    P = Plan(("1", "b", "2", "a", "c"))
    setup_res.write_text(
        """Nom;val
E_1;2.
E_2;1.1
E_3;0.0
E_4;0.9
E_5;1.0
"""
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert P.from_csv(
            b"""Name;1;b;2;a;c
E_1;1.0;0.0;0.0;0.0;1.0
E_2;0.0;1.0;0.0;0.0;1.0
E_3;0.0;0.0;1.0;0.0;1.0
E_4;0.0;0.0;0.0;1.0;1.0
E_5;0.0;0.0;0.0;0.0;1.0"""
        )
        val = P.read_results_from_csv(setup_res, 2)
        R, O, E, constants, f1, f2, r = P.solve2(2, val)
        assert constants == [
            ("c", 1.0),
        ]
        assert O == [
            "E_1",
            "E_2",
            "E_3",
            "E_4",
            "E_5",
        ]
        assert r == 5
        assert fabs(f1) < 1e-14
        assert fabs(f2) < 1e-14
        assert E.col_names == [
            "mean",
            "1",
            "b",
            "2",
            "a",
            "1.1",
            "1.b",
            "1.2",
            "1.a",
            "b.b",
            "b.2",
            "b.a",
            "2.2",
            "2.a",
            "a.a",
        ]
    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "lecture de 5 expériences"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"5 lignes lues dans le fichier {setup_res}",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "les variables constantes dans les équations sont:\nc: 1.0",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"après la résolution l'écart maximal est: {f1}\n"
            f"l'écart minimal est: {f2}\n"
            f"le rang de la matrice est: {r}",
        ),
    ]
    assert_almost_equal(
        R,
        np.array(
            [
                1.0,
                0.5,
                0.05,
                -0.5,
                -0.05,
                0.5,
                0.0,
                0.0,
                0.0,
                0.05,
                0.0,
                0.0,
                -0.5,
                0.0,
                -0.05,
            ]
        ),
    )
