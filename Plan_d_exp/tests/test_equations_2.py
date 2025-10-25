import logging
from math import fabs
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from Plan_d_exp.src.equations import (
    Courbe1,
    Courbe2,
    Equations,
    Plan,
    Plan_tri,
    def_scr,
    genere_def_scr_int,
    vers_int,
)


@pytest.fixture
def setup_res(tmp_path):
    tmp_res = tmp_path / Path("result.csv")
    tmp_res.write_text("Nom;val")
    return tmp_res


def test_Plan_tri(caplog) -> None:
    PT = Plan_tri(("1", "b", "2", "a", "c"))
    with caplog.at_level(logging.ERROR, logger="Plan_d_exp.src.equations"):
        PT.add({"1": 0.5, "b": 0, "2": 1, "a": 0.1, "c": 0}, "E_1")
    assert caplog.record_tuples == [
        (
            "Plan_d_exp.src.equations",
            logging.ERROR,
            """pour l'équation 'E_1' les valeurs suivantes affectées aux variables
ne sont pas acceptées pour un Plan_tri:
'1': 0.5, 'a': 0.1""",
        )
    ]


def test_Plan_tri_1(caplog) -> None:
    PT = Plan_tri(("1", "b", "2", "a", "c"))
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert PT.bit_index == {
            "1": (1, 2, 3),
            "b": (4, 8, 12),
            "2": (16, 32, 48),
            "a": (64, 128, 192),
            "c": (256, 512, 768),
        }
        assert PT.to_bitlevels({"1": 0, "b": 1, "2": -1, "a": 1, "c": 0}) == (
            2 + 12 + 16 + 192 + 512
        )
        assert PT.add({"1": 0, "b": 1, "2": -1, "a": 1, "c": 0}, "E_1") == 1
        assert PT.sbitlevels == {
            734,
        }
        assert PT.lbitlevels == [
            734,
        ]
        assert PT.add({"1": -1, "b": 1, "2": -1, "a": 1, "c": 0}, "E_2") == 2
        assert PT.sbitlevels == {
            734,
            733,
        }
        assert PT.lbitlevels == [734, 733]


def test_Plan_tri_2(caplog) -> None:
    PT = Plan_tri(("1", "b", "2", "a", "c"))
    with caplog.at_level(logging.ERROR, logger="Plan_d_exp.src.equations"):
        assert PT.add({"1": 0, "b": 0, "2": -1, "a": 1, "c": 0}, "E_1") == 1
        assert PT.add({"1": 0, "b": 0, "2": -1, "a": 1, "c": 0}, "E_2") == 0
    assert caplog.record_tuples == [
        (
            "Plan_d_exp.src.equations",
            logging.ERROR,
            "pour l'équation 'E_2': une équation avec les mêmes coefficients existe déjà",
        )
    ]


def test_Plan_tri_3(caplog) -> None:
    PT = Plan_tri(("1", "b", "2", "a", "c"))
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert PT.add({"1": 0, "b": 0, "2": -1, "a": 1, "c": 0}, "E_1") == 1
        assert PT.add({"1": 1, "b": 0, "2": 1, "a": 1, "c": 1}, "E_2") == 2
        assert PT.add({"1": -1, "b": 0, "2": -1, "a": -1, "c": 0}, "E_3") == 3
        assert PT.add({"1": 0, "b": 1, "2": 1, "a": 1, "c": 1}, "E_4") == 4
        assert PT.add({"1": 0, "b": 1, "2": -1, "a": -1, "c": 0}, "E_5") == 5
        assert PT.add({"1": 1, "b": -1, "2": 0, "a": 0, "c": -1}, "E_6") == 6
        PT.search_stat(
            {"E_1": 1.0, "E_2": 1.1, "E_3": 1.2, "E_4": 1.3, "E_5": 1.4, "E_6": 1.5}
        )
        assert set(PT.pos_index.keys()) == {"1", "b", "2", "a", "c"}
        res = {
            "1": (
                np.array(
                    [
                        2,
                    ]
                ),
                np.array([0, 3, 4]),
                np.array([1, 5]),
            ),
            "b": (
                np.array(
                    [
                        5,
                    ]
                ),
                np.array([0, 1, 2]),
                np.array([3, 4]),
            ),
            "2": (
                np.array([0, 2, 4]),
                np.array(
                    [
                        5,
                    ]
                ),
                np.array([1, 3]),
            ),
            "a": (
                np.array([2, 4]),
                np.array(
                    [
                        5,
                    ]
                ),
                np.array([0, 1, 3]),
            ),
            "c": (
                np.array(
                    [
                        5,
                    ]
                ),
                np.array([0, 2, 4]),
                np.array([1, 3]),
            ),
        }
        for k, v in res.items():
            assert_almost_equal(v[0], PT.pos_index[k][0])
            assert_almost_equal(v[1], PT.pos_index[k][1])
            assert_almost_equal(v[2], PT.pos_index[k][2])
        res2 = {
            "1": (
                np.array(
                    [
                        1.2,
                    ]
                ),
                np.array([1.0, 1.3, 1.4]),
                np.array([1.1, 1.5]),
            ),
            "b": (
                np.array(
                    [
                        1.5,
                    ]
                ),
                np.array([1.0, 1.1, 1.2]),
                np.array([1.3, 1.4]),
            ),
            "2": (
                np.array([1.0, 1.2, 1.4]),
                np.array(
                    [
                        1.5,
                    ]
                ),
                np.array([1.1, 1.3]),
            ),
            "a": (
                np.array([1.2, 1.4]),
                np.array(
                    [
                        1.5,
                    ]
                ),
                np.array([1.0, 1.1, 1.3]),
            ),
            "c": (
                np.array(
                    [
                        1.5,
                    ]
                ),
                np.array([1.0, 1.2, 1.4]),
                np.array([1.1, 1.3]),
            ),
        }
        for k, v in res2.items():
            assert_almost_equal(v[0], PT.res_index[k][0])
            assert_almost_equal(v[1], PT.res_index[k][1])
            assert_almost_equal(v[2], PT.res_index[k][2])
    assert PT.Courbes == {
        "1": (None, None, None),
        "2": (None, None, None),
        "a": (None, None, None),
        "b": (None, None, None),
        "c": (None, None, None),
    }
    assert len(PT.ddd.keys()) == 0

    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "6 résultats seront traités"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, à la valeur -1: la taille des résultats est 1 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, à la valeur 0: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, à la valeur 1: la taille des résultats est 2 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, à la valeur -1: la taille des résultats est 1 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, à la valeur 0: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, à la valeur 1: la taille des résultats est 2 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, à la valeur -1: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, à la valeur 0: la taille des résultats est 1 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, à la valeur 1: la taille des résultats est 2 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, à la valeur -1: la taille des résultats est 2 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, à la valeur 0: la taille des résultats est 1 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, à la valeur 1: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, à la valeur -1: la taille des résultats est 1 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, à la valeur 0: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, à la valeur 1: la taille des résultats est 2 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, entre 0 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, entre 0 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, entre 0 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, entre 0 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, entre 0 et 1: il n'y a pas de données",
        ),
    ]


def test_Plan_tri_4(caplog) -> None:
    PT = Plan_tri(("1", "b", "2", "a", "c"))
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        assert PT.add({"1": 0, "b": 0, "2": -1, "a": 1, "c": 0}, "E_1") == 1
        assert PT.add({"1": 1, "b": 0, "2": 1, "a": 1, "c": 1}, "E_2") == 2
        assert PT.add({"1": -1, "b": 0, "2": -1, "a": -1, "c": 0}, "E_3") == 3
        assert PT.add({"1": 0, "b": 1, "2": 1, "a": 1, "c": 1}, "E_4") == 4
        assert PT.add({"1": 0, "b": 1, "2": -1, "a": -1, "c": 0}, "E_5") == 5
        assert PT.add({"1": 1, "b": -1, "2": 0, "a": 0, "c": -1}, "E_6") == 6
        PT.search_stat(
            {
                "E_1": 1.0,
                "E_2": 1.1,
                "E_3": 1.2,
                "E_4": 1.3,
                "E_5": 1.4,
                "E_6": 1.5,
                "E_7": 1.6,
            }
        )
    assert caplog.record_tuples[0] == (
        "Plan_d_exp.src.equations",
        logging.WARNING,
        "les résultats suivants: E_7 n'ont pas d'équation associée, ils sont ignorés",
    )
    assert caplog.record_tuples[1] == (
        "Plan_d_exp.src.equations",
        logging.INFO,
        "6 résultats seront traités",
    )


def test_Plan_tri_5(caplog) -> None:
    PT = Plan_tri(("1", "b", "2", "a", "c"))
    with caplog.at_level(logging.DEBUG, logger="Plan_d_exp.src.equations"):
        PT.generate_product(
            {"1": 0, "b": 0, "2": -1}, {"a": [-1, 0, 1], "c": [-1, 0, 1]}, "E"
        )
        assert PT.Equations_table == [
            {"1": 0, "b": 0, "2": -1, "a": -1, "c": -1},
            {"1": 0, "b": 0, "2": -1, "a": -1, "c": 0},
            {"1": 0, "b": 0, "2": -1, "a": -1, "c": 1},
            {"1": 0, "b": 0, "2": -1, "a": 0, "c": -1},
            {"1": 0, "b": 0, "2": -1, "a": 0, "c": 0},
            {"1": 0, "b": 0, "2": -1, "a": 0, "c": 1},
            {"1": 0, "b": 0, "2": -1, "a": 1, "c": -1},
            {"1": 0, "b": 0, "2": -1, "a": 1, "c": 0},
            {"1": 0, "b": 0, "2": -1, "a": 1, "c": 1},
        ]
        assert PT.lbitlevels == [
            2 + 8 + 16 + 64 + 256,
            26 + 64 + 512,
            26 + 64 + 768,
            26 + 128 + 256,
            26 + 128 + 512,
            26 + 128 + 768,
            26 + 192 + 256,
            26 + 192 + 512,
            26 + 192 + 768,
        ]
        PT.search_stat(
            {
                "E_1": 1.0,
                "E_2": 1.1,
                "E_3": 1.2,
                "E_4": 1.3,
                "E_5": 1.4,
                "E_6": 1.5,
                "E_7": 1.6,
                "E_8": 1.7,
                "E_9": 1.8,
            }
        )
        res = {
            "1": (
                np.array([], dtype=np.float64),
                np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]),
                np.array([], dtype=np.float64),
            ),
            "2": (
                np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            ),
            "a": (
                np.array([1.0, 1.1, 1.2]),
                np.array([1.3, 1.4, 1.5]),
                np.array([1.6, 1.7, 1.8]),
            ),
            "b": (
                np.array([], dtype=np.float64),
                np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]),
                np.array([], dtype=np.float64),
            ),
            "c": (
                np.array([1.0, 1.3, 1.6]),
                np.array([1.1, 1.4, 1.7]),
                np.array([1.2, 1.5, 1.8]),
            ),
        }
        for k, v in res.items():
            assert_almost_equal(v[0], PT.res_index[k][0])
            assert_almost_equal(v[1], PT.res_index[k][1])
            assert_almost_equal(v[2], PT.res_index[k][2])
        assert set(PT.dd_index.keys()) == {3, 4}
        assert_almost_equal(
            PT.dd_index[3][0], np.array([1.3, 1.4, 1.5, 1.6, 1.6, 1.7, 1.7, 1.8, 1.8])
        )
        assert_almost_equal(
            PT.dd_index[3][1],
            np.array([1.0, 1.1, 1.2, 1.0, 1.3, 1.1, 1.4, 1.2, 1.5]),
        )
        assert_almost_equal(
            PT.dd_index[3][2],
            np.array([128, 128, 128, 192, 192, 192, 192, 192, 192], dtype=np.uint64),
        )
        assert_almost_equal(
            PT.dd_index[3][3],
            np.array([64, 64, 64, 64, 128, 64, 128, 64, 128], dtype=np.uint64),
        )
        assert_almost_equal(
            PT.dd_index[4][0],
            np.array([1.1, 1.2, 1.2, 1.4, 1.5, 1.5, 1.7, 1.8, 1.8]),
        )
        assert_almost_equal(
            PT.dd_index[4][1],
            np.array([1.0, 1.0, 1.1, 1.3, 1.3, 1.4, 1.6, 1.6, 1.7]),
        )
        assert_almost_equal(
            PT.dd_index[4][2],
            np.array([512, 768, 768, 512, 768, 768, 512, 768, 768], dtype=np.uint64),
        )
        assert_almost_equal(
            PT.dd_index[4][3],
            np.array([256, 256, 512, 256, 256, 512, 256, 256, 512], dtype=np.uint64),
        )
        assert set(PT.ddd.keys()) == {
            ("a", -1, 0),
            ("a", -1, 1),
            ("a", 0, 1),
            ("c", -1, 0),
            ("c", -1, 1),
            ("c", 0, 1),
        }
        assert_almost_equal(PT.ddd[("a", -1, 0)], np.array([0.3, 0.3, 0.3]))
        assert_almost_equal(PT.ddd[("a", -1, 1)], np.array([0.6, 0.6, 0.6]))
        assert_almost_equal(PT.ddd[("a", 0, 1)], np.array([0.3, 0.3, 0.3]))
        assert_almost_equal(PT.ddd[("c", -1, 0)], np.array([0.1, 0.1, 0.1]))
        assert_almost_equal(PT.ddd[("c", -1, 1)], np.array([0.2, 0.2, 0.2]))
        assert_almost_equal(PT.ddd[("c", 0, 1)], np.array([0.1, 0.1, 0.1]))
    assert PT.Courbes == {
        "1": (
            None,
            Courbe1(
                name="1",
                value="ERROR",
                bit=0,
                max_val=1.8,
                min_val=1.0,
                pct_25=1.2,
                pct_50=1.4,
                pct_75=1.6,
                sample_size=9,
                mu_samp=1.4,
                var_samp=0.075,
            ),
            None,
        ),
        "b": (
            None,
            Courbe1(
                name="b",
                value="ERROR",
                bit=0,
                max_val=1.8,
                min_val=1.0,
                pct_25=1.2,
                pct_50=1.4,
                pct_75=1.6,
                sample_size=9,
                mu_samp=1.4,
                var_samp=0.075,
            ),
            None,
        ),
        "2": (
            Courbe1(
                name="2",
                value="ERROR",
                bit=-1,
                max_val=1.8,
                min_val=1.0,
                pct_25=1.2,
                pct_50=1.4,
                pct_75=1.6,
                sample_size=9,
                mu_samp=1.4,
                var_samp=0.075,
            ),
            None,
            None,
        ),
        "a": (None, None, None),
        "c": (None, None, None),
    }
    assert PT.Courbes_ddd == {}
    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "9 résultats seront traités"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, à la valeur -1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, à la valeur 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, à la valeur -1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, à la valeur 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, à la valeur 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, à la valeur 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, à la valeur -1: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, à la valeur 0: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, à la valeur 1: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, à la valeur -1: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, à la valeur 0: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, à la valeur 1: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, entre 0 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, entre 0 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, entre 0 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, entre -1 et 0: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, entre -1 et 1: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, entre 0 et 1: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, entre -1 et 0: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, entre -1 et 1: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, entre 0 et 1: la taille des résultats est 3 < 5",
        ),
    ]


def test_Plan_tri_6(caplog) -> None:
    PT = Plan_tri(("1", "b", "2", "a", "c"))
    with caplog.at_level(logging.DEBUG, logger="Plan_d_exp.src.equations"):
        assert PT.add({"1": 0, "b": 0, "2": -1, "a": 1, "c": 0}, "E_1") == 1
        assert PT.add({"1": 0, "b": 0, "2": 1, "a": 1, "c": 1}, "E_2") == 2
        assert PT.add({"1": 0, "b": 0, "2": -1, "a": -1, "c": 0}, "E_3") == 3
        assert PT.add({"1": 0, "b": 1, "2": 1, "a": 1, "c": 1}, "E_4") == 4
        assert PT.add({"1": 0, "b": 1, "2": -1, "a": -1, "c": 0}, "E_5") == 5
        assert PT.add({"1": 0, "b": -1, "2": 0, "a": 0, "c": -1}, "E_6") == 6
        PT.search_stat(
            {"E_1": 1.0, "E_2": 1.1, "E_3": 1.2, "E_4": 1.3, "E_5": 1.4, "E_6": 1.5}
        )
        assert set(PT.pos_index.keys()) == {"1", "b", "2", "a", "c"}
        res = {
            "1": (
                np.array([]),
                np.array([0, 1, 2, 3, 4, 5]),
                np.array([]),
            ),
            "b": (
                np.array(
                    [
                        5,
                    ]
                ),
                np.array([0, 1, 2]),
                np.array([3, 4]),
            ),
            "2": (
                np.array([0, 2, 4]),
                np.array(
                    [
                        5,
                    ]
                ),
                np.array([1, 3]),
            ),
            "a": (
                np.array([2, 4]),
                np.array(
                    [
                        5,
                    ]
                ),
                np.array([0, 1, 3]),
            ),
            "c": (
                np.array(
                    [
                        5,
                    ]
                ),
                np.array([0, 2, 4]),
                np.array([1, 3]),
            ),
        }
        for k, v in res.items():
            assert_almost_equal(v[0], PT.pos_index[k][0])
            assert_almost_equal(v[1], PT.pos_index[k][1])
            assert_almost_equal(v[2], PT.pos_index[k][2])
        res2 = {
            "1": (
                np.array([]),
                np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
                np.array([]),
            ),
            "b": (
                np.array(
                    [
                        1.5,
                    ]
                ),
                np.array([1.0, 1.1, 1.2]),
                np.array([1.3, 1.4]),
            ),
            "2": (
                np.array([1.0, 1.2, 1.4]),
                np.array(
                    [
                        1.5,
                    ]
                ),
                np.array([1.1, 1.3]),
            ),
            "a": (
                np.array([1.2, 1.4]),
                np.array(
                    [
                        1.5,
                    ]
                ),
                np.array([1.0, 1.1, 1.3]),
            ),
            "c": (
                np.array(
                    [
                        1.5,
                    ]
                ),
                np.array([1.0, 1.2, 1.4]),
                np.array([1.1, 1.3]),
            ),
        }
        for k, v in res2.items():
            assert_almost_equal(v[0], PT.res_index[k][0])
            assert_almost_equal(v[1], PT.res_index[k][1])
            assert_almost_equal(v[2], PT.res_index[k][2])
        assert set(PT.ddd.keys()) == {
            ("a", -1, 1),
            ("b", 0, 1),
        }
        assert_almost_equal(
            PT.ddd[("a", -1, 1)],
            np.array(
                [
                    -0.2,
                ]
            ),
        )
        assert_almost_equal(
            PT.ddd[("b", 0, 1)],
            np.array([0.2, 0.2]),
        )
        assert PT.Courbes == {
            "1": (
                None,
                Courbe1(
                    name="1",
                    value="ERROR",
                    bit=0,
                    max_val=1.5,
                    min_val=1.0,
                    pct_25=1.125,
                    pct_50=1.25,
                    pct_75=1.375,
                    sample_size=6,
                    mu_samp=1.25,
                    var_samp=0.03499999999999999,
                ),
                None,
            ),
            "2": (None, None, None),
            "a": (None, None, None),
            "b": (None, None, None),
            "c": (None, None, None),
        }
    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "6 résultats seront traités"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, à la valeur -1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, à la valeur 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, à la valeur -1: la taille des résultats est 1 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, à la valeur 0: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, à la valeur 1: la taille des résultats est 2 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, à la valeur -1: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, à la valeur 0: la taille des résultats est 1 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, à la valeur 1: la taille des résultats est 2 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, à la valeur -1: la taille des résultats est 2 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, à la valeur 0: la taille des résultats est 1 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, à la valeur 1: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, à la valeur -1: la taille des résultats est 1 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, à la valeur 0: la taille des résultats est 3 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, à la valeur 1: la taille des résultats est 2 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 1, entre 0 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable b, entre 0 et 1: la taille des résultats est 2 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable 2, entre 0 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, entre -1 et 1: la taille des résultats est 1 < 5",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable a, entre 0 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, entre -1 et 0: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, entre -1 et 1: il n'y a pas de données",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "pour la variable c, entre 0 et 1: il n'y a pas de données",
        ),
    ]


def test_Plan_tri_7(caplog) -> None:
    PT = Plan_tri(("1", "b", "2", "a", "c", "e"))
    with caplog.at_level(logging.DEBUG, logger="Plan_d_exp.src.equations"):
        PT.generate_product(
            {},
            {
                "1": [-1, 0, 1],
                "b": [-1, 0, 1],
                "2": [-1, 0, 1],
                "a": [-1, 0, 1],
                "c": [-1, 0, 1],
                "e": [-1, 0, 1],
            },
            "E",
        )
        assert len(PT.Equations_table) == 729
        assert len(PT.lbitlevels) == 729
        PT.search_stat(dict(((f"E_{i}", 1 + 0.1 * i) for i in range(1, 730))))
        for k, v in PT.res_index.items():
            for a in v:
                assert len(a.shape) == 1
                assert a.shape[0] == 243
        assert set(PT.ddd.keys()) == set(
            (
                (a, i, j)
                for a in ("1", "b", "2", "a", "c", "e")
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                if j > i
            )
        )
        for k, v in PT.ddd.items():
            assert len(v.shape) == 1
            assert v.shape[0] == 243
        assert set(PT.Courbes_ddd.keys()) == set(
            (
                (a, i, j)
                for a in ("1", "b", "2", "a", "c", "e")
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                if j > i
            )
        )


def test_Plan_tri_8(caplog) -> None:
    PT = Plan_tri(("1", "b", "2", "a", "c", "e"))
    with caplog.at_level(logging.DEBUG, logger="Plan_d_exp.src.equations"):
        PT.generate_product(
            {},
            {
                "1": [-1, 0, 1],
                "b": [-1, 0, 1],
                "2": [-1, 0, 1],
                "a": [-1, 0, 1],
                "c": [-1, 0, 1],
                "e": [-1, 0, 1],
            },
            "E",
        )
        assert len(PT.Equations_table) == 729
        assert len(PT.lbitlevels) == 729
        PT.search_stat(dict(((f"E_{i}", 1 + 1.1e-5 * i**2) for i in range(1, 244))))
        for k, v in PT.res_index.items():
            if k == "1":
                assert len(v[0].shape) == 1
                assert len(v[1].shape) == 1
                assert len(v[1].shape) == 1
                assert v[0].shape[0] == 243
                assert v[1].shape[0] == 0
                assert v[2].shape[0] == 0
                continue
            for a in v:
                assert len(a.shape) == 1
                assert a.shape[0] == 81
        assert set(PT.ddd.keys()) == set(
            (
                (a, i, j)
                for a in ("b", "2", "a", "c", "e")
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                if j > i
            )
        )
        for k, v in PT.ddd.items():
            if k != "1":
                assert len(v.shape) == 1
                assert v.shape[0] == 81
            else:
                assert len(v.shape) == 1
                assert v.shape[0] == 0
        assert set(PT.Courbes_ddd.keys()) == set(
            (
                (a, i, j)
                for a in ("b", "2", "a", "c", "e")
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                if j > i
            )
        )
        for name in ("b", "2", "a", "c", "e"):
            assert PT.Courbes_ddd[(name, -1, 0)].name == name
            assert PT.Courbes_ddd[(name, -1, 0)].bit1 == -1
            assert PT.Courbes_ddd[(name, -1, 0)].bit2 == 0
            assert PT.Courbes_ddd[(name, -1, 0)].sample_size == 81
            assert PT.Courbes_ddd[(name, -1, 1)].name == name
            assert PT.Courbes_ddd[(name, -1, 1)].bit1 == -1
            assert PT.Courbes_ddd[(name, -1, 1)].bit2 == 1
            assert PT.Courbes_ddd[(name, -1, 1)].sample_size == 81
            assert PT.Courbes_ddd[(name, 0, 1)].name == name
            assert PT.Courbes_ddd[(name, 0, 1)].bit1 == 0
            assert PT.Courbes_ddd[(name, 0, 1)].bit2 == 1
            assert PT.Courbes_ddd[(name, 0, 1)].sample_size == 81


def test_csv(tmp_path, caplog) -> None:
    PT = Plan_tri(("1", "b", "2", "a", "c", "e"))
    Pa = tmp_path / Path("nom_fic.csv")
    PT.Courbes = {
        "1": (
            None,
            Courbe1(
                name="1",
                value="ERROR",
                bit=0,
                max_val=1.8,
                min_val=1.0,
                pct_25=1.2,
                pct_50=1.4,
                pct_75=1.6,
                sample_size=9,
                mu_samp=1.4,
                var_samp=0.075333,
            ),
            None,
        ),
        "b": (
            None,
            Courbe1(
                name="b",
                value="ERROR",
                bit=0,
                max_val=1.8,
                min_val=1.0,
                pct_25=1.2,
                pct_50=1.4,
                pct_75=1.6,
                sample_size=9,
                mu_samp=1.4,
                var_samp=0.075222,
            ),
            None,
        ),
        "2": (
            Courbe1(
                name="2",
                value="ERROR",
                bit=-1,
                max_val=1.8,
                min_val=1.0,
                pct_25=1.2,
                pct_50=1.4,
                pct_75=1.6,
                sample_size=9,
                mu_samp=1.3999999,
                var_samp=0.075111,
            ),
            None,
            None,
        ),
        "a": (None, None, None),
        "c": (None, None, None),
    }
    conv = dict(
        (
            ((a, b), f"{a}::{b}")
            for a in ("1", "b", "2", "a", "c", "e")
            for b in (-1, 0, 1)
        )
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        PT.to_csv1(Pa, conv)
    cont = Pa.read_text()
    assert (
        cont
        == """name;value;bit;max_val;min_val;pct_25;pct_50;pct_75;sample_size;mu_samp;var_samp
1;1::-1;-1;-;-;-;-;-;-;-;-
1;1::0;0;1.800;1.000;1.200;1.400;1.600;9;1.400;0.075
1;1::1;1;-;-;-;-;-;-;-;-
b;b::-1;-1;-;-;-;-;-;-;-;-
b;b::0;0;1.800;1.000;1.200;1.400;1.600;9;1.400;0.075
b;b::1;1;-;-;-;-;-;-;-;-
2;2::-1;-1;1.800;1.000;1.200;1.400;1.600;9;1.400;0.075
2;2::0;0;-;-;-;-;-;-;-;-
2;2::1;1;-;-;-;-;-;-;-;-
a;a::-1;-1;-;-;-;-;-;-;-;-
a;a::0;0;-;-;-;-;-;-;-;-
a;a::1;1;-;-;-;-;-;-;-;-
c;c::-1;-1;-;-;-;-;-;-;-;-
c;c::0;0;-;-;-;-;-;-;-;-
c;c::1;1;-;-;-;-;-;-;-;-
e;e::-1;-1;-;-;-;-;-;-;-;-
e;e::0;0;-;-;-;-;-;-;-;-
e;e::1;1;-;-;-;-;-;-;-;-"""
    )
    assert caplog.record_tuples == [
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"écriture de 20 lignes dans {Pa}",
        ),
    ]


def test_csv2(tmp_path, caplog) -> None:
    PT = Plan_tri(("1", "b", "2", "a", "c", "e"))
    Pa = tmp_path / Path("nom_fic.csv")
    PT.Courbes_ddd = {
        ("1", -1, 0): Courbe2(
            name="1",
            value1="ERROR",
            value2="ERROR",
            bit1=-1,
            bit2=0,
            max_val=1.8,
            min_val=1.0,
            pct_25=1.2,
            pct_50=1.4,
            pct_75=1.6,
            sample_size=9,
            mu_samp=1.4,
            var_samp=0.075333,
        ),
        ("b", -1, 1): Courbe2(
            name="b",
            value1="ERROR",
            value2="ERROR",
            bit1=-1,
            bit2=1,
            max_val=1.8,
            min_val=1.0,
            pct_25=1.2,
            pct_50=1.4,
            pct_75=1.6,
            sample_size=9,
            mu_samp=1.4,
            var_samp=0.075222,
        ),
        ("2", 0, 1): Courbe2(
            name="2",
            value1="ERROR",
            value2="ERROR",
            bit1=0,
            bit2=1,
            max_val=1.8,
            min_val=1.0,
            pct_25=1.2,
            pct_50=1.4,
            pct_75=1.6,
            sample_size=9,
            mu_samp=1.3999999,
            var_samp=0.075111,
        ),
    }
    conv = dict(
        (
            ((a, b), f"{a}::{b}")
            for a in ("1", "b", "2", "a", "c", "e")
            for b in (-1, 0, 1)
        )
    )
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        PT.to_csv2(Pa, conv)
    cont = Pa.read_text()
    assert (
        cont
        == """name;value1;value2;bit1;bit2;max_val;min_val;pct_25;pct_50;pct_75;sample_size;mu_samp;var_samp
1;1::-1;1::0;-1;0;1800.00;1000.00;1200.00;1400.00;1600.00;9;1400.00;75.33
1;1::-1;1::1;-1;1;-;-;-;-;-;-;-;-
1;1::0;1::1;0;1;-;-;-;-;-;-;-;-
b;b::-1;b::0;-1;0;-;-;-;-;-;-;-;-
b;b::-1;b::1;-1;1;1800.00;1000.00;1200.00;1400.00;1600.00;9;1400.00;75.22
b;b::0;b::1;0;1;-;-;-;-;-;-;-;-
2;2::-1;2::0;-1;0;-;-;-;-;-;-;-;-
2;2::-1;2::1;-1;1;-;-;-;-;-;-;-;-
2;2::0;2::1;0;1;1800.00;1000.00;1200.00;1400.00;1600.00;9;1400.00;75.11
a;a::-1;a::0;-1;0;-;-;-;-;-;-;-;-
a;a::-1;a::1;-1;1;-;-;-;-;-;-;-;-
a;a::0;a::1;0;1;-;-;-;-;-;-;-;-
c;c::-1;c::0;-1;0;-;-;-;-;-;-;-;-
c;c::-1;c::1;-1;1;-;-;-;-;-;-;-;-
c;c::0;c::1;0;1;-;-;-;-;-;-;-;-
e;e::-1;e::0;-1;0;-;-;-;-;-;-;-;-
e;e::-1;e::1;-1;1;-;-;-;-;-;-;-;-
e;e::0;e::1;0;1;-;-;-;-;-;-;-;-"""
    )
    assert caplog.record_tuples == [
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"écriture de 20 lignes dans {Pa}",
        ),
    ]
