import logging
from itertools import product
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
    Equations_tri,
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


def test_write_minmax_plan(tmp_path, caplog):
    PT = Plan(("1", "b", "2", "a", "c"))
    PT.E.order = 2
    E2 = Equations(("1", "b", "2", "a"), 1)
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        a = np.linspace(1.0, 1.28, 15)
        conv = set(product(("1", "b", "2"), (-1, 0, 1)))
        conv.add(("a", -1))
        conv.add(("a", 0))
        fmin = tmp_path / Path("Plan_min.csv")
        fmax = tmp_path / Path("Plan_max.csv")
        assert PT.write_minmax_plan(a, conv, E2, 2, 5, fmin, fmax, "E") == 54
    assert caplog.record_tuples == [
        ("Plan_d_exp.src.equations", logging.INFO, "le nombre d'équation est: 54"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "création du meshgrid des valeurs des variables",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "applatissement du meshgrid des valeurs des variables",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            "calcul de la matrice des valeur de la matrice des inconnues",
        ),
        ("Plan_d_exp.src.equations", logging.INFO, "recherche des extrèmes"),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"début d'écriture du fichier du plan max {fmax}",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            """Name;1;b;2;a;c
E_54;1;1;1;0;0
E_1;-1;-1;-1;-1;0
E_53;1;1;1;-1;0
E_48;0;1;1;0;0
E_36;1;0;1;0;0""",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"fin écriture du fichier {fmax}",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            """nom;val
E_54;11.100000000000001
E_1;8.700000000000001
E_53;7.660000000000001
E_48;6.720000000000001
E_36;6.5600000000000005""",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"début d'écriture du fichier du plan min {fmin}",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            """Name;1;b;2;a;c
E_28;0;0;0;0;0
E_22;-1;0;0;0;0
E_10;0;-1;0;0;0
E_26;0;0;-1;0;0
E_27;0;0;0;-1;0""",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            f"fin écriture du fichier {fmin}",
        ),
        (
            "Plan_d_exp.src.equations",
            logging.INFO,
            """nom;val
E_28;1.0
E_22;1.08
E_10;1.1399999999999997
E_26;1.1800000000000002
E_27;1.2000000000000002""",
        ),
    ]


def test_write_minmax_plan_1(tmp_path, caplog):
    PT = Plan(("1", "b", "2", "a", "c"))
    E2 = Equations(("1", "b", "2", "d"), 1)
    with caplog.at_level(logging.INFO, logger="Plan_d_exp.src.equations"):
        a = np.linspace(1.0, 1.28, 15)
        conv = set(product(("1", "b", "2"), (-1, 0, 1)))
        conv.add(("a", -1))
        conv.add(("a", 0))
        fmin = tmp_path / Path("Plan_min.csv")
        fmax = tmp_path / Path("Plan_max.csv")
        assert PT.write_minmax_plan(a, conv, E2, 1, 5, fmin, fmax, "E") == 0
    assert caplog.record_tuples == [
        (
            "Plan_d_exp.src.equations",
            logging.ERROR,
            "les variables: 'd' ne font pas partie du plan",
        ),
    ]
