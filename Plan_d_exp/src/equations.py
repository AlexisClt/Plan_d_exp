# main class
import logging
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import astuple, dataclass, field, fields
from decimal import ROUND_DOWN
from decimal import Decimal as D
from functools import cached_property, reduce
from itertools import combinations
from itertools import combinations_with_replacement as cwr
from itertools import cycle, groupby, product, repeat
from math import fabs, floor, sqrt
from pathlib import Path
from pickle import dumps, loads
from string import ascii_uppercase
from typing import (
    Any,
    DefaultDict,
    Iterable,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import numpy.typing as npt
from numpy.linalg import lstsq
from numpy.testing import assert_almost_equal
from scipy.special import gamma, ndtr, ndtri, stdtr, stdtrit
from scipy.stats import f_oneway, normaltest, probplot

logger = logging.getLogger(__name__)


def chain2(i1: Iterable[str], i2: Iterable[str]) -> Iterable[str]:
    for ele in i1:
        yield ele
    for ele in i2:
        yield ele


def_scr = {
    4: ("0+--", "-0-+", "--0-", "-++0"),
    5: ("0++--", "+0--+", "+-0+-", "+-+0+", "++++0"),
    6: ("0+----", "+0-++-", "--0+--", "-++0+-", "+-+-0-", "++++-0"),
    7: (
        "0+-+-+-",
        "-0+-++-",
        "+-0++++",
        "+--0+--",
        "--++0--",
        "-+-++0+",
        "+++++-0",
    ),
    8: (
        "0-++-+++",
        "-0-++++-",
        "--0++--+",
        "+-+0++--",
        "--+-0-+-",
        "+---+0++",
        "-++-++0+",
        "+++++-+0",
    ),
    9: (
        "0++++++++",
        "+0+-+--+-",
        "-+0-+-+--",
        "--+0+---+",
        "+-+-0++--",
        "----+0+++",
        "++--++0-+",
        "---+++-0-",
        "-++--+-+0",
    ),
    10: (
        "0++-++++-+",
        "+0-++-++--",
        "-+0---+---",
        "-++0+--++-",
        "----0++++-",
        "-+-++0+-++",
        "++----0+++",
        "++++-++0+-",
        "++--++--0-",
        "+-+-+-+-+0",
    ),
    11: (
        "0-+-----+-+",
        "-0--+----++",
        "--0++++---+",
        "+--+0+-++++",
        "--++-0-+-+-",
        "---++-0++--",
        "-+++--+0+++",
        "-+---+-+0-+",
        "+-----++-0+",
        "++-+------0",
    ),
    12: (
        "0--+-+-+++-+",
        "-0+++++++---",
        "++0-++-++-++",
        "+--0+-+-+--+",
        "++++0-++++++",
        "+-+-+0++-+-+",
        "++++-+0----+",
        "--+++--0--++",
        "+-++++--0++-",
        "++-++--+-0--",
        "-+-++++-+-0-",
        "+--+-+++--+0",
    ),
}


def vers_int(chaine: str) -> Sequence[int]:
    """
    converti les caractères +, - 0 en entiers +1, -1, 0
    """
    ret: Sequence[int] = []
    for c in chaine:
        if c == "0":
            ret.append(0)
        elif c == "+":
            ret.append(1)
        elif c == "-":
            ret.append(-1)
        else:
            raise ValueError(f"le caractere '{c}' n'est pas dans '-+0'")
    return ret


def genere_def_scr_int(
    desr: Mapping[int, Sequence[str]]
) -> Mapping[int, Sequence[Sequence[int]]]:
    """
    converti un dictionnaire indéxé par un nombre entier ayant pour valeur
    associée un tuple de chaine de caractères de même longueur contenant
    que des 0, des + et des -. Les + sont converti en +1 les - en -1, les 0
    en 0.
    """
    ret: Mapping[int, Sequence[Sequence[int]]] = {}
    if len(desr) == 0:
        return ret
    for order, lst_scr in desr.items():
        cont: List[List[int]] = []
        if len(lst_scr) == 0:
            return ret
        taille = len(lst_scr[0])
        contp = [vers_int(i) for i in lst_scr]
        contm = [[-j for j in i] for i in contp]
        for tp, tm in zip(contp, contm):
            cont.append(tp)
            cont.append(tm)
        cont.append(vers_int("0" * taille))
        ret[order] = cont
    return ret


def_scr_int = {
    4: [
        [0, 1, -1, -1],
        [0, -1, 1, 1],
        [-1, 0, -1, 1],
        [1, 0, 1, -1],
        [-1, -1, 0, -1],
        [1, 1, 0, 1],
        [-1, 1, 1, 0],
        [1, -1, -1, 0],
        [0, 0, 0, 0],
    ],
    5: [
        [0, 1, 1, -1, -1],
        [0, -1, -1, 1, 1],
        [1, 0, -1, -1, 1],
        [-1, 0, 1, 1, -1],
        [1, -1, 0, 1, -1],
        [-1, 1, 0, -1, 1],
        [1, -1, 1, 0, 1],
        [-1, 1, -1, 0, -1],
        [1, 1, 1, 1, 0],
        [-1, -1, -1, -1, 0],
        [0, 0, 0, 0, 0],
    ],
    6: [
        [0, 1, -1, -1, -1, -1],
        [0, -1, 1, 1, 1, 1],
        [1, 0, -1, 1, 1, -1],
        [-1, 0, 1, -1, -1, 1],
        [-1, -1, 0, 1, -1, -1],
        [1, 1, 0, -1, 1, 1],
        [-1, 1, 1, 0, 1, -1],
        [1, -1, -1, 0, -1, 1],
        [1, -1, 1, -1, 0, -1],
        [-1, 1, -1, 1, 0, 1],
        [1, 1, 1, 1, -1, 0],
        [-1, -1, -1, -1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    7: [
        [0, 1, -1, 1, -1, 1, -1],
        [0, -1, 1, -1, 1, -1, 1],
        [-1, 0, 1, -1, 1, 1, -1],
        [1, 0, -1, 1, -1, -1, 1],
        [1, -1, 0, 1, 1, 1, 1],
        [-1, 1, 0, -1, -1, -1, -1],
        [1, -1, -1, 0, 1, -1, -1],
        [-1, 1, 1, 0, -1, 1, 1],
        [-1, -1, 1, 1, 0, -1, -1],
        [1, 1, -1, -1, 0, 1, 1],
        [-1, 1, -1, 1, 1, 0, 1],
        [1, -1, 1, -1, -1, 0, -1],
        [1, 1, 1, 1, 1, -1, 0],
        [-1, -1, -1, -1, -1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    8: [
        [0, -1, 1, 1, -1, 1, 1, 1],
        [0, 1, -1, -1, 1, -1, -1, -1],
        [-1, 0, -1, 1, 1, 1, 1, -1],
        [1, 0, 1, -1, -1, -1, -1, 1],
        [-1, -1, 0, 1, 1, -1, -1, 1],
        [1, 1, 0, -1, -1, 1, 1, -1],
        [1, -1, 1, 0, 1, 1, -1, -1],
        [-1, 1, -1, 0, -1, -1, 1, 1],
        [-1, -1, 1, -1, 0, -1, 1, -1],
        [1, 1, -1, 1, 0, 1, -1, 1],
        [1, -1, -1, -1, 1, 0, 1, 1],
        [-1, 1, 1, 1, -1, 0, -1, -1],
        [-1, 1, 1, -1, 1, 1, 0, 1],
        [1, -1, -1, 1, -1, -1, 0, -1],
        [1, 1, 1, 1, 1, -1, 1, 0],
        [-1, -1, -1, -1, -1, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    9: [
        [0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 0, 1, -1, 1, -1, -1, 1, -1],
        [-1, 0, -1, 1, -1, 1, 1, -1, 1],
        [-1, 1, 0, -1, 1, -1, 1, -1, -1],
        [1, -1, 0, 1, -1, 1, -1, 1, 1],
        [-1, -1, 1, 0, 1, -1, -1, -1, 1],
        [1, 1, -1, 0, -1, 1, 1, 1, -1],
        [1, -1, 1, -1, 0, 1, 1, -1, -1],
        [-1, 1, -1, 1, 0, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, -1, 0, -1, -1, -1],
        [1, 1, -1, -1, 1, 1, 0, -1, 1],
        [-1, -1, 1, 1, -1, -1, 0, 1, -1],
        [-1, -1, -1, 1, 1, 1, -1, 0, -1],
        [1, 1, 1, -1, -1, -1, 1, 0, 1],
        [-1, 1, 1, -1, -1, 1, -1, 1, 0],
        [1, -1, -1, 1, 1, -1, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    10: [
        [0, 1, 1, -1, 1, 1, 1, 1, -1, 1],
        [0, -1, -1, 1, -1, -1, -1, -1, 1, -1],
        [1, 0, -1, 1, 1, -1, 1, 1, -1, -1],
        [-1, 0, 1, -1, -1, 1, -1, -1, 1, 1],
        [-1, 1, 0, -1, -1, -1, 1, -1, -1, -1],
        [1, -1, 0, 1, 1, 1, -1, 1, 1, 1],
        [-1, 1, 1, 0, 1, -1, -1, 1, 1, -1],
        [1, -1, -1, 0, -1, 1, 1, -1, -1, 1],
        [-1, -1, -1, -1, 0, 1, 1, 1, 1, -1],
        [1, 1, 1, 1, 0, -1, -1, -1, -1, 1],
        [-1, 1, -1, 1, 1, 0, 1, -1, 1, 1],
        [1, -1, 1, -1, -1, 0, -1, 1, -1, -1],
        [1, 1, -1, -1, -1, -1, 0, 1, 1, 1],
        [-1, -1, 1, 1, 1, 1, 0, -1, -1, -1],
        [1, 1, 1, 1, -1, 1, 1, 0, 1, -1],
        [-1, -1, -1, -1, 1, -1, -1, 0, -1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1, 0, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1, 0, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, 0],
        [-1, 1, -1, 1, -1, 1, -1, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    11: [
        [0, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1],
        [0, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1],
        [-1, 0, -1, -1, 1, -1, -1, -1, -1, 1, 1],
        [1, 0, 1, 1, -1, 1, 1, 1, 1, -1, -1],
        [-1, -1, 0, 1, 1, 1, 1, -1, -1, -1, 1],
        [1, 1, 0, -1, -1, -1, -1, 1, 1, 1, -1],
        [1, -1, -1, 1, 0, 1, -1, 1, 1, 1, 1],
        [-1, 1, 1, -1, 0, -1, 1, -1, -1, -1, -1],
        [-1, -1, 1, 1, -1, 0, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 0, 1, -1, 1, -1, 1],
        [-1, -1, -1, 1, 1, -1, 0, 1, 1, -1, -1],
        [1, 1, 1, -1, -1, 1, 0, -1, -1, 1, 1],
        [-1, 1, 1, 1, -1, -1, 1, 0, 1, 1, 1],
        [1, -1, -1, -1, 1, 1, -1, 0, -1, -1, -1],
        [-1, 1, -1, -1, -1, 1, -1, 1, 0, -1, 1],
        [1, -1, 1, 1, 1, -1, 1, -1, 0, 1, -1],
        [1, -1, -1, -1, -1, -1, 1, 1, -1, 0, 1],
        [-1, 1, 1, 1, 1, 1, -1, -1, 1, 0, -1],
        [1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 0],
        [-1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    12: [
        [0, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1],
        [0, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1],
        [-1, 0, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1],
        [1, 0, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 0, -1, 1, 1, -1, 1, 1, -1, 1, 1],
        [-1, -1, 0, 1, -1, -1, 1, -1, -1, 1, -1, -1],
        [1, -1, -1, 0, 1, -1, 1, -1, 1, -1, -1, 1],
        [-1, 1, 1, 0, -1, 1, -1, 1, -1, 1, 1, -1],
        [1, 1, 1, 1, 0, -1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, -1, -1],
        [1, -1, 1, -1, 1, 0, 1, 1, -1, 1, -1, 1],
        [-1, 1, -1, 1, -1, 0, -1, -1, 1, -1, 1, -1],
        [1, 1, 1, 1, -1, 1, 0, -1, -1, -1, -1, 1],
        [-1, -1, -1, -1, 1, -1, 0, 1, 1, 1, 1, -1],
        [-1, -1, 1, 1, 1, -1, -1, 0, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1, 0, 1, 1, -1, -1],
        [1, -1, 1, 1, 1, 1, -1, -1, 0, 1, 1, -1],
        [-1, 1, -1, -1, -1, -1, 1, 1, 0, -1, -1, 1],
        [1, 1, -1, 1, 1, -1, -1, 1, -1, 0, -1, -1],
        [-1, -1, 1, -1, -1, 1, 1, -1, 1, 0, 1, 1],
        [-1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 0, -1],
        [1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 0, 1],
        [1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 0],
        [-1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
}


def limi_1(
    nb: int, order: int
) -> Mapping[int, MutableSequence[Sequence[Sequence[int]]]]:
    res: Mapping[int, MutableSequence[Sequence[Sequence[int]]]] = {}
    res = dict(
        (
            (
                i,
                [
                    tuple(Counter(b).items())
                    for b in (list(a) for a in cwr(list(range(nb)), i))
                    if len(b) != 0
                ],
            )
            for i in range(1, order + 1)
        )
    )
    return res


def limi_3(
    nb: int,
    ord: int,
) -> Mapping[int, MutableSequence[Sequence[Sequence[int]]]]:
    res: Mapping[int, MutableSequence[Sequence[Sequence[int]]]] = dict(
        ((i, []) for i in range(1, 2 * ord + 1))
    )
    res1: Mapping[int, MutableSequence[Sequence[Sequence[int]]]] = {}
    for j in range(1, ord + 1):
        for tpl in combinations(list(range(nb)), j):
            for pwr in product((1, 2), repeat=j):
                res[sum(pwr)].append(tuple(zip(tpl, pwr)))
    for k, v in res.items():
        res1[k] = sorted(v)
    return res1


@contextmanager
def save_figure(plt: Any, nom_fic: Path) -> Any:
    plt.close("all")
    fig, ax = plt.subplots(layout="constrained")
    try:
        yield (fig, ax)
        logger.info(f"écriture de {nom_fic}")
        fig.savefig(nom_fic.as_posix())
        plt.show()
    except:
        raise


def visu_distri(
    plt: Any,
    sns: Any,
    data: npt.NDArray[np.float64],
    bins: int,
    nom_fic: Path,
    nom: str,
    col: str = "skyblue",
) -> None:
    """
    permet de visualiser une distribution avec seaborn
    et sauve dans nom_fic.png
    """
    plt.close("all")
    hist_plot = sns.histplot(data, kde=True, bins=bins, color=col, edgecolor="black")
    plt.title(f"{nom}\nHistogramme + courbe de densité")
    plt.xlabel("Valeur")
    plt.ylabel("Fréquence / Densité")
    plt.grid(True)
    fig = hist_plot.get_figure()
    logger.info(f"écriture de {nom_fic}")
    fig.savefig(nom_fic.as_posix())
    plt.show()


def visu2_distri(
    plt: Any,
    data: npt.NDArray[np.float64],
    nom_fic: Path,
    nom: str,
) -> None:
    """
    permet de visualiser une distribution avec seaborn
    et sauve dans nom_fic.png
    """
    with save_figure(plt, nom_fic) as (fig, ax):
        probplot(data, plot=ax)
        ax.set_title(f"{nom}\nProbality Plot")
        ax.grid(True)


class report_visu_distri:
    """
    class pour générer un rapport avec beaucoup d'images
    """

    def __init__(self) -> None:
        self.lst_name: Sequence[str] = []
        self.lst_distri: Sequence[str] = []
        self.lst_probplot: Sequence[str] = []

    def add_images(self, name: str, fic1: str, fic2: str) -> None:
        self.lst_name.append(name)
        self.lst_distri.append(fic1)
        self.lst_probplot.append(fic2)


def monome(datas: List[Tuple[int, int]]) -> str:
    """
    monome for excel
    """
    ret: List[str] = []
    for ind in datas:
        ret.append(power_case(*ind))
    return "*".join(ret)


def power_case(i: int, j: int) -> str:
    """
    fonction pour renvoyer ascii_upercase[i+2]^j
    """
    if j > 1:
        return ascii_uppercase[i + 1] + "2^" + f"{j}"
    else:
        return ascii_uppercase[i + 1] + "2"


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
        indexes2 = self.check_indexes(indexes, order)
        self.indexes = indexes2
        self.set_indexes = set(indexes2)
        self.order = order
        self.ind_indexes = dict((i[1], i[0]) for i in enumerate(indexes2))
        self.latex_indexes = dict(((i, i) for i in indexes2))
        self.fic_names = dict(((i, i.replace(":", "_")) for i in indexes2))
        self.dct_cwr_ind = limi_1(len(indexes2), order)
        self.powers = sorted(self.dct_cwr_ind.keys())
        self.power = "**"

    def check_indexes(self, indexes: Sequence[str], order: int) -> Sequence[str]:
        """
        check for datas feed to __init__
        return indexes2 if no errors
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
        indexes2 = tuple((a.strip() for a in indexes))
        msg1 = ", ".join(
            [f"{i}" for i, a in enumerate(indexes2, start=1) if len(a) == 0]
        )
        if len(msg1) != 0:
            raise ValueError(f"indexes number {msg1} has length equal zero")
        msg0 = "(" + ", ".join([f'"{a}"' for a in indexes2]) + ")"
        msg = "\n".join(
            [
                f'Index "{a[1]}" appears {a[0]} times in {msg0}'
                for a in sorted(
                    [
                        (b[1], b[0])
                        for b in list(Counter(indexes2).items())
                        if b[1] != 1
                    ],
                    reverse=True,
                )
            ]
        )

        if len(msg) != 0:
            raise ValueError(msg)
        return indexes2

    @cached_property
    def col_names(self) -> Sequence[str]:
        col_names = [
            "mean",
        ]

        for i in self.powers:
            col_names.extend(
                [
                    reduce(
                        lambda x, y: x + "." + y,
                        (
                            (
                                "{0}{2}{1}".format(self.indexes[b[0]], b[1], self.power)
                                if (b[1] > 1)
                                else "{0}".format(self.indexes[b[0]])
                            )
                            for b in a
                        ),
                    )
                    for a in self.dct_cwr_ind[i]
                    if len(a) != 0
                ]
            )
        return col_names

    #    def to_excel_formula(self, data: np.ndarray[Tuple[int, int], np.dtype[Any]]) -> str:
    def to_excel_formula(self, data: np.ndarray) -> str:
        """
        Generates formula associated with the equation
        First variable starts at second column
        """
        res = f"={data[0,0]}"
        res += "".join(
            (
                f"{a:+}*{b}"
                for a, b in zip(
                    data[0, 1:],
                    (monome(f) for e, d in self.dct_cwr_ind.items() for f in d),
                )
            )
        )
        return res

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

        for i in self.powers:
            ret.extend(
                [
                    reduce(
                        lambda x, y: x * y,
                        (datas[self.indexes[b[0]]] ** b[1] for b in a),
                        1.0,
                    )
                    for a in self.dct_cwr_ind[i]
                    if len(a) != 0
                ]
            )

        return ret

    def generate_array(self, data: np.ndarray) -> np.ndarray:
        """
        Generate the coefficients of several equations
        raise error if number row of data is not len(self.indexes)
        """
        if data.ndim != 2:
            raise ValueError(
                f"""le nombre de dimensions de l'array passée à generate_array n'est
pas accepté: {data.ndim} est strictement supérieur à 2"""
            )
        if data.shape[1] != len(self.indexes):
            raise ValueError(
                f"""la taille de l'array passée à generate_array n'est
pas acceptée: {data.shape}. La deuxième dimension attendue est {len(self.indexes)}"""
            )
        one_ret = np.ones(data.shape[0])
        ret = [
            one_ret,
        ]
        for i in self.powers:
            ret.extend(
                [
                    reduce(
                        lambda x, y: x * y,
                        (data[:, b[0]] ** b[1] for b in a),
                        one_ret,
                    )
                    for a in self.dct_cwr_ind[i]
                    if len(a) != 0
                ]
            )
        if data.shape[0] == 1:
            return np.ravel(np.column_stack(ret))
        else:
            return np.reshape(np.column_stack(ret), (data.shape[0], -1))

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

    def generate_def_scr(
        self, data1: Mapping[str, float], data2: Sequence[str]
    ) -> MutableSequence[Mapping[str, float]]:
        """
        genere une liste de dictionnaire pour un plan d'expérience de type
        definitive screening de taille len(data2) compris entre 4 et 12 inclus
        data1 est constant et data2 va varié tel un plan d'expérience définitive screening
        """
        if len(data2) < 4:
            raise ValueError(f"la longueur de {data2} est strictement inférieure à 4")
        if len(data2) > 12:
            raise ValueError(f"la longueur de {data2} est strictement supérieure à 12")
        self.check_datas(data1, dict([(k, 0) for k in data2]))
        res: MutableSequence[Mapping[str, float]] = []
        constant = list(data1.items())
        for ele in def_scr_int[len(data2)]:
            res.append(dict(constant + list(zip(data2, ele))))
        return res


class Equations_tri(Equations):
    """
    class de Equations qui se spécialise dans les plan à 3 niveaux
    """

    def __init__(self, indexes: Sequence[str], order: int) -> None:
        """
        Init of class
        indexes are the indexes of the X_i s
        order is the maximum order of the equation
        """
        indexes2 = self.check_indexes(indexes, order)
        self.indexes = indexes2
        self.set_indexes = set(indexes2)
        self.order = order
        self.ind_indexes = dict((i[1], i[0]) for i in enumerate(indexes2))
        self.latex_indexes = dict(((i, i) for i in indexes2))
        self.fic_names = dict(((i, i.replace(":", "_")) for i in indexes2))
        self.dct_cwr_ind = limi_3(len(indexes2), order)
        self.powers = sorted(self.dct_cwr_ind.keys())
        self.power = "**"


@dataclass(frozen=True)
class Prec_Bougie:
    """
    valeur pour la précision de to_csv
    """

    coef_inv: float
    prec: D


PB0 = Prec_Bougie(1, D("0.001"))
PB2 = Prec_Bougie(1000, D("0.01"))


@dataclass(frozen=True)
class Bougie:
    """
    données pour tracer les bougies
    """

    name: str
    max_val: float
    min_val: float
    pct_25: float
    pct_50: float
    pct_75: float

    def to_csv1(self, PB: Prec_Bougie) -> Sequence[str]:
        """
        fonction générique pour écrire une ligne de fichier excel
        """
        return [f"{D(a*PB.coef_inv).quantize(PB.prec)}" for a in astuple(self)[1:6]]

    def to_csv_0(self) -> Sequence[str]:
        return [f"{D(a).quantize(D('0.001'))}" for a in astuple(self)[1:6]]

    def to_csv_02(self) -> Sequence[str]:
        return [f"{D(1000*a).quantize(D('0.01'))}" for a in astuple(self)[1:6]]

    def plot_single_barh(self, ax: Any, pos: float, colo: str, labe: str) -> None:
        """
        affiche dans ax une bar horyzontal à la position pos
        """
        ax.barh(
            pos,
            width=(self.pct_75 - self.pct_50),
            left=self.pct_50,
            facecolor=colo,
            edgecolor="k",
            xerr=[
                [
                    0.0,
                ],
                [
                    (self.max_val - self.pct_75),
                ],
            ],
            label=labe,
        )
        ax.barh(
            pos,
            width=(self.pct_50 - self.pct_25),
            left=self.pct_25,
            facecolor=colo,
            edgecolor="k",
        )
        ax.barh(
            pos,
            width=0.0,
            left=self.pct_25,
            facecolor=colo,
            edgecolor="k",
            xerr=[
                [
                    (self.pct_25 - self.min_val),
                ],
                [
                    0.0,
                ],
            ],
        )


@dataclass(frozen=True)
class Prec_Courbe1:
    """
    valeur pour la précision de to_csv
    """

    PB: Prec_Bougie
    mu_coef_inv: float
    mu_prec: D
    var_coef_inv: float
    var_prec: D


PC1_0 = Prec_Courbe1(PB0, 1.0, D("0.001"), 1000.0, D("1.0"))
PC1_1 = Prec_Courbe1(PB0, 1.0, D("0.001"), 1.0, D("0.001"))


@dataclass(frozen=True)
class Courbe1(Bougie):
    """
    données pour tracer les bougies
    """

    value: str
    bit: int
    sample_size: int
    mu_samp: float
    var_samp: float

    def to_csv(
        self, conv: Mapping[Tuple[str, int], str], PC1: Prec_Courbe1 = PC1_0
    ) -> str:
        ret: List[str] = []
        ret.append(conv.get((self.name, self.bit), "ERR"))
        ret.append(f"{self.bit}")
        ret.extend(self.to_csv1(PC1.PB))
        ret.append(f"{self.sample_size}")
        ret.extend(
            (
                f"{D(PC1.mu_coef_inv*a).quantize(PC1.mu_prec)}"
                for a in astuple(self)[9:10]
            )
        )
        ret.extend(
            (
                f"{D(PC1.var_coef_inv*a).quantize(PC1.var_prec)}"
                for a in astuple(self)[10:]
            )
        )
        return ";".join(ret)


@dataclass(frozen=True)
class Prec_Proba1:
    """
    valeur pour la précision de to_csv
    """

    b_coef_inv: float
    b_prec: D


PP1_0 = Prec_Proba1(1.0, D("0.001"))


@dataclass
class Proba1:
    """
    données pour des intervales de tolérance entre -1 et 1
    pour des 2 jeux de résultats indépendants
    """

    name: str
    valuem: str
    valuep: str
    t_0: float
    v: float
    p_value: float
    normaltest_pvaluem: float
    normaltest_pvaluep: float
    alphas: Sequence[float] = field(default_factory=list)
    stdinvt: Sequence[float] = field(default_factory=list)
    b_inf: Sequence[float] = field(default_factory=list)
    b_sup: Sequence[float] = field(default_factory=list)

    def to_csv1(self, conv: Mapping[Tuple[str, int], str]) -> str:
        ret: List[str] = []
        ret.append(conv.get((self.name, -1), self.valuem))
        ret.append(conv.get((self.name, 1), self.valuep))
        ret.append(f"{D(self.t_0).quantize(D('0.001'))}")
        ret.append(f"{self.v}")
        ret.append(f"{D(self.p_value).quantize(D('0.1'))}%")
        ret.append(f"{D(self.normaltest_pvaluem).quantize(D('0.1'))}%")
        ret.append(f"{D(self.normaltest_pvaluep).quantize(D('0.1'))}%")
        return ";".join(ret)

    def to_csv2(
        self, conv: Mapping[Tuple[str, int], str], PP1: Prec_Proba1 = PP1_0
    ) -> str:
        ret: List[str] = []
        ret.append(f"{self.v}")
        for st in zip(self.stdinvt, self.b_inf, self.b_sup):
            ret.extend((f"{D(PP1.b_coef_inv*a).quantize(PP1.b_prec)}" for a in st))
        return ";".join(ret)


@dataclass(frozen=True)
class Prec_Courbe2:
    """
    valeur pour la précision de to_csv
    """

    PB: Prec_Bougie
    mu_coef_inv: float
    mu_prec: D
    var_coef_inv: float
    var_prec: D


PC2_0 = Prec_Courbe2(PB2, 1000.0, D("0.01"), 100000.0, D("1.0"))
PC2_1 = Prec_Courbe2(PB2, 1000.0, D("0.01"), 1000.0, D("1.00"))


@dataclass(frozen=True)
class Courbe2(Bougie):
    """
    données pour tracer les bougies des comparaisons doubles
    """

    value1: str
    value2: str
    bit1: int
    bit2: int
    sample_size: int
    mu_samp: float
    var_samp: float

    def to_csv(
        self, conv: Mapping[Tuple[str, int], str], PC2: Prec_Courbe2 = PC2_0
    ) -> str:
        ret: List[str] = []
        ret.append(conv.get((self.name, self.bit1), "ERR"))
        ret.append(conv.get((self.name, self.bit2), "ERR"))
        ret.append(f"{self.bit1}")
        ret.append(f"{self.bit2}")
        ret.extend(self.to_csv1(PC2.PB))
        ret.append(f"{self.sample_size}")
        ret.extend(
            (
                f"{D(PC2.mu_coef_inv*a).quantize(PC2.mu_prec)}"
                for a in astuple(self)[11:12]
            )
        )
        ret.extend(
            (
                f"{D(PC2.var_coef_inv*a).quantize(PC2.var_prec)}"
                for a in astuple(self)[12:]
            )
        )
        return ";".join(ret)


@dataclass(frozen=True)
class Prec_Proba2:
    """
    valeur pour la précision de to_csv
    """

    b_coef_inv: float
    b_prec: D


PP2_0 = Prec_Proba2(1000.0, D("0.1"))


@dataclass
class Proba2:
    """
    données pour des intervales de tolérance entre -1 et 1
    pour des 2 jeux de résultats indépendants
    """

    name: str
    valuem: str
    valuep: str
    t_0: float
    n: float
    p_value: float
    normaltest_pvalue: float
    alphas: Sequence[float] = field(default_factory=list)
    stdinvt: Sequence[float] = field(default_factory=list)
    b_inf: Sequence[float] = field(default_factory=list)
    b_sup: Sequence[float] = field(default_factory=list)

    def to_csv1(self, conv: Mapping[Tuple[str, int], str]) -> str:
        ret: List[str] = []
        ret.append(conv.get((self.name, -1), self.valuem))
        ret.append(conv.get((self.name, 1), self.valuep))
        ret.append(f"{D(self.t_0).quantize(D('0.001'))}")
        ret.append(f"{self.n}")
        ret.append(f"{D(self.p_value).quantize(D('0.1'))}%")
        ret.append(f"{D(self.normaltest_pvalue).quantize(D('0.1'))}%")
        return ";".join(ret)

    def to_csv2(
        self, conv: Mapping[Tuple[str, int], str], PP2: Prec_Proba2 = PP2_0
    ) -> str:
        ret: List[str] = []
        ret.append(f"{self.n}")
        for st in zip(self.stdinvt, self.b_inf, self.b_sup):
            ret.extend((f"{D(PP2.b_coef_inv*a).quantize(PP2.b_prec)}" for a in st))
        return ";".join(ret)


class Plan:
    """
    Manage and solve several equations
    """

    def __init__(self, indexes: Sequence[str] = ("a", "b")) -> None:
        self.Equations_table: MutableSequence[Mapping[str, float]] = []
        self.Equations_name: MutableSequence[str] = []
        self.set_name: MutableSet[str] = set([])
        self.E = Equations(indexes, 1)

    def clear(self) -> None:
        """
        clean evrything
        """
        ind = loads(dumps(self.E.indexes))
        self.__init__(ind)

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
                    + [str(b.get(c, "ERROR")) for c in self.E.indexes]
                )
                for a, b in zip(self.Equations_name, self.Equations_table)
            )
        )
        return ret

    def add(self, datas: Mapping[str, float], name: str) -> int:
        """Add an experiment in Plan"""

        if name in self.set_name:
            raise ValueError(f"une expérience avec le nom {name} existe déjà")
        self.E.generate_line(datas)
        self.Equations_table.append(datas)
        self.Equations_name.append(name)
        self.set_name.add(name)

        return len(self.Equations_table)

    def from_csv(self, cont: bytes, sep: str = ";") -> bool:
        """
        lit cont qui doit être un bytes
        met à jour self.E self.Equations_name et self.Equation_table
        """
        ret = True
        bsep = sep.encode()
        if len(cont) == 0:
            logger.error("fichier csv plan vide")
            return False
        lines = cont.split(b"\n")
        if not bsep in lines[0]:
            logger.error(
                f"la première ligne du fichier ne contient pas de nom de variables"
            )
            return False
        desi_col = tuple((a.strip().decode() for a in lines[0].split(bsep)[1:]))
        while len(desi_col[-1]) == 0:
            desi_col = desi_col[:-1]
        if len(desi_col) == 0:
            logger.error(f"fichier csv contient aucun nom de variable")
            return False
        try:
            self.E = Equations(desi_col, 1)
        except ValueError as e:
            logger.error(
                f"""le fichier csv contient des noms de variables vides:
{e}"""
            )
            return False
        for i, l in enumerate(lines[1:], start=2):
            l1 = l.strip()
            if len(l1) == 0:
                logger.warning(f"la ligne {i} est vide")
                continue
            col = [a.strip().decode() for a in l1.split(bsep)]
            while len(col[-1]) == 0:
                col.pop(-1)
            if len(col[1:]) != len(desi_col):
                logger.error(
                    f"""la ligne {i}:
'{l.decode()}'
ne contient pas suffisamment de colonnes: {len(col)-1} au lieu de {len(desi_col)}"""
                )
                ret = False
                continue
            colf: List[int] = []
            for j, val in enumerate(col[1:], start=2):
                try:
                    colf.append(float(val))
                except ValueError as e:
                    logger.error(
                        f"à la ligne {i} colonne {j}: impossible de convertir '{val}'"
                    )
                    ret = False
            if len(colf) == len(desi_col):
                self.add(dict(zip(desi_col, colf)), col[0])
        logger.info(f"lecture de {len(self.Equations_table)} expériences")
        return ret

    def read_results_from_csv(self, fic: Path, col_no: int) -> Mapping[str, float]:
        """
        lecture de la 1ère colonne pour le nom de l'expérience
        lecture de la col_no ième colonne pour les résultats
        la 1ère ligne est exclue de la lecture
        """
        ret: Mapping[str, float] = {}
        if col_no < 2:
            logger.error(f"col_no = {col_no} or il doit être suppérieur ou égal à 2")
            return ret
        cont: List[bytes] = fic.read_bytes().splitlines()
        if len(cont) < 2:
            logger.info(f"le fichier {fic} ne contient pas plus d'une ligne")
            return ret
        no_lignes_vides: List[int] = []
        lignes_erreur: List[Tuple[int, str]] = []
        lignes_erreur_nom: List[Tuple[int, str]] = []
        lignes_erreur_conversion: List[Tuple[int, str]] = []
        for i, l in enumerate(cont[1:], start=2):
            l0 = l.strip()
            ll = l0.split(b";")
            if len(l0) == 0:
                no_lignes_vides.append(i)
            elif len(ll) < col_no:
                lignes_erreur.append((i, l0.decode()))
            else:
                nom = ll[0].strip().decode()
                if not nom in self.set_name:
                    lignes_erreur_nom.append((i, nom))
                try:
                    ret[nom] = float(ll[col_no - 1].strip())
                except ValueError as e:
                    lignes_erreur_conversion.append(
                        (
                            i,
                            f"{ll[col_no-1].strip().decode()}",
                        )
                    )
        if len(no_lignes_vides) != 0:
            msg = ", ".join((f"{i}" for i in no_lignes_vides))
            logger.info(
                f"le fichier {fic} contient {len(no_lignes_vides)} ligne(s) vide(s) numérotée(s): {msg}"
            )
        if len(lignes_erreur) != 0:
            msg = "\n".join(
                (f"ligne no: {i} qui contient '{a}'" for i, a in lignes_erreur)
            )
            logger.warning(
                f"le fichier {fic} contient {len(lignes_erreur)} ligne(s) sans colonne no {col_no}:\n{msg}"
            )
        if len(lignes_erreur_nom) != 0:
            msg = "\n".join(
                (f"ligne no: {i} qui contient '{a}'" for i, a in lignes_erreur_nom)
            )
            logger.warning(
                f"le fichier {fic} contient {len(lignes_erreur_nom)} ligne(s) dont le nom n'est pas reconnu:\n{msg}"
            )
        if len(lignes_erreur_conversion) != 0:
            msg = "\n".join(
                (
                    f"ligne no: {i} qui contient '{a}'"
                    for i, a in lignes_erreur_conversion
                )
            )
            logger.warning(
                f"le fichier {fic} contient {len(lignes_erreur_conversion)} ligne(s) dont la valeur n'est pas convertissable en réel:\n{msg}"
            )
        logger.info(f"{len(ret)} lignes lues dans le fichier {fic}")
        return ret

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
        self.E.order = order
        M: MutableSequence[float] = []

        for e in self.Equations_table:
            M.extend(E1.generate_line(e))

        M1 = np.array(M).reshape(len(self.Equations_table), len(E1.col_names))
        a, residuals, rank, eign = lstsq(M1, b)
        max_diff = np.max(M1 @ a - b)
        min_diff = np.min(M1 @ a - b)

        return (a, max_diff, min_diff, rank)

    def solve2(self, order: int, b: Mapping[str, float]) -> Tuple[
        npt.NDArray[np.float64],
        Sequence[str],
        Equations,
        Sequence[Tuple[str, float]],
        float,
        float,
        int,
    ]:
        """Solve and find the coefficients
        only for equations in b.keys()
        return:
        the solution a of M @ a = b
        the maximum difference between M @ a and b
        the minimum difference between M @ a and b
        the rank"""

        new_index: List[str] = []
        scv = self.search_constant_var()
        cvar = set(scv)
        lcvar = set((a[0] for a in scv))
        for a in self.E.indexes:
            if not a in lcvar:
                new_index.append(a)
        E1 = Equations(new_index, order)
        M: MutableSequence[float] = []
        b2: MutableSequence[float] = []
        b3: npt.NDArray[np.float64] = np.array([])
        pos: Optional[int]
        equation_ignore: List[str] = []
        equation_OK: List[str] = []
        name_index = dict(((a[1], a[0]) for a in enumerate(self.Equations_name)))
        for name, val in sorted(b.items()):
            pos = name_index.get(name, None)
            if not pos == None:
                equation_OK.append(name)
                Et = dict(
                    (
                        (a, b)
                        for a, b in self.Equations_table[pos].items()
                        if not a in lcvar
                    )
                )
                M.extend(E1.generate_line(Et))
                b2.append(val)
            else:
                equation_ignore.append(name)

        if len(equation_ignore) != 0:
            msg = ", ".join(equation_ignore)
            logger.warning(f"les équations suivantes seront ignorées:\n{msg}")
        M1 = np.array(M).reshape(len(equation_OK), len(E1.col_names))
        b3 = np.array(b2, np.float64)
        a, residuals, rank, eign = lstsq(M1, b3, rcond=None)
        max_diff = np.max(M1 @ a - b3)
        min_diff = np.min(M1 @ a - b3)
        logger.info(
            f"""après la résolution l'écart maximal est: {max_diff}
l'écart minimal est: {min_diff}
le rang de la matrice est: {rank}"""
        )
        lst_cvar = sorted(cvar, key=lambda x: self.E.ind_indexes[x[0]])
        return (a, equation_OK, E1, lst_cvar, max_diff, min_diff, rank)

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
        self.E.order = order
        M: MutableSequence[float] = []

        for e in self.Equations_table:
            M.extend(E1.generate_line(e))

        M1 = np.array(M).reshape(len(self.Equations_table), len(E1.col_names))
        b = np.ones((len(self.Equations_table), 1))
        a, residuals, rank, eign = lstsq(M1, b, rcond=None)
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
        new_names = [(template_name + f"_{i}") for i in range(1, len(gen_cir) + 1)]
        snew_names = set(new_names)
        int_names = self.set_name & snew_names
        if len(int_names) != 0:
            msg = ", ".join(sorted(int_names))
            logger.warning(f"les noms des expériences {msg} existent déjà")
        for d, n in zip(gen_cir, new_names):
            self.add(d, n)

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
        new_names = [(template_name + f"_{i}") for i in range(1, len(gen_prod) + 1)]
        snew_names = set(new_names)
        int_names = self.set_name & snew_names
        if len(int_names) != 0:
            msg = ", ".join(sorted(int_names))
            logger.warning(f"les noms des expériences {msg} existent déjà")
        for d, n in zip(gen_prod, new_names):
            self.add(d, n)

    def generate_def_scr(
        self, data1: Mapping[str, float], data2: Sequence[str], template_name
    ) -> MutableSequence[Mapping[str, float]]:
        """
        add design experiment from definite screening paper
        """
        gen_scr = self.E.generate_def_scr(data1, data2)
        new_names = [(template_name + f"_{i}") for i in range(1, len(gen_scr) + 1)]
        snew_names = set(new_names)
        int_names = self.set_name & snew_names
        if len(int_names) != 0:
            msg = ", ".join(sorted(int_names))
            logger.warning(f"les noms des expériences {msg} existent déjà")
        for d, n in zip(gen_scr, new_names):
            self.add(d, n)

    def search_constant_var(self) -> Sequence[Tuple[str, float]]:
        """
        recherche à travers les equations les variables constantes
        retourne les noms des variables qui restent constantes pour
        toute les équations, la valeur constante est aussi ajoutée
        """
        ret: List[Tuple[str, float]] = []
        cou: DefaultDict[List[float]] = defaultdict(lambda: list())
        for eq in self.Equations_table:
            for var, val in eq.items():
                cou[var].append(val)
        if len(cou) == 0:
            return ret
        for var, lval in cou.items():
            val0 = lval[0]
            difference = np.array(lval) - val0
            zer = np.zeros_like(difference)
            if all(np.isclose(difference, zer)):
                ret.append((var, val0))
        if len(ret) != 0:
            logger.info(
                f"les variables constantes dans les équations sont:\n{'\n'.join((f"{a[0]}: {a[1]}" for a in ret))}"
            )
        return ret

    def write_minmax_plan(
        self,
        a: npt.NDArray[np.float64],
        conv: Set[Tuple[str, int]],
        E: Equations,
        order: int,
        out_size: int,
        fic_out_min: Path,
        fic_out_max: Path,
        Nom_eq: str,
    ) -> int:
        """
        évalue toutes les équations du plan générable par conv avec a vecteur des coefficients
        renvoyé par solve et solve2. Ecrit 2 fichiers contenant les out_size min ou max
        expériences nommées avec Nom_eq.
        renvoie le nombre d'équations évaluées.
        si out_size <= -1 alors fic_out_min est écrit avec toutes les correspondance nom,résultats
        si out_size > 0 alors seul fic_out_min est écrit avec le plan d'expérience
        """
        self.clear()
        Eq0 = dict(((a, 0) for a in self.E.indexes))
        old_indexes = loads(dumps(self.E.indexes))
        # logger.debug(f"conv = {conv}")
        dcon = dict(
            (
                (k, sorted((a[1] for a in g)))
                for k, g in groupby(sorted(conv), key=lambda x: x[0])
            )
        )
        # logger.debug(f"dcon = {dcon}")
        if not set(E.indexes) <= set(self.E.indexes):
            msg0 = ", ".join(
                (f"'{a}'" for a in sorted(set(E.indexes) - set(self.E.indexes)))
            )
            logger.error(f"les variables: {msg0} ne font pas partie du plan")
            return 0
        if not set(E.indexes) <= set(dcon.keys()):
            msg0 = ", ".join(
                (f"'{a}'" for a in sorted(set(dcon.keys()) - set(E.indexes)))
            )
            logger.warning(f"les variables: {msg0} de sont pas assignées")
            return 0
        new_indexes = list(filter(lambda x: x in dcon.keys(), E.indexes))
        lcon = [dcon[nom] for nom in new_indexes]
        # logger.debug(f"lcon = {lcon}")
        size = reduce(lambda x, y: x * y, (len(b) for b in lcon), 1)
        # logger.debug(f"size = {size}")
        logger.info(f"le nombre d'équation est: {size}")
        if size <= out_size:
            logger.warning(
                f"""le nombre de sortie d'équation demandée {out_size} en sortie
est supérieur au nombre d'équation générée: {size}
toutes les données seront écrites"""
            )
            out_size = -1
        noms = [f"{Nom_eq}_{i}" for i in range(1, size + 1)]
        logger.info("création du meshgrid des valeurs des variables")
        mesh = np.meshgrid(*lcon)
        logger.info("applatissement du meshgrid des valeurs des variables")
        matr = np.hstack(
            [np.reshape(np.ravel(mesh[i]), (size, 1)) for i in range(len(new_indexes))]
        )
        logger.info("calcul de la matrice des valeur de la matrice des inconnues")
        self.E.__init__(new_indexes, order)
        res = self.E.generate_array(matr) @ a
        self.E.__init__(old_indexes, order)
        logger.info("recherche des extrèmes")
        ind = np.argsort(res)
        if out_size > 0:
            logger.info(f"début d'écriture du fichier du plan max {fic_out_max}")
            indmax = ind[-out_size:]
            matrmax = matr[indmax]
            nomsmax = [noms[i] for i in indmax]
            for a, nom in zip(matrmax.tolist()[::-1], nomsmax[::-1]):
                Eq1 = loads(dumps(Eq0))
                Eq1.update(dict(zip(new_indexes, a)))
                self.add(Eq1, nom)
            msg = self.to_csv()
            fic_out_max.write_text(msg)
            logger.info(f"{msg}")
            logger.info(f"fin écriture du fichier {fic_out_max}")
            logger.info(
                "nom;val\n"
                + "\n".join(
                    (
                        f"{a[0]};{a[1]}"
                        for a in zip(nomsmax[::-1], res[indmax].tolist()[::-1])
                    )
                )
            )
            self.clear()
            logger.info(f"début d'écriture du fichier du plan min {fic_out_min}")
            indmin = ind[:out_size]
            matrmin = matr[indmin]
            nomsmin = [noms[i] for i in indmin]
            for a, nom in zip(matrmin.tolist(), nomsmin):
                Eq1 = loads(dumps(Eq0))
                Eq1.update(dict(zip(new_indexes, a)))
                self.add(Eq1, nom)
            msg = self.to_csv()
            fic_out_min.write_text(msg)
            logger.info(f"{msg}")
            logger.info(f"fin écriture du fichier {fic_out_min}")
            logger.info(
                "nom;val\n"
                + "\n".join((f"{a[0]};{a[1]}" for a in zip(nomsmin, res[indmin])))
            )
        elif out_size < 0:
            logger.info(f"début d'écriture du fichier {fic_out_max}")
            matrall = matr[ind]
            nomsall = [noms[i] for i in ind]
            fic_out_max.write_text(
                "nom;val\n"
                + "\n".join((f"{a[0]};{a[1]}" for a in zip(nomsall, res[ind])))
            )
            logger.info(
                f"""fin d'écriture du fichier des résultats {fic_out_max}
écriture de {len(noms) + 1} lignes"""
            )
            for a, nom in zip(matrall.tolist(), nomsall):
                Eq1 = loads(dumps(Eq0))
                Eq1.update(dict(zip(new_indexes, a)))
                self.add(Eq1, nom)
            msg = self.to_csv()
            fic_out_min.write_text(msg)
            logger.info(
                f"fin d'écriture de {len(nom)} lignes du plan complet dans {fic_out_min}"
            )
        return size


class Plan_tri(Plan):
    """
    Manage and solve several equations
    """

    def __init__(self, indexes: Sequence[str]) -> None:
        """
        init du plan à 3 niveau (-1, 0, 1)
        """
        super().__init__(indexes)
        self.E = Equations_tri(indexes, 1)
        self.sbitlevels: Set[int] = set()
        self.lbitlevels: List[int] = []
        self.abitlevels: npt.NDArray[np.uint64] = np.array([])
        self.bit_index: Mapping[str, Tuple[int, int, int]] = dict(
            (
                (a, (1 << i, 2 << i, 3 << i))
                for i, a in ((j << 1, b) for j, b in enumerate(indexes))
            )
        )  # dictionnaire des bits representant -1, 0, 1
        self.pos_index: Mapping[
            str,
            Tuple[
                npt.NDArray[np.uint64],
                npt.NDArray[np.uint64],
                npt.NDArray[np.uint64],
            ],
        ] = {}
        self.res_index: Mapping[
            str,
            Tuple[
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
            ],
        ] = {}  # key: nom, val: courbes pour -1, 0 et 1
        self.dd_index: Mapping[
            str,
            Tuple[
                npt.NDArray[np.uint64],
                npt.NDArray[np.uint64],
                npt.NDArray[np.uint64],
                npt.NDArray[np.uint64],
            ],
        ] = {}
        self.ddd: Mapping[
            Tuple[str, int, int],
            npt.NDArray[np.float64],
        ] = {}  # key: (nom, niv0 , niv1), val courbes de delta
        self.Courbes: Mapping[str, Sequence[Optional[Courbe1]]] = {}
        self.Courbes_ddd: Mapping[Tuple[str, int, int], Sequence[Optional[Courbe2]]] = (
            {}
        )
        self.prob: Mapping[str, Optional[Proba1]] = {}
        self.prob_ddd: Mapping[str, Optional[Proba2]] = {}
        self.f_oneway: Mapping[str, Sequence[Optional[float]]] = {}

    def to_bitlevels(self, datas: Mapping[str, float]) -> int:
        """
        renvoie un entier représentant les niveaux de l'expérience
        """
        ret: int = 0
        for ind in self.E.indexes:
            val = datas[ind]
            if fabs(val) < 1e-7:
                ret |= self.bit_index[ind][1]
            elif fabs(val - 1.0) < 1.0e-7:
                ret |= self.bit_index[ind][2]
            elif fabs(val + 1.0) < 1.0e-7:
                ret |= self.bit_index[ind][0]
            else:
                raise ValueError(
                    f"erreur de programmation à l'index '{ind}' sa valeur {val} n'est ni 1, 0, -1"
                )
        return ret

    def add(self, datas: Mapping[str, float], name: str) -> int:
        """
        ajout une expérience à 3 niveaux
        """
        self.E.generate_line(datas)
        lst_err = ", ".join(
            (
                f"'{n}': {val}"
                for n, val in ((m, datas[m]) for m in self.E.indexes)
                if not (
                    (fabs(val) < 1e-7)
                    or (fabs(val - 1.0) < 1.0e-7)
                    or (fabs(val + 1.0) < 1.0e-7)
                )
            )
        )
        if len(lst_err) != 0:
            logger.error(
                f"""pour l'équation '{name}' les valeurs suivantes affectées aux variables
ne sont pas acceptées pour un Plan_tri:
{lst_err}"""
            )
            return 0
        bile = self.to_bitlevels(datas)
        if bile in self.sbitlevels:
            logger.error(
                f"pour l'équation '{name}': une équation avec les mêmes coefficients existe déjà"
            )
            return 0
        self.sbitlevels.add(bile)
        self.lbitlevels.append(bile)
        return super().add(datas, name)

    def search_stat(self, res: Mapping[str, float]) -> None:
        """
        recherche les niveau pour réaliser des comparaisons sur les influences des
        facteur
        res est un dictionnaire avec comme clé le nom des expériences et valeur
        le résultat de l'expérience
        si il n'y a pas de résultat d'expérience cette dernière est ignorée
        met à jour:
        self.pos_index,
        self.res_index
        self.dd_index
        self.ddd
        """
        r_index = set(res.keys())
        if not (r_index <= self.set_name):
            msg = ", ".join(sorted(r_index - self.set_name))
            logger.warning(
                f"les résultats suivants: {msg} n'ont pas d'équation associée, ils sont ignorés"
            )
        expind = dict(((a[1], a[0]) for a in enumerate(self.Equations_name)))
        aindexes = np.sort(
            np.array(
                list(
                    filter(
                        lambda x: x is not None,
                        (expind.get(a, None) for a in res.keys()),
                    )
                ),
                dtype=np.uint,
            )
        )
        neq = np.shape(aindexes)[0]
        logger.info(f"{neq} résultats seront traités")
        position = np.arange(0, neq)
        abitlevels = np.array(self.lbitlevels, dtype=np.uint64)[aindexes]
        # logger.debug(f"abitlevels = {abitlevels}")
        ares = np.array([res[self.Equations_name[int(i)]] for i in aindexes])
        ar_all = np.array(
            [self.bit_index[name] for name in self.E.indexes], dtype=np.uint64
        )
        # logger.debug(f"ar_all = {ar_all}")
        arp_un = np.ravel(ar_all[:, 2])
        a1 = np.bitwise_and(abitlevels.reshape(neq, 1), arp_un).reshape(
            neq, len(self.E.indexes), 1
        )
        a2 = np.bitwise_xor(a1, ar_all)
        # a2[no_eq, no_variable, -1 ou 0 ou 1]
        for i, name in enumerate(self.E.indexes):
            col0 = np.ravel(a2[:, i, 0])
            self.pos_index[name] = (
                position[a2[:, i, 0] == 0],
                position[a2[:, i, 1] == 0],
                position[a2[:, i, 2] == 0],
            )
        for name, pos in self.pos_index.items():
            self.res_index[name] = (ares[pos[0]], ares[pos[1]], ares[pos[2]])
        # traitement des comparaisons doubles
        b1 = np.bitwise_xor(abitlevels.reshape(neq, 1), abitlevels)
        # logger.debug(f"b1 = {b1}")
        # b2 = np.bitwise_and(b1.reshape(neq, neq, 1), ~arp_un)
        b2b = np.broadcast_to(
            np.bitwise_and(b1.reshape(neq, neq, 1), ~arp_un),
            (3, neq, neq, len(self.E.indexes)),
        )
        # logger.debug(f"b2 = {b2}")
        # b2b(0 ou 1 ou 2, num_eq, num_eq, no_variable) = 0 si ne diffère que d'une variable
        d1_index = np.indices((neq, neq, len(self.E.indexes)))
        # logger.debug(f"d1_index.shape = {d1_index.shape}")
        # logger.debug(f"d1_index[0] = {d1_index[0]}")
        # logger.debug(f"d1_index[1] = {d1_index[1]}")
        # logger.debug(f"d1_index[2] = {d1_index[2]}")
        # logger.debug(f"b2b == 0 = {b2b==0}")
        # d2_index0 = d1_index[0][b2 == 0]
        # d2_index1 = d1_index[1][b2 == 0]
        # d2_index2 = d1_index[2][b2 == 0]
        # logger.debug(f"b2b == 0 = {b2b == 0}")
        d2b_index = np.reshape(d1_index[b2b == 0], (3, -1))
        # logger.debug(f"d2_index0 = {d2_index0}")
        # logger.debug(f"d2_index1 = {d2_index1}")
        # logger.debug(f"d2_index2 = {d2_index2}")
        # logger.debug(f"d2b_index = {d2b_index}")
        # d3m = d2_index0 > d2_index1
        d3bm = np.broadcast_to(d2b_index[0] > d2b_index[1], d2b_index.shape)
        # logger.debug(f"d3bm = {d3bm}")
        # logger.debug(f"d3bm = {d3bm}")
        # d3_index0 = d2_index0[d3m]
        # d3_index1 = d2_index1[d3m]
        # d3_index2 = d2_index2[d3m]
        # logger.debug(f"d2b_index[d3bm] = {d2b_index[d3bm]}")
        d3b_index = np.reshape(d2b_index[d3bm], (3, -1))
        # logger.debug(f"d3_index0 = {d3_index0}")
        # logger.debug(f"d3_index1 = {d3_index1}")
        # logger.debug(f"d3_index2 = {d3_index2}")
        # logger.debug(f"d3b_index = {d3b_index}")
        for i, name in enumerate(self.E.indexes):
            # dm4 = d3_index2 == i
            dm4 = d3b_index[2] == i
            # logger.debug(f"i = {i}, dm4 = {dm4}")
            if not all(np.logical_not(dm4)):
                d4_index0 = d3b_index[0][dm4]
                d4_index1 = d3b_index[1][dm4]
                # logger.debug(f"d4_index0 = {d4_index0}")
                # logger.debug(f"d4_index1 = {d4_index1}")
                d4_res0 = ares[d4_index0]
                d4_res1 = ares[d4_index1]
                # logger.debug(f"ares = {ares}")
                # logger.debug(f"d4_res0 = {d4_res0}")
                # logger.debug(f"d4_res1 = {d4_res1}")
                m_un, zer, p_un = self.bit_index[name]
                # logger.debug(f"abitlevels = {abitlevels}")
                d4_bl0 = np.bitwise_and(abitlevels[d4_index0], p_un)
                d4_bl1 = np.bitwise_and(abitlevels[d4_index1], p_un)
                # logger.debug(f"d4_bl0 = {d4_bl0}")
                # logger.debug(f"d4_bl1 = {d4_bl1}")
                self.dd_index[i] = (d4_res0, d4_res1, d4_bl0, d4_bl1)
                d4_res1m0 = d4_res1 - d4_res0
                # logger.debug(f"d4_res1m0 = {d4_res1m0}")
                d4_res0m1 = -d4_res1m0
                # logger.debug(f"d4_res0m1 = {d4_res0m1}")
                m4_bl0_m = np.bitwise_xor(d4_bl0, m_un) == 0
                m4_bl0_z = np.bitwise_xor(d4_bl0, zer) == 0
                m4_bl0_p = np.bitwise_xor(d4_bl0, p_un) == 0
                m4_bl1_m = np.bitwise_xor(d4_bl1, m_un) == 0
                m4_bl1_z = np.bitwise_xor(d4_bl1, zer) == 0
                m4_bl1_p = np.bitwise_xor(d4_bl1, p_un) == 0
                # cas 0:-1 , 1:0
                m_m_z = np.logical_and(m4_bl0_m, m4_bl1_z)
                m_z_m = np.logical_and(m4_bl0_z, m4_bl1_m)
                res_m_z = np.concatenate(
                    (d4_res1m0[m_m_z], d4_res0m1[m_z_m]), axis=None
                )
                if res_m_z.shape[0] != 0:
                    self.ddd[(name, -1, 0)] = res_m_z
                # cas 0:-1 , 1:1
                m_m_p = np.logical_and(m4_bl0_m, m4_bl1_p)
                m_p_m = np.logical_and(m4_bl0_p, m4_bl1_m)
                res_m_p = np.concatenate(
                    (d4_res1m0[m_m_p], d4_res0m1[m_p_m]), axis=None
                )
                if res_m_p.shape[0] != 0:
                    self.ddd[(name, -1, 1)] = res_m_p
                # cas 0:0 , 1:1
                m_z_p = np.logical_and(m4_bl0_z, m4_bl1_p)
                m_p_z = np.logical_and(m4_bl0_p, m4_bl1_z)
                res_z_p = np.concatenate(
                    (d4_res1m0[m_z_p], d4_res0m1[m_p_z]), axis=None
                )
                if res_z_p.shape[0] != 0:
                    self.ddd[(name, 0, 1)] = res_z_p

        # logger.debug(f"self.dd_index = {self.dd_index}")
        # logger.debug(f"self.ddd = {self.ddd}")
        for nom in self.E.indexes:
            res_m, res_z, res_p = self.res_index.get(nom, (None, None, None))
            ret1: List[Optional[Courbe1]] = [None, None, None]
            if res_m is not None:
                smp_siz = res_m.shape[0]
                if smp_siz >= 5:
                    percentile_25, percentile_50, percentile_75 = np.percentile(
                        res_m, [25, 50, 75]
                    )
                    mu_samp = np.mean(res_m, dtype=np.float64)
                    var_samp = np.sum((res_m - mu_samp) ** 2) / (smp_siz - 1)
                    ret1[0] = Courbe1(
                        nom,
                        np.max(res_m),
                        np.min(res_m),
                        percentile_25,
                        percentile_50,
                        percentile_75,
                        "ERROR",
                        -1,
                        smp_siz,
                        mu_samp,
                        var_samp,
                    )
                else:
                    if smp_siz != 0:
                        logger.info(
                            f"pour la variable {nom}, à la valeur -1: la taille des résultats est {smp_siz} < 5"
                        )
                    else:
                        logger.info(
                            f"pour la variable {nom}, à la valeur -1: il n'y a pas de données"
                        )
            if res_z is not None:
                smp_siz = res_z.shape[0]
                if smp_siz >= 5:
                    percentile_25, percentile_50, percentile_75 = np.percentile(
                        res_z, [25, 50, 75]
                    )
                    mu_samp = np.mean(res_z, dtype=np.float64)
                    var_samp = np.sum((res_z - mu_samp) ** 2) / (smp_siz - 1)
                    ret1[1] = Courbe1(
                        nom,
                        np.max(res_z),
                        np.min(res_z),
                        percentile_25,
                        percentile_50,
                        percentile_75,
                        "ERROR",
                        0,
                        smp_siz,
                        mu_samp,
                        var_samp,
                    )
                else:
                    if smp_siz != 0:
                        logger.info(
                            f"pour la variable {nom}, à la valeur 0: la taille des résultats est {smp_siz} < 5"
                        )
                    else:
                        logger.info(
                            f"pour la variable {nom}, à la valeur 0: il n'y a pas de données"
                        )
            if res_p is not None:
                smp_siz = res_p.shape[0]
                if smp_siz >= 5:
                    percentile_25, percentile_50, percentile_75 = np.percentile(
                        res_p, [25, 50, 75]
                    )
                    mu_samp = np.mean(res_p, dtype=np.float64)
                    var_samp = np.sum((res_p - mu_samp) ** 2) / (smp_siz - 1)
                    ret1[2] = Courbe1(
                        nom,
                        np.max(res_p),
                        np.min(res_p),
                        percentile_25,
                        percentile_50,
                        percentile_75,
                        "ERROR",
                        1,
                        smp_siz,
                        mu_samp,
                        var_samp,
                    )
                else:
                    if smp_siz != 0:
                        logger.info(
                            f"pour la variable {nom}, à la valeur 1: la taille des résultats est {smp_siz} < 5"
                        )
                    else:
                        logger.info(
                            f"pour la variable {nom}, à la valeur 1: il n'y a pas de données"
                        )
            self.Courbes[nom] = tuple(loads(dumps(ret1)))
            # if not any((res_m is None, res_z is None, res_p is None)) and all( (res_p.shape[0] > 1, res_z.shape[0] > 1, res_m.shape[0] > 1)):
            #    fo = f_oneway(res_m, res_z, res_p)
            #    self.f_oneway[nom] = (fo.statistic, fo.pvalue)
            # else:
            #    self.f_oneway[nom] = (None, None)
        for nom in self.E.indexes:
            res_mz = self.ddd.get((nom, -1, 0), None)
            res_mp = self.ddd.get((nom, -1, 1), None)
            res_zp = self.ddd.get((nom, 0, 1), None)
            if res_mz is not None:
                smp_siz = res_mz.shape[0]
                if smp_siz >= 5:
                    percentile_25, percentile_50, percentile_75 = np.percentile(
                        res_mz, [25, 50, 75]
                    )
                    mu_samp = np.mean(res_mz, dtype=np.float64)
                    var_samp = np.sum((res_mz - mu_samp) ** 2) / (smp_siz - 1)
                    self.Courbes_ddd[(nom, -1, 0)] = Courbe2(
                        nom,
                        np.max(res_mz),
                        np.min(res_mz),
                        percentile_25,
                        percentile_50,
                        percentile_75,
                        "ERROR1",
                        "ERROR2",
                        -1,
                        0,
                        smp_siz,
                        mu_samp,
                        var_samp,
                    )
                else:
                    if smp_siz != 0:
                        logger.info(
                            f"pour la variable {nom}, entre -1 et 0: la taille des résultats est {smp_siz} < 5"
                        )
                    else:
                        logger.info(
                            f"pour la variable {nom}, entre -1 et 0: il n'y a pas de données"
                        )
            else:
                logger.info(
                    f"pour la variable {nom}, entre -1 et 0: il n'y a pas de données"
                )
            if res_mp is not None:
                smp_siz = res_mp.shape[0]
                if smp_siz >= 5:
                    percentile_25, percentile_50, percentile_75 = np.percentile(
                        res_mp, [25, 50, 75]
                    )
                    mu_samp = np.mean(res_mp, dtype=np.float64)
                    var_samp = np.sum((res_mp - mu_samp) ** 2) / (smp_siz - 1)
                    self.Courbes_ddd[(nom, -1, 1)] = Courbe2(
                        nom,
                        np.max(res_mp),
                        np.min(res_mp),
                        percentile_25,
                        percentile_50,
                        percentile_75,
                        "ERROR1",
                        "ERROR2",
                        -1,
                        1,
                        smp_siz,
                        mu_samp,
                        var_samp,
                    )
                else:
                    if smp_siz != 0:
                        logger.info(
                            f"pour la variable {nom}, entre -1 et 1: la taille des résultats est {smp_siz} < 5"
                        )
                    else:
                        logger.info(
                            f"pour la variable {nom}, entre -1 et 1: il n'y a pas de données"
                        )
            else:
                logger.info(
                    f"pour la variable {nom}, entre -1 et 1: il n'y a pas de données"
                )
            if res_zp is not None:
                smp_siz = res_zp.shape[0]
                if smp_siz >= 5:
                    percentile_25, percentile_50, percentile_75 = np.percentile(
                        res_zp, [25, 50, 75]
                    )
                    mu_samp = np.mean(res_zp, dtype=np.float64)
                    var_samp = np.sum((res_zp - mu_samp) ** 2) / (smp_siz - 1)
                    self.Courbes_ddd[(nom, 0, 1)] = Courbe2(
                        nom,
                        np.max(res_zp),
                        np.min(res_zp),
                        percentile_25,
                        percentile_50,
                        percentile_75,
                        "ERROR1",
                        "ERROR2",
                        0,
                        1,
                        smp_siz,
                        mu_samp,
                        var_samp,
                    )
                else:
                    if smp_siz != 0:
                        logger.info(
                            f"pour la variable {nom}, entre 0 et 1: la taille des résultats est {smp_siz} < 5"
                        )
                    else:
                        logger.info(
                            f"pour la variable {nom}, entre 0 et 1: il n'y a pas de données"
                        )
            else:
                logger.info(
                    f"pour la variable {nom}, entre 0 et 1: il n'y a pas de données"
                )

    def search_prob(self, alphas: Sequence[float]) -> None:
        """
        à l'aide self.res_index, self.Courbes et
        self.ddd, self.Courbes_ddd
        rempli self.prob et self.prob_ddd
        """
        for nom in self.E.indexes:
            res_m, res_z, res_p = self.res_index.get(nom, (None, None, None))
            c1m, c1z, c1p = self.Courbes.get(nom, (None, None, None))
            if all(((a is not None) for a in (res_p, res_m, c1m, c1p))):
                np = c1p.sample_size
                nm = c1m.sample_size
                if all((a > 8 for a in (np, nm))):
                    valp = c1p.value
                    valm = c1m.value
                    S2p = c1p.var_samp
                    S2m = c1m.var_samp
                    yp = c1p.mu_samp
                    ym = c1m.mu_samp
                    Smp = S2p / np + S2m / nm
                    sqSmp = sqrt(Smp)
                    t_0 = (yp - ym) / sqSmp
                    v = floor(
                        Smp**2
                        / ((S2p / np) ** 2 / (np - 1) + (S2m / nm) ** 2 / (nm - 1))
                    )
                    p_value = 100.0 * stdtr(v, t_0)
                    ntm = 100.0 * normaltest(res_m).pvalue
                    ntp = 100.0 * normaltest(res_p).pvalue
                    stdinvt = [stdtrit(v, a / 2) for a in alphas]
                    b_inf = [(yp - ym + t * sqSmp) for t in stdinvt]
                    b_sup = [(yp - ym - t * sqSmp) for t in stdinvt]
                    self.prob[nom] = Proba1(
                        nom,
                        valm,
                        valp,
                        t_0,
                        v,
                        p_value,
                        ntm,
                        ntp,
                        alphas,
                        stdinvt,
                        b_inf,
                        b_sup,
                    )
                else:
                    logger.info(
                        f"taille des échantillons inférieurs à 8 pour la variable {nom}"
                    )
            else:
                logger.info(
                    f"impossible de calculer les probabilité indépendantes pour la variable {nom}"
                )
            res_mp = self.ddd.get((nom, -1, 1), None)
            cmp = self.Courbes_ddd.get((nom, -1, 1), None)
            if all(((a is not None) for a in (res_mp, cmp))):
                n = cmp.sample_size
                if n > 8:
                    valm = cmp.value1
                    valp = cmp.value2
                    S = sqrt(cmp.var_samp / n)
                    d = cmp.mu_samp
                    t_0 = d / S
                    p_value = 100.0 * stdtr(n - 1, t_0)
                    nt = 100.0 * normaltest(res_mp).pvalue
                    stdinvt = [stdtrit(n - 1, a / 2) for a in alphas]
                    b_inf = [(d + t * S) for t in stdinvt]
                    b_sup = [(d - t * S) for t in stdinvt]
                    self.prob_ddd[nom] = Proba2(
                        nom,
                        valm,
                        valp,
                        t_0,
                        n,
                        p_value,
                        nt,
                        alphas,
                        stdinvt,
                        b_inf,
                        b_sup,
                    )
                else:
                    logger.info(
                        f"taille de l'échantillons par pair inférieurs à 8 pour la variable {nom}"
                    )
            else:
                logger.info(
                    f"impossible de calculer les probabilité par pair pour la variable {nom}"
                )

    def to_csv1(
        self,
        nom_fic: Path,
        conv: Mapping[Tuple[str, int], str],
        PC1: Prec_Courbe2 = PC1_1,
    ) -> None:
        """
        Sortie de self.Courbes sous format csv
        """
        ret: MutableSequence[str] = []
        lfi0 = [a.name for a in fields(Courbe1)]
        lfi = lfi0[:1] + lfi0[6:8] + lfi0[1:6] + lfi0[8:]
        empty_fields = ";".join(repeat("-", len(lfi) - 3))
        ret.append(";".join(lfi))
        for nom in self.E.indexes:
            nom1 = self.E.latex_indexes.get(nom, nom)
            for res, bit in zip(self.Courbes.get(nom, (None, None, None)), (-1, 0, 1)):
                if res != None:
                    ret.append(nom1 + ";" + res.to_csv(conv, PC1))
                else:
                    val1 = conv.get((nom, bit), "ERR")
                    ret.append(f"{nom1};{val1};{bit};" + empty_fields)
        logger.info(f"écriture de {1 + len(ret)} lignes dans {nom_fic}")
        nom_fic.write_text("\n".join(ret))

    def to_csv2(
        self,
        nom_fic: Path,
        conv: Mapping[Tuple[str, int], str],
        PC2: Prec_Courbe2 = PC2_1,
    ) -> None:
        """
        Sortie de self.Courbes_ddd sous format csv
        """
        ret: MutableSequence[str] = []
        lfi0 = [a.name for a in fields(Courbe2)]
        lfi = lfi0[:1] + lfi0[6:10] + lfi0[1:6] + lfi0[10:]
        empty_fields = ";".join(repeat("-", len(lfi) - 5))
        ret.append(";".join(lfi))
        for nom in self.E.indexes:
            nom1 = self.E.latex_indexes.get(nom, nom)
            for bit in ((-1, 0), (-1, 1), (0, 1)):
                res = self.Courbes_ddd.get((nom, bit[0], bit[1]), None)
                if res != None:
                    ret.append(nom1 + ";" + res.to_csv(conv, PC2))
                else:
                    val1 = conv.get((nom, bit[0]), "ERR")
                    val2 = conv.get((nom, bit[1]), "ERR")
                    ret.append(
                        f"{nom1};{val1};{val2};{bit[0]};{bit[1]};" + empty_fields
                    )
        logger.info(f"écriture de {1 + len(ret)} lignes dans {nom_fic}")
        nom_fic.write_text("\n".join(ret))

    def to_csv_prob_1(self, nom_fic: Path, conv: Mapping[Tuple[str, int], str]) -> None:
        """
        première sortie de self.prob sous format csv
        """
        ret: MutableSequence[str] = []
        lfi = [a.name for a in fields(Proba1)[:8]]
        empty_fields = ";".join(repeat("-", len(lfi) - 3))
        ret.append(";".join(lfi))
        for nom in self.E.indexes:
            nom1 = self.E.latex_indexes.get(nom, nom)
            res = self.prob.get(nom, None)
            if res != None:
                ret.append(nom1 + ";" + res.to_csv1(conv))
            else:
                val1 = conv.get((nom, -1), "ERR")
                val2 = conv.get((nom, 1), "ERR")
                ret.append(f"{nom1};{val1};{val2};" + empty_fields)
        logger.info(f"écriture de {len(ret)} lignes dans {nom_fic}")
        nom_fic.write_text("\n".join(ret))

    def to_csv_prob_2(self, nom_fic: Path, conv: Mapping[Tuple[str, int], str]) -> None:
        """
        deuxième sortie de self.prob sous format csv
        """
        ret: MutableSequence[str] = []
        lfi = ["name", "v"]
        std_non_nul = [k for k, a in self.prob.items() if len(a.stdinvt) != 0]
        empty_fields = ""
        if len(std_non_nul) != 0:
            p1 = self.prob[std_non_nul[0]]
            for pro in p1.alphas:
                proba = f"{D(100. - 100*pro).quantize(D('0.01'))}"
                proba2 = f"{D(pro/2.).quantize(D('0.00001'))}"
                lfi.extend([f"t_{proba2}", f"val_inf_{proba}", f"val_sup_{proba}"])
                empty_fields += ";".join(repeat("-", 3)) + ";"
        ret.append(";".join(lfi))
        for nom in self.E.indexes:
            nom1 = self.E.latex_indexes.get(nom, nom)
            res = self.prob.get(nom, None)
            if res != None:
                ret.append(nom1 + ";" + res.to_csv2(conv))
            else:
                ret.append(f"{nom1};-;" + empty_fields)
        logger.info(f"écriture de {len(ret)} lignes dans {nom_fic}")
        nom_fic.write_text("\n".join(ret))

    def to_csv_prob_3(self, nom_fic: Path, conv: Mapping[Tuple[str, int], str]) -> None:
        """
        première sortie de self.prob_ddd sous format csv
        """
        ret: MutableSequence[str] = []
        lfi = [a.name for a in fields(Proba2)[:7]]
        empty_fields = ";".join(repeat("-", len(lfi) - 3))
        ret.append(";".join(lfi))
        for nom in self.E.indexes:
            nom1 = self.E.latex_indexes.get(nom, nom)
            res = self.prob_ddd.get(nom, None)
            if res != None:
                ret.append(nom1 + ";" + res.to_csv1(conv))
            else:
                val1 = conv.get((nom, -1), "ERR")
                val2 = conv.get((nom, 1), "ERR")
                ret.append(f"{nom1};{val1};{val2};" + empty_fields)
        logger.info(f"écriture de {len(ret)} lignes dans {nom_fic}")
        nom_fic.write_text("\n".join(ret))

    def to_csv_prob_4(self, nom_fic: Path, conv: Mapping[Tuple[str, int], str]) -> None:
        """
        deuxième sortie de self.prob sous format csv
        """
        ret: MutableSequence[str] = []
        lfi = ["name", "n"]
        std_non_nul = [k for k, a in self.prob_ddd.items() if len(a.stdinvt) != 0]
        empty_fields = ""
        if len(std_non_nul) != 0:
            p1 = self.prob_ddd[std_non_nul[0]]
            for pro in p1.alphas:
                proba = f"{D(100. - 100*pro).quantize(D('0.01'))}"
                proba2 = f"{D(pro/2.).quantize(D('0.00001'))}"
                lfi.extend([f"t_{proba2}", f"val_inf_{proba}", f"val_sup_{proba}"])
                empty_fields += ";".join(repeat("-", 3)) + ";"
        ret.append(";".join(lfi))
        for nom in self.E.indexes:
            nom1 = self.E.latex_indexes.get(nom, nom)
            res = self.prob_ddd.get(nom, None)
            if res != None:
                ret.append(nom1 + ";" + res.to_csv2(conv))
            else:
                ret.append(f"{nom1};-;" + empty_fields)
        logger.info(f"écriture de {len(ret)} lignes dans {nom_fic}")
        nom_fic.write_text("\n".join(ret))

    def visu_all_distri(
        self, plt: Any, sns: Any, dir_stat: Path, conv: Mapping[Tuple[str, int], str]
    ) -> report_visu_distri:
        """
        écriture de graphes histogramme et densité de probabilité
        """
        rep = report_visu_distri()
        for nom in self.E.indexes:
            nom0 = nom.replace("::", "_").replace("cotes_variables_", "")
            nom1 = self.E.fic_names.get(nom, nom0)
            nom2 = self.E.latex_indexes.get(nom, nom0)
            res_m, res_z, res_p = self.res_index.get(nom, (None, None, None))
            c1m, c1z, c1p = self.Courbes.get(nom, (None, None, None))
            if (res_m is not None) and (c1m is not None):
                visu_distri(
                    plt,
                    sns,
                    res_m,
                    floor((c1m.max_val - c1m.min_val) / 0.01),
                    dir_stat / Path(f"distri_sample_m_{nom1}.png"),
                    nom2 + " à " + conv.get((nom, -1), "N.A.") + " (-1)",
                    "yellowgreen",
                )
                rep.add_images(
                    nom2,
                    f"distri_sample_m_{nom1}.png",
                    f"probplot_sample_m_{nom1}.png",
                )
            if res_m is not None:
                visu2_distri(
                    plt,
                    res_m,
                    dir_stat / Path(f"probplot_sample_m_{nom1}.png"),
                    nom2 + " à " + conv.get((nom, -1), "N.A.") + " (-1)",
                )
            if (res_p is not None) and (c1p is not None):
                visu_distri(
                    plt,
                    sns,
                    res_p,
                    floor((c1p.max_val - c1p.min_val) / 0.01),
                    dir_stat / Path(f"distri_sample_p_{nom1}.png"),
                    nom2 + " à " + conv.get((nom, 1), "N.A.") + " (+1)",
                    "dodgerblue",
                )
                rep.add_images(
                    nom2,
                    f"distri_sample_p_{nom1}.png",
                    f"probplot_sample_p_{nom1}.png",
                )
            if res_p is not None:
                visu2_distri(
                    plt,
                    res_p,
                    dir_stat / Path(f"probplot_sample_p_{nom1}.png"),
                    nom2 + " à " + conv.get((nom, 1), "N.A.") + " (+1)",
                )
            res_mp = self.ddd.get((nom, -1, 1), None)
            cmp = self.Courbes_ddd.get((nom, -1, 1), None)
            if (res_mp is not None) and (cmp is not None):
                visu_distri(
                    plt,
                    sns,
                    res_mp,
                    floor((cmp.max_val - cmp.min_val) / 0.01),
                    dir_stat / Path(f"distri_pair_{nom1}.png"),
                    nom2
                    + " de "
                    + conv.get((nom, -1), "N.A.")
                    + " à "
                    + conv.get((nom, 1), "N.A."),
                    "magenta",
                )
                rep.add_images(
                    nom2,
                    f"distri_pair_{nom1}.png",
                    f"probplot_pair_{nom1}.png",
                )
            if res_mp is not None:
                visu2_distri(
                    plt,
                    res_mp,
                    dir_stat / Path(f"probplot_pair_{nom1}.png"),
                    nom2
                    + " de "
                    + conv.get((nom, -1), "N.A.")
                    + " à "
                    + conv.get((nom, 1), "N.A."),
                )
        return rep

    def plot_barh_sample(
        self,
        ax: Any,
        pos: float,
        nom_var: str,
        delta: float,
        posx_label: float,
        conv: Mapping[Tuple[str, int], str],
    ) -> Tuple[float, List[Tuple[float, str]]]:
        """
        affiche 1, 2 ou 3 bar pour chaque variable
        renvoie la dernière position où un bar horizontale est positionnée
        """
        ret: List[Tuple[float, str]] = []
        res_m, res_z, res_p = self.Courbes.get(nom_var, (None, None, None))
        desi = self.E.latex_indexes.get(nom_var, nom_var)
        if any((b is not None) for b in (res_m, res_z, res_p)):
            t1 = ax.text(
                posx_label,
                pos,
                desi,
                size="small",
                verticalalignment="center",
                backgroundcolor="w",
                bbox=dict(facecolor="w", alpha=0.5, mutation_aspect=0.5),
            )
            t1.set_linespacing(1.0)
            pos += 1.35 * delta
        if res_p is not None:
            lab = conv.get((nom_var, 1), res_p.value)
            res_p.plot_single_barh(ax, pos, "dodgerblue", lab)
            ret.append((pos, lab))
            pos += delta
        if res_z is not None:
            lab = conv.get((nom_var, 0), res_z.value)
            res_z.plot_single_barh(ax, pos, "lightcoral", lab)
            ret.append((pos, lab))
            pos += delta
        if res_m is not None:
            lab = conv.get((nom_var, -1), res_m.value)
            res_m.plot_single_barh(ax, pos, "yellowgreen", lab)
            ret.append((pos, lab))
            pos += delta
        return (pos, ret)

    def plot_barh_sample_some(
        self,
        ax: Any,
        delta: float,
        posy_label: float,
        v_names: Sequence[str],
        conv: Mapping[Tuple[str, int], str],
    ) -> List[Tuple[float, str]]:
        """
        affiche 1, 2 ou 3 bar pour chaque variable
        renvoie la dernière position où un bar horizontale est positionnée
        """
        pos = 0.0
        ret: List[Tuple[float, str]] = []
        for nom_var in v_names[::-1]:
            pos, ret0 = self.plot_barh_sample(ax, pos, nom_var, delta, posy_label, conv)
            pos += 0.85 * delta
            ret.extend(ret0)
        return ret

    def plot_barh_sample_all(
        self,
        ax: Any,
        delta: float,
        posy_label: float,
        conv: Mapping[Tuple[str, int], str],
    ) -> List[Tuple[float, str]]:
        """
        affiche 1, 2 ou 3 bar pour chaque variable
        renvoie la dernière position où un bar horizontale est positionnée
        """
        pos = 0.0
        ret: List[Tuple[float, str]] = []
        for nom_var in self.E.indexes:
            pos, ret0 = self.plot_barh_sample(ax, pos, nom_var, delta, posy_label, conv)
            pos += 1.05 * delta
            ret.extend(ret0)
        return ret

    def plot_barh_pair(
        self,
        ax: Any,
        pos: float,
        nom_var: str,
        delta: float,
        posx_label: float,
        conv: Mapping[Tuple[str, int], str],
    ) -> Tuple[float, List[Tuple[float, str]]]:
        """
        affiche 1, 2 ou 3 bar pour chaque variable
        renvoie la dernière position où un bar horizontale est positionnée
        """
        ret: List[Tuple[float, str]] = []
        cmp = self.Courbes_ddd.get((nom_var, -1, 1), None)
        desi = self.E.latex_indexes.get(nom_var, nom_var)
        if cmp is not None:
            lab1 = conv.get((nom_var, -1), cmp.value1)
            lab2 = conv.get((nom_var, 1), cmp.value2)
            size_lab = max((len(lab1), len(lab2))) - 2
            lab = f"{lab1}\n->" + " " * size_lab + f"\n{lab2}"
            t1 = ax.text(
                posx_label,
                pos,
                desi,
                size="small",
                verticalalignment="center",
                bbox=dict(facecolor="w", alpha=0.5, mutation_aspect=0.5),
            )
            pos += 1.25 * delta
            cmp.plot_single_barh(ax, pos, "magenta", lab)
            ret.append((pos, lab))
            pos += delta
        return (pos, ret)

    def plot_barh_pair_some(
        self,
        ax: Any,
        delta: float,
        posy_label: float,
        v_names: Sequence[str],
        conv: Mapping[Tuple[str, int], str],
    ) -> List[Tuple[float, str]]:
        """
        affiche 1, 2 ou 3 bar pour chaque variable
        renvoie la dernière position où un bar horizontale est positionnée
        """
        pos = 0.0
        ret: List[Tuple[float, str]] = []
        for nom_var in v_names[::-1]:
            pos, ret0 = self.plot_barh_pair(ax, pos, nom_var, delta, posy_label, conv)
            pos += 1.05 * delta
            ret.extend(ret0)
        ax.axvline(x=0, color="red", linewidth=1)
        ax.text(
            0.0,
            -delta / 2.0,
            "0",
            size="x-large",
            color="red",
            va="center",
            ha="center",
        )
        return ret


if __name__ == "__main__":
    """
    permet de générer le tableau des valeur du definitive screening
    """
    Path("def_src_int.txt").write_text("{}".format(genere_def_scr_int(def_scr)))
