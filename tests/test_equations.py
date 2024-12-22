import logging

import pytest

from src.equations import Equations


def test_Equations_1():
    with pytest.raises(ValueError, match="indexes is empty"):
        E = Equations((), 0)


def test_Equations_2():
    with pytest.raises(ValueError, match="-1 : Wrong value for order"):
        E = Equations((1,), -1)


def test_Equations_3(caplog):
    with caplog.at_level(logging.INFO, logger="src.equations"):
        E = Equations((1,), 0)
        assert caplog.record_tuples == [
            ("src.equations", logging.WARNING, "order is 0")
        ]
