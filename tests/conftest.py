import os

os.environ["CERES_DISABLE_BG"] = "1"  # no background pipeline task in tests (set before backend.app is imported)

import pytest

from backend.db import init_db


@pytest.fixture
def conn():
    c = init_db(":memory:")
    yield c
    c.close()
