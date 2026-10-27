import pytest

from backend.db import init_db


@pytest.fixture
def conn():
    c = init_db(":memory:")
    yield c
    c.close()
