import pytest
from pfas import model as m


class InputSet(set):
    def __radd__(self, other):
        return set(other) | set(self)


@pytest.fixture
def Field():
    class F:
        def __init__(self, default=None):
            self.default = default
    return F


@pytest.fixture
def producerX(Field):
    """Producer of x."""
    class P:
        model_fields = {}
        outputs = ["x"]
        inputs = InputSet()
        def compute(self):
            return {"x": 1}
    return P


@pytest.fixture
def producerY(Field):
    """Producer of y from x."""
    class Q:
        model_fields = {"x": Field()}
        outputs = ["y"]
        inputs = InputSet({"x"})
        def __init__(self, x):
            self.x = x
        def compute(self):
            return {"y": self.x + 1}
    return Q


@pytest.fixture
def register_chain(producerX, producerY):
    old = list(m.ALL_COMPONENTS)
    m.ALL_COMPONENTS.clear()
    m.ALL_COMPONENTS.extend([producerX, producerY])
    yield
    m.ALL_COMPONENTS.clear()
    m.ALL_COMPONENTS.extend(old)

