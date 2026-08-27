import pytest
from pfas.model import Model


@pytest.fixture
def Field():
    """Minimal field object with a default attribute."""
    class F:
        def __init__(self, default=None):
            self.default = default
    return F


@pytest.fixture
def dummyA(Field):
    """Dummy component A producing x."""
    class A:
        model_fields = {"a": Field()}
        def __init__(self, a):
            self.a = a
        def compute(self):
            return {"x": self.a + 1}
        @property
        def outputs(self):
            return ["x"]
    return A


@pytest.fixture
def dummyB(Field):
    """Dummy component B consuming x and producing y."""
    class B:
        model_fields = {"x": Field()}
        def __init__(self, x):
            self.x = x
        def compute(self):
            return {"y": self.x * 2}
        @property
        def outputs(self):
            return ["y"]
    return B


@pytest.fixture
def dummyC(Field):
    """Dummy component C consuming x and y and producing z."""
    class C:
        model_fields = {"x": Field(), "y": Field()}
        def __init__(self, x, y):
            self.x = x
            self.y = y
        def compute(self):
            return {"z": self.x + self.y}
        @property
        def outputs(self):
            return ["z"]
    return C


def test_model_basic_chain(dummyA, dummyB, dummyC):
    """Test basic chaining of components."""
    m = Model()
    m.compute(dummyA, a=1)
    m.compute(dummyB)
    m.compute(dummyC)
    assert m.x == 2
    assert m.y == 4
    assert m.z == 6


def test_model_kwargs_override(dummyA, dummyB):
    """Test overriding values via kwargs."""
    m = Model()
    m.compute(dummyA, a=10)
    m.compute(dummyB, x=5)
    assert m.x == 5
    assert m.y == 22


def test_model_missing_argument_error(dummyA):
    """Test missing required argument raises error."""
    m = Model()
    with pytest.raises(TypeError):
        m.compute(dummyA)


def test_model_unknown_kwargs(dummyA):
    """Test unknown kwargs raise error."""
    m = Model()
    with pytest.raises(ValueError):
        m.compute(dummyA, unknown=123)


def test_model_generated_data_persists(dummyA, dummyB):
    """Test generated data persists across components."""
    m = Model()
    m.compute(dummyA, a=3)
    assert m.x == 4
    m.compute(dummyB)
    assert m.y == 8


def test_model_getattr_access(dummyA):
    """Test attribute access via __getattr__."""
    m = Model()
    m.compute(dummyA, a=2)
    assert m.x == 3
    assert m.generated_data["x"] == 3

def test_model_find_add_components_chain(register_chain):
    """Test _find_add_components recurses with dummy chain."""
    from pfas.model import Model
    m = Model()
    missing = {"y"}
    all_keys = set()
    with pytest.raises(RecursionError):
        list(m._find_add_components(missing, all_keys))

