import pytest
from pydantic import BaseModel

from pfas.model import Model


class ComponentAB(BaseModel, validate_assignment=True, extra="forbid"):
    a: int

    @property
    def outputs(self):
        return ["b"]

    def compute(self):
        return {"b": self.a + 1}

class ComponentBC(BaseModel, validate_assignment=True, extra="forbid"):
    b: int

    def compute(self):
        return {"c": self.b + 1}

    @property
    def outputs(self):
        return ["c"]

class ComponentBD(BaseModel, validate_assignment=True, extra="forbid"):
    b: int

    def compute(self):
        return {"d": self.b + 2}

    @property
    def outputs(self):
        return ["d"]

class ComponentBCE(BaseModel, validate_assignment=True, extra="forbid"):
    b: int
    c: int = 10

    def compute(self):
        return {"e": self.b*self.c}

    @property
    def outputs(self):
        return ["e"]


class ComponentCDE(BaseModel, validate_assignment=True, extra="forbid"):
    c: int
    d: int

    def compute(self):
        return {"e": self.c + self.d}

    @property
    def outputs(self):
        return ["e"]

def test_model_basic_chain():
    m = Model()
    m.compute(ComponentAB, a=0)
    m.compute(ComponentBC)
    m.compute(ComponentCDE, d=1)
    assert m.b == 1
    assert m.c == 2
    assert m.e == 3
    assert "c" in m.generated_data

def test_model_refuse_inputs():
    m = Model()
    m.compute(ComponentAB, a=0)
    with pytest.raises(ValueError):
        m.compute(ComponentBC, a=0, b=1)

def test_model_missing_args():
    m = Model()
    with pytest.raises(TypeError):
        m.compute(ComponentAB)

def test_model_unknown_kwargs():
    m = Model()
    with pytest.raises(ValueError):
        m.compute(ComponentAB, a=0, abcd=1234)

    with pytest.raises(ValueError):
        m.compute(ComponentAB, a=0)
        m.compute(ComponentBC, a=1)

def test_multiple_definition():
    m = Model()
    m.compute(ComponentAB, a=1)
    with pytest.raises(ValueError):
        m.compute(ComponentBC, b=1)


def test_model_defaults():
    m = Model()

    m.compute(ComponentBCE, b=1)
    assert m.c == 10
    assert m.e == 10

    m = Model()
    m.compute(ComponentBCE, b=1, c=1)
    assert m.c == 1
    assert m.e == 1

    m = Model()
    m.compute(ComponentAB, a=1)
    m.compute(ComponentBC)
    m.compute(ComponentBCE)
    assert m.b == 2
    assert m.c == 3
    assert m.e == 6

