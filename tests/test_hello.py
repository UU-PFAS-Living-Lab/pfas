from pytest import mark

from pfas import hello_world

@mark.parametrize("target", ("you", "world", "me"))
def test_hello(target):
    hello_world(target)
