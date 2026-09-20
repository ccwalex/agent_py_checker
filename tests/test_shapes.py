from python_checker.shapes.dims import Dim, Shape, unify_shapes
from python_checker.shapes.transfers import apply_transfer


def test_unify_concrete_dims():
    left = Shape((Dim.concrete(3), Dim.concrete(4)))
    right = Shape((Dim.concrete(3), Dim.concrete(4)))
    result = unify_shapes(left, right)
    assert result.ok
    assert result.shape == left


def test_matmul_shape():
    a = Shape((Dim.concrete(2), Dim.concrete(3)))
    b = Shape((Dim.concrete(3), Dim.concrete(5)))
    result = apply_transfer("matmul", [a, b], {})
    assert result.ok
    assert str(result.shape) == "(2, 5)"


def test_matmul_mismatch():
    a = Shape((Dim.concrete(2), Dim.concrete(3)))
    b = Shape((Dim.concrete(4), Dim.concrete(5)))
    result = apply_transfer("matmul", [a, b], {})
    assert not result.ok
    assert not result.uncovered
