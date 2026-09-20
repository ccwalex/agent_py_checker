from python_checker import check_source
from python_checker.report import Status


def test_for_iter_ndarray_rejected():
    code = """
import numpy as np
arr = np.zeros(3)
for k in arr:
    print(k)
"""
    result = check_source(code, packs=["numpy"], ruff=False)
    assert result.status == Status.ERROR
    assert any(f.code == "NATURE_MISMATCH" and f.rule_id == "ops.for_iter" for f in result.findings)


def test_for_iter_unknown_list_ok():
    code = """
items = [1, 2, 3]
for k in items:
    print(k)
"""
    result = check_source(code, packs=["numpy"], ruff=False)
    assert not any(f.rule_id == "ops.for_iter" for f in result.findings)


def test_for_iter_tensor_rejected():
    code = """
import torch
t = torch.randn(3)
for k in t:
    print(k)
"""
    result = check_source(code, packs=["pytorch"], ruff=False)
    assert result.status == Status.ERROR
    assert any(f.code == "NATURE_MISMATCH" and f.rule_id == "ops.for_iter" for f in result.findings)


def test_ndarray_into_torch_matmul_rejected():
    code = """
import numpy as np
import torch
arr = np.zeros((2, 3))
t = torch.randn(3, 4)
out = torch.matmul(arr, t)
"""
    result = check_source(code, packs=["numpy", "pytorch"], ruff=False)
    assert result.status == Status.ERROR
    assert any(
        f.code == "NATURE_MISMATCH" and "torch.matmul" in f.rule_id for f in result.findings
    )


def test_from_numpy_accepts_ndarray():
    code = """
import numpy as np
import torch
arr = np.zeros(3)
t = torch.from_numpy(arr)
"""
    result = check_source(code, packs=["numpy", "pytorch"], ruff=False)
    assert not any(
        f.code == "NATURE_MISMATCH" and "from_numpy" in f.rule_id for f in result.findings
    )


def test_tensor_view_receiver_checked():
    code = """
import numpy as np
arr = np.zeros((2, 3))
arr.view(6)
"""
    result = check_source(code, packs=["numpy"], ruff=False)
    assert not any(f.code == "NATURE_MISMATCH" and "view" in f.rule_id for f in result.findings)


def test_tensor_view_ok():
    code = """
import torch
t = torch.randn(2, 3)
t.view(6)
"""
    result = check_source(code, packs=["pytorch"], ruff=False)
    assert not any(
        f.code == "NATURE_MISMATCH" and f.rule_id == "pytorch.torch.Tensor.view"
        for f in result.findings
    )


def test_for_loop_target_bound_as_scalar():
    code = """
import numpy as np
arr = np.zeros(3)
for k in arr:
    _ = k
"""
    from python_checker.facts import FactsStore, Nature
    from python_checker.parse import parse_source
    from python_checker.passes.imports import collect_import_facts
    from python_checker.passes.natures import propagate_natures
    from python_checker.passes.variables import check_variables

    parsed = parse_source(code)
    facts = FactsStore()
    collect_import_facts(parsed, facts)
    propagate_natures(parsed, facts, ["numpy"])
    check_variables(parsed, facts, ["numpy"])
    assert facts.symbols["k"].nature == Nature.SCALAR
