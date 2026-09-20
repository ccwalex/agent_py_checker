from python_checker import check_source
from python_checker.facts import FactsStore, Nature
from python_checker.parse import parse_source
from python_checker.passes.imports import collect_import_facts
from python_checker.passes.natures import propagate_natures
from python_checker.report import Status


def test_list_to_torch_tensor_ok():
    code = """
import torch
t = torch.tensor([1, 2, 3])
"""
    result = check_source(code, packs=["pytorch"], ruff=False)
    assert not any(
        f.code == "NATURE_MISMATCH" and "torch.tensor" in f.rule_id for f in result.findings
    )


def test_dataframe_to_from_numpy_rejected():
    code = """
import pandas as pd
import torch
df = pd.DataFrame([1, 2, 3])
t = torch.from_numpy(df)
"""
    result = check_source(code, packs=["pandas", "pytorch"], ruff=False)
    assert result.status == Status.ERROR
    assert any(
        f.code == "NATURE_MISMATCH" and "from_numpy" in f.rule_id for f in result.findings
    )


def test_tensor_numpy_propagates_ndarray():
    code = """
import torch
t = torch.randn(3)
arr = t.numpy()
"""
    parsed = parse_source(code)
    facts = FactsStore()
    collect_import_facts(parsed, facts)
    propagate_natures(parsed, facts, ["pytorch"])
    assert facts.symbols["arr"].nature == Nature.NDARRAY


def test_dataframe_values_propagates_ndarray():
    code = """
import pandas as pd
df = pd.DataFrame([1, 2, 3])
arr = df.values
"""
    parsed = parse_source(code)
    facts = FactsStore()
    collect_import_facts(parsed, facts)
    propagate_natures(parsed, facts, ["pandas"])
    assert facts.symbols["arr"].nature == Nature.NDARRAY


def test_dataframe_values_into_torch_matmul_rejected():
    code = """
import pandas as pd
import torch
df = pd.DataFrame([1, 2, 3])
t = torch.randn(3, 4)
out = torch.matmul(df.values, t)
"""
    result = check_source(code, packs=["pandas", "pytorch"], ruff=False)
    assert result.status == Status.ERROR
    assert any(f.code == "NATURE_MISMATCH" and "matmul" in f.rule_id for f in result.findings)


def test_dataframe_to_numpy_then_from_numpy_ok():
    code = """
import pandas as pd
import torch
df = pd.DataFrame([1, 2, 3])
arr = df.to_numpy()
t = torch.from_numpy(arr)
"""
    result = check_source(code, packs=["pandas", "pytorch"], ruff=False)
    assert not any(
        f.code == "NATURE_MISMATCH"
        and ("from_numpy" in f.rule_id or "to_numpy" in f.rule_id)
        for f in result.findings
    )


def test_ndarray_to_dataframe_ok():
    code = """
import numpy as np
import pandas as pd
arr = np.array([1, 2, 3])
df = pd.DataFrame(arr)
"""
    result = check_source(code, packs=["numpy", "pandas"], ruff=False)
    assert not any(
        f.code == "NATURE_MISMATCH" and "DataFrame" in f.rule_id for f in result.findings
    )


def test_list_literal_has_list_nature():
    code = """
items = [1, 2, 3]
"""
    parsed = parse_source(code)
    facts = FactsStore()
    collect_import_facts(parsed, facts)
    propagate_natures(parsed, facts, [])
    assert facts.symbols["items"].nature == Nature.LIST


def test_tensor_to_asarray_ok():
    code = """
import numpy as np
import torch
t = torch.randn(3)
arr = np.asarray(t)
"""
    result = check_source(code, packs=["numpy", "pytorch"], ruff=False)
    assert not any(
        f.code == "NATURE_MISMATCH" and "asarray" in f.rule_id for f in result.findings
    )


def _nature_after_propagate(code: str, symbol: str, packs: list[str]) -> Nature:
    parsed = parse_source(code)
    facts = FactsStore()
    collect_import_facts(parsed, facts)
    propagate_natures(parsed, facts, packs)
    return facts.symbols[symbol].nature


def test_chained_cpu_numpy_propagates_ndarray():
    code = """
import torch
t = torch.randn(3)
arr = t.cpu().numpy()
"""
    assert _nature_after_propagate(code, "arr", ["pytorch"]) == Nature.NDARRAY


def test_chained_detach_cpu_numpy_propagates_ndarray():
    code = """
import torch
t = torch.randn(3)
arr = t.detach().cpu().numpy()
"""
    assert _nature_after_propagate(code, "arr", ["pytorch"]) == Nature.NDARRAY


def test_tensor_item_propagates_scalar():
    code = """
import torch
t = torch.tensor(1.0)
x = t.item()
"""
    assert _nature_after_propagate(code, "x", ["pytorch"]) == Nature.SCALAR


def test_ndarray_item_propagates_scalar():
    code = """
import numpy as np
arr = np.array(1.0)
x = arr.item()
"""
    assert _nature_after_propagate(code, "x", ["numpy"]) == Nature.SCALAR


def test_ndarray_tolist_propagates_list():
    code = """
import numpy as np
arr = np.array([1, 2, 3])
items = arr.tolist()
"""
    assert _nature_after_propagate(code, "items", ["numpy"]) == Nature.LIST


def test_dataframe_tolist_propagates_list():
    code = """
import pandas as pd
df = pd.DataFrame([1, 2, 3])
items = df.tolist()
"""
    assert _nature_after_propagate(code, "items", ["pandas"]) == Nature.LIST


def test_scalar_into_torch_matmul_rejected():
    code = """
import torch
t = torch.tensor(1.0)
x = t.item()
out = torch.matmul(x, torch.randn(1, 2))
"""
    result = check_source(code, packs=["pytorch"], ruff=False)
    assert result.status == Status.ERROR
    assert any(f.code == "NATURE_MISMATCH" and "matmul" in f.rule_id for f in result.findings)
