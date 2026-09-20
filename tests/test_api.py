from python_checker import check_source
from python_checker.report import Status


def test_shape_mismatch_detected():
    code = """
import torch
a = torch.randn(2, 3)
b = torch.randn(4, 5)
c = torch.matmul(a, b)
"""
    result = check_source(code, packs=["pytorch"], ruff=False)
    assert result.status == Status.ERROR
    assert any(f.code == "SHAPE_MISMATCH" for f in result.findings)


def test_openai_missing_messages():
    code = """
from openai import OpenAI
client = OpenAI()
client.chat.completions.create(model="gpt-4o")
"""
    result = check_source(code, packs=["openai"], ruff=False)
    assert any(f.code == "MISSING_KWARG" for f in result.findings)


def test_catalog_miss_uncovered():
    code = """
import torch
x = torch.randn(2, 3)
y = torch.special.gammaln(x)
"""
    result = check_source(code, packs=["pytorch"], ruff=False)
    assert result.status == Status.UNCOVERED
    assert any(f.code == "CATALOG_MISS" for f in result.findings)


def test_forbidden_eval():
    code = "eval('1+1')"
    result = check_source(code, packs=[], ruff=False)
    assert any(f.code == "FORBIDDEN_CALL" for f in result.findings)
