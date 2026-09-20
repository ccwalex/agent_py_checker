from python_checker.facts import FactsStore, Nature
from python_checker.parse import parse_source
from python_checker.passes.imports import collect_import_facts
from python_checker.passes.natures import propagate_natures


def test_import_alias_and_nature():
    code = """
import pandas as pd
df = pd.read_csv("data.csv")
"""
    parsed = parse_source(code)
    facts = FactsStore()
    collect_import_facts(parsed, facts)
    propagate_natures(parsed, facts, ["pandas"])
    assert facts.import_map["pd"] == "pandas"
    assert facts.symbols["df"].nature == Nature.DATAFRAME
