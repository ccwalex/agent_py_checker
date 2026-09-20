# Python Checker

Deterministic Python checker for agent gates and local CLI use.

## Features

- **Ruff integration** for generic lint (`F`, `E9`)
- **Library rule packs** for numpy, pandas, sklearn, openai, fastapi, pytorch
- **Static PyTorch shape tracking** via symbolic dimensions
- **Ternary verdicts**: `ok`, `error`, `uncovered` (escalate)

## Agent gate

- Any `error` finding → **reject**
- Any `uncovered` finding (and no errors) → **escalate**
- Otherwise → **accept**

CLI exit codes: `0` ok, `1` error, `2` uncovered-only.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
python-checker script.py --format json
python-checker script.py --format text --no-ruff
python-checker script.py --packs pytorch,numpy
```

```python
from python_checker import check_source

result = check_source(code, packs=["pytorch"], ruff=True)
print(result.status, result.summary)
```

## Adding a rule

Add YAML under `python_checker/rules/<pack>/`:

```yaml
api: pandas.read_csv
effects:
  nature: DataFrame
required_kwargs:
  - filepath_or_buffer
```

Nature matching fields:

- `kind`: `call` (default) or structural ops like `for_iter`
- `accepts_nature`: per-slot allowlists, e.g. `[[Tensor], [Tensor]]` for two args
- `forbids_nature`: denylist for structural rules (see `rules/ops/for_iter.yaml`)
- `requires_nature`: legacy sugar for `accepts_nature: [[Tensor], ...]`

Example — forbid iterating array-like values:

```yaml
api: for_iter
kind: for_iter
forbids_nature:
  - NdArray
  - Tensor
  - DataFrame
  - Series
```

Example — conversion API with explicit allowed input nature:

```yaml
api: torch.from_numpy
kind: call
accepts_nature:
  - [NdArray]
effects:
  nature: Tensor
```

Conversion rules also cover attribute accessors (`kind: attr`), chained tensor calls (`.cpu().numpy()` via passthrough rules), and scalar/list extractors (`.item()`, `.tolist()`). The `conversions` rule pack is always loaded.

Supported natures include `List`, `Scalar`, `NdArray`, `Series`, `DataFrame`, and `Tensor`.

Optional fields: `shape`, `construct`, `call`.

## Ruff mapping

Ruff diagnostics are mapped to findings with `rule_id: ruff.<CODE>` and never marked `uncovered`.
