# Pytest primer & alphaBT tests walkthrough

A short intro to **pytest** and how the tests in this folder are structured.

---

## 1. Pytest in 5 minutes

### What is pytest?

**pytest** is a test framework for Python. You write normal Python functions (or methods) and use `assert` to check behavior. Pytest discovers and runs them, and reports which passed or failed.

### Basic rules

| Rule | Meaning |
|------|--------|
| **Discovery** | Files named `test_*.py` or `*_test.py` are test modules. Functions named `test_*` are tests. |
| **No boilerplate** | You don’t subclass anything; just define `def test_something():` and use `assert`. |
| **Assert** | Use `assert condition, "optional message"`. If the condition is false, the test fails. |

### Minimal example

```python
# test_example.py
def test_addition():
    assert 1 + 1 == 2

def test_list_contains():
    items = ["a", "b", "c"]
    assert "b" in items
```

Run from project root:

```bash
pytest src/tests/test_example.py -v
```

- `-v` = verbose (shows each test name).
- Omit the file to run all tests under `src/tests`.

### Testing that something raises an error

Use `pytest.raises` so the test **expects** an exception and fails if it doesn’t happen:

```python
import pytest

def test_divide_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        1 / 0
```

To also check the error message:

```python
with pytest.raises(ValueError, match="Missing required columns"):
    some_function(bad_input)
```

### Testing that a warning is emitted

```python
def test_warns_when_k_too_large():
    with pytest.warns(UserWarning):
        result = select_top_k_stocks(df, k=100)
```

### Running tests

From repo root (where `pyproject.toml` is):

```bash
# All tests
pytest

# One file
pytest src/tests/test_stock_selection.py

# One class
pytest src/tests/test_stock_selection.py::TestSelectTopKStocks

# One test
pytest src/tests/test_stock_selection.py::TestSelectTopKStocks::test_selects_correct_count

# Verbose
pytest -v
```

Your pytest config in `pyproject.toml` sets:

- `testpaths = ["src/tests"]` → only `src/tests` is collected.
- `pythonpath = ["src"]` → `src` is on `PYTHONPATH`, so imports like `from selection.stock_selection import ...` work.

---

## 2. Fixtures: shared test data and setup

A **fixture** is a function that provides something (data, config, client) to tests. You ask for it by name as an argument; pytest calls the function and passes the result in.

### Defining fixtures

In **conftest.py** (or any test file), use `@pytest.fixture`:

```python
@pytest.fixture
def sample_input_data():
    """DataFrame with 9 stocks × 2 quarters."""
    rows = []
    for q in [Q1, Q2]:
        for name in STOCK_NAMES:
            rows.append({"quarter": q, "co_name": name, ...})
    return pd.DataFrame(rows)
```

### Using fixtures in tests

Use the **fixture name as the parameter**; pytest injects the return value:

```python
def test_something(sample_input_data):
    # sample_input_data is the DataFrame from the fixture
    result = select_top_k_stocks(sample_input_data, k=5)
    assert len(result) == 10  # 5 per quarter × 2 quarters
```

### Fixtures using other fixtures

A fixture can depend on another by taking it as an argument:

```python
@pytest.fixture
def base_config(minimal_index_exit_config):
    return {
        ...
        "index_exit": minimal_index_exit_config,
    }
```

Pytest builds the dependency graph and runs fixtures in the right order.

### Why conftest.py?

**conftest.py** is special: pytest loads it automatically and makes its fixtures available to **all** test modules in that directory (and below). So shared data (sample DataFrames, configs, etc.) lives in `conftest.py` and any test can request it by name.

---

## 3. Organizing tests: classes and parametrize

### Test classes (optional grouping)

Tests can be grouped in classes. The class name doesn’t need to start with `Test` for discovery (only the method names do), but many projects use `TestXxx` for clarity:

```python
class TestSelectTopKStocks:
    def test_selects_correct_count(self):
        ...

    def test_equal_weights(self):
        ...
```

This keeps related tests together and makes it easy to run one group:

```bash
pytest src/tests/test_stock_selection.py::TestSelectTopKStocks
```

### Parametrize: one test, many inputs

Use `@pytest.mark.parametrize` to run the **same test** with different arguments:

```python
@pytest.mark.parametrize("name,expected_type", [
    ("TPE", optuna.samplers.TPESampler),
    ("Random", optuna.samplers.RandomSampler),
    ("CmaEs", optuna.samplers.CmaEsSampler),
])
def test_valid_samplers(self, name, expected_type):
    sampler = get_sampler(name)
    assert isinstance(sampler, expected_type)
```

This runs `test_valid_samplers` three times (once per row). No need to write three separate test methods.

---

## 4. What’s in this repo’s tests

### conftest.py — shared fixtures

- **Constants**: `Q1`, `Q2`, date ranges, `STOCK_NAMES`, `MCAP_CATEGORIES`, `VOLATILITIES`, `PROBABILITIES`.
- **Input/price/index data**: `sample_input_data`, `sample_input_data_volatility`, `sample_price_data`, `sample_index_data` — all deterministic (no randomness, no real files).
- **Preselected portfolios**: `sample_preselected_data`, `sample_preselected_no_category`.
- **Config**: `minimal_index_exit_config`, `base_config` (minimal valid backtest config).
- **Reporting**: `constant_portfolio`, `linear_growth_portfolio`, `drawdown_portfolio`, `comparison_df_identical`, `daily_returns_df`, `monthly_returns_df`, `trade_results_with_mcap`, etc.
- **Other**: `price_data_for_atr`, `index_data_for_volatility`, `sized_trades` for simulation.

Any test in `src/tests` can use these by adding the fixture name as an argument (e.g. `def test_foo(sample_input_data, base_config):`).

### Test modules (by area)

| File | What it tests |
|------|----------------|
| **test_stock_selection.py** | `select_top_k_stocks`, `get_entry_start_date`, `validate_price_data_coverage`, `filter_tradeable_stocks`. |
| **test_validate_preselected.py** | `validate_preselected_input`: required columns, tiered mode + category, weight/category warnings. |
| **test_config_schema.py** | Pydantic `BacktestConfig` (and related models): defaults, roundtrip, validation errors, typos. |
| **test_run_tuning.py** | `DataCache`, `_parse_objective_config`, `get_sampler` (including parametrized sampler types). |
| **test_tpsl.py** | TP/SL logic (flat/tiered, index exit config validation). |
| **test_backtest_core.py** / **test_backtest_core_integration.py** | Core backtest and integration with config/data. |
| **test_reporting_metrics.py** / **test_reporting_analytics.py** | Reporting metrics and analytics (CAGR, drawdown, etc.). |
| **test_dynamic_levels.py** | Dynamic levels (e.g. ATR-based). |
| **test_simulation.py** | Simulation with position sizing. |
| **test_charts.py** | Chart generation. |
| **test_export_config.py** | Config export. |
| **test_analyse_target.py** | Target analysis. |

### Patterns you’ll see

1. **Happy path**: call the function with valid input (often from a fixture or a small helper like `_make_input_data`), assert on the result (length, columns, values).
2. **Error path**: `with pytest.raises(ValueError, match="...")` when invalid input or config should raise.
3. **Warnings**: `with pytest.warns(UserWarning):` or `warnings.catch_warnings` to assert that a warning is (or isn’t) emitted.
4. **Fixtures from conftest**: e.g. `sample_input_data`, `base_config`, `sample_price_data` used as arguments.
5. **Local helpers**: e.g. `_make_input_data()`, `_make_preselected()` in the test file to build minimal inputs for that module only.
6. **Parametrize**: e.g. in `test_run_tuning.py` for multiple sampler names and expected types.

---

## 5. Quick reference

| Goal | How |
|------|-----|
| Run all tests | `pytest` |
| Run one file | `pytest src/tests/test_stock_selection.py` |
| Run one test | `pytest src/tests/test_stock_selection.py::TestSelectTopKStocks::test_selects_correct_count` |
| Verbose | `pytest -v` |
| Expect exception | `with pytest.raises(ValueError, match="..."):` |
| Expect warning | `with pytest.warns(UserWarning):` |
| Use shared data | Add fixture name as argument: `def test_foo(sample_input_data):` |
| Same test, many inputs | `@pytest.mark.parametrize("arg1,arg2", [(a1, a2), ...])` |

Once you’re comfortable with these, you can read any `test_*.py` file and follow along: look at the fixture names in the function arguments, then the asserts and `pytest.raises` / `pytest.warns` blocks to see what behavior is being tested.
