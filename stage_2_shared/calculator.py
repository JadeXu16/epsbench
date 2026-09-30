"""Safe arithmetic DAG executor for dynamic EPS derivation."""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set


_ALLOWED_OPS = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def _safe_eval(expr: str, env: dict[str, float]) -> float:
    """Evaluate a pure arithmetic expression with variable substitution."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {expr!r}") from e

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Non-numeric constant: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"Unknown variable: {node.id!r}")
            return float(env[node.id])
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_OPS):
                raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
            left  = _eval(node.left)
            right = _eval(node.right)
            op = node.op
            if isinstance(op, ast.Add):  return left + right
            if isinstance(op, ast.Sub):  return left - right
            if isinstance(op, ast.Mult): return left * right
            if isinstance(op, ast.Div):
                if right == 0:
                    raise ValueError("Division by zero")
                return left / right
            if isinstance(op, ast.Pow):  return left ** right
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_OPS):
                raise ValueError(f"Disallowed unary op: {type(node.op).__name__}")
            val = _eval(node.operand)
            if isinstance(node.op, ast.USub): return -val
            if isinstance(node.op, ast.UAdd): return +val
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    return _eval(tree)


def _referenced_names(expr: str) -> Set[str]:
    """Return all Name nodes referenced in an arithmetic expression."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _topological_sort(formulas: dict[str, str], leaf_names: Set[str]) -> List[str]:
    """Return formula keys in evaluation order (dependencies first)."""
    all_known = set(leaf_names) | set(formulas.keys())

    for name, expr in formulas.items():
        for ref in _referenced_names(expr):
            if ref not in all_known:
                raise ValueError(
                    f"Formula '{name}' references unknown variable '{ref}'. "
                    f"Known: {sorted(all_known)}"
                )

    deps: dict[str, Set[str]] = {
        name: _referenced_names(expr) - leaf_names
        for name, expr in formulas.items()
    }
    order: List[str] = []
    ready = [n for n, d in deps.items() if not d]

    while ready:
        node = ready.pop()
        order.append(node)
        for other, other_deps in deps.items():
            if node in other_deps:
                other_deps.discard(node)
                if not other_deps:
                    ready.append(other)

    if len(order) != len(formulas):
        cycle_nodes = set(formulas) - set(order)
        raise ValueError(f"Circular dependency detected among: {sorted(cycle_nodes)}")

    return order


def calculate(
    formulas: Dict[str, str],
    variables: Dict[str, float],
) -> Dict[str, float]:
    """Execute the formula DAG and return all computed values."""
    env: dict[str, float] = {k: float(v) for k, v in variables.items()}
    leaf_vars = set(env.keys()) - set(formulas.keys())
    order = _topological_sort(formulas, leaf_vars)

    for name in order:
        env[name] = _safe_eval(formulas[name], env)

    return env


def calculate_from_prediction(
    prediction: Dict[str, float],
    schema: dict,
) -> Dict[str, float]:
    """Resolve agent predictions → absolute variable values → run the DAG."""
    baseline: Dict[str, float] = {k: float(v) for k, v in schema["variables_baseline"].items()}
    pred_schema: dict = schema.get("prediction_schema", {})
    static_rules: dict = schema.get("static_rules", {})
    formulas: dict = schema.get("formulas", {})

    baseline_full: Dict[str, float] = dict(baseline)
    _changed = True
    while _changed:
        _changed = False
        for _var, _expr in formulas.items():
            if _var not in baseline_full:
                try:
                    baseline_full[_var] = float(eval(_expr, {"__builtins__": {}}, baseline_full))
                    _changed = True
                except Exception:
                    pass

    resolved: Dict[str, float] = {}

    def _propagate_formulas() -> None:
        """Iteratively evaluate formula nodes until no new nodes can be resolved."""
        env = {**baseline, **resolved}
        changed = True
        while changed:
            changed = False
            for var, expr in formulas.items():
                if var in resolved:
                    continue
                try:
                    env[var] = float(eval(expr, {"__builtins__": {}}, env))
                    resolved[var] = env[var]
                    changed = True
                except Exception:
                    pass

    for field_name, meta in pred_schema.items():
        if meta.get("type") != "yoy":
            continue
        yoy = float(prediction.get(field_name, 0.0))
        constituents = meta.get("constituents") or [meta["variable"]]
        for c in constituents:
            base_val = baseline.get(c, 0.0)
            resolved[c] = base_val * (1.0 + yoy)

    for field_name, meta in pred_schema.items():
        if meta.get("type") != "rate_override":
            continue
        var = meta["variable"]
        default = meta.get("default", baseline.get(var, 0.0))
        value = prediction.get(field_name)
        resolved[var] = float(value) if value is not None else float(default)

    for field_name, meta in pred_schema.items():
        if meta.get("type") != "absolute":
            continue
        var = meta["variable"]
        value = prediction.get(field_name)
        resolved[var] = float(value) if value is not None else float(baseline.get(var, 0.0))

    _propagate_formulas()

    for field_name, meta in pred_schema.items():
        if meta.get("type") != "bps":
            continue
        anchor = meta["anchor"]
        delta_bps = float(prediction.get(field_name, 0.0))
        anchor_base = baseline_full.get(anchor, 0.0)
        anchor_current = resolved.get(anchor, baseline_full.get(anchor, anchor_base))
        constituents = meta.get("constituents") or [meta["variable"]]
        constituents_base_sum = sum(baseline.get(c, 0.0) for c in constituents)
        base_ratio = constituents_base_sum / anchor_base if anchor_base != 0.0 else 0.0
        current_ratio = base_ratio + delta_bps / 10_000.0
        new_sum = anchor_current * current_ratio
        for c in constituents:
            c_base = baseline.get(c, 0.0)
            if constituents_base_sum != 0.0:
                resolved[c] = new_sum * (c_base / constituents_base_sum)
            else:
                resolved[c] = 0.0

    _propagate_formulas()

    for var, rule in static_rules.items():
        if isinstance(rule, str):
            strategy, rule_dict = rule, {}
        else:
            strategy = rule.get("strategy", "carry_forward")
            rule_dict = rule

        if strategy == "carry_forward":
            resolved[var] = baseline.get(var, 0.0)

        elif strategy == "oracle":
            resolved[var] = float(rule_dict.get("value", baseline.get(var, 0.0)))

        elif strategy == "historical_peg":
            anchor = rule_dict.get("anchor")
            if not anchor:
                resolved[var] = baseline.get(var, 0.0)
            else:
                anchor_base = baseline_full.get(anchor, 0.0)
                anchor_current = resolved.get(anchor, baseline_full.get(anchor, anchor_base))
                base_ratio = baseline.get(var, 0.0) / anchor_base if anchor_base != 0.0 else 0.0
                resolved[var] = anchor_current * base_ratio

        elif strategy == "ttm_average":
            ttm_value = rule_dict.get("ttm_value")
            resolved[var] = float(ttm_value) if ttm_value is not None else baseline.get(var, 0.0)

        else:
            resolved[var] = baseline.get(var, 0.0)

    _propagate_formulas()

    for var, val in baseline.items():
        if var not in resolved:
            resolved[var] = val

    return calculate(schema["formulas"], resolved)


def validate_formula(
    formulas: Dict[str, str],
    variables: Dict[str, float],
    eps_actual: float,
    tolerance: float = 0.01,
) -> dict:
    """Run calculate() and compare result['eps'] against eps_actual."""
    result = calculate(formulas, variables)
    eps_computed = result.get("eps")
    if eps_computed is None:
        raise ValueError("Formula DAG does not produce an 'eps' node.")

    error_pct = (
        abs(eps_computed - eps_actual) / abs(eps_actual)
        if eps_actual != 0 else float("inf")
    )
    return {
        "eps_computed": round(eps_computed, 4),
        "eps_actual":   round(eps_actual,   4),
        "error_pct":    round(error_pct,    6),
        "passed":       error_pct <= tolerance,
    }
