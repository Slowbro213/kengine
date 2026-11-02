"""A whitelisting arithmetic evaluator: no `eval`, bounded work (spec §5.5 #2)."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable
from fractions import Fraction

from kengine.numbers import decimal_places, parse_number

type Number = Fraction | float

MAX_LENGTH = 500
MAX_EXPONENT = 1000
MAX_RESULT_DIGITS = 10_000

_SYMBOLS = str.maketrans({"×": "*", "÷": "/", "−": "-"})
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)%")


class ArithError(ValueError):
    pass


def safe_eval(expression: str) -> Number:
    """Evaluate plain arithmetic exactly where possible.

    Integers and decimals become `Fraction`s; `sqrt` and fractional powers give floats.
    """
    if len(expression) > MAX_LENGTH:
        raise ArithError(f"expression longer than {MAX_LENGTH} characters")
    text = expression.translate(_SYMBOLS).replace("^", "**")
    text = _THOUSANDS_RE.sub("", text)
    text = _PERCENT_RE.sub(r"(\1/100)", text)
    try:
        tree = ast.parse(text.strip(), mode="eval")
        result = _eval(tree.body)
    except ArithError:
        raise
    except SyntaxError:
        raise ArithError("not a valid expression") from None
    except (RecursionError, MemoryError):
        raise ArithError("expression too deeply nested") from None
    except (OverflowError, ValueError, ZeroDivisionError) as e:
        raise ArithError(str(e) or type(e).__name__) from None
    if isinstance(result, float) and not math.isfinite(result):
        raise ArithError("result is not finite")
    return result


def _eval(node: ast.expr) -> Number:
    match node:
        case ast.Constant(value=bool()):
            raise ArithError("booleans aren't numbers")
        case ast.Constant(value=int() as v):
            return Fraction(v)
        case ast.Constant(value=float() as v):
            return Fraction(repr(v))
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_eval(operand)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return _eval(operand)
        case ast.BinOp(left=left, op=op, right=right):
            return _binop(op, _eval(left), _eval(right))
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in _FUNCTIONS:
            if any(isinstance(a, ast.Starred) for a in args):
                raise ArithError("starred arguments aren't allowed")
            return _FUNCTIONS[name]([_eval(a) for a in args])
        case ast.Call():
            raise ArithError("only sqrt, abs, round, min and max can be called")
        case ast.Name(id=name):
            raise ArithError(f"unknown name {name!r}")
        case _:
            raise ArithError(f"{type(node).__name__} isn't allowed")


def _binop(op: ast.operator, a: Number, b: Number) -> Number:
    match op:
        case ast.Add():
            return a + b
        case ast.Sub():
            return a - b
        case ast.Mult():
            return a * b
        case ast.Div():
            _nonzero(b)
            return a / b
        case ast.FloorDiv():
            _nonzero(b)
            return Fraction(a // b)
        case ast.Mod():
            _nonzero(b)
            return a % b
        case ast.Pow():
            return _pow(a, b)
        case _:
            raise ArithError(f"operator {type(op).__name__} isn't allowed")


def _nonzero(b: Number) -> None:
    if b == 0:
        raise ArithError("division by zero")


def _pow(base: Number, exp: Number) -> Number:
    if isinstance(exp, Fraction) and exp.denominator == 1:
        if abs(exp) > MAX_EXPONENT:
            raise ArithError("result too large")
        if abs(exp) * _log10_at_least_2(base) > MAX_RESULT_DIGITS:
            raise ArithError("result too large")
        if base == 0 and exp < 0:
            raise ArithError("division by zero")
        if isinstance(base, Fraction):
            return base ** int(exp)
        return base ** float(exp)
    if base < 0:
        raise ArithError("fractional power of a negative number")
    if abs(exp) * _log10_at_least_2(base) > MAX_RESULT_DIGITS:
        raise ArithError("result too large")
    return float(base) ** float(exp)


def _log10_at_least_2(x: Number) -> float:
    """log10(max(|x|, 2)), without converting huge Fractions to float."""
    x = abs(x)
    if x < 2:
        return math.log10(2)
    if isinstance(x, Fraction):
        return math.log10(x.numerator) - math.log10(x.denominator)
    return math.log10(x)


def _as_number(x: Number | int) -> Number:
    return Fraction(x) if isinstance(x, int) else x


def _sqrt(args: list[Number]) -> Number:
    (x,) = _arity("sqrt", args, 1, 1)
    if x < 0:
        raise ArithError("square root of a negative number")
    return math.sqrt(x)


def _abs(args: list[Number]) -> Number:
    (x,) = _arity("abs", args, 1, 1)
    return abs(x)


def _round(args: list[Number]) -> Number:
    args = _arity("round", args, 1, 2)
    if len(args) == 1:
        return Fraction(round(args[0]))
    x, digits = args
    if not (isinstance(digits, Fraction) and digits.denominator == 1):
        raise ArithError("round() needs a whole number of digits")
    return _as_number(round(x, int(digits)))


def _min(args: list[Number]) -> Number:
    return min(_arity("min", args, 1, None))


def _max(args: list[Number]) -> Number:
    return max(_arity("max", args, 1, None))


def _arity(name: str, args: list[Number], lo: int, hi: int | None) -> list[Number]:
    if len(args) < lo or (hi is not None and len(args) > hi):
        raise ArithError(f"wrong number of arguments to {name}()")
    return args


_FUNCTIONS: dict[str, Callable[[list[Number]], Number]] = {
    "sqrt": _sqrt,
    "abs": _abs,
    "round": _round,
    "min": _min,
    "max": _max,
}


def results_match(actual: Number, claimed: str) -> bool | None:
    """Whether `claimed` agrees with `actual` to the precision it was written with.

    None when `claimed` has no number in it.
    """
    value = parse_number(claimed)
    places = decimal_places(claimed)
    if value is None or places is None:
        return None
    diff = abs(Fraction(actual) - value)
    if places == 0:
        return diff <= Fraction(1, 10**9) * max(1, abs(Fraction(actual)))
    return diff <= Fraction(1, 2) / 10**places


def format_number(x: Number) -> str:
    """Whole Fractions as integers; anything else to at most 6 decimal places."""
    if isinstance(x, Fraction) and x.denominator == 1:
        return str(x.numerator)
    text = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text
