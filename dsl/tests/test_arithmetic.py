import numpy as np
import pytest
from conftest import run, approx, type_of
from interpreter import SDFTypeError

# ── float arithmetic ──────────────────────────────────────────────────────────

def test_float_add(interp):
    assert run(interp, 'return 3.0 + 2.0;') == approx(5.0)

def test_float_sub(interp):
    assert run(interp, 'return 5.0 - 1.5;') == approx(3.5)

def test_float_mul(interp):
    assert run(interp, 'return 4.0 * 2.5;') == approx(10.0)

def test_float_div(interp):
    assert run(interp, 'return 9.0 / 3.0;') == approx(3.0)

def test_unary_minus_float(interp):
    assert run(interp, 'return -5.0;') == approx(-5.0)

def test_unary_minus_expression(interp):
    assert run(interp, 'let x = 3.0; return -x;') == approx(-3.0)

def test_double_unary_minus_via_sub(interp):
    # a - -b == a + b
    assert run(interp, 'return 3.0 - -2.0;') == approx(5.0)

# ── operator precedence ───────────────────────────────────────────────────────

def test_mul_before_add(interp):
    assert run(interp, 'return 2.0 + 3.0 * 4.0;') == approx(14.0)

def test_grouping_overrides_precedence(interp):
    assert run(interp, 'return (2.0 + 3.0) * 4.0;') == approx(20.0)

def test_chained_ops(interp):
    assert run(interp, 'return 1.0 + 2.0 + 3.0 + 4.0;') == approx(10.0)

def test_mixed_precedence(interp):
    assert run(interp, 'return 10.0 - 2.0 * 3.0 + 1.0;') == approx(5.0)

# ── vector × scalar ───────────────────────────────────────────────────────────

def test_vec3_mul_scalar(interp):
    v = run(interp, 'return <1.0, 2.0, 3.0> * 2.0;')
    assert list(v) == approx([2.0, 4.0, 6.0])

def test_scalar_mul_vec3(interp):
    v = run(interp, 'return 3.0 * <1.0, 2.0, 3.0>;')
    assert list(v) == approx([3.0, 6.0, 9.0])

def test_vec3_div_scalar(interp):
    v = run(interp, 'return <6.0, 4.0, 2.0> / 2.0;')
    assert list(v) == approx([3.0, 2.0, 1.0])

def test_scalar_add_vec3(interp):
    v = run(interp, 'return 10.0 + <1.0, 2.0, 3.0>;')
    assert list(v) == approx([11.0, 12.0, 13.0])

def test_vec3_sub_scalar(interp):
    v = run(interp, 'return <5.0, 6.0, 7.0> - 1.0;')
    assert list(v) == approx([4.0, 5.0, 6.0])

# ── vector × vector (element-wise) ───────────────────────────────────────────

def test_vec3_add_vec3(interp):
    v = run(interp, 'return <1.0, 2.0, 3.0> + <4.0, 5.0, 6.0>;')
    assert list(v) == approx([5.0, 7.0, 9.0])

def test_vec3_sub_vec3(interp):
    v = run(interp, 'return <5.0, 5.0, 5.0> - <1.0, 2.0, 3.0>;')
    assert list(v) == approx([4.0, 3.0, 2.0])

def test_vec3_mul_vec3(interp):
    v = run(interp, 'return <2.0, 3.0, 4.0> * <1.0, 2.0, 3.0>;')
    assert list(v) == approx([2.0, 6.0, 12.0])

def test_vec3_div_vec3(interp):
    v = run(interp, 'return <6.0, 8.0, 9.0> / <2.0, 4.0, 3.0>;')
    assert list(v) == approx([3.0, 2.0, 3.0])

def test_unary_minus_vec3(interp):
    v = run(interp, 'return -<1.0, 2.0, 3.0>;')
    assert list(v) == approx([-1.0, -2.0, -3.0])

# ── float / vec1 interoperability ─────────────────────────────────────────────

def test_float_plus_vec1(interp):
    # float + vec1 should work; result usable as scalar-ish
    r = run(interp, 'let s = <2.0>; return 1.0 + s;')
    assert type_of(r) in ('float', 'vec1')

def test_vec1_mul_vec3_broadcasts(interp):
    # vec1 acts as a scalar and broadcasts across vec3
    v = run(interp, 'let s = <3.0>; return s * <1.0, 2.0, 3.0>;')
    assert type_of(v) == 'vec3'
    assert list(v) == approx([3.0, 6.0, 9.0])

def test_vec1_add_vec3_broadcasts(interp):
    v = run(interp, 'let s = <10.0>; return s + <1.0, 2.0, 3.0>;')
    assert type_of(v) == 'vec3'
    assert list(v) == approx([11.0, 12.0, 13.0])

def test_swizzle_x_is_float(interp):
    r = run(interp, 'let v = <7.0, 8.0, 9.0>; return v.x;')
    assert type_of(r) == 'float'
    assert r == approx(7.0)

def test_swizzle_x_arithmetic_with_float(interp):
    # vec3.x (float) + float should just work
    r = run(interp, 'let v = <2.0, 0.0, 0.0>; return v.x + 3.0;')
    assert r == approx(5.0)
