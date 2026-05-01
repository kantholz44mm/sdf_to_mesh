import math
import numpy as np
import pytest
from conftest import run, approx, type_of
from interpreter import SDFTypeError

PI  = math.pi
PI2 = math.pi / 2

# ── unary element-wise: scalars ───────────────────────────────────────────────

def test_abs_positive(interp):
    assert run(interp, 'return abs(3.0);') == approx(3.0)

def test_abs_negative(interp):
    assert run(interp, 'return abs(-5.0);') == approx(5.0)

def test_sqrt(interp):
    assert run(interp, 'return sqrt(9.0);') == approx(3.0)

def test_floor(interp):
    assert run(interp, 'return floor(3.7);') == approx(3.0)

def test_ceil(interp):
    assert run(interp, 'return ceil(3.2);') == approx(4.0)

def test_round_up(interp):
    assert run(interp, 'return round(3.6);') == approx(4.0)

def test_round_down(interp):
    assert run(interp, 'return round(3.4);') == approx(3.0)

def test_sign_positive(interp):
    assert run(interp, 'return sign(7.0);') == approx(1.0)

def test_sign_negative(interp):
    assert run(interp, 'return sign(-7.0);') == approx(-1.0)

def test_sign_zero(interp):
    assert run(interp, 'return sign(0.0);') == approx(0.0)

def test_fract(interp):
    assert run(interp, 'return fract(3.75);') == approx(0.75)

def test_exp(interp):
    assert run(interp, 'return exp(1.0);') == approx(math.e)

def test_log(interp):
    assert run(interp, 'return log(2.718281828);') == approx(1.0, rel=1e-4)

def test_log2(interp):
    assert run(interp, 'return log2(8.0);') == approx(3.0)

def test_sin(interp):
    assert run(interp, 'return sin(0.0);') == approx(0.0)

def test_cos(interp):
    assert run(interp, 'return cos(0.0);') == approx(1.0)

def test_tan(interp):
    assert run(interp, 'return tan(0.7853981633974483);') == approx(1.0)

def test_asin(interp):
    assert run(interp, 'return asin(1.0);') == approx(PI2)

def test_acos(interp):
    assert run(interp, 'return acos(1.0);') == approx(0.0)

def test_atan(interp):
    assert run(interp, 'return atan(1.0);') == approx(PI / 4)

def test_degrees(interp):
    assert run(interp, 'return degrees(3.141592653589793);') == approx(180.0)

def test_radians(interp):
    assert run(interp, 'return radians(180.0);') == approx(PI)

# ── unary element-wise: vectors ───────────────────────────────────────────────

def test_abs_vec3(interp):
    v = run(interp, 'return abs(<-1.0, 2.0, -3.0>);')
    assert type_of(v) == 'vec3'
    assert list(v) == approx([1.0, 2.0, 3.0])

def test_cos_vec3(interp):
    v = run(interp, 'return cos(<0.0, 0.0, 0.0>);')
    assert type_of(v) == 'vec3'
    assert list(v) == approx([1.0, 1.0, 1.0])

def test_sqrt_vec3(interp):
    v = run(interp, 'return sqrt(<1.0, 4.0, 9.0>);')
    assert list(v) == approx([1.0, 2.0, 3.0])

def test_floor_vec3(interp):
    v = run(interp, 'return floor(<1.1, 2.9, 3.5>);')
    assert list(v) == approx([1.0, 2.0, 3.0])

def test_sign_vec3(interp):
    v = run(interp, 'return sign(<-2.0, 0.0, 5.0>);')
    assert list(v) == approx([-1.0, 0.0, 1.0])

def test_fract_vec3(interp):
    v = run(interp, 'return fract(<1.25, 2.5, 3.75>);')
    assert list(v) == approx([0.25, 0.5, 0.75])

def test_unary_math_wrong_arity_raises(interp):
    with pytest.raises(SDFTypeError, match='expected 1'):
        run(interp, 'return cos(1.0, 2.0);')

def test_unary_math_on_sdf_raises(interp):
    with pytest.raises(SDFTypeError):
        run(interp, 'return cos(sphere(1.0));')

# ── two-arg element-wise ──────────────────────────────────────────────────────

def test_pow_scalar(interp):
    assert run(interp, 'return pow(2.0, 10.0);') == approx(1024.0)

def test_pow_vec3(interp):
    v = run(interp, 'return pow(<2.0, 3.0, 4.0>, <2.0, 2.0, 2.0>);')
    assert list(v) == approx([4.0, 9.0, 16.0])

def test_atan2(interp):
    assert run(interp, 'return atan2(1.0, 1.0);') == approx(PI / 4)

def test_step_below_edge(interp):
    assert run(interp, 'return step(0.5, 0.3);') == approx(0.0)

def test_step_above_edge(interp):
    assert run(interp, 'return step(0.5, 0.7);') == approx(1.0)

def test_step_vec3(interp):
    v = run(interp, 'return step(<0.5, 0.5, 0.5>, <0.3, 0.5, 0.7>);')
    assert list(v) == approx([0.0, 1.0, 1.0])

# ── min / max ─────────────────────────────────────────────────────────────────

def test_min_two_scalars(interp):
    assert run(interp, 'return min(3.0, 5.0);') == approx(3.0)

def test_max_two_scalars(interp):
    assert run(interp, 'return max(3.0, 5.0);') == approx(5.0)

def test_min_element_wise_vec3(interp):
    v = run(interp, 'return min(<3.0, 1.0, 4.0>, <2.0, 5.0, 3.0>);')
    assert list(v) == approx([2.0, 1.0, 3.0])

def test_max_element_wise_vec3(interp):
    v = run(interp, 'return max(<3.0, 1.0, 4.0>, <2.0, 5.0, 3.0>);')
    assert list(v) == approx([3.0, 5.0, 4.0])

def test_min_reduce_vec3(interp):
    assert run(interp, 'return min(<7.0, 2.0, 5.0>);') == approx(2.0)

def test_max_reduce_vec3(interp):
    assert run(interp, 'return max(<7.0, 2.0, 5.0>);') == approx(7.0)

# ── clamp / mix ───────────────────────────────────────────────────────────────

def test_clamp_below(interp):
    assert run(interp, 'return clamp(-5.0, 0.0, 1.0);') == approx(0.0)

def test_clamp_inside(interp):
    assert run(interp, 'return clamp(0.5, 0.0, 1.0);') == approx(0.5)

def test_clamp_above(interp):
    assert run(interp, 'return clamp(5.0, 0.0, 1.0);') == approx(1.0)

def test_clamp_vec3(interp):
    v = run(interp, 'return clamp(<-1.0, 0.5, 2.0>, 0.0, 1.0);')
    assert list(v) == approx([0.0, 0.5, 1.0])

def test_mix_scalar_at_0(interp):
    assert run(interp, 'return mix(1.0, 3.0, 0.0);') == approx(1.0)

def test_mix_scalar_at_1(interp):
    assert run(interp, 'return mix(1.0, 3.0, 1.0);') == approx(3.0)

def test_mix_scalar_midpoint(interp):
    assert run(interp, 'return mix(0.0, 10.0, 0.5);') == approx(5.0)

def test_mix_vec3(interp):
    v = run(interp, 'return mix(<0.0, 0.0, 0.0>, <2.0, 4.0, 6.0>, 0.5);')
    assert list(v) == approx([1.0, 2.0, 3.0])

# ── vector ops ────────────────────────────────────────────────────────────────

def test_dot_product(interp):
    assert run(interp, 'return dot(<1.0, 2.0, 3.0>, <4.0, 5.0, 6.0>);') == approx(32.0)

def test_dot_orthogonal(interp):
    assert run(interp, 'return dot(<1.0, 0.0, 0.0>, <0.0, 1.0, 0.0>);') == approx(0.0)

def test_cross_product(interp):
    v = run(interp, 'return cross(<1.0, 0.0, 0.0>, <0.0, 1.0, 0.0>);')
    assert list(v) == approx([0.0, 0.0, 1.0])

def test_cross_anticommutative(interp):
    v = run(interp, 'return cross(<0.0, 1.0, 0.0>, <1.0, 0.0, 0.0>);')
    assert list(v) == approx([0.0, 0.0, -1.0])

def test_length_unit_vector(interp):
    assert run(interp, 'return length(<1.0, 0.0, 0.0>);') == approx(1.0)

def test_length_345(interp):
    assert run(interp, 'return length(<3.0, 4.0, 0.0>);') == approx(5.0)

def test_normalize_returns_unit(interp):
    v = run(interp, 'return normalize(<3.0, 4.0, 0.0>);')
    assert list(v) == approx([0.6, 0.8, 0.0])
    assert math.isclose(sum(x**2 for x in v), 1.0, rel_tol=1e-5)
