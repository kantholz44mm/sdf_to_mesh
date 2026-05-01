import numpy as np
from conftest import run, approx, type_of

def test_integer_literal(interp):
    assert run(interp, 'return 42.0;') == approx(42.0)

def test_float_literal(interp):
    assert run(interp, 'return 3.14;') == approx(3.14)

def test_scientific_notation(interp):
    assert run(interp, 'return 1e3;') == approx(1000.0)

def test_scientific_negative_exp(interp):
    assert run(interp, 'return 1.5e-2;') == approx(0.015)

def test_unary_minus_literal(interp):
    assert run(interp, 'return -2.5;') == approx(-2.5)

def test_vec2_literal(interp):
    v = run(interp, 'return <1.0, 2.0>;')
    assert type_of(v) == 'vec2'
    assert list(v) == approx([1.0, 2.0])

def test_vec3_literal(interp):
    v = run(interp, 'return <1.0, 2.0, 3.0>;')
    assert type_of(v) == 'vec3'
    assert list(v) == approx([1.0, 2.0, 3.0])

def test_vec4_literal(interp):
    v = run(interp, 'return <1.0, 2.0, 3.0, 4.0>;')
    assert type_of(v) == 'vec4'
    assert list(v) == approx([1.0, 2.0, 3.0, 4.0])

def test_vec_with_expression_elements(interp):
    v = run(interp, 'return <1.0 + 1.0, 2.0 * 3.0, 10.0 / 2.0>;')
    assert list(v) == approx([2.0, 6.0, 5.0])

def test_vec_flattening_from_swizzle(interp):
    # <vec.xy, 0.0> expands the swizzle → vec3
    v = run(interp, 'let p = <3.0, 4.0, 5.0>; return <p.x, p.y, 0.0>;')
    assert type_of(v) == 'vec3'
    assert list(v) == approx([3.0, 4.0, 0.0])

def test_let_binding(interp):
    assert run(interp, 'let x = 7.0; return x;') == approx(7.0)

def test_multiple_let_bindings(interp):
    v = run(interp, 'let a = 1.0; let b = 2.0; return <a, b>;')
    assert list(v) == approx([1.0, 2.0])
