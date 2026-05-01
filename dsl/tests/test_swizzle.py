import pytest
from conftest import run, approx, type_of
from interpreter import SDFTypeError

def test_swizzle_x(interp):
    r = run(interp, 'let v = <10.0, 20.0, 30.0>; return v.x;')
    assert r == approx(10.0) and type_of(r) == 'float'

def test_swizzle_y(interp):
    r = run(interp, 'let v = <10.0, 20.0, 30.0>; return v.y;')
    assert r == approx(20.0)

def test_swizzle_z(interp):
    r = run(interp, 'let v = <10.0, 20.0, 30.0>; return v.z;')
    assert r == approx(30.0)

def test_swizzle_w(interp):
    r = run(interp, 'let v = <1.0, 2.0, 3.0, 4.0>; return v.w;')
    assert r == approx(4.0)

def test_swizzle_xy(interp):
    v = run(interp, 'let p = <1.0, 2.0, 3.0>; return p.xy;')
    assert type_of(v) == 'vec2'
    assert list(v) == approx([1.0, 2.0])

def test_swizzle_xyz(interp):
    v = run(interp, 'let p = <1.0, 2.0, 3.0>; return p.xyz;')
    assert type_of(v) == 'vec3'
    assert list(v) == approx([1.0, 2.0, 3.0])

def test_swizzle_zyx_reverses(interp):
    v = run(interp, 'let p = <1.0, 2.0, 3.0>; return p.zyx;')
    assert list(v) == approx([3.0, 2.0, 1.0])

def test_swizzle_xzx(interp):
    # repeated components are allowed
    v = run(interp, 'let p = <1.0, 2.0, 3.0>; return p.xzx;')
    assert list(v) == approx([1.0, 3.0, 1.0])

def test_swizzle_rgb_aliases(interp):
    v = run(interp, 'let c = <0.2, 0.5, 0.8>; return c.rgb;')
    assert list(v) == approx([0.2, 0.5, 0.8])

def test_swizzle_gb(interp):
    v = run(interp, 'let c = <0.2, 0.5, 0.8>; return c.gb;')
    assert type_of(v) == 'vec2'
    assert list(v) == approx([0.5, 0.8])

def test_swizzle_numeric_index(interp):
    v = run(interp, 'let p = <10.0, 20.0, 30.0>; return p.01;')
    assert type_of(v) == 'vec2'
    assert list(v) == approx([10.0, 20.0])

def test_swizzle_on_vec4(interp):
    r = run(interp, 'let v = <1.0, 2.0, 3.0, 99.0>; return v.w;')
    assert r == approx(99.0)

def test_swizzle_out_of_range_raises(interp):
    with pytest.raises(SDFTypeError, match='out of range'):
        run(interp, 'let v = <1.0, 2.0>; return v.z;')

def test_swizzle_on_sdf_raises(interp):
    with pytest.raises((SDFTypeError, Exception)):
        run(interp, 'let s = sphere(1.0); return s.x;')

def test_swizzle_chained_into_vector(interp):
    # construct a new vec from swizzled parts
    v = run(interp, 'let p = <3.0, 1.0, 2.0>; return <p.z, p.x, p.y>;')
    assert list(v) == approx([2.0, 3.0, 1.0])
