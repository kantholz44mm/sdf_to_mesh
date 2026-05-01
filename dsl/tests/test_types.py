import pytest
from conftest import run, type_of
from interpreter import SDFTypeError, SDFNameError, types_compatible

# ── types_compatible ──────────────────────────────────────────────────────────

def test_float_compatible_with_float():
    assert types_compatible('float', 'float')

def test_vec1_compatible_with_float():
    assert types_compatible('float', 'vec1')
    assert types_compatible('vec1', 'float')

def test_vec3_not_compatible_with_vec2():
    assert not types_compatible('vec3', 'vec2')

def test_sdf_not_compatible_with_float():
    assert not types_compatible('sdf', 'float')

# ── arithmetic type errors ─────────────────────────────────────────────────────

def test_vec2_plus_vec3_raises(interp):
    with pytest.raises(SDFTypeError, match='dimension mismatch'):
        run(interp, 'return <1.0, 2.0> + <1.0, 2.0, 3.0>;')

def test_vec2_minus_vec4_raises(interp):
    with pytest.raises(SDFTypeError, match='dimension mismatch'):
        run(interp, 'return <1.0, 2.0> - <1.0, 2.0, 3.0, 4.0>;')

def test_vec3_mul_vec2_raises(interp):
    with pytest.raises(SDFTypeError, match='dimension mismatch'):
        run(interp, 'return <1.0, 2.0, 3.0> * <1.0, 2.0>;')

def test_arithmetic_on_sdf_raises(interp):
    with pytest.raises(SDFTypeError, match='arithmetic on SDF'):
        run(interp, 'let s = sphere(1.0); return s + 1.0;')

def test_sdf_times_scalar_raises(interp):
    with pytest.raises(SDFTypeError, match='arithmetic on SDF'):
        run(interp, 'let s = sphere(1.0); return s * 2.0;')

def test_negate_sdf_raises(interp):
    with pytest.raises(SDFTypeError, match="negate"):
        run(interp, 'let s = sphere(1.0); return -s;')

# ── function call type errors ──────────────────────────────────────────────────

def test_wrong_arity_raises(interp):
    with pytest.raises(SDFTypeError, match='expected'):
        run(interp, '''
def f(a: float) -> float = { return a; }
return f(1.0, 2.0);
''')

def test_wrong_param_type_raises(interp):
    with pytest.raises(SDFTypeError, match="expected vec3"):
        run(interp, '''
def f(v: vec3) -> float = { return 1.0; }
return f(1.0);
''')

def test_passing_sdf_where_float_expected_raises(interp):
    with pytest.raises(SDFTypeError):
        run(interp, '''
def f(x: float) -> float = { return x; }
return f(sphere(1.0));
''')

def test_return_type_mismatch_raises(interp):
    with pytest.raises(SDFTypeError, match='declared return'):
        run(interp, '''
def f() -> vec3 = { return 1.0; }
return f();
''')

def test_return_sdf_where_float_declared_raises(interp):
    with pytest.raises(SDFTypeError, match='declared return'):
        run(interp, '''
def f() -> float = { return sphere(1.0); }
return f();
''')

# ── name errors ───────────────────────────────────────────────────────────────

def test_undefined_variable_raises(interp):
    with pytest.raises(SDFNameError, match="undefined"):
        run(interp, 'return xyz_undefined;')

def test_undefined_function_raises(interp):
    with pytest.raises(SDFNameError, match="undefined"):
        run(interp, 'return undefined_fn(1.0);')

# ── vec1 / float are NOT type errors ─────────────────────────────────────────

def test_float_param_accepts_vec1(interp):
    r = run(interp, '''
def f(x: float) -> float = { return x; }
return f(<2.0>);
''')
    assert r is not None

def test_vec1_param_accepts_float(interp):
    r = run(interp, '''
def f(x: vec1) -> float = { return x; }
return f(3.0);
''')
    assert r is not None
