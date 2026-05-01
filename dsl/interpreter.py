"""interpreter.py — SDF DSL interpreter.

Differences from grammar.tx (pragmatic, semantics identical):
  · Reference  : name=ID  (avoids textX cross-ref scoping issues; runtime lookup)
  · DataType   : regex    (returns plain string instead of an unattributed class)
  · AddOp/MulOp: regex    (returns plain string)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from textx import metamodel_from_str
import sdf as sdflib

# ── Grammar ───────────────────────────────────────────────────────────────────

_GRAMMAR = r"""
Program:
    (definitions+=FunctionDefinition | statements+=Assignment)* value=ReturnStatement
;
FunctionDefinition:
    'def' name=ID '(' parameters*=ParameterDefinition[','] ')' '->'
    datatype=/float|vec[1-4]|sdf/ '=' body=FunctionBody
;
ParameterDefinition:
    name=ID ':' datatype=/float|vec[1-4]|sdf/
;
FunctionBody:
    '{' assignments*=Assignment (value=ReturnStatement)? '}'
;
ReturnStatement:
    'return' expression=Expression ';'
;
Assignment:
    'let' name=ID '=' expression=Expression ';'
;
Expression:
    terms+=Term (ops+=/[+\-]/ terms+=Term)*
;
Term:
    factors+=Factor (ops+=/[*\/]/ factors+=Factor)*
;
Factor:
    neg?='-' operand=Primary
;
Primary:
    operand=PrimaryOperand swizzle=Swizzle?
;
PrimaryOperand:
    Vector | FunctionCall | '(' Expression ')' | Reference | NumberLit
;
FunctionCall:
    name=ID '(' args*=Expression[','] ')'
;
Swizzle:
    '.' components=/[xyzwrgba0123]+/
;
Vector:
    '<' elements+=Expression[','] '>'
;
Reference:
    name=ID
;
NumberLit:
    value=/\d+(\.\d+)?([eE][-+]?\d+)?/
;
"""

# ── Type system ───────────────────────────────────────────────────────────────

def type_of(val) -> str:
    if callable(val):
        return 'sdf'
    if isinstance(val, np.ndarray):
        return f'vec{val.shape[0]}'
    return 'float'

def types_compatible(expected: str, actual: str) -> bool:
    return expected == actual or frozenset([expected, actual]) == frozenset(['float', 'vec1'])

# ── Errors ────────────────────────────────────────────────────────────────────

class SDFTypeError(TypeError):
    pass

class SDFNameError(NameError):
    pass

# ── Swizzle ───────────────────────────────────────────────────────────────────

_SWIZZLE_IDX = {
    'x': 0, 'r': 0, '0': 0,
    'y': 1, 'g': 1, '1': 1,
    'z': 2, 'b': 2, '2': 2,
    'w': 3, 'a': 3, '3': 3,
}

def _swizzle(val, components: str):
    if isinstance(val, (int, float)):
        val = np.array([float(val)])
    if not isinstance(val, np.ndarray):
        raise SDFTypeError(f"swizzle on non-numeric value (type: {type_of(val)})")
    indices = []
    for c in components:
        if c not in _SWIZZLE_IDX:
            raise SDFTypeError(f"unknown swizzle component '{c}'")
        idx = _SWIZZLE_IDX[c]
        if idx >= val.shape[0]:
            raise SDFTypeError(
                f"swizzle '.{components}': index {idx} out of range for {type_of(val)}")
        indices.append(idx)
    result = val[np.array(indices)]
    return float(result[0]) if len(indices) == 1 else result.copy()

# ── Arithmetic ────────────────────────────────────────────────────────────────

def _numeric_pair(op, a, b):
    if type_of(a) == 'sdf' or type_of(b) == 'sdf':
        raise SDFTypeError(f"'{op}': arithmetic on SDF values is not allowed")
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.asarray(a, float), np.asarray(b, float)
    return float(a), float(b)

_MULTI_VEC = {'vec2', 'vec3', 'vec4'}  # vec1 broadcasts like a scalar

def _dim_check(op, a, b):
    ta, tb = type_of(a), type_of(b)
    if ta in _MULTI_VEC and tb in _MULTI_VEC and ta != tb:
        raise SDFTypeError(f"'{op}': dimension mismatch ({ta} vs {tb})")

def _add(a, b):
    _dim_check('+', a, b);  a, b = _numeric_pair('+', a, b);  return a + b

def _sub(a, b):
    _dim_check('-', a, b);  a, b = _numeric_pair('-', a, b);  return a - b

def _mul(a, b):
    _dim_check('*', a, b);  a, b = _numeric_pair('*', a, b);  return a * b

def _div(a, b):
    _dim_check('/', a, b);  a, b = _numeric_pair('/', a, b);  return a / b

def _neg(val):
    if type_of(val) == 'sdf':
        raise SDFTypeError("unary '-': cannot negate an SDF")
    return -val if isinstance(val, np.ndarray) else -float(val)

# ── Element-wise helpers ──────────────────────────────────────────────────────

def _ew1(fn, args, name):
    if len(args) != 1:
        raise SDFTypeError(f"'{name}': expected 1 argument, got {len(args)}")
    v = args[0]
    if type_of(v) == 'sdf':
        raise SDFTypeError(f"'{name}': argument must be numeric, not sdf")
    return fn(v) if isinstance(v, np.ndarray) else float(fn(float(v)))

def _ew2(fn, args, name):
    if len(args) != 2:
        raise SDFTypeError(f"'{name}': expected 2 arguments, got {len(args)}")
    _dim_check(name, args[0], args[1])
    a, b = _numeric_pair(name, args[0], args[1])
    r = fn(a, b)
    return float(r) if isinstance(r, np.ndarray) and r.shape == () else r

# ── Built-in argument coercers ────────────────────────────────────────────────

def _req_sdf(val, fn, n=1):
    if type_of(val) != 'sdf':
        raise SDFTypeError(f"'{fn}' arg {n}: expected sdf, got {type_of(val)}")
    return val

def _req_scalar(val, fn, n=1):
    t = type_of(val)
    if t not in ('float', 'vec1'):
        raise SDFTypeError(f"'{fn}' arg {n}: expected float, got {t}")
    return float(val[0]) if isinstance(val, np.ndarray) else float(val)

def _req_vec(val, fn, n=1, dim=None):
    if isinstance(val, (int, float)):
        val = np.array([float(val)])
    if not isinstance(val, np.ndarray):
        raise SDFTypeError(f"'{fn}' arg {n}: expected vector, got {type_of(val)}")
    if dim is not None and val.shape[0] != dim:
        raise SDFTypeError(f"'{fn}' arg {n}: expected vec{dim}, got {type_of(val)}")
    return val.astype(float)

def _as_arr(v):
    return v if isinstance(v, np.ndarray) else np.array([float(v)])

# ── Built-ins registry ────────────────────────────────────────────────────────

def _build_builtins() -> dict:
    b = {}

    # element-wise unary math
    for _nm, _fn in [
        ('abs',     np.abs),      ('sqrt',    np.sqrt),
        ('exp',     np.exp),      ('log',     np.log),
        ('log2',    np.log2),     ('floor',   np.floor),
        ('ceil',    np.ceil),     ('round',   np.round),
        ('sign',    np.sign),     ('sin',     np.sin),
        ('cos',     np.cos),      ('tan',     np.tan),
        ('asin',    np.arcsin),   ('acos',    np.arccos),
        ('atan',    np.arctan),   ('degrees', np.degrees),
        ('radians', np.radians),  ('fract',   lambda v: v - np.floor(v)),
    ]:
        b[_nm] = (lambda fn=_fn, nm=_nm: lambda args: _ew1(fn, args, nm))()

    # two-arg element-wise
    b['pow']   = lambda args: _ew2(np.power,   args, 'pow')
    b['atan2'] = lambda args: _ew2(np.arctan2, args, 'atan2')
    b['step']  = lambda args: _ew2(lambda e, x: np.where(x >= e, 1.0, 0.0), args, 'step')

    # min / max — 1 arg reduces, 2 args are element-wise
    def _minmax(np_scalar, np_ew, name):
        def _f(args):
            if len(args) == 1:   return float(np_scalar(args[0]))
            if len(args) == 2:   return _ew2(np_ew, args, name)
            raise SDFTypeError(f"'{name}': expected 1 or 2 arguments")
        return _f
    b['min'] = _minmax(np.min, np.minimum, 'min')
    b['max'] = _minmax(np.max, np.maximum, 'max')

    # three-arg
    b['clamp'] = lambda args: np.clip(_as_arr(args[0]), float(args[1]), float(args[2])) \
                              if isinstance(args[0], np.ndarray) \
                              else float(np.clip(float(args[0]), float(args[1]), float(args[2])))
    b['mix']   = lambda args: _as_arr(args[0]) * (1.0 - float(args[2])) \
                              + _as_arr(args[1]) * float(args[2]) \
                              if isinstance(args[0], np.ndarray) or isinstance(args[1], np.ndarray) \
                              else float(args[0]) * (1.0 - float(args[2])) + float(args[1]) * float(args[2])

    # vector ops
    b['dot']       = lambda args: float(np.dot(_req_vec(args[0],'dot',1), _req_vec(args[1],'dot',2)))
    b['cross']     = lambda args: np.cross(_req_vec(args[0],'cross',1,3), _req_vec(args[1],'cross',2,3))
    b['length']    = lambda args: float(np.linalg.norm(_req_vec(args[0], 'length')))
    b['normalize'] = lambda args: (lambda v: v / np.linalg.norm(v))(_req_vec(args[0], 'normalize'))

    # SDF primitives
    def _sphere(args):
        r = _req_scalar(args[0], 'sphere')
        return sdflib.sphere(r) if len(args) == 1 else sdflib.sphere(r, center=_req_vec(args[1],'sphere',2,3))
    b['sphere']   = _sphere
    b['box']      = lambda args: sdflib.box(_req_vec(args[0], 'box'))
    b['cylinder'] = lambda args: sdflib.cylinder(_req_scalar(args[0],'cylinder'), _req_scalar(args[1],'cylinder',2))
    b['capsule']  = lambda args: sdflib.capsule(_req_scalar(args[0],'capsule'),   _req_scalar(args[1],'capsule',2))
    b['torus']    = lambda args: sdflib.torus(_req_scalar(args[0],'torus'),       _req_scalar(args[1],'torus',2))
    b['plane']    = lambda args: sdflib.plane(
        _req_vec(args[0],'plane',1,3),
        _req_scalar(args[1],'plane',2) if len(args) > 1 else 0.0)

    # SDF booleans
    for _nm, _fn in [('union',        sdflib.union),
                     ('intersection', sdflib.intersection),
                     ('difference',   sdflib.difference)]:
        b[_nm] = (lambda fn=_fn, nm=_nm:
            lambda args: fn(_req_sdf(args[0],nm), _req_sdf(args[1],nm,2)))()

    # SDF smooth booleans
    for _nm, _fn in [('smooth_union',        sdflib.smooth_union),
                     ('smooth_intersection', sdflib.smooth_intersection),
                     ('smooth_difference',   sdflib.smooth_difference)]:
        b[_nm] = (lambda fn=_fn, nm=_nm:
            lambda args: fn(_req_sdf(args[0],nm), _req_sdf(args[1],nm,2), _req_scalar(args[2],nm,3)))()

    # SDF transforms
    b['translate'] = lambda args: sdflib.translate(_req_sdf(args[0],'translate'), _req_vec(args[1],'translate',2,3))
    b['rotate']    = lambda args: sdflib.rotate(_req_sdf(args[0],'rotate'), _req_vec(args[1],'rotate',2,3), _req_scalar(args[2],'rotate',3))
    b['scale']     = lambda args: sdflib.scale(_req_sdf(args[0],'scale'),
        args[1] if isinstance(args[1], np.ndarray) else _req_scalar(args[1],'scale',2))
    b['mirror']    = lambda args: sdflib.mirror(_req_sdf(args[0],'mirror'), int(_req_scalar(args[1],'mirror',2)))
    b['elongate']  = lambda args: sdflib.elongate(_req_sdf(args[0],'elongate'), _req_vec(args[1],'elongate',2,3))
    b['twist']     = lambda args: sdflib.twist(_req_sdf(args[0],'twist'), _req_scalar(args[1],'twist',2))

    # SDF domain
    b['repeat'] = lambda args: sdflib.repeat(_req_sdf(args[0],'repeat'), _req_vec(args[1],'repeat',2,3), _req_vec(args[2],'repeat',3,3))
    b['onion']  = lambda args: sdflib.onion(_req_sdf(args[0],'onion'),   _req_scalar(args[1],'onion',2))
    b['offset'] = lambda args: sdflib.offset(_req_sdf(args[0],'offset'), _req_scalar(args[1],'offset',2))

    # meshing
    def _mc(args):
        s        = _req_sdf(args[0], 'marching_cubes')
        center   = _req_vec(args[1],'marching_cubes',2,3) if len(args)>1 else np.zeros(3)
        size     = _req_vec(args[2],'marching_cubes',3,3) if len(args)>2 else np.ones(3)*2
        res      = _req_scalar(args[3],'marching_cubes',4) if len(args)>3 else 0.05
        filename = str(args[4])                            if len(args)>4 else 'mesh.stl'
        sdflib.marching_cubes(s, center=center, size=size, resolution=res, filename=filename)
        return None
    b['marching_cubes'] = _mc

    return b

# ── Environment ───────────────────────────────────────────────────────────────

class Env:
    __slots__ = ('_b', '_p')
    def __init__(self, parent=None): self._b = {}; self._p = parent
    def get(self, name):
        if name in self._b: return self._b[name]
        if self._p:         return self._p.get(name)
        raise SDFNameError(f"undefined name '{name}'")
    def put(self, name, val): self._b[name] = val

# ── Interpreter ───────────────────────────────────────────────────────────────

class Interpreter:
    def __init__(self):
        self._mm       = metamodel_from_str(_GRAMMAR)
        self._builtins = _build_builtins()
        self._user_fns = {}
        self._global   = Env()

    # public
    def run_file(self, path: str):
        return self._run(self._mm.model_from_file(path))

    def run_str(self, source: str):
        return self._run(self._mm.model_from_str(source))

    # program — fresh state per run so programs don't pollute each other
    def _run(self, prog):
        self._user_fns = {}
        self._global   = Env()
        for defn in prog.definitions:
            self._user_fns[defn.name] = defn
        for stmt in prog.statements:
            self._global.put(stmt.name, self._expr(stmt.expression, self._global))
        return self._expr(prog.value.expression, self._global)

    # expressions
    def _expr(self, node, env):
        val = self._term(node.terms[0], env)
        for op, t in zip(node.ops, node.terms[1:]):
            rhs = self._term(t, env)
            val = _add(val, rhs) if op == '+' else _sub(val, rhs)
        return val

    def _term(self, node, env):
        val = self._factor(node.factors[0], env)
        for op, f in zip(node.ops, node.factors[1:]):
            rhs = self._factor(f, env)
            val = _mul(val, rhs) if op == '*' else _div(val, rhs)
        return val

    def _factor(self, node, env):
        val = self._primary(node.operand, env)
        return _neg(val) if node.neg else val

    def _primary(self, node, env):
        val = self._operand(node.operand, env)
        if node.swizzle:
            val = _swizzle(val, node.swizzle.components)
        return val

    def _operand(self, op, env):
        cn = op.__class__.__name__
        if cn == 'Vector':
            return self._eval_vector(op, env)
        if cn == 'FunctionCall':
            args = [self._expr(a, env) for a in op.args]
            return self._call(op.name, args, env)
        if cn == 'Reference':
            return env.get(op.name)
        if cn == 'NumberLit':
            return float(op.value)
        # grouped '(' Expression ')'
        return self._expr(op, env)

    def _eval_vector(self, node, env):
        flat = []
        for elem in node.elements:
            v = self._expr(elem, env)
            if isinstance(v, np.ndarray):
                flat.extend(v.tolist())
            else:
                flat.append(float(v))
        if len(flat) > 4:
            raise SDFTypeError(f"vector literal has {len(flat)} components; max is 4")
        return np.array(flat, dtype=float)

    # function dispatch
    def _call(self, name: str, args: list, env):
        if name in self._user_fns:
            return self._call_user(self._user_fns[name], args)
        if name in self._builtins:
            return self._builtins[name](args)
        raise SDFNameError(f"undefined function '{name}'")

    def _call_user(self, defn, args: list):
        if len(args) != len(defn.parameters):
            raise SDFTypeError(
                f"'{defn.name}': expected {len(defn.parameters)} args, got {len(args)}")
        # parameter type check
        for param, arg in zip(defn.parameters, args):
            actual = type_of(arg)
            expected = str(param.datatype)
            if not types_compatible(expected, actual):
                raise SDFTypeError(
                    f"'{defn.name}' param '{param.name}': "
                    f"expected {expected}, got {actual}")
        # build frame
        frame = Env(parent=self._global)
        for param, arg in zip(defn.parameters, args):
            frame.put(param.name, arg)
        # execute body
        for stmt in defn.body.assignments:
            frame.put(stmt.name, self._expr(stmt.expression, frame))
        # return
        result = self._expr(defn.body.value.expression, frame) if defn.body.value else None
        # return type check
        declared = str(defn.datatype)
        if result is not None and not types_compatible(declared, type_of(result)):
            raise SDFTypeError(
                f"'{defn.name}': declared return {declared}, got {type_of(result)}")
        return result

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python interpreter.py <file.sdf>")
        sys.exit(1)
    interp = Interpreter()
    try:
        result = interp.run_file(sys.argv[1])
        if result is not None:
            print(f"=> {result}  (type: {type_of(result)})")
    except (SDFTypeError, SDFNameError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
