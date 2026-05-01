use std::collections::HashMap;
use glam::{Vec2, Vec3, Vec4};

use crate::sdf::{ast::*, primitives::{self, Sdf}};

// ── Value type ───────────────────────────────────────────────────────────────

#[derive(Clone)]
pub enum Value {
    Float(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    Sdf(Sdf),
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Float(_) => "float",
        Value::Vec2(_)  => "vec2",
        Value::Vec3(_)  => "vec3",
        Value::Vec4(_)  => "vec4",
        Value::Sdf(_)   => "sdf",
    }
}

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum EvalError {
    UndefinedName(String),
    UnknownFunction(String),
    TypeMismatch { expected: &'static str, got: &'static str },
    WrongArgCount { func: String, expected: usize, got: usize },
    NotAnSdf,
    DuplicateName(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::UndefinedName(n)    => write!(f, "undefined name: {n}"),
            EvalError::UnknownFunction(n)  => write!(f, "unknown function: {n}"),
            EvalError::TypeMismatch { expected, got } => write!(f, "type mismatch: expected {expected}, got {got}"),
            EvalError::WrongArgCount { func, expected, got } => write!(f, "{func}: expected {expected} args, got {got}"),
            EvalError::NotAnSdf            => write!(f, "program did not return an sdf"),
            EvalError::DuplicateName(n)    => write!(f, "redefined name: {n}"),
        }
    }
}

impl std::error::Error for EvalError {}

// ── Public entry point ───────────────────────────────────────────────────────

pub fn eval_program(program: &Program) -> Result<Sdf, EvalError> {
    let funcs: HashMap<&str, &FunctionDef> = program.functions.iter()
        .map(|f| (f.name.as_str(), f))
        .collect();

    let mut env: HashMap<String, Value> = HashMap::new();
    env.insert(String::from("PI"), Value::Float(3.14159265359));
    env.insert(String::from("pi"), Value::Float(3.14159265359));
    env.insert(String::from("E"), Value::Float(2.7182818284));
    env.insert(String::from("e"), Value::Float(2.7182818284));

    for a in &program.assignments {
        let val = eval_expr(&a.expr, &env, &funcs)?;
        if let Some(_previous) = env.insert(a.name.clone(), val) {
            return Err(EvalError::DuplicateName(a.name.clone()));
        }
    }

    match eval_expr(&program.return_expr, &env, &funcs)? {
        Value::Sdf(s) => Ok(s),
        other         => Err(EvalError::TypeMismatch { expected: "sdf", got: type_name(&other) }),
    }
}

// ── Expression evaluation ────────────────────────────────────────────────────

type Env = HashMap<String, Value>;
type Funcs<'a> = HashMap<&'a str, &'a FunctionDef>;

fn eval_expr(expr: &Expression, env: &Env, funcs: &Funcs) -> Result<Value, EvalError> {
    match expr {
        Expression::Number(n) => Ok(Value::Float(*n as f32)),

        Expression::Reference(name) => env
            .get(name.as_str())
            .cloned()
            .ok_or_else(|| { EvalError::UndefinedName(name.clone())}),

        Expression::Vector(elems) => {
            let floats: Result<Vec<f32>, _> = elems.iter()
                .map(|e| eval_expr(e, env, funcs).and_then(|v| match v {
                    Value::Float(f) => Ok(f),
                    other => Err(EvalError::TypeMismatch { expected: "float", got: type_name(&other) }),
                }))
                .collect();
            let fs = floats?;
            match fs.as_slice() {
                [x, y]       => Ok(Value::Vec2(Vec2::new(*x, *y))),
                [x, y, z]    => Ok(Value::Vec3(Vec3::new(*x, *y, *z))),
                [x, y, z, w] => Ok(Value::Vec4(Vec4::new(*x, *y, *z, *w))),
                _ => Err(EvalError::TypeMismatch { expected: "2–4 element vector", got: "wrong size" }),
            }
        }

        Expression::Unary { op: UnaryOp::Neg, expr } => match eval_expr(expr, env, funcs)? {
            Value::Float(f) => Ok(Value::Float(-f)),
            Value::Vec2(v)  => Ok(Value::Vec2(-v)),
            Value::Vec3(v)  => Ok(Value::Vec3(-v)),
            Value::Vec4(v)  => Ok(Value::Vec4(-v)),
            other => Err(EvalError::TypeMismatch { expected: "numeric", got: type_name(&other) }),
        },

        Expression::Binary { op, left, right } => {
            let lv = eval_expr(left, env, funcs)?;
            let rv = eval_expr(right, env, funcs)?;
            eval_binary(op, lv, rv)
        }

        Expression::Call { name, args } => {
            let vals: Result<Vec<Value>, _> = args.iter()
                .map(|a| eval_expr(a, env, funcs))
                .collect();
            let vals = vals?;
            if let Some(&func) = funcs.get(name.as_str()) {
                eval_user_func(func, env, vals, funcs)
            } else {
                eval_builtin(name, vals)
            }
        }

        Expression::Swizzle { expr, components } => {
            let val = eval_expr(expr, env, funcs)?;
            eval_swizzle(val, components)
        }
    }
}

// ── Binary arithmetic ────────────────────────────────────────────────────────

fn eval_binary(op: &BinaryOp, lv: Value, rv: Value) -> Result<Value, EvalError> {
    macro_rules! arith {
        ($a:expr, $b:expr, $op:expr, $wrap:expr) => {
            Ok($wrap(match $op {
                BinaryOp::Add => $a + $b,
                BinaryOp::Sub => $a - $b,
                BinaryOp::Mul => $a * $b,
                BinaryOp::Div => $a / $b,
            }))
        };
    }
    match (&lv, &rv) {
        (Value::Float(a), Value::Float(b)) => arith!(*a, *b, op, Value::Float),
        (Value::Vec2(a),  Value::Vec2(b))  => arith!(*a, *b, op, Value::Vec2),
        (Value::Vec3(a),  Value::Vec3(b))  => arith!(*a, *b, op, Value::Vec3),
        (Value::Vec4(a),  Value::Vec4(b))  => arith!(*a, *b, op, Value::Vec4),
        // scalar broadcast
        (Value::Float(s), Value::Vec2(v))  => arith!(Vec2::splat(*s), *v, op, Value::Vec2),
        (Value::Vec2(v),  Value::Float(s)) => arith!(*v, Vec2::splat(*s), op, Value::Vec2),
        (Value::Float(s), Value::Vec3(v))  => arith!(Vec3::splat(*s), *v, op, Value::Vec3),
        (Value::Vec3(v),  Value::Float(s)) => arith!(*v, Vec3::splat(*s), op, Value::Vec3),
        (Value::Float(s), Value::Vec4(v))  => arith!(Vec4::splat(*s), *v, op, Value::Vec4),
        (Value::Vec4(v),  Value::Float(s)) => arith!(*v, Vec4::splat(*s), op, Value::Vec4),
        _ => Err(EvalError::TypeMismatch { expected: "compatible numeric types", got: "incompatible" }),
    }
}

// ── User-defined function call ───────────────────────────────────────────────

fn eval_user_func(func: &FunctionDef, env: &Env, args: Vec<Value>, funcs: &Funcs) -> Result<Value, EvalError> {
    if args.len() != func.params.len() {
        return Err(EvalError::WrongArgCount {
            func: func.name.clone(),
            expected: func.params.len(),
            got: args.len(),
        });
    }

    let mut env = env.clone();
    for (param, assigned_value) in func.params.iter().zip(args) {
        if let Some(_previous) = env.insert(param.name.clone(), assigned_value) {
            return Err(EvalError::DuplicateName(param.name.clone()))
        }
    }

    for a in &func.body.assignments {
        let val = eval_expr(&a.expr, &env, funcs)?;
        env.insert(a.name.clone(), val);
    }
    eval_expr(&func.body.return_expr, &env, funcs)
}

// ── Built-in function dispatch ───────────────────────────────────────────────

fn eval_builtin(name: &str, args: Vec<Value>) -> Result<Value, EvalError> {
    macro_rules! float_arg {
        ($i:expr) => {
            match &args[$i] {
                Value::Float(f) => *f,
                v => return Err(EvalError::TypeMismatch { expected: "float", got: type_name(v) }),
            }
        };
    }
    macro_rules! vec3_arg {
        ($i:expr) => {
            match &args[$i] {
                Value::Vec3(v) => *v,
                v => return Err(EvalError::TypeMismatch { expected: "vec3", got: type_name(v) }),
            }
        };
    }
    macro_rules! sdf_arg {
        ($i:expr) => {
            match &args[$i] {
                Value::Sdf(s) => s.clone(),
                v => return Err(EvalError::TypeMismatch { expected: "sdf", got: type_name(v) }),
            }
        };
    }
    macro_rules! check_argc {
        ($n:expr) => {
            if args.len() != $n {
                return Err(EvalError::WrongArgCount {
                    func: name.to_string(),
                    expected: $n,
                    got: args.len(),
                });
            }
        };
    }

    match name {
        // ── Primitives ───────────────────────────────────────────────────────
        "sphere" => {
            if args.is_empty() || args.len() > 2 {
                return Err(EvalError::WrongArgCount { func: name.into(), expected: 1, got: args.len() });
            }
            let r = float_arg!(0);
            let c = if args.len() == 2 { vec3_arg!(1) } else { Vec3::ZERO };
            Ok(Value::Sdf(primitives::sphere(r, c)))
        }
        "box" | "cuboid" => {
            check_argc!(1);
            let size = match &args[0] {
                Value::Vec3(v)  => *v,
                Value::Float(f) => Vec3::splat(*f),
                v => return Err(EvalError::TypeMismatch { expected: "vec3 or float", got: type_name(v) }),
            };
            Ok(Value::Sdf(primitives::cuboid(size)))
        }
        "cylinder" => { check_argc!(2); Ok(Value::Sdf(primitives::cylinder(float_arg!(0), float_arg!(1)))) }
        "capsule"  => { check_argc!(2); Ok(Value::Sdf(primitives::capsule(float_arg!(0), float_arg!(1)))) }
        "torus"    => { check_argc!(2); Ok(Value::Sdf(primitives::torus(float_arg!(0), float_arg!(1)))) }
        "plane"    => { check_argc!(2); Ok(Value::Sdf(primitives::plane(vec3_arg!(0), float_arg!(1)))) }

        // ── Transforms ───────────────────────────────────────────────────────
        "translate" => { check_argc!(2); Ok(Value::Sdf(primitives::translate(sdf_arg!(0), vec3_arg!(1)))) }
        "rotate"    => { check_argc!(3); Ok(Value::Sdf(primitives::rotate(sdf_arg!(0), vec3_arg!(1), float_arg!(2)))) }
        "scale"     => { check_argc!(2); Ok(Value::Sdf(primitives::scale(sdf_arg!(0), float_arg!(1)))) }
        "elongate"  => { check_argc!(2); Ok(Value::Sdf(primitives::elongate(sdf_arg!(0), vec3_arg!(1)))) }
        "twist"     => { check_argc!(2); Ok(Value::Sdf(primitives::twist(sdf_arg!(0), float_arg!(1)))) }
        "mirror"    => { check_argc!(2); Ok(Value::Sdf(primitives::mirror(sdf_arg!(0), float_arg!(1) as usize))) }

        // ── Booleans ─────────────────────────────────────────────────────────
        "union"        => { check_argc!(2); Ok(Value::Sdf(primitives::union(sdf_arg!(0), sdf_arg!(1)))) }
        "intersection" => { check_argc!(2); Ok(Value::Sdf(primitives::intersection(sdf_arg!(0), sdf_arg!(1)))) }
        "difference"   => { check_argc!(2); Ok(Value::Sdf(primitives::difference(sdf_arg!(0), sdf_arg!(1)))) }

        // ── Smooth booleans ───────────────────────────────────────────────────
        "smooth_union"        => { check_argc!(3); Ok(Value::Sdf(primitives::smooth_union(sdf_arg!(0), sdf_arg!(1), float_arg!(2)))) }
        "smooth_intersection" => { check_argc!(3); Ok(Value::Sdf(primitives::smooth_intersection(sdf_arg!(0), sdf_arg!(1), float_arg!(2)))) }
        "smooth_difference"   => { check_argc!(3); Ok(Value::Sdf(primitives::smooth_difference(sdf_arg!(0), sdf_arg!(1), float_arg!(2)))) }

        // ── Domain operations ─────────────────────────────────────────────────
        "repeat" => { check_argc!(3); Ok(Value::Sdf(primitives::repeat(sdf_arg!(0), vec3_arg!(1), vec3_arg!(2)))) }
        "onion"  => { check_argc!(2); Ok(Value::Sdf(primitives::onion(sdf_arg!(0), float_arg!(1)))) }
        "offset" => { check_argc!(2); Ok(Value::Sdf(primitives::offset(sdf_arg!(0), float_arg!(1)))) }

        // ── Math utilities ────────────────────────────────────────────────────
        "normalize" => match args.as_slice() {
            [Value::Vec3(v)] => Ok(Value::Vec3(v.normalize())),
            [Value::Vec2(v)] => Ok(Value::Vec2(v.normalize())),
            _ => Err(EvalError::TypeMismatch { expected: "vec2 or vec3", got: "other" }),
        },
        "length" => match args.as_slice() {
            [Value::Vec3(v)] => Ok(Value::Float(v.length())),
            [Value::Vec2(v)] => Ok(Value::Float(v.length())),
            _ => Err(EvalError::TypeMismatch { expected: "vec2 or vec3", got: "other" }),
        },
        "abs" => match args.as_slice() {
            [Value::Float(f)] => Ok(Value::Float(f.abs())),
            [Value::Vec2(v)]  => Ok(Value::Vec2(v.abs())),
            [Value::Vec3(v)]  => Ok(Value::Vec3(v.abs())),
            [Value::Vec4(v)]  => Ok(Value::Vec4(v.abs())),
            _ => Err(EvalError::TypeMismatch { expected: "numeric", got: "other" }),
        },
        "min" => match args.as_slice() {
            [Value::Float(a), Value::Float(b)] => Ok(Value::Float(a.min(*b))),
            [Value::Vec3(a),  Value::Vec3(b)]  => Ok(Value::Vec3(a.min(*b))),
            _ => Err(EvalError::TypeMismatch { expected: "float,float or vec3,vec3", got: "other" }),
        },
        "max" => match args.as_slice() {
            [Value::Float(a), Value::Float(b)] => Ok(Value::Float(a.max(*b))),
            [Value::Vec3(a),  Value::Vec3(b)]  => Ok(Value::Vec3(a.max(*b))),
            _ => Err(EvalError::TypeMismatch { expected: "float,float or vec3,vec3", got: "other" }),
        },
        "clamp" => match args.as_slice() {
            [Value::Float(v), Value::Float(lo), Value::Float(hi)] => Ok(Value::Float(v.clamp(*lo, *hi))),
            _ => Err(EvalError::TypeMismatch { expected: "float, float, float", got: "other" }),
        },
        "mix" | "lerp" => match args.as_slice() {
            [Value::Float(a), Value::Float(b), Value::Float(t)] => Ok(Value::Float(a + (b - a) * t)),
            [Value::Vec3(a),  Value::Vec3(b),  Value::Float(t)] => Ok(Value::Vec3(a.lerp(*b, *t))),
            _ => Err(EvalError::TypeMismatch { expected: "(float|vec3), (float|vec3), float", got: "other" }),
        },
        "dot" => match args.as_slice() {
            [Value::Vec2(a), Value::Vec2(b)] => Ok(Value::Float(a.dot(*b))),
            [Value::Vec3(a), Value::Vec3(b)] => Ok(Value::Float(a.dot(*b))),
            _ => Err(EvalError::TypeMismatch { expected: "vec, vec", got: "other" }),
        },
        "cross" => match args.as_slice() {
            [Value::Vec3(a), Value::Vec3(b)] => Ok(Value::Vec3(a.cross(*b))),
            _ => Err(EvalError::TypeMismatch { expected: "vec3, vec3", got: "other" }),
        },
        "sqrt"  => { check_argc!(1); Ok(Value::Float(float_arg!(0).sqrt())) }
        "sin"   => { check_argc!(1); Ok(Value::Float(float_arg!(0).sin())) }
        "cos"   => { check_argc!(1); Ok(Value::Float(float_arg!(0).cos())) }
        "tan"   => { check_argc!(1); Ok(Value::Float(float_arg!(0).tan())) }
        "asin"  => { check_argc!(1); Ok(Value::Float(float_arg!(0).asin())) }
        "acos"  => { check_argc!(1); Ok(Value::Float(float_arg!(0).acos())) }
        "atan"  => { check_argc!(1); Ok(Value::Float(float_arg!(0).atan())) }
        "atan2" => { check_argc!(2); Ok(Value::Float(float_arg!(0).atan2(float_arg!(1)))) }
        "floor" => { check_argc!(1); Ok(Value::Float(float_arg!(0).floor())) }
        "ceil"  => { check_argc!(1); Ok(Value::Float(float_arg!(0).ceil())) }
        "round" => { check_argc!(1); Ok(Value::Float(float_arg!(0).round())) }
        "pow"   => { check_argc!(2); Ok(Value::Float(float_arg!(0).powf(float_arg!(1)))) }
        "sign"  => { check_argc!(1); Ok(Value::Float(float_arg!(0).signum())) }
        "exp"   => { check_argc!(1); Ok(Value::Float(float_arg!(0).exp())) }
        "log"   => { check_argc!(1); Ok(Value::Float(float_arg!(0).ln())) }

        _ => Err(EvalError::UnknownFunction(name.to_string())),
    }
}

// ── Swizzle ───────────────────────────────────────────────────────────────────

fn eval_swizzle(val: Value, components: &arrayvec::ArrayVec<SwizzleIndex, 4>) -> Result<Value, EvalError> {
    let v4 = match val {
        Value::Float(f) => Vec4::new(f, 0.0, 0.0, 0.0),
        Value::Vec2(v)  => Vec4::new(v.x, v.y, 0.0, 0.0),
        Value::Vec3(v)  => Vec4::new(v.x, v.y, v.z, 0.0),
        Value::Vec4(v)  => v,
        other => return Err(EvalError::TypeMismatch { expected: "numeric", got: type_name(&other) }),
    };

    let fs: Vec<f32> = components.iter().map(|idx| match idx {
        SwizzleIndex::First  => v4.x,
        SwizzleIndex::Second => v4.y,
        SwizzleIndex::Third  => v4.z,
        SwizzleIndex::Fourth => v4.w,
    }).collect();

    match fs.as_slice() {
        [x]       => Ok(Value::Float(*x)),
        [x, y]    => Ok(Value::Vec2(Vec2::new(*x, *y))),
        [x, y, z] => Ok(Value::Vec3(Vec3::new(*x, *y, *z))),
        [x, y, z, w] => Ok(Value::Vec4(Vec4::new(*x, *y, *z, *w))),
        _ => unreachable!(),
    }
}
