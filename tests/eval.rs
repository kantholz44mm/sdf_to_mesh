use glam::Vec3;
use mesh::sdf::{eval, parser};

fn compile(src: &str) -> Result<mesh::sdf::primitives::Sdf, eval::EvalError> {
    let prog = parser::parse(src).expect("parse error");
    eval::eval_program(&prog)
}

// ── Happy-path evaluation ─────────────────────────────────────────────────────

#[test]
fn eval_sphere_produces_valid_sdf() {
    let sdf = compile("return sphere(1.0);").unwrap();
    assert!(sdf(Vec3::ZERO) < 0.0, "center of unit sphere should be inside");
    assert!(sdf(Vec3::new(2.0, 0.0, 0.0)) > 0.0, "point at distance 2 should be outside");
}

#[test]
fn eval_arithmetic_in_radius() {
    let sdf = compile("return sphere(0.5 + 0.5);").unwrap();
    assert!(sdf(Vec3::ZERO) < 0.0);
    assert!((sdf(Vec3::X)).abs() < 1e-4, "point at radius 1 should be on surface");
}

#[test]
fn eval_negation() {
    let sdf = compile("let r = -(-1.0);\nreturn sphere(r);").unwrap();
    assert!(sdf(Vec3::ZERO) < 0.0);
}

#[test]
fn eval_variable_binding() {
    let sdf = compile("let r = 2.0;\nreturn sphere(r);").unwrap();
    assert!(sdf(Vec3::new(1.5, 0.0, 0.0)) < 0.0);
    assert!((sdf(Vec3::new(2.0, 0.0, 0.0))).abs() < 1e-4);
}

#[test]
fn eval_boolean_union() {
    let src = "let a = sphere(1.0);\nlet b = translate(sphere(1.0), <4.0, 0.0, 0.0>);\nreturn union(a, b);";
    let sdf = compile(src).unwrap();
    assert!(sdf(Vec3::ZERO) < 0.0, "inside first sphere");
    assert!(sdf(Vec3::new(4.0, 0.0, 0.0)) < 0.0, "inside second sphere");
    assert!(sdf(Vec3::new(2.0, 0.0, 0.0)) > 0.0, "between the two spheres");
}

#[test]
fn eval_builtin_constants_pi() {
    // PI / PI = 1.0
    let src = "let r = PI / PI;\nreturn sphere(r);";
    let sdf = compile(src).unwrap();
    assert!((sdf(Vec3::X)).abs() < 1e-4, "PI/PI = 1 → unit sphere surface at X");
}

#[test]
fn eval_math_builtins() {
    // sqrt(4) = 2 → sphere of radius 2
    let src = "let r = sqrt(4.0);\nreturn sphere(r);";
    let sdf = compile(src).unwrap();
    assert!((sdf(Vec3::new(2.0, 0.0, 0.0))).abs() < 1e-4);
}

#[test]
fn eval_vector_literal() {
    // sphere radius 1 centered at (2,0,0): origin is at distance 2, sdf = 1 > 0
    let sdf = compile("return sphere(1.0, <2.0, 0.0, 0.0>);").unwrap();
    assert!(sdf(Vec3::new(2.0, 0.0, 0.0)) < 0.0, "center at (2,0,0) should be inside");
    assert!(sdf(Vec3::ZERO) > 0.0, "origin should be outside");
}

#[test]
fn eval_swizzle_xy() {
    let src = "let v = <3.0, 0.0, 0.0>;\nlet r = v.x;\nreturn sphere(r);";
    let sdf = compile(src).unwrap();
    // r = 3.0 → sphere radius 3
    assert!(sdf(Vec3::new(2.5, 0.0, 0.0)) < 0.0);
    assert!(sdf(Vec3::new(4.0, 0.0, 0.0)) > 0.0);
}

#[test]
fn eval_user_defined_function() {
    let src = "def my_sphere(r: float) -> sdf {\n    return sphere(r);\n}\nreturn my_sphere(1.5);";
    let sdf = compile(src).unwrap();
    assert!(sdf(Vec3::ZERO) < 0.0);
    assert!((sdf(Vec3::new(1.5, 0.0, 0.0))).abs() < 1e-4);
}

#[test]
fn eval_user_function_with_local_bindings() {
    let src = "def big_sphere(base: float) -> sdf {\n    let doubled = base * 2.0;\n    return sphere(doubled);\n}\nreturn big_sphere(1.0);";
    let sdf = compile(src).unwrap();
    // radius = 2.0
    assert!((sdf(Vec3::new(2.0, 0.0, 0.0))).abs() < 1e-4);
}

#[test]
fn eval_nested_transforms() {
    // scale wraps the translated sphere: effective center at (0, 10, 0), radius 2
    let src = "let s = sphere(1.0);\nlet t = translate(s, <0.0, 5.0, 0.0>);\nlet sc = scale(t, 2.0);\nreturn sc;";
    let sdf = compile(src).unwrap();
    assert!(sdf(Vec3::new(0.0, 10.0, 0.0)) < 0.0);
}

#[test]
fn eval_smooth_union() {
    let src = "let a = sphere(1.0);\nlet b = translate(sphere(1.0), <1.5, 0.0, 0.0>);\nreturn smooth_union(a, b, 0.5);";
    let sdf = compile(src).unwrap();
    assert!(sdf(Vec3::ZERO) < 0.0);
    assert!(sdf(Vec3::new(1.5, 0.0, 0.0)) < 0.0);
}

// ── Error cases ───────────────────────────────────────────────────────────────

#[test]
fn eval_error_undefined_name() {
    let prog = parser::parse("return foo;").unwrap();
    let err = eval::eval_program(&prog).err().expect("expected EvalError");
    assert!(matches!(err, eval::EvalError::UndefinedName(_)));
}

#[test]
fn eval_error_unknown_function() {
    let prog = parser::parse("return does_not_exist(1.0);").unwrap();
    let err = eval::eval_program(&prog).err().expect("expected EvalError");
    assert!(matches!(err, eval::EvalError::UnknownFunction(_)));
}

#[test]
fn eval_error_wrong_arg_count_too_few() {
    let prog = parser::parse("return sphere();").unwrap();
    let err = eval::eval_program(&prog).err().expect("expected EvalError");
    assert!(matches!(err, eval::EvalError::WrongArgCount { .. }));
}

#[test]
fn eval_error_wrong_arg_count_too_many() {
    let prog = parser::parse("return cylinder(1.0, 2.0, 3.0);").unwrap();
    let err = eval::eval_program(&prog).err().expect("expected EvalError");
    assert!(matches!(err, eval::EvalError::WrongArgCount { .. }));
}

#[test]
fn eval_error_type_mismatch_sdf_as_float() {
    // sphere() expects a float as first arg, not an sdf
    let prog = parser::parse("let s = sphere(1.0);\nreturn sphere(s);").unwrap();
    let err = eval::eval_program(&prog).err().expect("expected EvalError");
    assert!(matches!(err, eval::EvalError::TypeMismatch { .. }));
}

#[test]
fn eval_error_not_an_sdf_returned() {
    // Returning a float literal instead of an sdf
    let prog = parser::parse("let x = sphere(1.0);\nreturn 1.0;").unwrap();
    let err = eval::eval_program(&prog).err().expect("expected EvalError");
    assert!(matches!(err, eval::EvalError::TypeMismatch { .. }));
}

#[test]
fn eval_error_duplicate_name() {
    let prog = parser::parse("let x = 1.0;\nlet x = 2.0;\nreturn sphere(x);").unwrap();
    let err = eval::eval_program(&prog).err().expect("expected EvalError");
    assert!(matches!(err, eval::EvalError::DuplicateName(_)));
}

// ── Parse success checks ──────────────────────────────────────────────────────

#[test]
fn parse_all_primitives() {
    let sources = [
        "return sphere(1.0);",
        "return box(1.0);",
        "return cuboid(<1.0, 2.0, 3.0>);",
        "return cylinder(1.0, 2.0);",
        "return capsule(0.5, 2.0);",
        "return torus(2.0, 0.5);",
        "return plane(<0.0, 1.0, 0.0>, 0.0);",
    ];
    for src in &sources {
        assert!(parser::parse(src).is_ok(), "failed to parse: {src}");
    }
}

#[test]
fn parse_all_transforms() {
    let sources = [
        "let s = sphere(1.0);\nreturn translate(s, <1.0, 0.0, 0.0>);",
        "let s = sphere(1.0);\nreturn scale(s, 2.0);",
        "let s = sphere(1.0);\nreturn rotate(s, <0.0, 1.0, 0.0>, 1.0);",
        "let s = sphere(1.0);\nreturn elongate(s, <0.0, 1.0, 0.0>);",
        "let s = sphere(1.0);\nreturn mirror(s, 0.0);",
        "let s = sphere(1.0);\nreturn twist(s, 0.5);",
    ];
    for src in &sources {
        assert!(parser::parse(src).is_ok(), "failed to parse: {src}");
    }
}

#[test]
fn parse_all_booleans() {
    let sources = [
        "let a = sphere(1.0);\nlet b = sphere(2.0);\nreturn union(a, b);",
        "let a = sphere(1.0);\nlet b = sphere(2.0);\nreturn intersection(a, b);",
        "let a = sphere(1.0);\nlet b = sphere(2.0);\nreturn difference(a, b);",
        "let a = sphere(1.0);\nlet b = sphere(2.0);\nreturn smooth_union(a, b, 0.5);",
        "let a = sphere(1.0);\nlet b = sphere(2.0);\nreturn smooth_intersection(a, b, 0.5);",
        "let a = sphere(1.0);\nlet b = sphere(2.0);\nreturn smooth_difference(a, b, 0.5);",
    ];
    for src in &sources {
        assert!(parser::parse(src).is_ok(), "failed to parse: {src}");
    }
}

#[test]
fn parse_domain_operations() {
    let sources = [
        "let s = sphere(1.0);\nreturn onion(s, 0.1);",
        "let s = sphere(1.0);\nreturn offset(s, 0.5);",
        "let s = sphere(0.5);\nreturn repeat(s, <4.0, 4.0, 4.0>, <2.0, 2.0, 2.0>);",
    ];
    for src in &sources {
        assert!(parser::parse(src).is_ok(), "failed to parse: {src}");
    }
}

#[test]
fn parse_user_defined_function() {
    let src = "def make_box(s: float) -> sdf {\n    return box(s);\n}\nreturn make_box(1.0);";
    assert!(parser::parse(src).is_ok());
}

#[test]
fn parse_swizzle_components() {
    let sources = [
        "let v = <1.0, 2.0, 3.0>;\nreturn sphere(v.x);",
        "let v = <1.0, 2.0, 3.0>;\nreturn sphere(v.y);",
        "let v = <1.0, 2.0, 3.0>;\nreturn sphere(v.z);",
    ];
    for src in &sources {
        assert!(parser::parse(src).is_ok(), "failed to parse: {src}");
    }
}
