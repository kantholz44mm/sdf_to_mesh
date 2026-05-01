use arrayvec::ArrayVec;
use pest::iterators::Pair;
use pest::Parser;

use crate::sdf::ast::{Assignment, BinaryOp, DataType, Expression, FunctionBody, FunctionDef, Parameter, Program, SwizzleIndex, UnaryOp};

#[derive(pest_derive::Parser)]
#[grammar = "sdf/resources/grammar.pest"]
pub struct SdfParser;

pub type ParseResult<A> = Result<A, (Pair<'static, Rule>, SemanticError)>;

pub enum SemanticError {
    MissingReturnStatement,
}

pub fn parse(src: &str) -> Result<Program, pest::error::Error<Rule>> {
    let pair = SdfParser::parse(Rule::Program, src)?.next().unwrap();
    build_program(pair).map_err(|_| panic!("semantic error"))
}

fn build_program(pair: Pair<Rule>) -> Result<Program, ()> {
    let mut functions = Vec::new();
    let mut assignments = Vec::new();
    let mut return_expr = None;

    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::FunctionDefinition => {
                functions.push(build_function(p)?);
            }
            Rule::Assignment => {
                assignments.push(build_assignment(p));
            }
            Rule::ReturnStatement => {
                return_expr = Some(build_return(p));
            }
            Rule::EOI => {}
            _ => {}
        }
    }

    Ok(Program {
        functions,
        assignments,
        return_expr: return_expr.expect("program must have a return statement"),
    })
}

fn build_function(pair: Pair<Rule>) -> Result<FunctionDef, ()> {
    let mut name = String::new();
    let mut params = Vec::new();
    let mut return_type = None;
    let mut body = None;

    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ID => {
                name = p.as_str().to_string();
            }
            Rule::ParameterList => {
                params = build_params(p);
            }
            Rule::DataType => {
                return_type = Some(build_type(p));
            }
            Rule::FunctionBody => {
                body = Some(build_function_body(p));
            }
            _ => {}
        }
    }

    Ok(FunctionDef {
        name,
        params,
        return_type: return_type.expect("missing return type"),
        body: body.expect("missing function body"),
    })
}

fn build_params(pair: Pair<Rule>) -> Vec<Parameter> {
    pair.into_inner()
        .map(|p| {
            let mut i = p.into_inner();
            let name = i.next().unwrap().as_str().to_string();
            let ty = build_type(i.next().unwrap());
            Parameter { name, ty }
        })
        .collect()
}

fn build_type(pair: Pair<Rule>) -> DataType {
    match pair.as_str() {
        "float" => DataType::Float,
        "vec1" => DataType::Vec1,
        "vec2" => DataType::Vec2,
        "vec3" => DataType::Vec3,
        "vec4" => DataType::Vec4,
        "sdf"  => DataType::Sdf,
        unknown => panic!("unknown type: {}", unknown),
    }
}

fn build_function_body(pair: Pair<Rule>) -> FunctionBody {
    let mut assignments = Vec::new();
    let mut return_expr = None;

    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::Assignment => {
                assignments.push(build_assignment(p));
            }
            Rule::ReturnStatement => {
                return_expr = Some(build_return(p));
            }
            _ => {}
        }
    }

    FunctionBody {
        assignments,
        return_expr: return_expr.expect("function body must have a return statement"),
    }
}

fn build_assignment(pair: Pair<Rule>) -> Assignment {
    let mut i = pair.into_inner();
    let name = i.next().unwrap().as_str().to_string();
    let expr = build_expression(i.next().unwrap());
    Assignment { name, expr }
}

fn build_return(pair: Pair<Rule>) -> Expression {
    build_expression(pair.into_inner().next().unwrap())
}

fn build_expression(pair: Pair<Rule>) -> Expression {
    let mut inner = pair.into_inner();

    let mut left = build_term(inner.next().unwrap());

    while let Some(op) = inner.next() {
        let right = build_term(inner.next().unwrap());

        left = Expression::Binary {
            left: Box::new(left),
            op: match op.as_str() {
                "+" => BinaryOp::Add,
                "-" => BinaryOp::Sub,
                _ => unreachable!(),
            },
            right: Box::new(right),
        };
    }

    left
}

fn build_term(pair: Pair<Rule>) -> Expression {
    let mut inner = pair.into_inner();

    let mut left = build_factor(inner.next().unwrap());

    while let Some(op) = inner.next() {
        let right = build_factor(inner.next().unwrap());

        left = Expression::Binary {
            left: Box::new(left),
            op: match op.as_str() {
                "*" => BinaryOp::Mul,
                "/" => BinaryOp::Div,
                _ => unreachable!(),
            },
            right: Box::new(right),
        };
    }

    left
}

fn build_factor(pair: Pair<Rule>) -> Expression {
    let mut inner = pair.into_inner();

    let mut negate = false;
    let mut next = inner.next().unwrap();

    if next.as_rule() == Rule::UnaryOp {
        negate = true;
        next = inner.next().unwrap();
    }

    let expr = build_primary(next);

    if negate {
        Expression::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(expr),
        }
    } else {
        expr
    }
}

fn build_primary(pair: Pair<Rule>) -> Expression {
    let mut inner = pair.into_inner();

    let mut expr = build_primary_operand(inner.next().unwrap());

    if let Some(swizzle) = inner.next() {
        let s = swizzle.as_str();
        let components = parse_swizzle(&s[1..]); // skip leading '.'
        expr = Expression::Swizzle {
            expr: Box::new(expr),
            components,
        };
    }

    expr
}

fn parse_swizzle(s: &str) -> ArrayVec<SwizzleIndex, 4> {
    s.chars()
        .map(|c| match c {
            'x' | 'r' | '0' => SwizzleIndex::First,
            'y' | 'g' | '1' => SwizzleIndex::Second,
            'z' | 'b' | '2' => SwizzleIndex::Third,
            'w' | 'a' | '3' => SwizzleIndex::Fourth,
            _ => panic!("invalid swizzle char: {}", c),
        })
        .collect()
}

fn build_primary_operand(pair: Pair<Rule>) -> Expression {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::FunctionCall => build_call(inner),
        Rule::Vector => build_vector(inner),
        Rule::NumberLit => Expression::Number(inner.as_str().parse().unwrap()),
        Rule::Reference => Expression::Reference(inner.as_str().to_string()),
        Rule::Expression => build_expression(inner),
        _ => panic!("unexpected node in PrimaryOperand: {:?}", inner.as_rule()),
    }
}

fn build_call(pair: Pair<Rule>) -> Expression {
    let mut inner = pair.into_inner();

    let name = inner.next().unwrap().as_str().to_string();

    let args = inner
        .next()
        .map(|arg_list| {
            arg_list
                .into_inner()
                .map(build_expression)
                .collect()
        })
        .unwrap_or_default();

    Expression::Call { name, args }
}

fn build_vector(pair: Pair<Rule>) -> Expression {
    let elems = pair
        .into_inner()
        .map(build_expression)
        .collect();

    Expression::Vector(elems)
}
