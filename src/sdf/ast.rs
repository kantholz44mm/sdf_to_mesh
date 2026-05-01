use arrayvec::ArrayVec;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub functions: Vec<FunctionDef>,
    pub assignments: Vec<Assignment>,
    pub return_expr: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: DataType,
    pub body: FunctionBody,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub ty: DataType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Float,
    Vec1,
    Vec2,
    Vec3,
    Vec4,
    Sdf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionBody {
    pub assignments: Vec<Assignment>,
    pub return_expr: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub name: String,
    pub expr: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Number(f64),
    Reference(String),
    Vector(Vec<Expression>),

    Call {
        name: String,
        args: Vec<Expression>,
    },

    Unary {
        op: UnaryOp,
        expr: Box<Expression>,
    },

    Binary {
        op: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },

    Swizzle {
        expr: Box<Expression>,
        components: ArrayVec<SwizzleIndex, 4>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum SwizzleIndex {
    First,
    Second,
    Third,
    Fourth,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}
