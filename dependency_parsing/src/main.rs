use polars::prelude::*;
use rustpython_parser::{ast, parser};

fn main() {
    let python_source = "print('Hello world')";
    let python_ast = parser::parse_expression(python_source).unwrap();
    let v = python_ast.node;

    println!("{:?}", v);
}
