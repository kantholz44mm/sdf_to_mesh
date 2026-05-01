use std::fs;
use glam::Vec3;
use mesh::{marching_cubes, sdf::{eval, parser}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = fs::read_to_string("example.sdf")?;
    let program = parser::parse(&input).unwrap();
    let scene = eval::eval_program(&program)?;
    let vertices = marching_cubes::march_sdf(scene, Vec3::ZERO, Vec3::splat(7.0), 0.01, "mesh.stl");
    marching_cubes::write_stl("mesh.stl", &vertices);
    Ok(())
}
