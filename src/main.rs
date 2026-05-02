use std::fs;
use glam::Vec3;
use mesh::{marching_cubes, sdf::{eval, parser, primitives::estimate_bounding_box_iterative}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = fs::read_to_string("example.sdf")?;
    let program = parser::parse(&input).unwrap();
    let scene = eval::eval_program(&program)?;

    if let Some(bounding_box) = estimate_bounding_box_iterative(&scene, 100.0) {
        let (lower_bound, upper_bound) = bounding_box;
        let size = upper_bound - lower_bound;
        let center = lower_bound + 0.5 * size;
        println!("Estimated bounding box: {size:.2?}");
        let (vertices, indices) = marching_cubes::march_sdf(scene.clone(), center, size, 0.01);
        marching_cubes::write_stl("mesh.stl", &vertices, &indices);
    } else {
        println!("Failed to determine bounding box!");
    }

    Ok(())
}
