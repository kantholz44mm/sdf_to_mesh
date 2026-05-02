use std::{collections::HashMap, fs::OpenOptions, time::Instant};

use glam::Vec3;
use meshopt::{typed_to_bytes, SimplifyOptions, VertexDataAdapter};
use rayon::prelude::*;
use spatialtree::{LodVec, OctTree, OctVec};
use stl_io::{Triangle, Vector};

use crate::sdf::primitives::Sdf;

type OctCoord = OctVec<u16>;

const NEIGHBORS: [[u8; 3]; 8] = [
    [0, 0, 0], [1, 0, 0],
    [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1],
    [0, 1, 1], [1, 1, 1],
];

struct Cell {
    corners: [f32; 8],
}

fn eval_cell(sdf: &Sdf, pos: OctCoord, world_min: Vec3, world_size: f32) -> Cell {
    let fc = pos.float_coords();
    let fs = pos.float_size();
    let mut corners = [0.0f32; 8];
    for (i, nb) in NEIGHBORS.iter().enumerate() {
        let p = Vec3::new(
            world_min.x + (fc[0] + nb[0] as f32 * fs) * world_size,
            world_min.y + (fc[1] + nb[1] as f32 * fs) * world_size,
            world_min.z + (fc[2] + nb[2] as f32 * fs) * world_size,
        );
        corners[i] = sdf(p);
    }
    Cell { corners }
}

fn straddles(cell: &Cell) -> bool {
    cell.corners.iter().any(|&c| c <= 0.0) && cell.corners.iter().any(|&c| c > 0.0)
}

// True if the cell might contain or touch the surface.
// Catches thin features (e.g. onion shells) where all corners are the same sign
// but a corner is close enough to the surface that it could pass through the cell.
fn near_surface(cell: &Cell, cell_size: f32) -> bool {
    let threshold = 3.0_f32.sqrt() * cell_size;
    straddles(cell) || cell.corners.iter().any(|&c| c.abs() < threshold)
}

fn march_cell(pos: OctCoord, cell: &Cell, world_min: Vec3, world_size: f32, out: &mut Vec<Vec3>) {
    let fc = pos.float_coords();
    let fs = pos.float_size();
    let origin = Vec3::new(
        world_min.x + fc[0] * world_size,
        world_min.y + fc[1] * world_size,
        world_min.z + fc[2] * world_size,
    );
    let cell_size = fs * world_size;

    let mut accumulator: usize = 0;
    for (i, &c) in cell.corners.iter().enumerate() {
        if c > 0.0 {
            accumulator |= 1 << i;
        }
    }

    let edges = MC_TABLE[accumulator];
    let mut i = 0;
    while i < 12 && edges[i] != -1 {
        for k in 0..3 {
            let e = MC_EDGES[edges[i + k] as usize];
            let t = cell.corners[e[0]] / (cell.corners[e[0]] - cell.corners[e[1]]);
            let nb0 = Vec3::from_array(NEIGHBORS[e[0]].map(|x| x as f32));
            let nb1 = Vec3::from_array(NEIGHBORS[e[1]].map(|x| x as f32));
            out.push(origin + (nb0 + t * (nb1 - nb0)) * cell_size);
        }
        i += 3;
    }
}

fn build_indexed(soup: &[Vec3]) -> (Vec<[f32; 3]>, Vec<u32>) {
    let positions: Vec<[f32; 3]> = soup.iter().map(|v| v.to_array()).collect();
    let (unique_count, remap) = meshopt::generate_vertex_remap(&positions, None);
    let unique_positions = meshopt::remap_vertex_buffer(&positions, unique_count, &remap);
    let indices = meshopt::remap_index_buffer(None, positions.len(), &remap);
    (unique_positions, indices)
}

pub fn optimize_mesh(soup: &[Vec3], max_error: f32) -> (Vec<Vec3>, Vec<u32>) {
    let (verts, indices) = build_indexed(soup);
    println!("Deduplicated: {} vertices, {} triangles", verts.len(), indices.len() / 3);

    let adapter = VertexDataAdapter::new(typed_to_bytes(&verts), std::mem::size_of::<[f32; 3]>(), 0).unwrap();
    let simplified = meshopt::simplify(&indices, &adapter, 3, max_error, SimplifyOptions::ErrorAbsolute, None);
    let simplified = if simplified.len() >= 3 { simplified } else { indices };

    let mut opt_indices = meshopt::optimize_vertex_cache(&simplified, verts.len());
    let opt_verts = meshopt::optimize_vertex_fetch::<[f32; 3]>(&mut opt_indices, &verts);

    println!("Optimized: {} triangles ({:.0}% of raw)", opt_indices.len() / 3,
        100.0 * opt_indices.len() as f32 / soup.len() as f32);

    let final_verts: Vec<Vec3> = opt_verts.into_iter().map(Vec3::from_array).collect();
    (final_verts, opt_indices)
}

pub fn march_sdf(sdf: Sdf, center: Vec3, size: Vec3, max_error: f32) -> (Vec<Vec3>, Vec<u32>) {
    let cube_size = size.max_element();
    let world_min = center - Vec3::splat(cube_size / 2.0);

    // cell_size at depth D = cube_size / 2^D; we want cell_size <= max_error
    let max_depth = ((cube_size / max_error).log2().ceil() as u8).clamp(1, 15);
    println!("Adaptive Marching Cubes: max_depth={}, cell_size={:.4}", max_depth, cube_size / (1u32 << max_depth) as f32);

    // BFS: evaluate candidate cells level-by-level.
    // Cells straddling the surface are subdivided until max_depth, then stored.
    let mut current: Vec<OctCoord> = (0..8)
        .map(|i| OctCoord::root().get_child(i))
        .collect();

    let mut final_cells: HashMap<OctCoord, Cell> = HashMap::new();
    let t0 = Instant::now();

    loop {
        if current.is_empty() {
            break;
        }
        let depth = current[0].depth();
        let mut next: Vec<OctCoord> = Vec::new();

        let cell_size = cube_size / (1u32 << depth) as f32;
        let (child_vecs, final_vecs): (Vec<Vec<OctCoord>>, Vec<Vec<(OctCoord, Cell)>>) = current
            .par_iter()
            .map(|&pos| {
                let cell = eval_cell(&sdf, pos, world_min, cube_size);
                if !near_surface(&cell, cell_size) {
                    return (vec![], vec![]);
                }
                if depth >= max_depth {
                    if straddles(&cell) {
                        (vec![], vec![(pos, cell)])
                    } else {
                        (vec![], vec![])
                    }
                } else {
                    ((0..8).map(|i| pos.get_child(i)).collect(), vec![])
                }
            })
            .unzip();

        next = child_vecs.into_iter().flatten().collect();
        for (pos, cell) in final_vecs.into_iter().flatten() {
            final_cells.insert(pos, cell);
        }

        if next.is_empty() {
            break;
        }
        current = next;
    }

    println!("BFS: {} leaf cells in {:.2?}", final_cells.len(), t0.elapsed());

    // Batch-insert into tree. Sort by position for spatial locality in insert_many.
    let mut tree = OctTree::<Cell, OctCoord>::new();
    let mut positions: Vec<OctCoord> = final_cells.keys().copied().collect();
    positions.sort_unstable_by_key(|p| p.pos);
    tree.insert_many(positions.into_iter(), |pos| {
        final_cells.remove(&pos).unwrap()
    });

    // Collect chunk data (iter_chunks requires &mut self so gather first).
    let chunks: Vec<(OctCoord, [f32; 8])> = tree
        .iter_chunks()
        .map(|(_, cc)| (cc.position(), cc.chunk.corners))
        .collect();

    let raw: Vec<Vec3> = chunks
        .into_par_iter()
        .flat_map_iter(|(pos, corners)| {
            let mut v = Vec::new();
            march_cell(pos, &Cell { corners }, world_min, cube_size, &mut v);
            v
        })
        .collect();

    println!("Generated {} triangles in {:.2?}", raw.len() / 3, t0.elapsed());

    optimize_mesh(&raw, max_error / 4.0)
}

pub fn write_stl(filename: &str, vertices: &[Vec3], indices: &[u32]) {
    let triangles: Vec<Triangle> = indices.chunks_exact(3).map(|tri| {
        let (a, b, c) = (
            vertices[tri[0] as usize],
            vertices[tri[1] as usize],
            vertices[tri[2] as usize],
        );
        let n = (b - a).cross(c - a).normalize_or_zero();
        Triangle {
            normal: Vector::new(n.to_array()),
            vertices: [
                Vector::new(a.to_array()),
                Vector::new(b.to_array()),
                Vector::new(c.to_array()),
            ],
        }
    }).collect();

    let mut file = OpenOptions::new()
        .write(true).truncate(true).create(true)
        .open(filename)
        .expect("could not open STL file for writing");
    stl_io::write_stl(&mut file, triangles.iter()).expect("STL write failed");
}

pub fn write_obj(filename: &str, vertices: &[Vec3], indices: &[u32]) {
    let start = Instant::now();
    let mut buf: Vec<u8> = Vec::new();
    for v in vertices {
        buf.extend(format!("v {} {} {}\n", v.x, v.y, v.z).as_bytes());
    }
    for face in indices.chunks_exact(3) {
        buf.extend(format!("f {} {} {}\n", face[0] + 1, face[1] + 1, face[2] + 1).as_bytes());
    }
    std::fs::write(filename, &buf).expect("could not write OBJ file");
    println!("Wrote {} in {:?}", filename, start.elapsed());
}

// Marching cubes lookup tables

const MC_EDGES: [[usize; 2]; 12] = [
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [3, 7], [2, 6],
];

#[rustfmt::skip]
const MC_TABLE: [[i32; 12]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,8,1,1,8,9,-1,-1,-1,-1,-1,-1],
    [2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,0,11,11,0,2,-1,-1,-1,-1,-1,-1],
    [3,2,11,1,0,9,-1,-1,-1,-1,-1,-1],
    [11,1,2,11,9,1,11,8,9,-1,-1,-1],
    [1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,3,8,2,1,10,-1,-1,-1,-1,-1,-1],
    [10,2,9,9,2,0,-1,-1,-1,-1,-1,-1],
    [8,2,3,8,10,2,8,9,10,-1,-1,-1],
    [11,3,10,10,3,1,-1,-1,-1,-1,-1,-1],
    [10,0,1,10,8,0,10,11,8,-1,-1,-1],
    [9,3,0,9,11,3,9,10,11,-1,-1,-1],
    [8,9,11,11,9,10,-1,-1,-1,-1,-1,-1],
    [4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,4,3,3,4,0,-1,-1,-1,-1,-1,-1],
    [4,8,7,0,9,1,-1,-1,-1,-1,-1,-1],
    [1,4,9,1,7,4,1,3,7,-1,-1,-1],
    [8,7,4,11,3,2,-1,-1,-1,-1,-1,-1],
    [4,11,7,4,2,11,4,0,2,-1,-1,-1],
    [0,9,1,8,7,4,11,3,2,-1,-1,-1],
    [7,4,11,11,4,2,2,4,9,2,9,1],
    [4,8,7,2,1,10,-1,-1,-1,-1,-1,-1],
    [7,4,3,3,4,0,10,2,1,-1,-1,-1],
    [10,2,9,9,2,0,7,4,8,-1,-1,-1],
    [10,2,3,10,3,4,3,7,4,9,10,4],
    [1,10,3,3,10,11,4,8,7,-1,-1,-1],
    [10,11,1,11,7,4,1,11,4,1,4,0],
    [7,4,8,9,3,0,9,11,3,9,10,11],
    [7,4,11,4,9,11,9,10,11,-1,-1,-1],
    [9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,4,5,8,0,3,-1,-1,-1,-1,-1,-1],
    [4,5,0,0,5,1,-1,-1,-1,-1,-1,-1],
    [5,8,4,5,3,8,5,1,3,-1,-1,-1],
    [9,4,5,11,3,2,-1,-1,-1,-1,-1,-1],
    [2,11,0,0,11,8,5,9,4,-1,-1,-1],
    [4,5,0,0,5,1,11,3,2,-1,-1,-1],
    [5,1,4,1,2,11,4,1,11,4,11,8],
    [1,10,2,5,9,4,-1,-1,-1,-1,-1,-1],
    [9,4,5,0,3,8,2,1,10,-1,-1,-1],
    [2,5,10,2,4,5,2,0,4,-1,-1,-1],
    [10,2,5,5,2,4,4,2,3,4,3,8],
    [11,3,10,10,3,1,4,5,9,-1,-1,-1],
    [4,5,9,10,0,1,10,8,0,10,11,8],
    [11,3,0,11,0,5,0,4,5,10,11,5],
    [4,5,8,5,10,8,10,11,8,-1,-1,-1],
    [8,7,9,9,7,5,-1,-1,-1,-1,-1,-1],
    [3,9,0,3,5,9,3,7,5,-1,-1,-1],
    [7,0,8,7,1,0,7,5,1,-1,-1,-1],
    [7,5,3,3,5,1,-1,-1,-1,-1,-1,-1],
    [5,9,7,7,9,8,2,11,3,-1,-1,-1],
    [2,11,7,2,7,9,7,5,9,0,2,9],
    [2,11,3,7,0,8,7,1,0,7,5,1],
    [2,11,1,11,7,1,7,5,1,-1,-1,-1],
    [8,7,9,9,7,5,2,1,10,-1,-1,-1],
    [10,2,1,3,9,0,3,5,9,3,7,5],
    [7,5,8,5,10,2,8,5,2,8,2,0],
    [10,2,5,2,3,5,3,7,5,-1,-1,-1],
    [8,7,5,8,5,9,11,3,10,3,1,10],
    [5,11,7,10,11,5,1,9,0,-1,-1,-1],
    [11,5,10,7,5,11,8,3,0,-1,-1,-1],
    [5,11,7,10,11,5,-1,-1,-1,-1,-1,-1],
    [6,7,11,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,11,6,3,8,0,-1,-1,-1,-1,-1,-1],
    [6,7,11,0,9,1,-1,-1,-1,-1,-1,-1],
    [9,1,8,8,1,3,6,7,11,-1,-1,-1],
    [3,2,7,7,2,6,-1,-1,-1,-1,-1,-1],
    [0,7,8,0,6,7,0,2,6,-1,-1,-1],
    [6,7,2,2,7,3,9,1,0,-1,-1,-1],
    [6,7,8,6,8,1,8,9,1,2,6,1],
    [11,6,7,10,2,1,-1,-1,-1,-1,-1,-1],
    [3,8,0,11,6,7,10,2,1,-1,-1,-1],
    [0,9,2,2,9,10,7,11,6,-1,-1,-1],
    [6,7,11,8,2,3,8,10,2,8,9,10],
    [7,10,6,7,1,10,7,3,1,-1,-1,-1],
    [8,0,7,7,0,6,6,0,1,6,1,10],
    [7,3,6,3,0,9,6,3,9,6,9,10],
    [6,7,10,7,8,10,8,9,10,-1,-1,-1],
    [11,6,8,8,6,4,-1,-1,-1,-1,-1,-1],
    [6,3,11,6,0,3,6,4,0,-1,-1,-1],
    [11,6,8,8,6,4,1,0,9,-1,-1,-1],
    [1,3,9,3,11,6,9,3,6,9,6,4],
    [2,8,3,2,4,8,2,6,4,-1,-1,-1],
    [4,0,6,6,0,2,-1,-1,-1,-1,-1,-1],
    [9,1,0,2,8,3,2,4,8,2,6,4],
    [9,1,4,1,2,4,2,6,4,-1,-1,-1],
    [4,8,6,6,8,11,1,10,2,-1,-1,-1],
    [1,10,2,6,3,11,6,0,3,6,4,0],
    [11,6,4,11,4,8,10,2,9,2,0,9],
    [10,4,9,6,4,10,11,2,3,-1,-1,-1],
    [4,8,3,4,3,10,3,1,10,6,4,10],
    [1,10,0,10,6,0,6,4,0,-1,-1,-1],
    [4,10,6,9,10,4,0,8,3,-1,-1,-1],
    [4,10,6,9,10,4,-1,-1,-1,-1,-1,-1],
    [6,7,11,4,5,9,-1,-1,-1,-1,-1,-1],
    [4,5,9,7,11,6,3,8,0,-1,-1,-1],
    [1,0,5,5,0,4,11,6,7,-1,-1,-1],
    [11,6,7,5,8,4,5,3,8,5,1,3],
    [3,2,7,7,2,6,9,4,5,-1,-1,-1],
    [5,9,4,0,7,8,0,6,7,0,2,6],
    [3,2,6,3,6,7,1,0,5,0,4,5],
    [6,1,2,5,1,6,4,7,8,-1,-1,-1],
    [10,2,1,6,7,11,4,5,9,-1,-1,-1],
    [0,3,8,4,5,9,11,6,7,10,2,1],
    [7,11,6,2,5,10,2,4,5,2,0,4],
    [8,4,7,5,10,6,3,11,2,-1,-1,-1],
    [9,4,5,7,10,6,7,1,10,7,3,1],
    [10,6,5,7,8,4,1,9,0,-1,-1,-1],
    [4,3,0,7,3,4,6,5,10,-1,-1,-1],
    [10,6,5,8,4,7,-1,-1,-1,-1,-1,-1],
    [9,6,5,9,11,6,9,8,11,-1,-1,-1],
    [11,6,3,3,6,0,0,6,5,0,5,9],
    [11,6,5,11,5,0,5,1,0,8,11,0],
    [11,6,3,6,5,3,5,1,3,-1,-1,-1],
    [9,8,5,8,3,2,5,8,2,5,2,6],
    [5,9,6,9,0,6,0,2,6,-1,-1,-1],
    [1,6,5,2,6,1,3,0,8,-1,-1,-1],
    [1,6,5,2,6,1,-1,-1,-1,-1,-1,-1],
    [2,1,10,9,6,5,9,11,6,9,8,11],
    [9,0,1,3,11,2,5,10,6,-1,-1,-1],
    [11,0,8,2,0,11,10,6,5,-1,-1,-1],
    [3,11,2,5,10,6,-1,-1,-1,-1,-1,-1],
    [1,8,3,9,8,1,5,10,6,-1,-1,-1],
    [6,5,10,0,1,9,-1,-1,-1,-1,-1,-1],
    [8,3,0,5,10,6,-1,-1,-1,-1,-1,-1],
    [6,5,10,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,3,8,6,10,5,-1,-1,-1,-1,-1,-1],
    [10,5,6,9,1,0,-1,-1,-1,-1,-1,-1],
    [3,8,1,1,8,9,6,10,5,-1,-1,-1],
    [2,11,3,6,10,5,-1,-1,-1,-1,-1,-1],
    [8,0,11,11,0,2,5,6,10,-1,-1,-1],
    [1,0,9,2,11,3,6,10,5,-1,-1,-1],
    [5,6,10,11,1,2,11,9,1,11,8,9],
    [5,6,1,1,6,2,-1,-1,-1,-1,-1,-1],
    [5,6,1,1,6,2,8,0,3,-1,-1,-1],
    [6,9,5,6,0,9,6,2,0,-1,-1,-1],
    [6,2,5,2,3,8,5,2,8,5,8,9],
    [3,6,11,3,5,6,3,1,5,-1,-1,-1],
    [8,0,1,8,1,6,1,5,6,11,8,6],
    [11,3,6,6,3,5,5,3,0,5,0,9],
    [5,6,9,6,11,9,11,8,9,-1,-1,-1],
    [5,6,10,7,4,8,-1,-1,-1,-1,-1,-1],
    [0,3,4,4,3,7,10,5,6,-1,-1,-1],
    [5,6,10,4,8,7,0,9,1,-1,-1,-1],
    [6,10,5,1,4,9,1,7,4,1,3,7],
    [7,4,8,6,10,5,2,11,3,-1,-1,-1],
    [10,5,6,4,11,7,4,2,11,4,0,2],
    [4,8,7,6,10,5,3,2,11,1,0,9],
    [1,2,10,11,7,6,9,5,4,-1,-1,-1],
    [2,1,6,6,1,5,8,7,4,-1,-1,-1],
    [0,3,7,0,7,4,2,1,6,1,5,6],
    [8,7,4,6,9,5,6,0,9,6,2,0],
    [7,2,3,6,2,7,5,4,9,-1,-1,-1],
    [4,8,7,3,6,11,3,5,6,3,1,5],
    [5,0,1,4,0,5,7,6,11,-1,-1,-1],
    [9,5,4,6,11,7,0,8,3,-1,-1,-1],
    [11,7,6,9,5,4,-1,-1,-1,-1,-1,-1],
    [6,10,4,4,10,9,-1,-1,-1,-1,-1,-1],
    [6,10,4,4,10,9,3,8,0,-1,-1,-1],
    [0,10,1,0,6,10,0,4,6,-1,-1,-1],
    [6,10,1,6,1,8,1,3,8,4,6,8],
    [9,4,10,10,4,6,3,2,11,-1,-1,-1],
    [2,11,8,2,8,0,6,10,4,10,9,4],
    [11,3,2,0,10,1,0,6,10,0,4,6],
    [6,8,4,11,8,6,2,10,1,-1,-1,-1],
    [4,1,9,4,2,1,4,6,2,-1,-1,-1],
    [3,8,0,4,1,9,4,2,1,4,6,2],
    [6,2,4,4,2,0,-1,-1,-1,-1,-1,-1],
    [3,8,2,8,4,2,4,6,2,-1,-1,-1],
    [4,6,9,6,11,3,9,6,3,9,3,1],
    [8,6,11,4,6,8,9,0,1,-1,-1,-1],
    [11,3,6,3,0,6,0,4,6,-1,-1,-1],
    [8,6,11,4,6,8,-1,-1,-1,-1,-1,-1],
    [10,7,6,10,8,7,10,9,8,-1,-1,-1],
    [3,7,0,7,6,10,0,7,10,0,10,9],
    [6,10,7,7,10,8,8,10,1,8,1,0],
    [6,10,7,10,1,7,1,3,7,-1,-1,-1],
    [3,2,11,10,7,6,10,8,7,10,9,8],
    [2,9,0,10,9,2,6,11,7,-1,-1,-1],
    [0,8,3,7,6,11,1,2,10,-1,-1,-1],
    [7,6,11,1,2,10,-1,-1,-1,-1,-1,-1],
    [2,1,9,2,9,7,9,8,7,6,2,7],
    [2,7,6,3,7,2,0,1,9,-1,-1,-1],
    [8,7,0,7,6,0,6,2,0,-1,-1,-1],
    [7,2,3,6,2,7,-1,-1,-1,-1,-1,-1],
    [8,1,9,3,1,8,11,7,6,-1,-1,-1],
    [11,7,6,1,9,0,-1,-1,-1,-1,-1,-1],
    [6,11,7,0,8,3,-1,-1,-1,-1,-1,-1],
    [11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,11,5,5,11,10,-1,-1,-1,-1,-1,-1],
    [10,5,11,11,5,7,0,3,8,-1,-1,-1],
    [7,11,5,5,11,10,0,9,1,-1,-1,-1],
    [7,11,10,7,10,5,3,8,1,8,9,1],
    [5,2,10,5,3,2,5,7,3,-1,-1,-1],
    [5,7,10,7,8,0,10,7,0,10,0,2],
    [0,9,1,5,2,10,5,3,2,5,7,3],
    [9,7,8,5,7,9,10,1,2,-1,-1,-1],
    [1,11,2,1,7,11,1,5,7,-1,-1,-1],
    [8,0,3,1,11,2,1,7,11,1,5,7],
    [7,11,2,7,2,9,2,0,9,5,7,9],
    [7,9,5,8,9,7,3,11,2,-1,-1,-1],
    [3,1,7,7,1,5,-1,-1,-1,-1,-1,-1],
    [8,0,7,0,1,7,1,5,7,-1,-1,-1],
    [0,9,3,9,5,3,5,7,3,-1,-1,-1],
    [9,7,8,5,7,9,-1,-1,-1,-1,-1,-1],
    [8,5,4,8,10,5,8,11,10,-1,-1,-1],
    [0,3,11,0,11,5,11,10,5,4,0,5],
    [1,0,9,8,5,4,8,10,5,8,11,10],
    [10,3,11,1,3,10,9,5,4,-1,-1,-1],
    [3,2,8,8,2,4,4,2,10,4,10,5],
    [10,5,2,5,4,2,4,0,2,-1,-1,-1],
    [5,4,9,8,3,0,10,1,2,-1,-1,-1],
    [2,10,1,4,9,5,-1,-1,-1,-1,-1,-1],
    [8,11,4,11,2,1,4,11,1,4,1,5],
    [0,5,4,1,5,0,2,3,11,-1,-1,-1],
    [0,11,2,8,11,0,4,9,5,-1,-1,-1],
    [5,4,9,2,3,11,-1,-1,-1,-1,-1,-1],
    [4,8,5,8,3,5,3,1,5,-1,-1,-1],
    [0,5,4,1,5,0,-1,-1,-1,-1,-1,-1],
    [5,4,9,3,0,8,-1,-1,-1,-1,-1,-1],
    [5,4,9,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,4,7,11,9,4,11,10,9,-1,-1,-1],
    [0,3,8,11,4,7,11,9,4,11,10,9],
    [11,10,7,10,1,0,7,10,0,7,0,4],
    [3,10,1,11,10,3,7,8,4,-1,-1,-1],
    [3,2,10,3,10,4,10,9,4,7,3,4],
    [9,2,10,0,2,9,8,4,7,-1,-1,-1],
    [3,4,7,0,4,3,1,2,10,-1,-1,-1],
    [7,8,4,10,1,2,-1,-1,-1,-1,-1,-1],
    [7,11,4,4,11,9,9,11,2,9,2,1],
    [1,9,0,4,7,8,2,3,11,-1,-1,-1],
    [7,11,4,11,2,4,2,0,4,-1,-1,-1],
    [4,7,8,2,3,11,-1,-1,-1,-1,-1,-1],
    [9,4,1,4,7,1,7,3,1,-1,-1,-1],
    [7,8,4,1,9,0,-1,-1,-1,-1,-1,-1],
    [3,4,7,0,4,3,-1,-1,-1,-1,-1,-1],
    [7,8,4,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,10,8,8,10,9,-1,-1,-1,-1,-1,-1],
    [0,3,9,3,11,9,11,10,9,-1,-1,-1],
    [1,0,10,0,8,10,8,11,10,-1,-1,-1],
    [10,3,11,1,3,10,-1,-1,-1,-1,-1,-1],
    [3,2,8,2,10,8,10,9,8,-1,-1,-1],
    [9,2,10,0,2,9,-1,-1,-1,-1,-1,-1],
    [8,3,0,10,1,2,-1,-1,-1,-1,-1,-1],
    [2,10,1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,1,11,1,9,11,9,8,11,-1,-1,-1],
    [11,2,3,9,0,1,-1,-1,-1,-1,-1,-1],
    [11,0,8,2,0,11,-1,-1,-1,-1,-1,-1],
    [3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,9,8,1,-1,-1,-1,-1,-1,-1],
    [1,9,0,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,3,0,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
];
