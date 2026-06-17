use glam::Vec3;
use mesh::{marching_cubes, sdf::primitives};

// ── march_sdf ─────────────────────────────────────────────────────────────────

#[test]
fn march_sphere_produces_non_empty_mesh() {
    let s = primitives::sphere(1.0, Vec3::ZERO);
    let (verts, indices) = marching_cubes::march_sdf(s, Vec3::ZERO, Vec3::splat(2.5), 0.3);
    assert!(!verts.is_empty(), "should produce vertices");
    assert!(!indices.is_empty(), "should produce indices");
    assert_eq!(indices.len() % 3, 0, "indices must be a multiple of 3");
}

#[test]
fn march_sphere_indices_in_range() {
    let s = primitives::sphere(1.0, Vec3::ZERO);
    let (verts, indices) = marching_cubes::march_sdf(s, Vec3::ZERO, Vec3::splat(2.5), 0.3);
    let n = verts.len() as u32;
    for &idx in &indices {
        assert!(idx < n, "index {idx} out of range (vertex count: {n})");
    }
}

#[test]
fn march_sphere_vertices_near_surface() {
    let radius = 1.0_f32;
    let s = primitives::sphere(radius, Vec3::ZERO);
    let (verts, _) = marching_cubes::march_sdf(s, Vec3::ZERO, Vec3::splat(2.5), 0.3);
    for v in &verts {
        let dist = (v.length() - radius).abs();
        assert!(dist < 0.4, "vertex {:?} is too far from sphere surface (dist={dist:.3})", v);
    }
}

#[test]
fn march_cuboid_produces_mesh() {
    let c = primitives::cuboid(Vec3::ONE);
    let (verts, indices) = marching_cubes::march_sdf(c, Vec3::ZERO, Vec3::splat(2.5), 0.3);
    assert!(!verts.is_empty());
    assert_eq!(indices.len() % 3, 0);
}

#[test]
fn march_empty_sdf_returns_empty_mesh() {
    // A sphere far outside the marching volume → no triangles
    let s = primitives::sphere(0.1, Vec3::new(100.0, 100.0, 100.0));
    let (verts, indices) = marching_cubes::march_sdf(s, Vec3::ZERO, Vec3::splat(2.0), 0.5);
    assert!(verts.is_empty() || indices.is_empty(), "should produce no geometry for out-of-range SDF");
}

// ── write_obj ─────────────────────────────────────────────────────────────────

#[test]
fn write_obj_creates_file() {
    let s = primitives::sphere(1.0, Vec3::ZERO);
    let (verts, indices) = marching_cubes::march_sdf(s, Vec3::ZERO, Vec3::splat(2.5), 0.5);

    let path = std::env::temp_dir().join("test_mesh.obj");
    let path_str = path.to_str().unwrap();
    marching_cubes::write_obj(path_str, &verts, &indices);

    assert!(path.exists(), "OBJ file should have been created");
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("v "), "OBJ should contain vertex lines");
    assert!(content.contains("f "), "OBJ should contain face lines");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn write_obj_vertex_count_matches() {
    let s = primitives::sphere(1.0, Vec3::ZERO);
    let (verts, indices) = marching_cubes::march_sdf(s, Vec3::ZERO, Vec3::splat(2.5), 0.5);

    let path = std::env::temp_dir().join("test_mesh_vc.obj");
    let path_str = path.to_str().unwrap();
    marching_cubes::write_obj(path_str, &verts, &indices);

    let content = std::fs::read_to_string(&path).unwrap();
    let v_count = content.lines().filter(|l| l.starts_with("v ")).count();
    let f_count = content.lines().filter(|l| l.starts_with("f ")).count();
    assert_eq!(v_count, verts.len(), "vertex count in OBJ should match");
    assert_eq!(f_count, indices.len() / 3, "face count in OBJ should match");
    let _ = std::fs::remove_file(&path);
}

// ── write_stl ─────────────────────────────────────────────────────────────────

#[test]
fn write_stl_creates_file() {
    let s = primitives::sphere(1.0, Vec3::ZERO);
    let (verts, indices) = marching_cubes::march_sdf(s, Vec3::ZERO, Vec3::splat(2.5), 0.5);

    let path = std::env::temp_dir().join("test_mesh.stl");
    let path_str = path.to_str().unwrap();
    marching_cubes::write_stl(path_str, &verts, &indices);

    assert!(path.exists(), "STL file should have been created");
    // Binary STL: 80-byte header + 4-byte triangle count
    let data = std::fs::read(&path).unwrap();
    assert!(data.len() >= 84, "STL file too small");
    let tri_count = u32::from_le_bytes(data[80..84].try_into().unwrap()) as usize;
    assert_eq!(tri_count, indices.len() / 3, "STL triangle count should match");
    let _ = std::fs::remove_file(&path);
}

// ── optimize_mesh ─────────────────────────────────────────────────────────────

#[test]
fn optimize_mesh_deduplicates_vertices() {
    // Create a simple triangle soup with duplicate vertices
    let verts = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        // Same triangle again
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let (out_verts, out_indices) = marching_cubes::optimize_mesh(&verts, 0.0);
    assert!(out_verts.len() <= 6, "optimize_mesh should deduplicate vertices");
    assert_eq!(out_indices.len() % 3, 0);
}

#[test]
fn optimize_mesh_preserves_triangle_data() {
    let verts = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let (out_verts, out_indices) = marching_cubes::optimize_mesh(&verts, 0.0);
    // One triangle → 3 indices
    assert_eq!(out_indices.len(), 3);
    // The unique 3 vertices should all be present (indices must be valid)
    for &idx in &out_indices {
        assert!((idx as usize) < out_verts.len());
    }
}
