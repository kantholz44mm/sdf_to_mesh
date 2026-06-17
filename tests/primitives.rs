use glam::Vec3;
use mesh::sdf::primitives::{
    capsule, cuboid, cylinder, difference, elongate, intersection, mirror, offset, onion,
    plane, repeat, scale, smooth_difference, smooth_intersection, smooth_union, sphere, torus,
    translate, union, estimate_bounding_box_iterative,
};

const EPS: f32 = 1e-4;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < EPS
}

// ── Sphere ──────────────────────────────────────────────────────────────────

#[test]
fn sphere_center_is_inside() {
    let s = sphere(1.0, Vec3::ZERO);
    assert!(s(Vec3::ZERO) < 0.0);
}

#[test]
fn sphere_surface_is_zero() {
    let s = sphere(1.0, Vec3::ZERO);
    assert!(approx(s(Vec3::X), 0.0));
    assert!(approx(s(Vec3::Y), 0.0));
    assert!(approx(s(Vec3::Z), 0.0));
}

#[test]
fn sphere_outside_is_positive() {
    let s = sphere(1.0, Vec3::ZERO);
    assert!(s(Vec3::splat(2.0)) > 0.0);
}

#[test]
fn sphere_with_center_offset() {
    let c = Vec3::new(3.0, 0.0, 0.0);
    let s = sphere(1.0, c);
    assert!(s(c) < 0.0);
    assert!(approx(s(c + Vec3::X), 0.0));
}

#[test]
fn sphere_distance_is_exact() {
    let s = sphere(1.0, Vec3::ZERO);
    // Point at distance 2 from origin, radius 1 → sdf = 1
    assert!(approx(s(Vec3::new(3.0, 0.0, 0.0)), 2.0));
}

// ── Cuboid ───────────────────────────────────────────────────────────────────

#[test]
fn cuboid_center_is_inside() {
    let c = cuboid(Vec3::ONE);
    assert!(c(Vec3::ZERO) < 0.0);
}

#[test]
fn cuboid_face_center_is_zero() {
    let c = cuboid(Vec3::ONE);
    assert!(approx(c(Vec3::X), 0.0));
}

#[test]
fn cuboid_corner_is_outside() {
    let c = cuboid(Vec3::ONE);
    assert!(c(Vec3::new(2.0, 2.0, 2.0)) > 0.0);
}

#[test]
fn cuboid_point_well_inside() {
    let c = cuboid(Vec3::ONE);
    // Distance from (0,0,0) to nearest face is 1.0, so sdf = -1.0
    assert!(approx(c(Vec3::ZERO), -1.0));
}

// ── Cylinder ─────────────────────────────────────────────────────────────────

#[test]
fn cylinder_center_is_inside() {
    let c = cylinder(1.0, 2.0);
    assert!(c(Vec3::ZERO) < 0.0);
}

#[test]
fn cylinder_side_surface_is_zero() {
    let c = cylinder(1.0, 2.0);
    // On the curved surface: x=1, y=0, z=0
    assert!(approx(c(Vec3::X), 0.0));
}

#[test]
fn cylinder_above_is_positive() {
    let c = cylinder(1.0, 2.0);
    // Above the top cap (height/2 = 1.0)
    assert!(c(Vec3::new(0.0, 2.0, 0.0)) > 0.0);
}

// ── Capsule ───────────────────────────────────────────────────────────────────

#[test]
fn capsule_center_is_inside() {
    let c = capsule(0.5, 2.0);
    assert!(c(Vec3::ZERO) < 0.0);
}

#[test]
fn capsule_pole_surface_is_zero() {
    // Capsule with radius 0.5 and height 2.0: half-height = 1.0
    // Top pole is at (0, 1.0, 0), surface at (0, 1.5, 0)
    let c = capsule(0.5, 2.0);
    assert!(approx(c(Vec3::new(0.0, 1.5, 0.0)), 0.0));
}

#[test]
fn capsule_far_away_is_positive() {
    let c = capsule(0.5, 2.0);
    assert!(c(Vec3::new(0.0, 10.0, 0.0)) > 0.0);
}

// ── Torus ─────────────────────────────────────────────────────────────────────

#[test]
fn torus_on_ring_axis_is_on_surface() {
    // Torus with major=2, minor=0.5: ring at xz-plane radius 2, tube radius 0.5
    // Point at (2, 0, 0) is the center of the tube → sdf = -0.5
    let t = torus(2.0, 0.5);
    assert!(approx(t(Vec3::new(2.0, 0.0, 0.0)), -0.5));
}

#[test]
fn torus_surface_point() {
    // (2.5, 0, 0) is on the outer surface of the tube
    let t = torus(2.0, 0.5);
    assert!(approx(t(Vec3::new(2.5, 0.0, 0.0)), 0.0));
}

#[test]
fn torus_origin_is_outside() {
    let t = torus(2.0, 0.5);
    assert!(t(Vec3::ZERO) > 0.0);
}

// ── Plane ─────────────────────────────────────────────────────────────────────

#[test]
fn plane_above_normal_is_positive() {
    // Plane with upward normal, offset 0 → y=0 plane
    let p = plane(Vec3::Y, 0.0);
    assert!(p(Vec3::new(0.0, 1.0, 0.0)) > 0.0);
}

#[test]
fn plane_below_normal_is_negative() {
    let p = plane(Vec3::Y, 0.0);
    assert!(p(Vec3::new(0.0, -1.0, 0.0)) < 0.0);
}

#[test]
fn plane_offset_shifts_surface() {
    // Plane normal Y, offset 1: surface at y = -1
    let p = plane(Vec3::Y, 1.0);
    assert!(approx(p(Vec3::new(0.0, -1.0, 0.0)), 0.0));
}

// ── Transformations ───────────────────────────────────────────────────────────

#[test]
fn translate_moves_sdf() {
    let s = sphere(1.0, Vec3::ZERO);
    let t = translate(s, Vec3::new(5.0, 0.0, 0.0));
    // Origin should now be outside
    assert!(t(Vec3::ZERO) > 0.0);
    // New center is at (5,0,0)
    assert!(t(Vec3::new(5.0, 0.0, 0.0)) < 0.0);
}

#[test]
fn scale_shrinks_sdf() {
    // Scale sphere(1) by 2 → effective radius 2
    let s = sphere(1.0, Vec3::ZERO);
    let scaled = scale(s, 2.0);
    // Point at radius 1.5 should now be inside (radius 2)
    assert!(scaled(Vec3::new(1.5, 0.0, 0.0)) < 0.0);
    // Point at radius 2 should be on surface
    assert!(approx(scaled(Vec3::new(2.0, 0.0, 0.0)), 0.0));
}

#[test]
fn mirror_x_reflects_sdf() {
    let s = sphere(1.0, Vec3::new(2.0, 0.0, 0.0));
    let m = mirror(s, 0); // mirror over x=0 plane
    // Both (2,0,0) and (-2,0,0) should be inside
    assert!(m(Vec3::new(2.0, 0.0, 0.0)) < 0.0);
    assert!(m(Vec3::new(-2.0, 0.0, 0.0)) < 0.0);
}

#[test]
fn offset_grows_sdf() {
    let s = sphere(1.0, Vec3::ZERO);
    let grown = offset(s, 0.5); // effectively radius 1.5
    assert!(approx(grown(Vec3::new(1.5, 0.0, 0.0)), 0.0));
}

#[test]
fn onion_hollows_sdf() {
    // Unit sphere, onion shell of thickness 0.1
    // Point at radius 0.5 (well inside) should now be outside the shell
    let s = sphere(1.0, Vec3::ZERO);
    let shell = onion(s, 0.1);
    // Inside the shell: sdf of sphere is -0.5 → abs - 0.1 = 0.4 > 0
    assert!(shell(Vec3::new(0.5, 0.0, 0.0)) > 0.0);
    // On the inner surface of the shell: sphere sdf = -0.1 → abs - 0.1 = 0
    assert!(approx(shell(Vec3::new(0.9, 0.0, 0.0)), 0.0));
}

#[test]
fn elongate_extends_sdf() {
    // A sphere elongated 2 units along Y becomes a capsule-like shape
    let s = sphere(1.0, Vec3::ZERO);
    let e = elongate(s, Vec3::new(0.0, 2.0, 0.0));
    // Point at (0, 2, 0) should be inside (elongation extends the domain)
    assert!(e(Vec3::new(0.0, 2.0, 0.0)) < 0.0);
    // Original surface at (0, 1, 0) should still be inside the elongated shape
    assert!(e(Vec3::new(0.0, 1.0, 0.0)) <= 0.0);
}

// ── Boolean operations ────────────────────────────────────────────────────────

#[test]
fn union_combines_shapes() {
    let a = sphere(1.0, Vec3::new(-3.0, 0.0, 0.0));
    let b = sphere(1.0, Vec3::new(3.0, 0.0, 0.0));
    let u = union(a, b);
    assert!(u(Vec3::new(-3.0, 0.0, 0.0)) < 0.0);
    assert!(u(Vec3::new(3.0, 0.0, 0.0)) < 0.0);
    assert!(u(Vec3::ZERO) > 0.0);
}

#[test]
fn union_returns_min() {
    let a = sphere(1.0, Vec3::ZERO);
    let b = sphere(2.0, Vec3::ZERO);
    let u = union(a.clone(), b.clone());
    let p = Vec3::new(1.5, 0.0, 0.0);
    assert!(approx(u(p), a(p).min(b(p))));
}

#[test]
fn intersection_carves_overlap() {
    let a = sphere(2.0, Vec3::new(-1.0, 0.0, 0.0));
    let b = sphere(2.0, Vec3::new(1.0, 0.0, 0.0));
    let i = intersection(a, b);
    // Origin is inside both → inside intersection
    assert!(i(Vec3::ZERO) < 0.0);
    // Far left is only in a, not in b → outside intersection
    assert!(i(Vec3::new(-3.0, 0.0, 0.0)) > 0.0);
}

#[test]
fn difference_subtracts_second_from_first() {
    let a = sphere(2.0, Vec3::ZERO);
    let b = sphere(1.0, Vec3::ZERO);
    let d = difference(a, b);
    // Between radius 1 and 2: inside a, outside b → inside difference
    assert!(d(Vec3::new(1.5, 0.0, 0.0)) < 0.0);
    // Inside b (radius < 1): outside difference
    assert!(d(Vec3::new(0.5, 0.0, 0.0)) > 0.0);
}

// ── Smooth boolean operations ─────────────────────────────────────────────────

#[test]
fn smooth_union_blends_at_junction() {
    // Two spheres touching at origin; smooth union should bridge them
    let a = sphere(1.0, Vec3::new(-0.9, 0.0, 0.0));
    let b = sphere(1.0, Vec3::new(0.9, 0.0, 0.0));
    let su = smooth_union(a.clone(), b.clone(), 0.5);
    let u = union(a.clone(), b.clone());
    // At origin both have same value; smooth_union should be ≤ union
    assert!(su(Vec3::ZERO) <= u(Vec3::ZERO));
}

#[test]
fn smooth_union_matches_union_far_from_junction() {
    let a = sphere(0.5, Vec3::new(-5.0, 0.0, 0.0));
    let b = sphere(0.5, Vec3::new(5.0, 0.0, 0.0));
    let su = smooth_union(a.clone(), b.clone(), 0.1);
    let u = union(a.clone(), b.clone());
    // Far from junction the blend is negligible
    let p = Vec3::new(-5.0, 0.0, 0.0);
    assert!((su(p) - u(p)).abs() < 0.01);
}

#[test]
fn smooth_intersection_inside_both() {
    let a = sphere(2.0, Vec3::ZERO);
    let b = sphere(2.0, Vec3::ZERO);
    let si = smooth_intersection(a, b, 0.5);
    assert!(si(Vec3::ZERO) < 0.0);
}

#[test]
fn smooth_difference_positive_inside_subtracted() {
    let a = sphere(2.0, Vec3::ZERO);
    let b = sphere(1.0, Vec3::ZERO);
    let sd = smooth_difference(a, b, 0.1);
    // Well inside b → should be positive (subtracted out)
    assert!(sd(Vec3::new(0.3, 0.0, 0.0)) > 0.0);
}

// ── Domain operations ─────────────────────────────────────────────────────────

#[test]
fn repeat_creates_copies() {
    // Sphere at origin, repeated every 4 units in X, limited to ±1 copies
    let s = sphere(0.5, Vec3::ZERO);
    let r = repeat(s, Vec3::new(4.0, 100.0, 100.0), Vec3::new(1.0, 0.0, 0.0));
    // Original at (0,0,0)
    assert!(r(Vec3::ZERO) < 0.0);
    // Copy at (4,0,0)
    assert!(r(Vec3::new(4.0, 0.0, 0.0)) < 0.0);
    // Between copies: outside
    assert!(r(Vec3::new(2.0, 0.0, 0.0)) > 0.0);
}

// ── Bounding box estimation ───────────────────────────────────────────────────

#[test]
fn bounding_box_sphere_is_found() {
    let s = sphere(1.0, Vec3::ZERO);
    let bbox = estimate_bounding_box_iterative(&s, 10.0);
    assert!(bbox.is_some(), "bounding box should be found for unit sphere");
}

#[test]
fn bounding_box_sphere_contains_surface() {
    let s = sphere(1.0, Vec3::ZERO);
    let (lo, hi) = estimate_bounding_box_iterative(&s, 10.0).unwrap();
    assert!(lo.x <= -1.0, "lower bound should be at most -1");
    assert!(hi.x >= 1.0, "upper bound should be at least 1");
}

#[test]
fn bounding_box_none_for_empty_sdf() {
    // A sphere far outside the search range should return None
    let s = sphere(0.1, Vec3::new(50.0, 50.0, 50.0));
    let bbox = estimate_bounding_box_iterative(&s, 3.0);
    assert!(bbox.is_none(), "sphere outside search range should yield None");
}
