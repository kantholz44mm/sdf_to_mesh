use std::sync::Arc;
use glam::{Vec2, Vec3};

pub type Sdf = Arc<dyn Fn(Vec3) -> f32 + Send + Sync>;

fn sdf<F: Fn(Vec3) -> f32 + Send + Sync + 'static>(f: F) -> Sdf {
    Arc::new(f)
}

// Primitives

pub fn sphere(radius: f32, center: Vec3) -> Sdf {
    sdf(move |p| (p - center).length() - radius)
}

pub fn cuboid(size: Vec3) -> Sdf {
    sdf(move |p| {
        let q = p.abs() - size;
        q.max(Vec3::ZERO).length() + q.x.max(q.y).max(q.z).min(0.0)
    })
}

pub fn cylinder(radius: f32, height: f32) -> Sdf {
    let h = height / 2.0;
    sdf(move |p| {
        let r = Vec2::new(p.x, p.z).length() - radius;
        let y = p.y.abs() - h;
        Vec2::new(r.max(0.0), y.max(0.0)).length() + r.max(y).min(0.0)
    })
}

pub fn capsule(radius: f32, height: f32) -> Sdf {
    let h = height / 2.0;
    sdf(move |p| {
        let cy = p.y.clamp(-h, h);
        Vec3::new(p.x, p.y - cy, p.z).length() - radius
    })
}

pub fn torus(major_radius: f32, minor_radius: f32) -> Sdf {
    sdf(move |p| {
        let q = Vec2::new(Vec2::new(p.x, p.z).length() - major_radius, p.y);
        q.length() - minor_radius
    })
}

pub fn plane(normal: Vec3, offset: f32) -> Sdf {
    let n = normal.normalize();
    sdf(move |p| p.dot(n) + offset)
}

// Transformations

pub fn translate(s: Sdf, offset: Vec3) -> Sdf {
    sdf(move |p| s(p - offset))
}

pub fn rotate(s: Sdf, axis: Vec3, angle: f32) -> Sdf {
    let r = glam::Mat3::from_axis_angle(axis.normalize(), angle);
    let r_inv = r.transpose();
    sdf(move |p| s(r_inv * p))
}

pub fn scale(s: Sdf, factor: f32) -> Sdf {
    sdf(move |p| s(p / factor) * factor)
}

pub fn mirror(s: Sdf, axis: usize) -> Sdf {
    sdf(move |p| {
        let mp = match axis {
            0 => Vec3::new(p.x.abs(), p.y, p.z),
            1 => Vec3::new(p.x, p.y.abs(), p.z),
            2 => Vec3::new(p.x, p.y, p.z.abs()),
            _ => panic!("axis must be 0, 1, or 2"),
        };
        s(mp)
    })
}

pub fn elongate(s: Sdf, size: Vec3) -> Sdf {
    sdf(move |p| s(p - p.clamp(-size, size)))
}

pub fn twist(s: Sdf, k: f32) -> Sdf {
    sdf(move |p| {
        let c = (k * p.z).cos();
        let sin = (k * p.z).sin();
        s(Vec3::new(c * p.x - sin * p.y, sin * p.x + c * p.y, p.z))
    })
}

// Booleans

pub fn union(a: Sdf, b: Sdf) -> Sdf {
    sdf(move |p| a(p).min(b(p)))
}

pub fn intersection(a: Sdf, b: Sdf) -> Sdf {
    sdf(move |p| a(p).max(b(p)))
}

pub fn difference(a: Sdf, b: Sdf) -> Sdf {
    sdf(move |p| a(p).max(-b(p)))
}

// Booleans, but smooth

pub fn smooth_union(a: Sdf, b: Sdf, k: f32) -> Sdf {
    sdf(move |p| {
        let (d1, d2) = (a(p), b(p));
        let h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
        d2 + h * (d1 - d2) - k * h * (1.0 - h)
    })
}

pub fn smooth_intersection(a: Sdf, b: Sdf, k: f32) -> Sdf {
    sdf(move |p| {
        let (d1, d2) = (a(p), b(p));
        let h = (0.5 - 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
        d2 + h * (d1 - d2) + k * h * (1.0 - h)
    })
}

pub fn smooth_difference(a: Sdf, b: Sdf, k: f32) -> Sdf {
    sdf(move |p| {
        let (d1, d2) = (a(p), b(p));
        let h = (0.5 - 0.5 * (d2 + d1) / k).clamp(0.0, 1.0);
        d2 + h * (-d1 - d2) + k * h * (1.0 - h)
    })
}

// Domain operations

pub fn repeat(s: Sdf, period: Vec3, count: Vec3) -> Sdf {
    sdf(move |p| s(p - period * (p / period).round().clamp(-count, count)))
}

pub fn onion(s: Sdf, thickness: f32) -> Sdf {
    sdf(move |p| s(p).abs() - thickness)
}

pub fn offset(s: Sdf, amount: f32) -> Sdf {
    sdf(move |p| s(p) - amount)
}

// Boundary estimation

fn estimate_bounding_box(s: &Sdf, search_range: f32, allowed_error: f32) -> Option<(Vec3, Vec3)> {
    let n = search_range.ceil() as usize;

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let p = Vec3::new(
                    (ix as f32 - (n as f32 - 1.0) / 2.0) * allowed_error,
                    (iy as f32 - (n as f32 - 1.0) / 2.0) * allowed_error,
                    (iz as f32 - (n as f32 - 1.0) / 2.0) * allowed_error,
                );

                if s(p) <= 0.0 {
                    min = min.min(p);
                    max = max.max(p);
                }
            }
        }
    }

    if min.is_finite() && max.is_finite() {
        Some((min - Vec3::splat(allowed_error), max + Vec3::splat(allowed_error)))
    } else {
        None
    }
}

pub fn estimate_bounding_box_iterative(s: &Sdf, search_range: f32) -> Option<(Vec3, Vec3)> {
    
    for subdivision in 2..=4 {
        let allowed_error = 1.0 / subdivision as f32;
        if let Some(bbox) = estimate_bounding_box(s, search_range, allowed_error) {
            return Some(bbox);
        }
    }

    return None;
}