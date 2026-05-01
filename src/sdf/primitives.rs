use std::sync::Arc;
use glam::{Vec2, Vec3};

pub type Sdf = Arc<dyn Fn(Vec3) -> f32 + Send + Sync>;

fn sdf<F: Fn(Vec3) -> f32 + Send + Sync + 'static>(f: F) -> Sdf {
    Arc::new(f)
}

// ── Primitives ──────────────────────────────────────────────────────────────

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

// ── Transforms ───────────────────────────────────────────────────────────────

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

// ── Booleans ─────────────────────────────────────────────────────────────────

pub fn union(a: Sdf, b: Sdf) -> Sdf {
    sdf(move |p| a(p).min(b(p)))
}

pub fn intersection(a: Sdf, b: Sdf) -> Sdf {
    sdf(move |p| a(p).max(b(p)))
}

pub fn difference(a: Sdf, b: Sdf) -> Sdf {
    sdf(move |p| a(p).max(-b(p)))
}

// ── Smooth booleans ───────────────────────────────────────────────────────────

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

// ── Domain operations ────────────────────────────────────────────────────────

pub fn repeat(s: Sdf, period: Vec3, count: Vec3) -> Sdf {
    sdf(move |p| s(p - period * (p / period).round().clamp(-count, count)))
}

pub fn onion(s: Sdf, thickness: f32) -> Sdf {
    sdf(move |p| s(p).abs() - thickness)
}

pub fn offset(s: Sdf, amount: f32) -> Sdf {
    sdf(move |p| s(p) - amount)
}