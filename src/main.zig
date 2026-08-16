const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

// ============================================================================
// Io Backend
// ============================================================================

/// The `std.Io` implementation is chosen at build time via `-Dio=`.
///
/// * `threaded`        - thread pool backed, supports concurrency (default)
/// * `single_threaded` - no concurrency support; `Io.async`/`Io.concurrent`
///                       are unavailable, but blocking file I/O and the
///                       futex-based `Io.Mutex` still work
/// * `evented`         - io_uring on Linux, kqueue on BSD, Dispatch on Darwin
const IoBackend = switch (build_options.io_impl) {
    .threaded, .single_threaded => struct {
        state: std.Io.Threaded,

        fn init(self: *@This(), gpa: std.mem.Allocator) !void {
            self.state = switch (build_options.io_impl) {
                .threaded => .init(gpa, .{}),
                else => .init_single_threaded,
            };
        }

        fn io(self: *@This()) std.Io {
            return self.state.io();
        }

        fn deinit(self: *@This()) void {
            self.state.deinit();
        }
    },
    .evented => struct {
        state: std.Io.Evented,

        fn init(self: *@This(), gpa: std.mem.Allocator) !void {
            comptime checkEventedSupported();
            try self.state.init(gpa, .{});
        }

        fn io(self: *@This()) std.Io {
            return self.state.io();
        }

        fn deinit(self: *@This()) void {
            self.state.deinit();
        }
    },
};

/// `std.Io.Evented` is not usable everywhere. Fail with an actionable message
/// instead of letting the standard library produce confusing errors.
fn checkEventedSupported() void {
    if (std.Io.Evented == void) @compileError(
        "-Dio=evented is not supported on this target; use -Dio=threaded",
    );
    // Zig 0.16.0 ships a std.Io.Uring that cannot compile at all: it lets
    // error.ReadOnlyFileSystem escape dirOpenDir and dirRealPathFile, which are
    // declared to return Dir.OpenError and Dir.RealPathFileError respectively.
    //
    // Upstream bug:  https://codeberg.org/ziglang/zig/issues/32023
    //                (duplicate of https://codeberg.org/ziglang/zig/issues/31828)
    // Fixed by:      https://codeberg.org/ziglang/zig/pulls/31764, merged to
    //                master on 2026-05-27, after 0.16.0 was tagged.
    //
    // Guarded on exactly 0.16.0: it is the only released version verified to be
    // affected, and master already carries the fix. `Uring` is looked up with
    // `@hasDecl` because master renamed the file to `IoUring.zig`.
    if (@hasDecl(std.Io, "Uring")) {
        const broken_release: std.SemanticVersion = .{ .major = 0, .minor = 16, .patch = 0 };
        if (std.Io.Evented == std.Io.Uring and builtin.zig_version.order(broken_release) == .eq) @compileError(
            "-Dio=evented does not compile on Zig 0.16.0: std.Io.Uring lets " ++
                "error.ReadOnlyFileSystem escape Dir.OpenError and Dir.RealPathFileError. " ++
                "This is an upstream standard library bug, not a zaytracer bug: " ++
                "https://codeberg.org/ziglang/zig/issues/32023 (fixed on master by PR 31764). " ++
                "Use -Dio=threaded on this compiler.",
        );
    }
}

// ============================================================================
// Math Utilities
// ============================================================================

fn degreesToRadians(degrees: f64) f64 {
    return degrees * std.math.pi / 180.0;
}

// ============================================================================
// Random Utilities
// ============================================================================

fn randomFloat(rng: std.Random) f64 {
    return rng.float(f64);
}

fn randomFloatRange(rng: std.Random, min: f64, max: f64) f64 {
    return min + (max - min) * randomFloat(rng);
}

fn randomVec3(rng: std.Random) Vec3 {
    return Vec3{
        randomFloat(rng),
        randomFloat(rng),
        randomFloat(rng),
    };
}

fn randomVec3Range(rng: std.Random, min: f64, max: f64) Vec3 {
    return Vec3{
        randomFloatRange(rng, min, max),
        randomFloatRange(rng, min, max),
        randomFloatRange(rng, min, max),
    };
}

fn randomInUnitSphere(rng: std.Random) Vec3 {
    while (true) {
        const p = randomVec3Range(rng, -1.0, 1.0);
        if (lengthSquared(p) < 1.0) {
            return p;
        }
    }
}

fn randomUnitVector(rng: std.Random) Vec3 {
    return unitVector(randomInUnitSphere(rng));
}

fn randomInUnitDisk(rng: std.Random) Vec3 {
    while (true) {
        const p = Vec3{
            randomFloatRange(rng, -1.0, 1.0),
            randomFloatRange(rng, -1.0, 1.0),
            0.0,
        };
        if (lengthSquared(p) < 1.0) {
            return p;
        }
    }
}

// ============================================================================
// Vec3 - 3D Vector/Point/Color (SIMD-optimized)
// ============================================================================

// SIMD vector type - uses hardware vector instructions (AVX/SSE/NEON)
const Vec3 = @Vector(3, f64);

// Type aliases for semantic clarity
const Point3 = Vec3;
const Color = Vec3;

// Helper for construction
pub inline fn vec3(x: f64, y: f64, z: f64) Vec3 {
    return Vec3{ x, y, z };
}

// Arithmetic operations (leverage SIMD operators)
pub inline fn add(a: Vec3, b: Vec3) Vec3 {
    return a + b;
}

pub inline fn sub(a: Vec3, b: Vec3) Vec3 {
    return a - b;
}

pub inline fn mul(v: Vec3, s: f64) Vec3 {
    return v * @as(Vec3, @splat(s));
}

pub inline fn mulVec(a: Vec3, b: Vec3) Vec3 {
    return a * b;
}

pub inline fn div(v: Vec3, s: f64) Vec3 {
    return v / @as(Vec3, @splat(s));
}

pub inline fn neg(v: Vec3) Vec3 {
    return -v;
}

// Reduction operations
pub inline fn dot(a: Vec3, b: Vec3) f64 {
    return @reduce(.Add, a * b);
}

pub inline fn lengthSquared(v: Vec3) f64 {
    return dot(v, v);
}

pub inline fn length(v: Vec3) f64 {
    return @sqrt(lengthSquared(v));
}

pub inline fn unitVector(v: Vec3) Vec3 {
    return div(v, length(v));
}

// Cross product (requires shuffling)
pub inline fn cross(a: Vec3, b: Vec3) Vec3 {
    const a_yzx = @shuffle(f64, a, undefined, [3]i32{ 1, 2, 0 });
    const a_zxy = @shuffle(f64, a, undefined, [3]i32{ 2, 0, 1 });
    const b_yzx = @shuffle(f64, b, undefined, [3]i32{ 1, 2, 0 });
    const b_zxy = @shuffle(f64, b, undefined, [3]i32{ 2, 0, 1 });
    return a_yzx * b_zxy - a_zxy * b_yzx;
}

// Comparison
pub inline fn nearZero(v: Vec3) bool {
    const s = @as(Vec3, @splat(1e-8));
    const abs_v = @abs(v);
    const cmp = abs_v < s;
    return @reduce(.And, cmp);
}

// ============================================================================
// Vector Utilities
// ============================================================================

fn reflect(v: Vec3, n: Vec3) Vec3 {
    return sub(v, mul(n, 2.0 * dot(v, n)));
}

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f64) Vec3 {
    const cos_theta = @min(dot(neg(uv), n), 1.0);
    const r_out_perp = mul(add(uv, mul(n, cos_theta)), etai_over_etat);
    const r_out_parallel = mul(n, -@sqrt(@abs(1.0 - lengthSquared(r_out_perp))));
    return add(r_out_perp, r_out_parallel);
}

fn reflectance(cosine: f64, refraction_index: f64) f64 {
    // Use Schlick's approximation for reflectance
    var r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * std.math.pow(f64, 1.0 - cosine, 5);
}

// ============================================================================
// Material
// ============================================================================

const MaterialType = enum {
    lambertian,
    metal,
    dielectric,
    diffuse_light,
};

/// Whether a material bends light along a single (possibly fuzzed) direction.
/// A photon has to bounce off one of these before it can form a caustic.
fn isSpecular(material_type: MaterialType) bool {
    return switch (material_type) {
        .lambertian, .diffuse_light => false,
        .metal, .dielectric => true,
    };
}

const Material = struct {
    material_type: MaterialType,
    albedo: Color,
    fuzz: f64, // Only used for metal
    refraction_index: f64, // Only used for dielectric
    emission: Color, // Only used for diffuse_light

    pub fn lambertian(albedo: Color) Material {
        return Material{
            .material_type = .lambertian,
            .albedo = albedo,
            .fuzz = 0.0,
            .refraction_index = 0.0,
            .emission = Color{ 0, 0, 0 },
        };
    }

    pub fn metal(albedo: Color, fuzz: f64) Material {
        return Material{
            .material_type = .metal,
            .albedo = albedo,
            .fuzz = if (fuzz < 1.0) fuzz else 1.0,
            .refraction_index = 0.0,
            .emission = Color{ 0, 0, 0 },
        };
    }

    pub fn dielectric(refraction_index: f64) Material {
        return Material{
            .material_type = .dielectric,
            .albedo = Color{ 1.0, 1.0, 1.0 },
            .fuzz = 0.0,
            .refraction_index = refraction_index,
            .emission = Color{ 0, 0, 0 },
        };
    }

    /// A surface that emits light and reflects none of it. This is the only
    /// thing in the renderer that puts light into a closed scene: without one,
    /// a sealed room is lit by nothing but its caustics.
    pub fn diffuseLight(color: Color, intensity: f64) Material {
        return Material{
            .material_type = .diffuse_light,
            .albedo = Color{ 0, 0, 0 },
            .fuzz = 0.0,
            .refraction_index = 0.0,
            .emission = mul(color, intensity),
        };
    }

    /// Light leaving the surface on its own account, before anything bounces.
    ///
    /// One-sided: only the face the geometric normal points out of emits. A
    /// panel hung below a ceiling would otherwise light the gap above it from
    /// a few centimetres away, and direct sampling divides by the square of
    /// that distance.
    pub fn emitted(self: Material, front_face: bool) Color {
        return switch (self.material_type) {
            .diffuse_light => if (front_face) self.emission else Color{ 0, 0, 0 },
            .lambertian, .metal, .dielectric => Color{ 0, 0, 0 },
        };
    }

    pub fn scatter(self: Material, ray_in: Ray, rec: HitRecord, attenuation: *Color, scattered: *Ray, rng: std.Random) bool {
        switch (self.material_type) {
            .lambertian => {
                var scatter_direction = add(rec.normal, randomUnitVector(rng));

                // Catch degenerate scatter direction
                if (nearZero(scatter_direction)) {
                    scatter_direction = rec.normal;
                }

                scattered.* = Ray.init(rec.point, scatter_direction);
                attenuation.* = self.albedo;
                return true;
            },
            .metal => {
                const reflected = reflect(unitVector(ray_in.direction), rec.normal);
                scattered.* = Ray.init(rec.point, add(reflected, mul(randomInUnitSphere(rng), self.fuzz)));
                attenuation.* = self.albedo;
                return dot(scattered.direction, rec.normal) > 0;
            },
            .dielectric => {
                attenuation.* = Color{ 1.0, 1.0, 1.0 };
                const ri = if (rec.front_face) (1.0 / self.refraction_index) else self.refraction_index;

                const unit_direction = unitVector(ray_in.direction);
                const cos_theta = @min(dot(neg(unit_direction), rec.normal), 1.0);
                const sin_theta = @sqrt(1.0 - cos_theta * cos_theta);

                const cannot_refract = ri * sin_theta > 1.0;
                const direction = if (cannot_refract or reflectance(cos_theta, ri) > randomFloat(rng))
                    reflect(unit_direction, rec.normal)
                else
                    refract(unit_direction, rec.normal, ri);

                scattered.* = Ray.init(rec.point, direction);
                return true;
            },
            .diffuse_light => {
                // A light source only emits; nothing bounces off it.
                return false;
            },
        }
    }
};

// ============================================================================
// Light Sources
// ============================================================================

const LightType = enum {
    point,
    quad,
};

const Light = struct {
    light_type: LightType,
    /// Where the photon pass emits from. For a quad that is its centre: the
    /// caustic map is built from a point source even when the light has area.
    position: Point3,
    intensity: Color,
    power: f64,

    /// Quad only: a corner of the emitting rectangle and its two edges.
    corner: Point3 = Point3{ 0, 0, 0 },
    edge_u: Vec3 = Vec3{ 0, 0, 0 },
    edge_v: Vec3 = Vec3{ 0, 0, 0 },
    /// Quad only: radiance leaving the surface. Should match the emission of
    /// the diffuse_light material on the quad's own geometry, or the lamp will
    /// not look as bright as the light it casts.
    emission: Color = Color{ 0, 0, 0 },

    pub fn pointLight(position: Point3, intensity: Color, power: f64) Light {
        return Light{
            .light_type = .point,
            .position = position,
            .intensity = intensity,
            .power = power,
        };
    }

    /// A rectangular area light spanning `corner + s*edge_u + t*edge_v` for
    /// s, t in [0, 1], emitting from the face `edge_u x edge_v` points out of.
    /// Pass the same corner and edges to `SceneData.addQuad` and the lamp's
    /// geometry will face the same way as the light it stands for.
    ///
    /// `emission` is the radiance leaving the surface; the flux the caustic
    /// pass emits is derived from it, so the lamp's brightness and the caustics
    /// it throws cannot drift apart.
    pub fn quadLight(corner: Point3, edge_u: Vec3, edge_v: Vec3, emission: Color) Light {
        const brightest = @max(@max(emission[0], emission[1]), emission[2]);
        const hue = if (brightest > 0.0) div(emission, brightest) else Color{ 1, 1, 1 };
        const quad_area = length(cross(edge_u, edge_v));

        return Light{
            .light_type = .quad,
            .position = add(corner, mul(add(edge_u, edge_v), 0.5)),
            .intensity = hue,
            // Radiant flux of a lambertian emitter: radiance x area x pi.
            .power = std.math.pi * quad_area * brightest,
            .corner = corner,
            .edge_u = edge_u,
            .edge_v = edge_v,
            .emission = emission,
        };
    }

    pub fn area(self: Light) f64 {
        return switch (self.light_type) {
            .point => 0.0,
            .quad => length(cross(self.edge_u, self.edge_v)),
        };
    }

    /// Light arriving straight from this source at a lambertian surface, found
    /// by sampling the source rather than waiting for a bounce to stumble into
    /// it. Without this a small lamp in a closed room is almost pure noise.
    ///
    /// Point lights return nothing: they exist only to drive the caustic map,
    /// and always have. Making them cast direct light too would relight every
    /// scene that uses one.
    pub fn sampleDirect(
        self: Light,
        world: BVH,
        point: Point3,
        normal: Vec3,
        albedo: Color,
        rng: std.Random,
    ) Color {
        if (self.light_type != .quad) return Color{ 0, 0, 0 };

        const on_light = add(add(
            self.corner,
            mul(self.edge_u, randomFloat(rng)),
        ), mul(self.edge_v, randomFloat(rng)));

        const to_light = sub(on_light, point);
        const distance_squared = lengthSquared(to_light);
        if (distance_squared <= 0.0) return Color{ 0, 0, 0 };

        const distance = @sqrt(distance_squared);
        const direction = div(to_light, distance);

        const cos_surface = dot(normal, direction);
        if (cos_surface <= 0.0) return Color{ 0, 0, 0 };

        // One-sided, matching the diffuse_light material: the emitting face is
        // the one the quad's normal points out of, which for a ceiling panel is
        // downwards. Without this a point on the ceiling a few centimetres
        // above the panel samples it across almost no distance at all, and the
        // 1/d^2 below turns into a white speck that no amount of sampling
        // averages away.
        const light_normal = unitVector(cross(self.edge_u, self.edge_v));
        const cos_light = -dot(light_normal, direction);
        if (cos_light <= 1e-8) return Color{ 0, 0, 0 };

        // Stop short of the light itself, or the lamp shadows its own surface.
        var occluder: HitRecord = undefined;
        if (world.hit(
            Ray.init(point, direction),
            Interval{ .min = 0.001, .max = distance - 0.001 },
            &occluder,
        )) {
            return Color{ 0, 0, 0 };
        }

        // Lambertian BRDF (albedo/pi) times the geometry term that converts the
        // area sample into a solid angle.
        const geometry = cos_surface * cos_light * self.area() / distance_squared;
        return mulVec(mul(albedo, geometry / std.math.pi), self.emission);
    }
};

// ============================================================================
// Photon
// ============================================================================

const Photon = struct {
    position: Point3,
    direction: Vec3, // Incoming direction
    power: Color, // RGB power/flux
};

// ============================================================================
// Ray
// ============================================================================

const Ray = struct {
    origin: Point3,
    direction: Vec3,

    pub fn init(origin: Point3, direction: Vec3) Ray {
        return Ray{ .origin = origin, .direction = direction };
    }

    pub inline fn at(self: Ray, t: f64) Point3 {
        return add(self.origin, mul(self.direction, t));
    }
};

// ============================================================================
// Interval
// ============================================================================

const Interval = struct {
    min: f64,
    max: f64,

    pub fn contains(self: Interval, x: f64) bool {
        return self.min <= x and x <= self.max;
    }

    pub fn surrounds(self: Interval, x: f64) bool {
        return self.min < x and x < self.max;
    }

    pub fn expand(self: Interval, delta: f64) Interval {
        const padding = delta / 2.0;
        return Interval{
            .min = self.min - padding,
            .max = self.max + padding,
        };
    }

    pub fn size(self: Interval) f64 {
        return self.max - self.min;
    }

    pub const empty = Interval{ .min = std.math.inf(f64), .max = -std.math.inf(f64) };
    pub const universe = Interval{ .min = -std.math.inf(f64), .max = std.math.inf(f64) };
};

// ============================================================================
// AABB - Axis-Aligned Bounding Box
// ============================================================================

const AABB = struct {
    x: Interval,
    y: Interval,
    z: Interval,

    pub fn init(x: Interval, y: Interval, z: Interval) AABB {
        return AABB{ .x = x, .y = y, .z = z };
    }

    pub fn fromPoints(a: Point3, b: Point3) AABB {
        // Create AABB from two corner points
        return AABB{
            .x = Interval{ .min = @min(a[0], b[0]), .max = @max(a[0], b[0]) },
            .y = Interval{ .min = @min(a[1], b[1]), .max = @max(a[1], b[1]) },
            .z = Interval{ .min = @min(a[2], b[2]), .max = @max(a[2], b[2]) },
        };
    }

    pub fn fromBoxes(box0: AABB, box1: AABB) AABB {
        // Create AABB that encloses two boxes
        return AABB{
            .x = Interval{
                .min = @min(box0.x.min, box1.x.min),
                .max = @max(box0.x.max, box1.x.max),
            },
            .y = Interval{
                .min = @min(box0.y.min, box1.y.min),
                .max = @max(box0.y.max, box1.y.max),
            },
            .z = Interval{
                .min = @min(box0.z.min, box1.z.min),
                .max = @max(box0.z.max, box1.z.max),
            },
        };
    }

    pub fn axis(self: AABB, n: usize) Interval {
        return switch (n) {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            else => unreachable,
        };
    }

    pub fn hit(self: AABB, ray: Ray, ray_t: Interval) bool {
        const ray_orig = ray.origin;
        const ray_dir = ray.direction;

        var t_min = ray_t.min;
        var t_max = ray_t.max;

        // Check intersection with each axis slab
        inline for (0..3) |axis_idx| {
            const ax = self.axis(axis_idx);
            const inv_d = 1.0 / ray_dir[axis_idx];

            const t0 = (ax.min - ray_orig[axis_idx]) * inv_d;
            const t1 = (ax.max - ray_orig[axis_idx]) * inv_d;

            if (inv_d < 0.0) {
                t_min = @max(t_min, t1);
                t_max = @min(t_max, t0);
            } else {
                t_min = @max(t_min, t0);
                t_max = @min(t_max, t1);
            }

            if (t_max <= t_min) {
                return false;
            }
        }

        return true;
    }

    pub fn longestAxis(self: AABB) usize {
        const x_size = self.x.size();
        const y_size = self.y.size();
        const z_size = self.z.size();

        if (x_size > y_size) {
            return if (x_size > z_size) 0 else 2;
        } else {
            return if (y_size > z_size) 1 else 2;
        }
    }

    pub const empty = AABB{
        .x = Interval.empty,
        .y = Interval.empty,
        .z = Interval.empty,
    };
};

// ============================================================================
// Hit Record
// ============================================================================

const HitRecord = struct {
    point: Point3,
    normal: Vec3,
    material: Material,
    t: f64,
    front_face: bool,

    pub fn setFaceNormal(self: *HitRecord, ray: Ray, outward_normal: Vec3) void {
        self.front_face = dot(ray.direction, outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else neg(outward_normal);
    }
};

// ============================================================================
// Hittable - Sphere
// ============================================================================

const Sphere = struct {
    center: Point3,
    radius: f64,
    material: Material,

    pub fn init(center: Point3, radius: f64, material: Material) Sphere {
        return Sphere{ .center = center, .radius = radius, .material = material };
    }

    pub fn boundingBox(self: Sphere) AABB {
        const rvec = Vec3{ self.radius, self.radius, self.radius };
        return AABB.fromPoints(sub(self.center, rvec), add(self.center, rvec));
    }

    pub fn hit(self: Sphere, ray: Ray, ray_t: Interval, rec: *HitRecord) bool {
        const oc = sub(self.center, ray.origin);
        const a = lengthSquared(ray.direction);
        const h = dot(ray.direction, oc);
        const c = lengthSquared(oc) - self.radius * self.radius;
        const discriminant = h * h - a * c;

        if (discriminant < 0) {
            return false;
        }

        const sqrtd = @sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range
        var root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root)) {
                return false;
            }
        }

        rec.t = root;
        rec.point = ray.at(rec.t);
        rec.material = self.material;
        const outward_normal = div(sub(rec.point, self.center), self.radius);
        rec.setFaceNormal(ray, outward_normal);

        return true;
    }
};

// ============================================================================
// Hittable - Triangle
// ============================================================================

const Triangle = struct {
    // Vertex positions
    v0: Point3,
    v1: Point3,
    v2: Point3,

    // Vertex normals (for smooth shading)
    n0: Vec3,
    n1: Vec3,
    n2: Vec3,
    has_normals: bool,

    // Material
    material: Material,
    material_index: u32, // Future: per-face materials (currently unused, set to 0)

    // Precomputed edges (performance optimization)
    edge1: Vec3, // v1 - v0
    edge2: Vec3, // v2 - v0

    /// Initialize triangle with flat shading (face normal)
    pub fn init(v0: Point3, v1: Point3, v2: Point3, material: Material) Triangle {
        const edge1 = sub(v1, v0);
        const edge2 = sub(v2, v0);
        const flat_normal = unitVector(cross(edge1, edge2));

        return Triangle{
            .v0 = v0,
            .v1 = v1,
            .v2 = v2,
            .n0 = flat_normal,
            .n1 = flat_normal,
            .n2 = flat_normal,
            .has_normals = false,
            .material = material,
            .material_index = 0,
            .edge1 = edge1,
            .edge2 = edge2,
        };
    }

    /// Initialize triangle with smooth shading (interpolated vertex normals)
    pub fn initWithNormals(
        v0: Point3,
        v1: Point3,
        v2: Point3,
        n0: Vec3,
        n1: Vec3,
        n2: Vec3,
        material: Material,
    ) Triangle {
        return Triangle{
            .v0 = v0,
            .v1 = v1,
            .v2 = v2,
            .n0 = unitVector(n0), // Normalize inputs
            .n1 = unitVector(n1),
            .n2 = unitVector(n2),
            .has_normals = true,
            .material = material,
            .material_index = 0,
            .edge1 = sub(v1, v0),
            .edge2 = sub(v2, v0),
        };
    }

    /// Compute axis-aligned bounding box for the triangle
    pub fn boundingBox(self: Triangle) AABB {
        // Find min/max of triangle vertices for each axis
        const min_x = @min(@min(self.v0[0], self.v1[0]), self.v2[0]);
        const max_x = @max(@max(self.v0[0], self.v1[0]), self.v2[0]);
        const min_y = @min(@min(self.v0[1], self.v1[1]), self.v2[1]);
        const max_y = @max(@max(self.v0[1], self.v1[1]), self.v2[1]);
        const min_z = @min(@min(self.v0[2], self.v1[2]), self.v2[2]);
        const max_z = @max(@max(self.v0[2], self.v1[2]), self.v2[2]);

        // Add epsilon to prevent degenerate boxes (flat triangles have zero volume otherwise)
        const epsilon = 0.0001;

        return AABB{
            .x = Interval{ .min = min_x - epsilon, .max = max_x + epsilon },
            .y = Interval{ .min = min_y - epsilon, .max = max_y + epsilon },
            .z = Interval{ .min = min_z - epsilon, .max = max_z + epsilon },
        };
    }

    /// Möller-Trumbore ray-triangle intersection algorithm
    /// Returns true if ray hits triangle, fills in hit record with barycentric-interpolated normal
    pub fn hit(self: Triangle, ray: Ray, ray_t: Interval, rec: *HitRecord) bool {
        const epsilon = 1e-8;

        // Step 1: Calculate determinant (tests if ray is parallel to triangle)
        const pvec = cross(ray.direction, self.edge2);
        const det = dot(self.edge1, pvec);

        // If determinant is near zero, ray is parallel to triangle
        if (@abs(det) < epsilon) {
            return false;
        }

        const inv_det = 1.0 / det;

        // Step 2: Calculate u parameter (first barycentric coordinate)
        const tvec = sub(ray.origin, self.v0);
        const u = dot(tvec, pvec) * inv_det;

        // Check if intersection is outside triangle (u bounds)
        if (u < 0.0 or u > 1.0) {
            return false;
        }

        // Step 3: Calculate v parameter (second barycentric coordinate)
        const qvec = cross(tvec, self.edge1);
        const v = dot(ray.direction, qvec) * inv_det;

        // Check if intersection is outside triangle (v bounds)
        if (v < 0.0 or u + v > 1.0) {
            return false;
        }

        // Step 4: Calculate t (distance along ray)
        const t = dot(self.edge2, qvec) * inv_det;

        // Check if intersection is within valid ray interval
        if (!ray_t.surrounds(t)) {
            return false;
        }

        // Step 5: Valid hit! Fill in hit record
        rec.t = t;
        rec.point = ray.at(t);
        rec.material = self.material;

        // Interpolate normal using barycentric coordinates
        // Barycentric coords: (w, u, v) where w = 1-u-v
        // These are weights for (v0, v1, v2)
        const w = 1.0 - u - v;

        const outward_normal = if (self.has_normals)
            // SMOOTH SHADING: Interpolate vertex normals
            unitVector(add(add(mul(self.n0, w), mul(self.n1, u)), mul(self.n2, v)))
        else
            // FLAT SHADING: Use face normal
            self.n0;

        rec.setFaceNormal(ray, outward_normal);

        return true;
    }
};

// ============================================================================
// Primitive - Union of all hittable primitives
// ============================================================================

const Primitive = union(enum) {
    sphere: Sphere,
    triangle: Triangle,

    pub fn boundingBox(self: Primitive) AABB {
        return switch (self) {
            .sphere => |s| s.boundingBox(),
            .triangle => |t| t.boundingBox(),
        };
    }

    pub fn material(self: Primitive) Material {
        return switch (self) {
            .sphere => |s| s.material,
            .triangle => |t| t.material,
        };
    }

    pub fn hit(self: Primitive, ray: Ray, ray_t: Interval, rec: *HitRecord) bool {
        return switch (self) {
            .sphere => |s| s.hit(ray, ray_t, rec),
            .triangle => |t| t.hit(ray, ray_t, rec),
        };
    }
};

// ============================================================================
// Hittable List
// ============================================================================

const HittableList = struct {
    spheres: []const Sphere,

    pub fn init(spheres: []const Sphere) HittableList {
        return HittableList{ .spheres = spheres };
    }

    pub fn hit(self: HittableList, ray: Ray, ray_t: Interval, rec: *HitRecord) bool {
        var temp_rec: HitRecord = undefined;
        var hit_anything = false;
        var closest_so_far = ray_t.max;

        for (self.spheres) |sphere| {
            if (sphere.hit(ray, Interval{ .min = ray_t.min, .max = closest_so_far }, &temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec.* = temp_rec;
            }
        }

        return hit_anything;
    }
};

// ============================================================================
// BVH - Bounding Volume Hierarchy
// ============================================================================

const BVHNode = struct {
    bbox: AABB,
    left: u32, // Index of left child (or first primitive index if leaf)
    right: u32, // Index of right child (or past-the-end primitive index if leaf)
    is_leaf: bool,

    pub fn makeLeaf(bbox: AABB, first: u32, count: u32) BVHNode {
        return BVHNode{
            .bbox = bbox,
            .left = first,
            .right = first + count,
            .is_leaf = true,
        };
    }

    pub fn makeInterior(bbox: AABB, left: u32, right: u32) BVHNode {
        return BVHNode{
            .bbox = bbox,
            .left = left,
            .right = right,
            .is_leaf = false,
        };
    }
};

const BVH = struct {
    nodes: []BVHNode,
    primitives: []const Primitive,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, primitives: []Primitive) !BVH {
        if (primitives.len == 0) {
            return BVH{
                .nodes = &[_]BVHNode{},
                .primitives = primitives,
                .allocator = allocator,
            };
        }

        // Allocate maximum possible nodes (2 * n - 1 for binary tree)
        var nodes = try allocator.alloc(BVHNode, 2 * primitives.len);
        var node_count: usize = 0;

        // Build the BVH tree
        _ = try buildBVH(allocator, primitives, 0, nodes, &node_count);

        // Trim to actual size
        nodes = try allocator.realloc(nodes, node_count);

        return BVH{
            .nodes = nodes,
            .primitives = primitives,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: BVH) void {
        self.allocator.free(self.nodes);
    }

    pub fn hit(self: BVH, ray: Ray, ray_t: Interval, rec: *HitRecord) bool {
        if (self.nodes.len == 0) return false;
        return hitNode(self, 0, ray, ray_t, rec);
    }

    fn hitNode(self: BVH, node_idx: u32, ray: Ray, ray_t: Interval, rec: *HitRecord) bool {
        const node = self.nodes[node_idx];

        // Early exit if ray doesn't hit bounding box
        if (!node.bbox.hit(ray, ray_t)) {
            return false;
        }

        if (node.is_leaf) {
            // Test all primitives in this leaf
            var hit_anything = false;
            var closest_so_far = ray_t.max;
            var temp_rec: HitRecord = undefined;

            var i = node.left;
            while (i < node.right) : (i += 1) {
                if (self.primitives[i].hit(ray, Interval{ .min = ray_t.min, .max = closest_so_far }, &temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec.* = temp_rec;
                }
            }

            return hit_anything;
        } else {
            // Test both children
            var temp_rec: HitRecord = undefined;
            const hit_left = hitNode(self, node.left, ray, ray_t, &temp_rec);
            const closest_so_far = if (hit_left) temp_rec.t else ray_t.max;

            const hit_right = hitNode(self, node.right, ray, Interval{ .min = ray_t.min, .max = closest_so_far }, rec);

            if (hit_right) {
                return true;
            } else if (hit_left) {
                rec.* = temp_rec;
                return true;
            }

            return false;
        }
    }
};

fn buildBVH(allocator: std.mem.Allocator, primitives: []Primitive, primitive_offset: u32, nodes: []BVHNode, node_count: *usize) !u32 {
    const node_idx = @as(u32, @intCast(node_count.*));
    node_count.* += 1;

    // Compute bounding box for all primitives
    var bbox = primitives[0].boundingBox();
    for (primitives[1..]) |primitive| {
        bbox = AABB.fromBoxes(bbox, primitive.boundingBox());
    }

    // Leaf node if few primitives
    const leaf_threshold = 4;
    if (primitives.len <= leaf_threshold) {
        nodes[node_idx] = BVHNode.makeLeaf(bbox, primitive_offset, @intCast(primitives.len));
        return node_idx;
    }

    // Choose split axis (longest bbox axis)
    const axis = bbox.longestAxis();

    // Sort primitives along chosen axis
    const SortContext = struct {
        axis_idx: usize,

        pub fn lessThan(ctx: @This(), a: Primitive, b: Primitive) bool {
            // Compute center/centroid for each primitive type
            const a_center = switch (a) {
                .sphere => |s| s.center,
                .triangle => |t| div(add(add(t.v0, t.v1), t.v2), 3.0), // Triangle centroid
            };
            const b_center = switch (b) {
                .sphere => |s| s.center,
                .triangle => |t| div(add(add(t.v0, t.v1), t.v2), 3.0),
            };
            return switch (ctx.axis_idx) {
                0 => a_center[0] < b_center[0],
                1 => a_center[1] < b_center[1],
                else => a_center[2] < b_center[2],
            };
        }
    };

    std.mem.sort(Primitive, primitives, SortContext{ .axis_idx = axis }, SortContext.lessThan);

    // Split in the middle
    const mid = primitives.len / 2;

    // Recursively build left and right subtrees
    const left_idx = try buildBVH(allocator, primitives[0..mid], primitive_offset, nodes, node_count);
    const right_idx = try buildBVH(allocator, primitives[mid..], primitive_offset + @as(u32, @intCast(mid)), nodes, node_count);

    // Create interior node
    nodes[node_idx] = BVHNode.makeInterior(bbox, left_idx, right_idx);

    return node_idx;
}

// ============================================================================
// Photon Map - Spatial Grid for Caustics
// ============================================================================

// Caustic photon map tuning. The map only holds LS+D photons (see tracePhoton),
// so the budget is spent on directions that can actually reach a specular
// object and the gather radius can be small enough to keep caustics sharp.

/// Photons emitted from the light before rendering.
const photons_emitted: u32 = 1_000_000;
/// Upper bound on how many LS+D photons the map can store. Only a fraction of
/// the emitted photons complete such a path, so this is well below the number
/// emitted.
const photon_map_capacity: usize = 200_000;

/// How much of the photon budget a scene wants. The defaults suit an open,
/// sky-lit scene where most emitted photons escape; a closed room converts a
/// far larger share of them into stored caustic photons and needs a different
/// balance, so this is a scene's to choose.
const PhotonBudget = struct {
    emitted: u32 = photons_emitted,
    capacity: usize = photon_map_capacity,
};
/// Cells per axis in the photon lookup grid. Sized so a cell is roughly the
/// gather radius: too coarse and every gather walks thousands of photons.
const photon_grid_size: u32 = 128;
/// Bounces a photon may take before it is dropped. Only specular bounces
/// continue a path, so this is a budget for reflections and refractions.
const photon_max_bounces: i32 = 8;
/// Radius of the disk searched when estimating caustic radiance. The estimate
/// uses a fixed radius rather than a k-nearest-neighbour query, so this trades
/// sharpness in the bright cores against noise in the sparse areas.
const caustic_gather_radius: f64 = 0.2;
/// Smallest world box, per axis, that the photon grid will cover. The grid is
/// always `photon_grid_size` cells across whatever box it is given, so this
/// doubles as the coarsest cell size the derivation will settle for: 32 units
/// over 128 cells is 0.25, close to the gather radius.
const min_photon_extent: f64 = 32.0;

/// The slice of the world the photon grid covers. A photon that lands outside
/// it is dropped, so getting this wrong costs caustics silently.
const PhotonBounds = struct {
    min: Point3,
    max: Point3,

    pub fn cellSize(self: PhotonBounds, grid_size: u32) Vec3 {
        return div(sub(self.max, self.min), @floatFromInt(grid_size));
    }
};

/// Work out which part of the world is worth covering with photon cells.
///
/// The scene's own bounding box is the wrong answer: the cover scene's ground
/// is a sphere of radius 1000, so the box spans 2000 units, its centre sits 1000
/// units underground, and every cell would be ~15 units across — hundreds of
/// times the gather radius. Caustics only appear near the geometry that focuses
/// them, so the box is anchored on the specular primitives, grown to at least
/// `min_photon_extent` to cover the splash around them, and then clipped to the
/// scene. It is never narrower than the specular geometry it has to hold.
fn derivePhotonBounds(primitives: []const Primitive, scene_box: AABB) PhotonBounds {
    if (primitives.len == 0) {
        const half = min_photon_extent / 2.0;
        return PhotonBounds{
            .min = Point3{ -half, -half, -half },
            .max = Point3{ half, half, half },
        };
    }

    // Photons are only stored after a specular bounce, so those are the objects
    // the grid has to be built around.
    var anchor_box = AABB.empty;
    var specular_count: usize = 0;
    for (primitives) |prim| {
        if (!isSpecular(prim.material().material_type)) continue;
        anchor_box = AABB.fromBoxes(anchor_box, prim.boundingBox());
        specular_count += 1;
    }
    if (specular_count == 0) anchor_box = scene_box;

    var min: Point3 = undefined;
    var max: Point3 = undefined;

    inline for (0..3) |axis| {
        const anchor = anchor_box.axis(axis);
        const scene = scene_box.axis(axis);
        const center = (anchor.min + anchor.max) / 2.0;
        const room = scene.size();
        const bounded = room > 0.0 and std.math.isFinite(room);

        // Never narrower than the specular geometry, and wide enough to catch
        // the splash around it.
        const size = @max(anchor.size(), min_photon_extent);

        if (bounded and room <= size) {
            // The whole scene fits in the grid, so cover all of it: a closed
            // room throws caustics onto its walls, not just onto the floor
            // under the glass.
            min[axis] = scene.min;
            max[axis] = scene.max;
        } else {
            var lo = center - size / 2.0;
            var hi = center + size / 2.0;

            // Clip to the scene rather than sliding the window back inside it.
            // The window is centred on the specular geometry and never
            // narrower, so clipping cannot cut any of it off, and it keeps the
            // box from stretching tens of units into the empty space under a
            // ground sphere just to preserve a nominal extent.
            if (bounded) {
                lo = @max(lo, scene.min);
                hi = @min(hi, scene.max);
            }

            min[axis] = lo;
            max[axis] = hi;
        }
    }

    return PhotonBounds{ .min = min, .max = max };
}

const PhotonMap = struct {
    photons: []Photon,
    allocator: std.mem.Allocator,
    bounds_min: Point3,
    bounds_max: Point3,
    grid_size: u32,
    grid_cells: []std.ArrayList(usize), // Each cell contains photon indices

    pub fn init(allocator: std.mem.Allocator, capacity: usize, bounds_min: Point3, bounds_max: Point3, grid_size: u32) !PhotonMap {
        const photons = try allocator.alloc(Photon, capacity);
        const num_cells = grid_size * grid_size * grid_size;
        var grid_cells = try allocator.alloc(std.ArrayList(usize), num_cells);

        for (0..num_cells) |i| {
            grid_cells[i] = .empty;
        }

        return PhotonMap{
            .photons = photons,
            .allocator = allocator,
            .bounds_min = bounds_min,
            .bounds_max = bounds_max,
            .grid_size = grid_size,
            .grid_cells = grid_cells,
        };
    }

    pub fn deinit(self: *PhotonMap) void {
        for (self.grid_cells) |*cell| {
            cell.deinit(self.allocator);
        }
        self.allocator.free(self.grid_cells);
        self.allocator.free(self.photons);
    }

    pub fn buildGrid(self: *PhotonMap, photon_count: usize) !void {
        // Clear existing grid
        for (self.grid_cells) |*cell| {
            cell.clearRetainingCapacity();
        }

        // Insert photons into grid cells
        var dropped: usize = 0;
        for (0..photon_count) |i| {
            const photon = self.photons[i];
            const cell_idx = self.getCellIndex(photon.position);
            if (cell_idx) |idx| {
                try self.grid_cells[idx].append(self.allocator, i);
            } else {
                // A photon outside the grid contributes to nothing. Losing a
                // few is harmless, but losing a scene's worth is the difference
                // between "no caustics here" and "the bounds are wrong", and
                // that should not be silent.
                dropped += 1;
            }
        }

        if (dropped > 0) {
            std.debug.print(
                "Warning: {d} of {d} photons landed outside the photon grid and were dropped\n",
                .{ dropped, photon_count },
            );
        }
    }

    fn getCellIndex(self: PhotonMap, pos: Point3) ?usize {
        const extent = sub(self.bounds_max, self.bounds_min);
        const rel_pos = sub(pos, self.bounds_min);

        const grid_f = @as(f64, @floatFromInt(self.grid_size));
        const ix = @as(i32, @intFromFloat((rel_pos[0] / extent[0]) * grid_f));
        const iy = @as(i32, @intFromFloat((rel_pos[1] / extent[1]) * grid_f));
        const iz = @as(i32, @intFromFloat((rel_pos[2] / extent[2]) * grid_f));

        if (ix < 0 or ix >= self.grid_size or
            iy < 0 or iy >= self.grid_size or
            iz < 0 or iz >= self.grid_size)
        {
            return null;
        }

        const ux = @as(usize, @intCast(ix));
        const uy = @as(usize, @intCast(iy));
        const uz = @as(usize, @intCast(iz));

        return ux + uy * self.grid_size + uz * self.grid_size * self.grid_size;
    }

    /// Grid coordinates of `pos`, clamped to the grid. Used to walk the cells a
    /// gather sphere actually overlaps.
    fn cellCoordsClamped(self: PhotonMap, pos: Point3) [3]u32 {
        const extent = sub(self.bounds_max, self.bounds_min);
        const rel_pos = sub(pos, self.bounds_min);
        const grid_f = @as(f64, @floatFromInt(self.grid_size));

        var coords: [3]u32 = undefined;
        inline for (0..3) |axis| {
            const scaled = (rel_pos[axis] / extent[axis]) * grid_f;
            coords[axis] = @intFromFloat(std.math.clamp(scaled, 0.0, grid_f - 1.0));
        }
        return coords;
    }

    pub fn estimateRadiance(self: PhotonMap, pos: Point3, normal: Vec3, albedo: Color, max_distance: f64) Color {
        // Nothing was stored outside the mapped volume, so nothing to gather.
        if (self.getCellIndex(pos) == null) return Color{ 0, 0, 0 };

        var flux = Color{ 0, 0, 0 };
        const max_dist_sq = max_distance * max_distance;
        var photon_count: usize = 0;

        // Walk every cell the gather sphere touches. The number of cells follows
        // from the radius instead of being fixed at 3x3x3, so the gather stays
        // correct when the radius is larger than a cell and cheap when it is
        // smaller.
        const radius_vec = Vec3{ max_distance, max_distance, max_distance };
        const lo = self.cellCoordsClamped(sub(pos, radius_vec));
        const hi = self.cellCoordsClamped(add(pos, radius_vec));

        var iz = lo[2];
        while (iz <= hi[2]) : (iz += 1) {
            var iy = lo[1];
            while (iy <= hi[1]) : (iy += 1) {
                var ix = lo[0];
                while (ix <= hi[0]) : (ix += 1) {
                    const cell_idx = @as(usize, ix) +
                        @as(usize, iy) * self.grid_size +
                        @as(usize, iz) * self.grid_size * self.grid_size;

                    for (self.grid_cells[cell_idx].items) |photon_idx| {
                        const photon = self.photons[photon_idx];
                        const diff = sub(photon.position, pos);
                        const dist_sq = lengthSquared(diff);

                        if (dist_sq < max_dist_sq) {
                            // Check if photon is on the correct side (similar hemisphere)
                            const cos_theta = dot(normal, neg(photon.direction));
                            if (cos_theta > 0) {
                                flux = add(flux, photon.power);
                                photon_count += 1;
                            }
                        }
                    }
                }
            }
        }

        if (photon_count > 0) {
            // Density estimate: the flux landing on the gather disk, turned into
            // reflected radiance by the surface BRDF. For a Lambertian surface
            // that BRDF is albedo / pi, which is what tints a caustic with the
            // colour of the surface it lands on.
            const area = std.math.pi * max_dist_sq;
            return div(mulVec(flux, albedo), area * std.math.pi);
        }

        return Color{ 0, 0, 0 };
    }
};

// ============================================================================
// Photon Tracing
// ============================================================================

/// Trace a photon and store it where it lands on a diffuse surface *after* at
/// least one specular bounce, i.e. on an `LS+D` path. Those paths are exactly
/// the ones that form caustics, and they are the ones a plain path tracer is
/// worst at finding.
///
/// A photon that reaches a diffuse surface directly (`LD`) is direct light, not
/// a caustic. Storing it made this a global photon map whose energy was added
/// on top of the path traced result, washing the scene out. The renderer has no
/// direct-lighting term for the point light, so that energy is dropped rather
/// than moved elsewhere: the lamp contributes caustics only, and everything
/// else is lit by the sky.
///
/// The path also ends at the first diffuse hit: continuing it would record
/// `LS+DD` indirect bounces, which is again not caustic energy.
fn tracePhoton(
    ray: Ray,
    world: BVH,
    depth: i32,
    power: Color,
    specular_bounces: u32,
    photons: []Photon,
    photon_count: *usize,
    max_photons: usize,
    rng: std.Random,
) void {
    if (depth <= 0 or photon_count.* >= max_photons) return;

    var rec: HitRecord = undefined;
    if (!world.hit(ray, Interval{ .min = 0.001, .max = std.math.inf(f64) }, &rec)) {
        return;
    }

    switch (rec.material.material_type) {
        .lambertian => {
            // End of the path either way: only a photon that already bounced off
            // a specular surface belongs in a caustic map.
            if (specular_bounces == 0) return;

            photons[photon_count.*] = Photon{
                .position = rec.point,
                .direction = ray.direction,
                .power = power,
            };
            photon_count.* += 1;
        },
        .diffuse_light => {
            // Absorbed. Storing it would paint a caustic onto the lamp itself,
            // and the light's own emission already accounts for this energy.
        },
        .metal => {
            // Reflect and continue (specular bounce for caustics)
            const reflected = reflect(unitVector(ray.direction), rec.normal);
            const scattered = Ray.init(rec.point, add(reflected, mul(randomInUnitSphere(rng), rec.material.fuzz)));

            if (dot(scattered.direction, rec.normal) > 0) {
                const new_power = mulVec(power, rec.material.albedo);
                tracePhoton(scattered, world, depth - 1, new_power, specular_bounces + 1, photons, photon_count, max_photons, rng);
            }
        },
        .dielectric => {
            // Refract/reflect and continue (specular bounce for caustics)
            const ri = if (rec.front_face) (1.0 / rec.material.refraction_index) else rec.material.refraction_index;
            const unit_direction = unitVector(ray.direction);
            const cos_theta = @min(dot(neg(unit_direction), rec.normal), 1.0);
            const sin_theta = @sqrt(1.0 - cos_theta * cos_theta);

            const cannot_refract = ri * sin_theta > 1.0;
            const direction = if (cannot_refract or reflectance(cos_theta, ri) > randomFloat(rng))
                reflect(unit_direction, rec.normal)
            else
                refract(unit_direction, rec.normal, ri);

            const scattered = Ray.init(rec.point, direction);
            // Dielectrics don't absorb light (for caustics)
            tracePhoton(scattered, world, depth - 1, power, specular_bounces + 1, photons, photon_count, max_photons, rng);
        },
    }
}

// ============================================================================
// Projection Map - Where a Light Can See Specular Geometry
// ============================================================================

/// Coarse map of the emission directions around a light that reach specular
/// geometry. Photons sent anywhere else can never complete an `LS+D` path, so
/// aiming the budget at the marked directions turns a mostly wasted emission
/// pass into one where nearly every photon has a chance to contribute.
///
/// Cells are uniform in (cos(theta), phi), so all of them subtend the same
/// solid angle: a marked cell can be picked uniformly and the emitted fraction
/// of the sphere is simply the marked fraction of the cells.
const ProjectionMap = struct {
    const theta_cells: usize = 64;
    const phi_cells: usize = 128;
    const cell_count: usize = theta_cells * phi_cells;
    /// Directions probed per cell when deciding whether it sees specular
    /// geometry. Probing is stochastic, so the marked set is dilated afterwards
    /// to recover cells that only clip the edge of an object.
    const probes_per_cell: usize = 16;

    /// Indices of the cells worth emitting into.
    cells: []u32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, origin: Point3, world: BVH, rng: std.Random) !ProjectionMap {
        const probed = try allocator.alloc(bool, cell_count);
        defer allocator.free(probed);
        @memset(probed, false);

        var rec: HitRecord = undefined;
        for (0..cell_count) |cell| {
            for (0..probes_per_cell) |_| {
                const ray = Ray.init(origin, sampleCell(cell, rng));
                const hit = world.hit(ray, Interval{ .min = 0.001, .max = std.math.inf(f64) }, &rec);
                if (hit and isSpecular(rec.material.material_type)) {
                    probed[cell] = true;
                    break;
                }
            }
        }

        // A cell is far coarser than most of the geometry, so grow the marked set
        // by one cell in every direction. Emitting into a few extra directions
        // only costs photons; missing a cell loses caustic energy.
        var marked_count: usize = 0;
        const marked = try allocator.alloc(bool, cell_count);
        defer allocator.free(marked);
        for (0..theta_cells) |i| {
            for (0..phi_cells) |j| {
                const cell = i * phi_cells + j;
                marked[cell] = neighbourhoodProbed(probed, i, j);
                if (marked[cell]) marked_count += 1;
            }
        }

        const cells = try allocator.alloc(u32, marked_count);
        var next: usize = 0;
        for (marked, 0..) |is_marked, cell| {
            if (is_marked) {
                cells[next] = @intCast(cell);
                next += 1;
            }
        }

        return ProjectionMap{ .cells = cells, .allocator = allocator };
    }

    pub fn deinit(self: ProjectionMap) void {
        self.allocator.free(self.cells);
    }

    fn neighbourhoodProbed(probed: []const bool, i: usize, j: usize) bool {
        var di: isize = -1;
        while (di <= 1) : (di += 1) {
            const ni = @as(isize, @intCast(i)) + di;
            if (ni < 0 or ni >= theta_cells) continue;

            var dj: isize = -1;
            while (dj <= 1) : (dj += 1) {
                // Azimuth wraps around; polar angle does not.
                const nj = @mod(@as(isize, @intCast(j)) + dj, @as(isize, phi_cells));
                if (probed[@as(usize, @intCast(ni)) * phi_cells + @as(usize, @intCast(nj))]) return true;
            }
        }
        return false;
    }

    /// A direction drawn uniformly from the solid angle of one cell.
    fn sampleCell(cell: usize, rng: std.Random) Vec3 {
        const i = @as(f64, @floatFromInt(cell / phi_cells));
        const j = @as(f64, @floatFromInt(cell % phi_cells));

        const cos_max = 1.0 - 2.0 * i / @as(f64, theta_cells);
        const cos_min = 1.0 - 2.0 * (i + 1.0) / @as(f64, theta_cells);

        const cos_theta = cos_min + (cos_max - cos_min) * randomFloat(rng);
        const phi = 2.0 * std.math.pi * (j + randomFloat(rng)) / @as(f64, phi_cells);
        const sin_theta = @sqrt(@max(0.0, 1.0 - cos_theta * cos_theta));

        return Vec3{ sin_theta * @cos(phi), cos_theta, sin_theta * @sin(phi) };
    }

    /// Fraction of the whole sphere the marked cells cover. Photon power is
    /// scaled by this, so restricting emission redistributes the light's power
    /// instead of adding energy to the scene.
    pub fn sphereFraction(self: ProjectionMap) f64 {
        return @as(f64, @floatFromInt(self.cells.len)) / @as(f64, cell_count);
    }

    pub fn sampleDirection(self: ProjectionMap, rng: std.Random) Vec3 {
        if (self.cells.len == 0) return randomUnitVector(rng);
        return sampleCell(self.cells[rng.uintLessThan(usize, self.cells.len)], rng);
    }
};

fn emitPhotonsFromLight(
    light: Light,
    world: BVH,
    photon_map: *PhotonMap,
    num_photons: u32,
    projection: ProjectionMap,
    rng: std.Random,
) usize {
    var photon_count: usize = 0;
    if (num_photons == 0) return 0;

    // Emission is restricted to the projection map, so a photon stands for the
    // light's power over that solid angle only: aiming the emission
    // redistributes the light's power instead of adding energy to the scene.
    // Without specular geometry there is nothing to aim at, and emission falls
    // back to the whole sphere.
    const emitted_fraction = if (projection.cells.len == 0) 1.0 else projection.sphereFraction();
    const photon_power = mul(light.intensity, light.power * emitted_fraction / @as(f64, @floatFromInt(num_photons)));

    for (0..num_photons) |_| {
        const ray = Ray.init(light.position, projection.sampleDirection(rng));
        tracePhoton(ray, world, photon_max_bounces, photon_power, 0, photon_map.photons, &photon_count, photon_map.photons.len, rng);
    }

    return photon_count;
}

// ============================================================================
// OBJ File Parser
// ============================================================================

const OBJParseError = error{
    InvalidFormat,
    MissingData,
    InvalidIndex,
} || std.mem.Allocator.Error || std.Io.File.OpenError || std.Io.File.Reader.Error;

const OBJData = struct {
    vertices: []Vec3,
    normals: []Vec3,
    faces: []Face,
    allocator: std.mem.Allocator,

    pub fn deinit(self: OBJData) void {
        self.allocator.free(self.vertices);
        self.allocator.free(self.normals);
        self.allocator.free(self.faces);
    }
};

const Face = struct {
    // Vertex indices (0-based after conversion from OBJ's 1-based)
    v0: u32,
    v1: u32,
    v2: u32,

    // Normal indices (0xFFFFFFFF if not present)
    n0: u32,
    n1: u32,
    n2: u32,

    pub fn hasNormals(self: Face) bool {
        return self.n0 != 0xFFFFFFFF and
            self.n1 != 0xFFFFFFFF and
            self.n2 != 0xFFFFFFFF;
    }
};

const VertexDescriptor = struct {
    v: u32, // vertex index (0-based)
    n: u32, // normal index (0-based or 0xFFFFFFFF)
};

fn parseOBJ(allocator: std.mem.Allocator, io: std.Io, filepath: []const u8) !OBJData {
    // Read entire file (reasonable for small-medium meshes up to 50MB)
    const max_file_size = 50 * 1024 * 1024;
    const contents = try std.Io.Dir.cwd().readFileAlloc(io, filepath, allocator, .limited(max_file_size));
    defer allocator.free(contents);

    // Dynamic arrays for parsed data
    var vertices = try std.ArrayList(Vec3).initCapacity(allocator, 100);
    defer vertices.deinit(allocator);
    var normals = try std.ArrayList(Vec3).initCapacity(allocator, 100);
    defer normals.deinit(allocator);
    var faces = try std.ArrayList(Face).initCapacity(allocator, 100);
    defer faces.deinit(allocator);

    // Parse line by line
    var line_iter = std.mem.splitScalar(u8, contents, '\n');
    while (line_iter.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0 or trimmed[0] == '#') continue;

        if (std.mem.startsWith(u8, trimmed, "v ")) {
            try parseVertex(&vertices, trimmed, allocator);
        } else if (std.mem.startsWith(u8, trimmed, "vn ")) {
            try parseNormal(&normals, trimmed, allocator);
        } else if (std.mem.startsWith(u8, trimmed, "f ")) {
            try parseFace(&faces, trimmed, allocator);
        }
        // Ignore: vt (textures), mtllib, usemtl, s, o, g
    }

    return OBJData{
        .vertices = try vertices.toOwnedSlice(allocator),
        .normals = try normals.toOwnedSlice(allocator),
        .faces = try faces.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

fn parseVertex(vertices: *std.ArrayList(Vec3), line: []const u8, allocator: std.mem.Allocator) !void {
    // Format: "v x y z [w]" - ignore optional w
    var iter = std.mem.tokenizeAny(u8, line, " \t");
    _ = iter.next(); // skip "v"

    const x_str = iter.next() orelse return error.InvalidFormat;
    const y_str = iter.next() orelse return error.InvalidFormat;
    const z_str = iter.next() orelse return error.InvalidFormat;

    const x = try std.fmt.parseFloat(f64, x_str);
    const y = try std.fmt.parseFloat(f64, y_str);
    const z = try std.fmt.parseFloat(f64, z_str);

    try vertices.append(allocator, Vec3{ x, y, z });
}

fn parseNormal(normals: *std.ArrayList(Vec3), line: []const u8, allocator: std.mem.Allocator) !void {
    // Format: "vn x y z"
    var iter = std.mem.tokenizeAny(u8, line, " \t");
    _ = iter.next(); // skip "vn"

    const x_str = iter.next() orelse return error.InvalidFormat;
    const y_str = iter.next() orelse return error.InvalidFormat;
    const z_str = iter.next() orelse return error.InvalidFormat;

    const x = try std.fmt.parseFloat(f64, x_str);
    const y = try std.fmt.parseFloat(f64, y_str);
    const z = try std.fmt.parseFloat(f64, z_str);

    try normals.append(allocator, Vec3{ x, y, z });
}

fn parseFace(faces: *std.ArrayList(Face), line: []const u8, allocator: std.mem.Allocator) !void {
    // Parse face: "f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3" (or variations)
    var iter = std.mem.tokenizeAny(u8, line, " \t");
    _ = iter.next(); // skip "f"

    const vert1 = iter.next() orelse return error.InvalidFormat;
    const vert2 = iter.next() orelse return error.InvalidFormat;
    const vert3 = iter.next() orelse return error.InvalidFormat;

    const v0_data = try parseVertexDescriptor(vert1);
    const v1_data = try parseVertexDescriptor(vert2);
    const v2_data = try parseVertexDescriptor(vert3);

    try faces.append(allocator, Face{
        .v0 = v0_data.v,
        .v1 = v1_data.v,
        .v2 = v2_data.v,
        .n0 = v0_data.n,
        .n1 = v1_data.n,
        .n2 = v2_data.n,
    });

    // Triangulate if more than 3 vertices (simple fan from v0)
    var prev = v2_data;
    while (iter.next()) |vert| {
        const curr = try parseVertexDescriptor(vert);
        try faces.append(allocator, Face{
            .v0 = v0_data.v,
            .v1 = prev.v,
            .v2 = curr.v,
            .n0 = v0_data.n,
            .n1 = prev.n,
            .n2 = curr.n,
        });
        prev = curr;
    }
}

fn parseVertexDescriptor(desc: []const u8) !VertexDescriptor {
    // Format: "v/vt/vn" or "v//vn" or "v"
    var iter = std.mem.splitScalar(u8, desc, '/');

    // Vertex index (required, OBJ uses 1-based indexing)
    const v_str = iter.next() orelse return error.InvalidFormat;
    const v_index = try std.fmt.parseInt(i32, v_str, 10);
    if (v_index <= 0) return error.InvalidIndex;

    // Texture coord (optional, skip)
    _ = iter.next();

    // Normal index (optional)
    var n_index: u32 = 0xFFFFFFFF; // sentinel for "not present"
    if (iter.next()) |n_str| {
        if (n_str.len > 0) {
            const n = try std.fmt.parseInt(i32, n_str, 10);
            if (n <= 0) return error.InvalidIndex;
            n_index = @intCast(n - 1); // Convert to 0-based
        }
    }

    return VertexDescriptor{
        .v = @intCast(v_index - 1), // Convert to 0-based
        .n = n_index,
    };
}

// ============================================================================
// Mesh - Collection of Triangles
// ============================================================================

const Mesh = struct {
    triangles: []Triangle,
    allocator: std.mem.Allocator,

    pub fn fromOBJ(
        allocator: std.mem.Allocator,
        io: std.Io,
        filepath: []const u8,
        material: Material,
    ) !Mesh {
        const obj_data = try parseOBJ(allocator, io, filepath);
        defer obj_data.deinit();

        std.debug.print("Loaded OBJ: {d} vertices, {d} normals, {d} faces\n", .{
            obj_data.vertices.len,
            obj_data.normals.len,
            obj_data.faces.len,
        });

        // Validate indices
        var has_normals_count: usize = 0;
        for (obj_data.faces) |face| {
            // Check vertex indices
            if (face.v0 >= obj_data.vertices.len or
                face.v1 >= obj_data.vertices.len or
                face.v2 >= obj_data.vertices.len)
            {
                return error.InvalidIndex;
            }

            // Check normal indices if present
            if (face.hasNormals()) {
                has_normals_count += 1;
                if (face.n0 >= obj_data.normals.len or
                    face.n1 >= obj_data.normals.len or
                    face.n2 >= obj_data.normals.len)
                {
                    return error.InvalidIndex;
                }
            }
        }

        const shading_type = if (has_normals_count == obj_data.faces.len)
            "smooth"
        else if (has_normals_count == 0)
            "flat"
        else
            "mixed";

        std.debug.print("Using {s} shading ({d}/{d} faces with normals)\n", .{
            shading_type,
            has_normals_count,
            obj_data.faces.len,
        });

        // Create triangles
        var triangles = try allocator.alloc(Triangle, obj_data.faces.len);
        errdefer allocator.free(triangles);

        for (obj_data.faces, 0..) |face, i| {
            const v0 = obj_data.vertices[face.v0];
            const v1 = obj_data.vertices[face.v1];
            const v2 = obj_data.vertices[face.v2];

            triangles[i] = if (face.hasNormals())
                Triangle.initWithNormals(
                    v0,
                    v1,
                    v2,
                    obj_data.normals[face.n0],
                    obj_data.normals[face.n1],
                    obj_data.normals[face.n2],
                    material,
                )
            else
                Triangle.init(v0, v1, v2, material);
        }

        return Mesh{
            .triangles = triangles,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: Mesh) void {
        self.allocator.free(self.triangles);
    }

    /// Scale all vertices uniformly by a factor
    pub fn scale(self: *Mesh, factor: f64) void {
        for (self.triangles) |*tri| {
            tri.v0 = mul(tri.v0, factor);
            tri.v1 = mul(tri.v1, factor);
            tri.v2 = mul(tri.v2, factor);
            // Recompute edges
            tri.edge1 = sub(tri.v1, tri.v0);
            tri.edge2 = sub(tri.v2, tri.v0);
        }
    }

    /// Translate all vertices by an offset
    pub fn translate(self: *Mesh, offset: Vec3) void {
        for (self.triangles) |*tri| {
            tri.v0 = add(tri.v0, offset);
            tri.v1 = add(tri.v1, offset);
            tri.v2 = add(tri.v2, offset);
            // Edges remain the same (translation doesn't affect them)
        }
    }

    /// Rotate mesh around Y-axis by angle in degrees
    pub fn rotateY(self: *Mesh, degrees: f64) void {
        const radians = degrees * std.math.pi / 180.0;
        const cos_theta = @cos(radians);
        const sin_theta = @sin(radians);

        for (self.triangles) |*tri| {
            // Rotate vertices
            tri.v0 = rotatePointY(tri.v0, cos_theta, sin_theta);
            tri.v1 = rotatePointY(tri.v1, cos_theta, sin_theta);
            tri.v2 = rotatePointY(tri.v2, cos_theta, sin_theta);

            // Rotate normals if present
            if (tri.has_normals) {
                tri.n0 = rotatePointY(tri.n0, cos_theta, sin_theta);
                tri.n1 = rotatePointY(tri.n1, cos_theta, sin_theta);
                tri.n2 = rotatePointY(tri.n2, cos_theta, sin_theta);
            }

            // Recompute edges
            tri.edge1 = sub(tri.v1, tri.v0);
            tri.edge2 = sub(tri.v2, tri.v0);
        }
    }

    /// The mesh's axis-aligned bounds, taken from the vertices themselves.
    ///
    /// Deliberately not a union of the triangles' bounding boxes: those are
    /// padded by an epsilon so the BVH never sees a zero-thickness slab, and
    /// placing a model with padded bounds would leave it hovering.
    pub fn bounds(self: Mesh) AABB {
        var box = AABB.empty;
        for (self.triangles) |tri| {
            inline for (.{ tri.v0, tri.v1, tri.v2 }) |v| {
                box.x.min = @min(box.x.min, v[0]);
                box.x.max = @max(box.x.max, v[0]);
                box.y.min = @min(box.y.min, v[1]);
                box.y.max = @max(box.y.max, v[1]);
                box.z.min = @min(box.z.min, v[2]);
                box.z.max = @max(box.z.max, v[2]);
            }
        }
        return box;
    }

    /// Centre the mesh on `center` and scale it until its longest side is
    /// `longest_side`.
    ///
    /// Scanned models arrive in whatever units the scanner used and wherever
    /// the scanner's origin happened to be: the XYZ RGB dragon spans about 200
    /// units across and does not straddle the origin. Placing one by hand means
    /// guessing constants that are wrong for the next model.
    pub fn fitTo(self: *Mesh, center: Point3, longest_side: f64) void {
        const box = self.bounds();
        const longest = @max(@max(box.x.size(), box.y.size()), box.z.size());
        if (!(longest > 0.0)) return;

        const box_center = Point3{
            (box.x.min + box.x.max) / 2.0,
            (box.y.min + box.y.max) / 2.0,
            (box.z.min + box.z.max) / 2.0,
        };

        // scale() works about the origin, so the mesh has to visit it first.
        self.translate(neg(box_center));
        self.scale(longest_side / longest);
        self.translate(center);
    }

    /// Drop the mesh straight down until its lowest point rests at `y`.
    pub fn placeOnGround(self: *Mesh, y: f64) void {
        const box = self.bounds();
        self.translate(Vec3{ 0, y - box.y.min, 0 });
    }

    /// Generate smooth vertex normals by averaging face normals
    pub fn generateSmoothNormals(self: *Mesh, allocator: std.mem.Allocator) !void {
        // HashMap to accumulate normals for each unique vertex position
        var vertex_normals = std.AutoHashMap(Vec3Key, Vec3).init(allocator);
        defer vertex_normals.deinit();

        // First pass: accumulate face normals at each vertex
        for (self.triangles) |*tri| {
            // Compute face normal
            const face_normal = unitVector(cross(tri.edge1, tri.edge2));

            // Accumulate at each vertex
            try accumulateNormal(&vertex_normals, tri.v0, face_normal);
            try accumulateNormal(&vertex_normals, tri.v1, face_normal);
            try accumulateNormal(&vertex_normals, tri.v2, face_normal);
        }

        // Second pass: assign averaged normals to triangles
        for (self.triangles) |*tri| {
            tri.n0 = unitVector(vertex_normals.get(Vec3Key.init(tri.v0)) orelse tri.n0);
            tri.n1 = unitVector(vertex_normals.get(Vec3Key.init(tri.v1)) orelse tri.n1);
            tri.n2 = unitVector(vertex_normals.get(Vec3Key.init(tri.v2)) orelse tri.n2);
            tri.has_normals = true;
        }

        std.debug.print("Generated smooth normals for {d} triangles\n", .{self.triangles.len});
    }
};

// Helper struct for using Vec3 as HashMap key
const Vec3Key = struct {
    x: u64,
    y: u64,
    z: u64,

    pub fn init(v: Vec3) Vec3Key {
        return Vec3Key{
            .x = @bitCast(@as(f64, v[0])),
            .y = @bitCast(@as(f64, v[1])),
            .z = @bitCast(@as(f64, v[2])),
        };
    }
};

fn accumulateNormal(map: *std.AutoHashMap(Vec3Key, Vec3), vertex: Vec3, normal: Vec3) !void {
    const key = Vec3Key.init(vertex);
    const existing = map.get(key) orelse Vec3{ 0, 0, 0 };
    try map.put(key, add(existing, normal));
}

fn rotatePointY(point: Vec3, cos_theta: f64, sin_theta: f64) Vec3 {
    // Y-axis rotation matrix: [cos 0 sin; 0 1 0; -sin 0 cos]
    return Vec3{
        point[0] * cos_theta + point[2] * sin_theta,
        point[1],
        -point[0] * sin_theta + point[2] * cos_theta,
    };
}

// ============================================================================
// Ray Color (Background)
// ============================================================================

/// What a ray sees when it hits nothing. The sky gradient lights every open
/// scene in the book; a closed room needs it gone, or light leaks in through
/// the wall the camera looks through.
const Background = union(enum) {
    sky,
    solid: Color,

    pub fn sample(self: Background, direction: Vec3) Color {
        return switch (self) {
            .sky => blk: {
                // Linear interpolation from white to blue based on height
                const unit_direction = unitVector(direction);
                const a = 0.5 * (unit_direction[1] + 1.0);
                const white = Color{ 1.0, 1.0, 1.0 };
                const blue = Color{ 0.5, 0.7, 1.0 };
                break :blk add(mul(white, 1.0 - a), mul(blue, a));
            },
            .solid => |color| color,
        };
    }
};

/// Everything a ray needs that does not change from ray to ray. Passing one
/// struct keeps the recursion signature from growing a parameter per feature.
const RenderContext = struct {
    world: BVH,
    background: Background = .sky,
    photon_map: ?*const PhotonMap = null,
    lights: []const Light = &.{},
};

fn rayColor(ctx: RenderContext, ray: Ray, depth: i32, rng: std.Random) Color {
    // A camera ray sees emitters directly: nothing has sampled them yet.
    return rayColorInner(ctx, ray, depth, rng, true);
}

/// `count_emission` says whether hitting a light should add its emission.
///
/// It must not, once the path has touched a diffuse surface. Two other
/// estimators already cover the light from there: direct sampling handles the
/// light seen straight from that surface, and the caustic map handles it seen
/// through glass or off metal. Letting the bounce count emission as well would
/// add both a second time — which is not only too bright but violently noisy,
/// since finding a small lamp by chance through a specular bounce is exactly
/// the rare, high-energy sample that shows up as a white speck.
///
/// So emission counts only while the path has been specular the whole way from
/// the camera: looking at the lamp, or at its reflection, or at it through
/// glass. Without a photon map nothing else covers specular paths, so there
/// they are counted and the renderer degrades to plain path tracing.
fn rayColorInner(ctx: RenderContext, ray: Ray, depth: i32, rng: std.Random, count_emission: bool) Color {
    // If we've exceeded the ray bounce limit, no more light is gathered
    if (depth <= 0) {
        return Color{ 0, 0, 0 };
    }

    var rec: HitRecord = undefined;

    if (ctx.world.hit(ray, Interval{ .min = 0.001, .max = std.math.inf(f64) }, &rec)) {
        var scattered: Ray = undefined;
        var attenuation: Color = undefined;

        const emitted = if (count_emission) rec.material.emitted(rec.front_face) else Color{ 0, 0, 0 };

        var direct = Color{ 0, 0, 0 };
        var caustics_contribution = Color{ 0, 0, 0 };

        if (rec.material.material_type == .lambertian) {
            for (ctx.lights) |light| {
                direct = add(direct, light.sampleDirect(
                    ctx.world,
                    rec.point,
                    rec.normal,
                    rec.material.albedo,
                    rng,
                ));
            }

            if (ctx.photon_map) |map| {
                // The map holds LS+D photons only, so this adds the caustics the
                // recursive estimate below cannot find, and nothing it can.
                caustics_contribution = map.estimateRadiance(
                    rec.point,
                    rec.normal,
                    rec.material.albedo,
                    caustic_gather_radius,
                );
            }
        }

        const local = add(add(emitted, direct), caustics_contribution);

        if (rec.material.scatter(ray, rec, &attenuation, &scattered, rng)) {
            const bounce_counts_emission = if (!isSpecular(rec.material.material_type))
                // Direct sampling covered the light here, and the caustic map
                // covers it arriving through anything specular further on.
                false
            else if (ctx.photon_map == null)
                // No caustic map: plain path tracing, where the bounce landing
                // on a light is the only way that light ever arrives.
                true
            else
                count_emission;

            const indirect = rayColorInner(ctx, scattered, depth - 1, rng, bounce_counts_emission);
            return add(local, mulVec(attenuation, indirect));
        }
        return local;
    }

    return ctx.background.sample(ray.direction);
}

// ============================================================================
// Camera
// ============================================================================

const Camera = struct {
    image_width: u32,
    image_height: u32,
    center: Point3,
    pixel00_loc: Point3,
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
    defocus_disk_u: Vec3,
    defocus_disk_v: Vec3,
    defocus_angle: f64,

    pub fn init(
        lookfrom: Point3,
        lookat: Point3,
        vup: Vec3,
        vfov: f64, // Vertical field of view in degrees
        aspect_ratio: f64,
        image_width: u32,
        defocus_angle: f64, // Variation angle of rays through each pixel
        focus_dist: f64, // Distance from camera lookfrom point to plane of perfect focus
    ) Camera {
        const image_height = @max(1, @as(u32, @intFromFloat(@as(f64, @floatFromInt(image_width)) / aspect_ratio)));

        const theta = degreesToRadians(vfov);
        const h = @tan(theta / 2.0);
        const viewport_height = 2.0 * h * focus_dist;
        const viewport_width = viewport_height * (@as(f64, @floatFromInt(image_width)) / @as(f64, @floatFromInt(image_height)));

        // Calculate camera basis vectors
        const w = unitVector(sub(lookfrom, lookat));
        const u = unitVector(cross(vup, w));
        const v = cross(w, u);

        const center = lookfrom;

        // Calculate the vectors across the horizontal and down the vertical viewport edges
        const viewport_u = mul(u, viewport_width);
        const viewport_v = mul(neg(v), viewport_height);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel
        const pixel_delta_u = div(viewport_u, @floatFromInt(image_width));
        const pixel_delta_v = div(viewport_v, @floatFromInt(image_height));

        // Calculate the location of the upper left pixel
        const viewport_upper_left = sub(sub(sub(center, mul(w, focus_dist)), div(viewport_u, 2.0)), div(viewport_v, 2.0));

        const pixel00_loc = add(viewport_upper_left, mul(add(pixel_delta_u, pixel_delta_v), 0.5));

        // Calculate the camera defocus disk basis vectors
        const defocus_radius = focus_dist * @tan(degreesToRadians(defocus_angle / 2.0));
        const defocus_disk_u = mul(u, defocus_radius);
        const defocus_disk_v = mul(v, defocus_radius);

        return Camera{
            .image_width = image_width,
            .image_height = image_height,
            .center = center,
            .pixel00_loc = pixel00_loc,
            .pixel_delta_u = pixel_delta_u,
            .pixel_delta_v = pixel_delta_v,
            .defocus_disk_u = defocus_disk_u,
            .defocus_disk_v = defocus_disk_v,
            .defocus_angle = defocus_angle,
        };
    }

    pub fn getRay(self: Camera, i: u32, j: u32, rng: std.Random) Ray {
        const offset_u = randomFloat(rng) - 0.5;
        const offset_v = randomFloat(rng) - 0.5;

        const pixel_sample = add(add(self.pixel00_loc, mul(self.pixel_delta_u, @as(f64, @floatFromInt(i)) + offset_u)), mul(self.pixel_delta_v, @as(f64, @floatFromInt(j)) + offset_v));

        const ray_origin = if (self.defocus_angle <= 0) self.center else self.defocusDiskSample(rng);
        const ray_direction = sub(pixel_sample, ray_origin);
        return Ray.init(ray_origin, ray_direction);
    }

    fn defocusDiskSample(self: Camera, rng: std.Random) Point3 {
        const p = randomInUnitDisk(rng);
        return add(add(self.center, mul(self.defocus_disk_u, p[0])), mul(self.defocus_disk_v, p[1]));
    }
};

// ============================================================================
// Scenes
// ============================================================================

/// A camera without the pixel grid attached. Scenes describe where to stand and
/// what to look at; the resolution comes from the build options, so the two are
/// kept apart until the render actually starts.
const CameraSpec = struct {
    lookfrom: Point3,
    lookat: Point3,
    vup: Vec3 = Vec3{ 0, 1, 0 },
    /// Vertical field of view in degrees.
    vfov: f64,
    aspect_ratio: f64 = 16.0 / 9.0,
    defocus_angle: f64 = 0.0,
    focus_dist: f64 = 10.0,

    pub fn toCamera(self: CameraSpec, image_width: u32) Camera {
        return Camera.init(
            self.lookfrom,
            self.lookat,
            self.vup,
            self.vfov,
            self.aspect_ratio,
            image_width,
            self.defocus_angle,
            self.focus_dist,
        );
    }
};

/// Everything a render needs from a scene, and everything it has to free
/// afterwards. Meshes are copied into `primitives` and released by the builder,
/// so a 250k-triangle model is not held twice for the whole render.
const SceneData = struct {
    allocator: std.mem.Allocator,
    primitives: std.ArrayList(Primitive),
    lights: std.ArrayList(Light),
    camera: CameraSpec,
    background: Background,
    photons: PhotonBudget,

    pub fn init(allocator: std.mem.Allocator, camera: CameraSpec) SceneData {
        return SceneData{
            .allocator = allocator,
            .primitives = .empty,
            .lights = .empty,
            .camera = camera,
            .background = .sky,
            .photons = .{},
        };
    }

    pub fn deinit(self: *SceneData) void {
        self.primitives.deinit(self.allocator);
        self.lights.deinit(self.allocator);
    }

    pub fn addSphere(self: *SceneData, sphere: Sphere) !void {
        try self.primitives.append(self.allocator, Primitive{ .sphere = sphere });
    }

    /// A rectangle as two triangles, spanning `corner + s*edge_u + t*edge_v`
    /// for s, t in [0, 1].
    pub fn addQuad(self: *SceneData, corner: Point3, edge_u: Vec3, edge_v: Vec3, material: Material) !void {
        const a = corner;
        const b = add(corner, edge_u);
        const c = add(add(corner, edge_u), edge_v);
        const d = add(corner, edge_v);

        try self.primitives.append(self.allocator, Primitive{ .triangle = Triangle.init(a, b, c, material) });
        try self.primitives.append(self.allocator, Primitive{ .triangle = Triangle.init(a, c, d, material) });
    }

    pub fn addLight(self: *SceneData, light: Light) !void {
        try self.lights.append(self.allocator, light);
    }

    /// Copy a mesh's triangles into the scene. The mesh itself stays owned by
    /// the caller, which is free to release it as soon as this returns.
    pub fn addMesh(self: *SceneData, mesh: Mesh) !void {
        try self.primitives.ensureUnusedCapacity(self.allocator, mesh.triangles.len);
        for (mesh.triangles) |tri| {
            self.primitives.appendAssumeCapacity(Primitive{ .triangle = tri });
        }
    }
};

const SceneBuildFn = *const fn (allocator: std.mem.Allocator, io: std.Io) anyerror!SceneData;

const Scene = struct {
    name: []const u8,
    description: []const u8,
    build: SceneBuildFn,
};

/// Rendered when `--scene` is not given. Must only use models committed to the
/// repository, because CI renders it without running `make models`.
const default_scene_name = "cover";

/// Fetched by `make models`, not committed. See models/manifest.tsv.
const dragon_model_path = "models/xyzrgb_dragon.obj";
const bunny_model_path = "models/stanford-bunny.obj";
const spot_model_path = "models/spot.obj";

/// Load a model that `make models` fetches rather than one the repository
/// carries. A missing file here is the expected first run, not a broken
/// install, so it says what to do about it instead of reporting FileNotFound
/// and leaving the reader to work out which file and why.
fn loadFetchedMesh(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    material: Material,
) !Mesh {
    return Mesh.fromOBJ(allocator, io, path, material) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print(
                "Error: {s} is not here. It is fetched rather than committed: run 'make models'.\n",
                .{path},
            );
            return err;
        },
        else => return err,
    };
}

const scenes = [_]Scene{
    .{
        .name = "cover",
        .description = "Ray Tracing in One Weekend cover: random spheres, a diamond teapot and a cube",
        .build = buildCoverScene,
    },
    .{
        .name = "cornell-box",
        .description = "Closed Cornell box lit by a ceiling panel, with a glass and a metal sphere",
        .build = buildCornellBoxScene,
    },
    .{
        .name = "glass-dragon",
        .description = "XYZ RGB Asian Dragon in glass, 250k triangles (needs 'make models')",
        .build = buildGlassDragonScene,
    },
    .{
        .name = "glass-bunny",
        .description = "Stanford bunny in glass, 69k triangles: the quick caustic scene (needs 'make models')",
        .build = buildGlassBunnyScene,
    },
    .{
        .name = "spot",
        .description = "Spot the cow under a daylit sky, with a glass sphere for company (needs 'make models')",
        .build = buildSpotScene,
    },
};

fn findScene(name: []const u8) ?*const Scene {
    for (&scenes) |*scene| {
        if (std.mem.eql(u8, scene.name, name)) return scene;
    }
    return null;
}

fn buildCoverScene(allocator: std.mem.Allocator, io: std.Io) !SceneData {
    var scene = SceneData.init(allocator, CameraSpec{
        .lookfrom = Point3{ 13, 2, 3 },
        .lookat = Point3{ 0, 0, 0 },
        .vfov = 20.0,
        .defocus_angle = 0.6,
        .focus_dist = 10.0,
    });
    errdefer scene.deinit();

    try scene.primitives.ensureTotalCapacity(allocator, 500);

    // Random number generator for scene generation
    var scene_prng = std.Random.DefaultPrng.init(0);
    const scene_rng = scene_prng.random();

    // Ground
    try scene.addSphere(Sphere.init(
        Point3{ 0, -1000, 0 },
        1000,
        Material.lambertian(Color{ 0.5, 0.5, 0.5 }),
    ));

    // Load test cube mesh
    {
        const cube_mesh = try Mesh.fromOBJ(
            allocator,
            io,
            "models/test_cube.obj",
            Material.lambertian(Color{ 0.8, 0.3, 0.3 }), // Red-ish
        );
        defer cube_mesh.deinit();

        try scene.addMesh(cube_mesh);
    }

    // Load teapot mesh
    {
        var teapot_mesh = try Mesh.fromOBJ(
            allocator,
            io,
            "models/teapot.obj",
            Material.dielectric(2.4), // Diamond (refractive index 2.4)
        );
        defer teapot_mesh.deinit();

        // Generate smooth normals for better shading
        try teapot_mesh.generateSmoothNormals(allocator);

        // Transform teapot: rotate, scale, and position
        // Rotate 30° around Y-axis, then scale and translate
        teapot_mesh.rotateY(30.0);
        teapot_mesh.scale(0.4);
        teapot_mesh.translate(Vec3{ 2.3, 1.0, 2.95 });

        try scene.addMesh(teapot_mesh);
    }

    // Random small spheres
    var a: i32 = -11;
    while (a < 11) : (a += 1) {
        var b: i32 = -11;
        while (b < 11) : (b += 1) {
            const choose_mat = randomFloat(scene_rng);
            const center = Point3{
                @as(f64, @floatFromInt(a)) + 0.9 * randomFloat(scene_rng),
                0.2,
                @as(f64, @floatFromInt(b)) + 0.9 * randomFloat(scene_rng),
            };

            if (length(sub(center, Point3{ 4, 0.2, 0 })) > 0.9) {
                if (choose_mat < 0.8) {
                    // Diffuse
                    const albedo = Color{
                        randomFloat(scene_rng) * randomFloat(scene_rng),
                        randomFloat(scene_rng) * randomFloat(scene_rng),
                        randomFloat(scene_rng) * randomFloat(scene_rng),
                    };
                    try scene.addSphere(Sphere.init(center, 0.2, Material.lambertian(albedo)));
                } else if (choose_mat < 0.95) {
                    // Metal
                    const albedo = Color{
                        randomFloatRange(scene_rng, 0.5, 1.0),
                        randomFloatRange(scene_rng, 0.5, 1.0),
                        randomFloatRange(scene_rng, 0.5, 1.0),
                    };
                    const fuzz = randomFloatRange(scene_rng, 0.0, 0.5);
                    try scene.addSphere(Sphere.init(center, 0.2, Material.metal(albedo, fuzz)));
                } else {
                    // Glass
                    try scene.addSphere(Sphere.init(center, 0.2, Material.dielectric(1.5)));
                }
            }
        }
    }

    // Three large spheres
    try scene.addSphere(Sphere.init(Point3{ 0, 1, 0 }, 1.0, Material.dielectric(1.5)));
    try scene.addSphere(Sphere.init(Point3{ -4, 1, 0 }, 1.0, Material.lambertian(Color{ 0.4, 0.2, 0.1 })));
    try scene.addSphere(Sphere.init(Point3{ 4, 1, 0 }, 1.0, Material.metal(Color{ 0.7, 0.6, 0.5 }, 0.0)));

    // Light source for caustics
    try scene.addLight(Light.pointLight(
        Point3{ 5, 10, 5 }, // Position above and to the side
        Color{ 1.0, 1.0, 1.0 }, // White light
        1000.0, // Power
    ));

    return scene;
}

/// The Cornell box, closed on five sides and lit only by the panel on its
/// ceiling. This is the scene the caustic map was built for: an open, sky-lit
/// field washes a caustic out, while a sealed room has nothing else to look at.
///
/// Dimensions follow the original measurements scaled to a 5.55-unit cube, and
/// the sixth wall is left off for the camera to look through. The background is
/// black, because a "closed" box that lets the sky in through that opening is
/// not closed at all.
fn buildCornellBoxScene(allocator: std.mem.Allocator, io: std.Io) !SceneData {
    _ = io;

    const size = 5.55;

    var scene = SceneData.init(allocator, CameraSpec{
        .lookfrom = Point3{ size / 2.0, size / 2.0, -8.0 },
        .lookat = Point3{ size / 2.0, size / 2.0, size / 2.0 },
        .vfov = 38.0,
        // The box is square, so anything wider just frames it in black.
        .aspect_ratio = 1.0,
    });
    errdefer scene.deinit();

    scene.background = .{ .solid = Color{ 0, 0, 0 } };
    // A closed room turns ~41% of emitted photons into stored caustic photons,
    // against ~7% for the open cover scene, so the map needs the extra room.
    scene.photons = .{ .emitted = 1_000_000, .capacity = 500_000 };

    const white = Material.lambertian(Color{ 0.73, 0.73, 0.73 });
    const red = Material.lambertian(Color{ 0.65, 0.05, 0.05 });
    const green = Material.lambertian(Color{ 0.12, 0.45, 0.15 });

    const x = Vec3{ size, 0, 0 };
    const y = Vec3{ 0, size, 0 };
    const z = Vec3{ 0, 0, size };

    try scene.addQuad(Point3{ 0, 0, 0 }, x, z, white); // floor
    try scene.addQuad(Point3{ 0, 0, size }, x, y, white); // back wall

    // The camera looks along +z, which puts +x on the left of the image, so
    // green goes on the x = 0 wall to land red on the left as in the original.
    try scene.addQuad(Point3{ 0, 0, 0 }, y, z, green);
    try scene.addQuad(Point3{ size, 0, 0 }, y, z, red);

    // The panel is set into the ceiling rather than hung below it. Hanging it
    // leaves a sliver of ceiling a few centimetres away that samples the light
    // across almost no distance, and direct lighting divides by the square of
    // that distance. Set into the opening the two are coplanar, every direction
    // from the ceiling to the panel is edge-on, and the term vanishes instead
    // of exploding.
    const panel_x0 = 1.925;
    const panel_z0 = 1.925;
    const panel_side = 1.7;
    const panel_x1 = panel_x0 + panel_side;
    const panel_z1 = panel_z0 + panel_side;

    // Ceiling, as four strips around the opening.
    try scene.addQuad(Point3{ 0, size, 0 }, x, Vec3{ 0, 0, panel_z0 }, white);
    try scene.addQuad(Point3{ 0, size, panel_z1 }, x, Vec3{ 0, 0, size - panel_z1 }, white);
    try scene.addQuad(Point3{ 0, size, panel_z0 }, Vec3{ panel_x0, 0, 0 }, Vec3{ 0, 0, panel_side }, white);
    try scene.addQuad(Point3{ panel_x1, size, panel_z0 }, Vec3{ size - panel_x1, 0, 0 }, Vec3{ 0, 0, panel_side }, white);

    const emission = Color{ 15.0, 15.0, 15.0 };
    const panel_corner = Point3{ panel_x0, size, panel_z0 };
    const panel_u = Vec3{ panel_side, 0, 0 };
    const panel_v = Vec3{ 0, 0, panel_side };

    // Same corner and edges for both, so the lamp faces the way the light says
    // it does: edge_u x edge_v points down into the room.
    try scene.addQuad(panel_corner, panel_u, panel_v, Material.diffuseLight(Color{ 1, 1, 1 }, 15.0));
    try scene.addLight(Light.quadLight(panel_corner, panel_u, panel_v, emission));

    // Glass focuses the panel into a caustic on the floor; the metal sphere is
    // there to bounce it around the room.
    try scene.addSphere(Sphere.init(Point3{ 1.85, 1.0, 2.6 }, 1.0, Material.dielectric(1.5)));
    try scene.addSphere(Sphere.init(Point3{ 3.9, 0.8, 3.7 }, 0.8, Material.metal(Color{ 0.8, 0.85, 0.88 }, 0.0)));

    return scene;
}

/// How a studio scene is set up around whichever model it is given.
const StudioOptions = struct {
    /// Degrees around Y before fitting, to turn the model towards the camera.
    rotate_y: f64 = 0.0,
    /// Longest side after fitting, in world units.
    size: f64 = 6.0,
    /// Half-width of the floor.
    floor: f64 = 14.0,
    floor_color: Color = Color{ 0.58, 0.56, 0.52 },
    /// Radiance of the panel overhead.
    emission: Color = Color{ 26.0, 25.0, 24.0 },
    photons: PhotonBudget = .{},
};

/// One model on a floor under a panel, against a near-black sky.
///
/// Lit this way rather than by daylight because an open, sky-lit field washes
/// a caustic out, which is the whole reason these scenes exist. The camera and
/// the lamp are placed from the model's own fitted bounds, so pointing this at
/// a different model needs no new constants: that is what makes it worth
/// sharing between the dragon and the bunny.
fn buildStudioScene(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
    material: Material,
    options: StudioOptions,
) !SceneData {
    var scene = SceneData.init(allocator, CameraSpec{
        // Replaced below, once the model's fitted size is known.
        .lookfrom = Point3{ 0, 3, 11 },
        .lookat = Point3{ 0, 1, 0 },
        .vfov = 32.0,
    });
    errdefer scene.deinit();

    scene.background = .{ .solid = Color{ 0.02, 0.025, 0.035 } };
    scene.photons = options.photons;

    // A flat floor rather than a sphere of radius 1000: the photon grid is
    // sized from the scene, and a floor that is honestly flat keeps that box
    // tight around the part of the world the caustic lands on.
    try scene.addQuad(
        Point3{ -options.floor, 0, -options.floor },
        Vec3{ 2 * options.floor, 0, 0 },
        Vec3{ 0, 0, 2 * options.floor },
        Material.lambertian(options.floor_color),
    );

    var model_box: AABB = undefined;
    {
        var model = try loadFetchedMesh(allocator, io, model_path, material);
        defer model.deinit();

        try model.generateSmoothNormals(allocator);
        model.rotateY(options.rotate_y);
        model.fitTo(Point3{ 0, 0, 0 }, options.size);
        model.placeOnGround(0.0);

        model_box = model.bounds();
        try scene.addMesh(model);
    }

    const center = Point3{
        (model_box.x.min + model_box.x.max) / 2.0,
        (model_box.y.min + model_box.y.max) / 2.0,
        (model_box.z.min + model_box.z.max) / 2.0,
    };
    const reach = @max(@max(model_box.x.size(), model_box.y.size()), model_box.z.size());

    // Frame whatever the model turned out to be rather than a hand-tuned guess.
    const distance = reach * 1.9;
    scene.camera = CameraSpec{
        .lookfrom = Point3{ center[0] - 0.35 * reach, center[1] + 0.55 * reach, center[2] + distance },
        .lookat = center,
        .vfov = 34.0,
        .defocus_angle = 0.25,
        .focus_dist = distance,
    };

    // A panel over the model, high enough to light the floor around it.
    const panel_side = reach * 0.55;
    const panel_corner = Point3{
        center[0] - panel_side / 2.0,
        model_box.y.max + reach * 0.9,
        center[2] - panel_side / 2.0,
    };
    const panel_u = Vec3{ panel_side, 0, 0 };
    const panel_v = Vec3{ 0, 0, panel_side };

    const brightest = @max(@max(options.emission[0], options.emission[1]), options.emission[2]);
    try scene.addQuad(panel_corner, panel_u, panel_v, Material.diffuseLight(div(options.emission, brightest), brightest));
    try scene.addLight(Light.quadLight(panel_corner, panel_u, panel_v, options.emission));

    return scene;
}

/// The glass dragon: the XYZ RGB Asian Dragon rendered as a dielectric, which
/// is the classic photon-mapping subject and the reason these scenes exist.
///
/// The model is fetched rather than committed, so it may not be there.
fn buildGlassDragonScene(allocator: std.mem.Allocator, io: std.Io) !SceneData {
    return buildStudioScene(allocator, io, dragon_model_path, Material.dielectric(1.5), .{
        .rotate_y = 140.0,
        .size = 6.0,
        // Glass over a large floor sends a lot of photons a long way, and a
        // closed budget would stop the emission early and dim the caustic.
        .photons = .{ .emitted = 2_000_000, .capacity = 700_000 },
    });
}

/// The Stanford bunny in glass. The same studio as the dragon, at 69,451
/// triangles instead of 249,882, which makes it the one to iterate on when
/// changing anything about caustics: it builds its BVH in a fraction of the
/// time and still throws a proper caustic.
fn buildGlassBunnyScene(allocator: std.mem.Allocator, io: std.Io) !SceneData {
    return buildStudioScene(allocator, io, bunny_model_path, Material.dielectric(1.5), .{
        // The bunny faces the camera-left as it comes; turn it to a three-quarter view.
        .rotate_y = 270.0,
        .size = 4.5,
        // A third of the emitted photons come back as caustic photons here,
        // against a tenth for the dragon: less glass to get lost inside.
        .photons = .{ .emitted = 2_000_000, .capacity = 700_000 },
    });
}

/// Spot, Keenan Crane's cow, lit by daylight rather than staged in the dark.
///
/// Deliberately the plain one: a lambertian mesh under the sky, with no
/// dielectric anywhere. It is also the only model here whose faces carry
/// texture indices (`f v/vt`), so it exercises a corner of the OBJ parser the
/// other models never reach.
fn buildSpotScene(allocator: std.mem.Allocator, io: std.Io) !SceneData {
    var scene = SceneData.init(allocator, CameraSpec{
        .lookfrom = Point3{ 0, 3, 11 },
        .lookat = Point3{ 0, 1, 0 },
        .vfov = 32.0,
    });
    errdefer scene.deinit();

    // One small glass sphere converts about a third of the emitted photons
    // into stored caustic photons, which the default capacity cannot hold.
    scene.photons = .{ .emitted = 1_000_000, .capacity = 400_000 };

    const floor = 30.0;
    try scene.addQuad(
        Point3{ -floor, 0, -floor },
        Vec3{ 2 * floor, 0, 0 },
        Vec3{ 0, 0, 2 * floor },
        Material.lambertian(Color{ 0.48, 0.52, 0.42 }),
    );

    var spot_box: AABB = undefined;
    {
        var spot = try loadFetchedMesh(
            allocator,
            io,
            spot_model_path,
            Material.lambertian(Color{ 0.76, 0.34, 0.30 }),
        );
        defer spot.deinit();

        try spot.generateSmoothNormals(allocator);
        spot.rotateY(215.0);
        spot.fitTo(Point3{ 0, 0, 0 }, 4.0);
        spot.placeOnGround(0.0);

        spot_box = spot.bounds();
        try scene.addMesh(spot);
    }

    const center = Point3{
        (spot_box.x.min + spot_box.x.max) / 2.0,
        (spot_box.y.min + spot_box.y.max) / 2.0,
        (spot_box.z.min + spot_box.z.max) / 2.0,
    };
    const reach = @max(@max(spot_box.x.size(), spot_box.y.size()), spot_box.z.size());
    const distance = reach * 2.1;

    scene.camera = CameraSpec{
        .lookfrom = Point3{ center[0] + 0.5 * reach, center[1] + 0.45 * reach, center[2] + distance },
        .lookat = center,
        .vfov = 34.0,
        .focus_dist = distance,
    };

    // A glass sphere beside her, so the scene still has a caustic to show and
    // the photon grid still has something to anchor on.
    try scene.addSphere(Sphere.init(
        Point3{ center[0] - reach * 0.42, reach * 0.26, center[2] + reach * 0.55 },
        reach * 0.26,
        Material.dielectric(1.5),
    ));

    try scene.addLight(Light.pointLight(
        Point3{ 5, 9, 4 },
        Color{ 1.0, 1.0, 1.0 },
        900.0,
    ));

    return scene;
}

// ============================================================================
// Timing
// ============================================================================

/// Stopwatch over the render's stages. Loading a 250k-triangle model, building
/// a BVH over it and shooting photons at it are three very different costs, and
/// one total tells you nothing about which of them to worry about.
const StageTimer = struct {
    io: std.Io,
    last: std.Io.Timestamp,

    pub fn start(io: std.Io) StageTimer {
        return StageTimer{ .io = io, .last = std.Io.Timestamp.now(io, .awake) };
    }

    pub fn lap(self: *StageTimer) std.Io.Duration {
        const now = std.Io.Timestamp.now(self.io, .awake);
        const elapsed = self.last.durationTo(now);
        self.last = now;
        return elapsed;
    }
};

// ============================================================================
// Command Line
// ============================================================================

const CliArgs = struct {
    scene: *const Scene,
};

/// Returns null when the program has already said everything it was asked to
/// (`--help`, `--list-scenes`) and should exit successfully without rendering.
fn parseArgs(allocator: std.mem.Allocator, args: std.process.Args) !?CliArgs {
    var it = try args.iterateAllocator(allocator);
    defer it.deinit();
    _ = it.skip(); // program name

    var scene_name: []const u8 = default_scene_name;

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return null;
        } else if (std.mem.eql(u8, arg, "--list-scenes")) {
            printScenes();
            return null;
        } else if (std.mem.startsWith(u8, arg, "--scene=")) {
            scene_name = arg["--scene=".len..];
        } else if (std.mem.eql(u8, arg, "--scene")) {
            scene_name = it.next() orelse {
                std.debug.print("Error: --scene needs a scene name\n\n", .{});
                printScenes();
                return error.MissingSceneName;
            };
        } else {
            std.debug.print("Error: unknown argument '{s}'\n\n", .{arg});
            printUsage();
            return error.UnknownArgument;
        }
    }

    // The scene name borrows the iterator's buffer, but the scene it resolves
    // to is a static entry in `scenes`, so nothing outlives `it`.
    const scene = findScene(scene_name) orelse {
        std.debug.print("Error: unknown scene '{s}'\n\n", .{scene_name});
        printScenes();
        return error.UnknownScene;
    };

    return CliArgs{ .scene = scene };
}

fn printUsage() void {
    std.debug.print(
        \\Usage: zaytracer [options]
        \\
        \\Renders a scene to image.ppm.
        \\
        \\Options:
        \\  --scene=<name>   Scene to render (default: {s})
        \\  --list-scenes    List the available scenes and exit
        \\  -h, --help       Show this help and exit
        \\
        \\Image size, sample count, multithreading and the std.Io backend are
        \\build options; see the README for -Dwidth, -Dsamples, -Dmultithreading
        \\and -Dio.
        \\
    , .{default_scene_name});
}

fn printScenes() void {
    std.debug.print("Available scenes:\n", .{});
    for (&scenes) |scene| {
        const marker = if (std.mem.eql(u8, scene.name, default_scene_name)) " (default)" else "";
        std.debug.print("  {s}{s}\n      {s}\n", .{ scene.name, marker, scene.description });
    }
}

// ============================================================================
// Color Utilities
// ============================================================================

fn writeColor(file: std.Io.File, io: std.Io, color: Color, samples_per_pixel: u32) !void {
    var r = color[0];
    var g = color[1];
    var b = color[2];

    // Divide the color by the number of samples
    const scale = 1.0 / @as(f64, @floatFromInt(samples_per_pixel));
    r *= scale;
    g *= scale;
    b *= scale;

    // Apply gamma correction (gamma=2.0, so sqrt)
    r = @sqrt(r);
    g = @sqrt(g);
    b = @sqrt(b);

    // Convert to 0-255 range
    const ir = @as(u8, @intFromFloat(256.0 * std.math.clamp(r, 0.0, 0.999)));
    const ig = @as(u8, @intFromFloat(256.0 * std.math.clamp(g, 0.0, 0.999)));
    const ib = @as(u8, @intFromFloat(256.0 * std.math.clamp(b, 0.0, 0.999)));

    var buf: [64]u8 = undefined;
    const line = try std.fmt.bufPrint(&buf, "{d} {d} {d}\n", .{ ir, ig, ib });
    try file.writeStreamingAll(io, line);
}

// ============================================================================
// Multithreading Support
// ============================================================================

const Tile = struct {
    start_x: u32,
    start_y: u32,
    end_x: u32,
    end_y: u32,
};

const TileQueue = struct {
    tiles: []const Tile,
    current_index: std.atomic.Value(usize),

    pub fn init(tiles: []const Tile) TileQueue {
        return TileQueue{
            .tiles = tiles,
            .current_index = std.atomic.Value(usize).init(0),
        };
    }

    pub fn getNextTile(self: *TileQueue) ?Tile {
        const index = self.current_index.fetchAdd(1, .monotonic);
        if (index >= self.tiles.len) {
            return null;
        }
        return self.tiles[index];
    }
};

const PixelBuffer = struct {
    pixels: []Color,
    width: u32,
    height: u32,

    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32) !PixelBuffer {
        const pixels = try allocator.alloc(Color, width * height);
        @memset(pixels, Color{ 0, 0, 0 });
        return PixelBuffer{
            .pixels = pixels,
            .width = width,
            .height = height,
        };
    }

    pub fn deinit(self: PixelBuffer, allocator: std.mem.Allocator) void {
        allocator.free(self.pixels);
    }

    pub fn set(self: *PixelBuffer, x: u32, y: u32, color: Color) void {
        self.pixels[y * self.width + x] = color;
    }

    pub fn get(self: PixelBuffer, x: u32, y: u32) Color {
        return self.pixels[y * self.width + x];
    }
};

const Progress = struct {
    completed_tiles: std.atomic.Value(usize),
    total_tiles: usize,
    mutex: std.Io.Mutex,
    io: std.Io,

    pub fn init(io: std.Io, total_tiles: usize) Progress {
        return Progress{
            .completed_tiles = std.atomic.Value(usize).init(0),
            .total_tiles = total_tiles,
            .mutex = .init,
            .io = io,
        };
    }

    pub fn increment(self: *Progress, thread_id: u32) void {
        const completed = self.completed_tiles.fetchAdd(1, .monotonic) + 1;

        // Print progress every 10 tiles to reduce output spam
        if (completed % 10 == 0 or completed == self.total_tiles) {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);

            const percentage = @as(f64, @floatFromInt(completed)) / @as(f64, @floatFromInt(self.total_tiles)) * 100.0;
            std.debug.print("\rProgress: {d:.1}% ({d}/{d} tiles) - Thread {d}    ", .{ percentage, completed, self.total_tiles, thread_id });
        }
    }

    pub fn finish(self: *Progress) void {
        std.debug.print("\rRendering complete: 100.0% ({d}/{d} tiles)\n", .{ self.total_tiles, self.total_tiles });
    }
};

const WorkerContext = struct {
    queue: *TileQueue,
    pixel_buffer: *PixelBuffer,
    camera: *const Camera,
    render: RenderContext,
    samples_per_pixel: u32,
    max_depth: i32,
    thread_id: u32,
    progress: *Progress,
};

fn generateTiles(allocator: std.mem.Allocator, width: u32, height: u32, tile_size: u32) ![]Tile {
    var tiles = try std.ArrayList(Tile).initCapacity(allocator, 100);

    var y: u32 = 0;
    while (y < height) {
        var x: u32 = 0;
        while (x < width) {
            try tiles.append(allocator, Tile{
                .start_x = x,
                .start_y = y,
                .end_x = @min(x + tile_size, width),
                .end_y = @min(y + tile_size, height),
            });
            x += tile_size;
        }
        y += tile_size;
    }

    return tiles.toOwnedSlice(allocator);
}

fn workerThread(ctx: *WorkerContext) void {
    // Each thread gets unique RNG seed
    var prng = std.Random.DefaultPrng.init(42 + @as(u64, ctx.thread_id) * 12345);
    const rng = prng.random();

    // Process tiles until queue is empty
    while (ctx.queue.getNextTile()) |tile| {
        // Render this tile
        var y = tile.start_y;
        while (y < tile.end_y) : (y += 1) {
            var x = tile.start_x;
            while (x < tile.end_x) : (x += 1) {
                var pixel_color = Color{ 0, 0, 0 };

                // Multiple samples per pixel
                var sample: u32 = 0;
                while (sample < ctx.samples_per_pixel) : (sample += 1) {
                    const ray = ctx.camera.getRay(x, y, rng);
                    pixel_color = add(pixel_color, rayColor(ctx.render, ray, ctx.max_depth, rng));
                }

                // Write to shared buffer (no race condition - each thread writes different tiles)
                ctx.pixel_buffer.set(x, y, pixel_color);
            }
        }

        // Update progress
        ctx.progress.increment(ctx.thread_id);
    }
}

// ============================================================================
// Main
// ============================================================================

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try parseArgs(allocator, init.args) orelse return;

    var io_backend: IoBackend = undefined;
    try io_backend.init(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    std.debug.print("Scene: {s}\n", .{args.scene.name});
    var timer = StageTimer.start(io);
    var scene = try args.scene.build(allocator, io);
    defer scene.deinit();
    std.debug.print("Scene built in {f}\n", .{timer.lap()});

    // Build BVH for efficient ray-object intersection
    std.debug.print("Building BVH from {d} primitives...\n", .{scene.primitives.items.len});
    const world = try BVH.init(allocator, scene.primitives.items);
    defer world.deinit();
    std.debug.print("BVH built with {d} nodes in {f}\n", .{ world.nodes.len, timer.lap() });

    // Build photon map for caustics
    std.debug.print("Building photon map for caustics...\n", .{});
    const scene_box = if (world.nodes.len > 0) world.nodes[0].bbox else AABB.empty;
    const photon_bounds = derivePhotonBounds(scene.primitives.items, scene_box);
    const cell = photon_bounds.cellSize(photon_grid_size);
    std.debug.print(
        "Photon grid covers ({d:.2}, {d:.2}, {d:.2})..({d:.2}, {d:.2}, {d:.2}), cells of {d:.3} x {d:.3} x {d:.3}\n",
        .{
            photon_bounds.min[0], photon_bounds.min[1], photon_bounds.min[2],
            photon_bounds.max[0], photon_bounds.max[1], photon_bounds.max[2],
            cell[0],              cell[1],              cell[2],
        },
    );
    var photon_map = try PhotonMap.init(
        allocator,
        scene.photons.capacity,
        photon_bounds.min,
        photon_bounds.max,
        photon_grid_size,
    );
    defer photon_map.deinit();

    // Emit photons from light
    var photon_prng = std.Random.DefaultPrng.init(12345);
    const photon_rng = photon_prng.random();

    // Caustics are gathered from a single light: a scene with none simply
    // renders without them.
    if (scene.lights.items.len > 0) {
        const light = scene.lights.items[0];
        if (scene.lights.items.len > 1) {
            std.debug.print("Note: {d} lights in scene, emitting photons from the first only\n", .{scene.lights.items.len});
        }

        const projection = try ProjectionMap.init(allocator, light.position, world, photon_rng);
        defer projection.deinit();
        std.debug.print(
            "Projection map: {d} of {d} directions reach specular geometry ({d:.2}% of the sphere)\n",
            .{ projection.cells.len, ProjectionMap.cell_count, projection.sphereFraction() * 100.0 },
        );

        const photon_count = emitPhotonsFromLight(light, world, &photon_map, scene.photons.emitted, projection, photon_rng);
        std.debug.print("Emitted {d} photons, stored {d} caustic photons\n", .{ scene.photons.emitted, photon_count });
        if (photon_count >= scene.photons.capacity) {
            // Emission stops once the map is full, so the remaining photons never
            // get traced and the caustics are missing that share of the light.
            std.debug.print("Warning: photon map hit its capacity of {d}; the caustics are missing the light that was never emitted\n", .{scene.photons.capacity});
        }

        // Build spatial grid for fast photon queries
        try photon_map.buildGrid(photon_count);
        std.debug.print("Built photon spatial grid, caustic pass took {f}\n", .{timer.lap()});
    } else {
        std.debug.print("Scene has no lights: skipping caustics\n", .{});
    }

    // Camera setup - scene chooses the framing, build options the quality
    // Preview: -Dwidth=400 -Dsamples=10 (fast, ~5-10 seconds)
    // Final: -Dwidth=1200 -Dsamples=500 (slow, ~minutes to hours)
    const camera = scene.camera.toCamera(build_options.image_width);

    const render_ctx = RenderContext{
        .world = world,
        .background = scene.background,
        .photon_map = &photon_map,
        .lights = scene.lights.items,
    };

    const samples_per_pixel: u32 = build_options.samples_per_pixel; // Configurable via -Dsamples=N
    const max_depth: i32 = 50;

    // Configuration (build-time constants from build.zig)
    const use_multithreading = build_options.use_multithreading; // Set via -Dmultithreading=true/false
    const tile_size: u32 = 32; // 32x32 pixel tiles (only used if multithreading)

    // Create output file
    const file = try std.Io.Dir.cwd().createFile(io, "image.ppm", .{});
    defer file.close(io);

    // Write PPM header
    var header_buf: [256]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf, "P3\n{d} {d}\n255\n", .{ camera.image_width, camera.image_height });
    try file.writeStreamingAll(io, header);

    if (comptime use_multithreading) {
        // ===== MULTI-THREADED PATH =====
        const num_threads = try std.Thread.getCpuCount();
        std.debug.print("Multi-threaded mode: Using {d} threads\n", .{num_threads});

        // Generate tiles
        const tiles = try generateTiles(allocator, camera.image_width, camera.image_height, tile_size);
        defer allocator.free(tiles);
        std.debug.print("Generated {d} tiles of size {d}x{d}\n", .{ tiles.len, tile_size, tile_size });

        // Create tile queue and progress tracker
        var tile_queue = TileQueue.init(tiles);
        var progress = Progress.init(io, tiles.len);

        // Create shared pixel buffer
        var pixel_buffer = try PixelBuffer.init(allocator, camera.image_width, camera.image_height);
        defer pixel_buffer.deinit(allocator);

        // Spawn worker threads
        var threads = try allocator.alloc(std.Thread, num_threads);
        defer allocator.free(threads);

        var contexts = try allocator.alloc(WorkerContext, num_threads);
        defer allocator.free(contexts);

        std.debug.print("Starting render...\n", .{});
        for (0..num_threads) |i| {
            contexts[i] = WorkerContext{
                .queue = &tile_queue,
                .pixel_buffer = &pixel_buffer,
                .camera = &camera,
                .render = render_ctx,
                .samples_per_pixel = samples_per_pixel,
                .max_depth = max_depth,
                .thread_id = @intCast(i),
                .progress = &progress,
            };

            threads[i] = try std.Thread.spawn(.{}, workerThread, .{&contexts[i]});
        }

        // Wait for all threads to finish
        for (threads) |thread| {
            thread.join();
        }

        progress.finish();
        std.debug.print("Writing to file...\n", .{});

        // Write all pixels from buffer
        var y: u32 = 0;
        while (y < camera.image_height) : (y += 1) {
            var x: u32 = 0;
            while (x < camera.image_width) : (x += 1) {
                const color = pixel_buffer.get(x, y);
                try writeColor(file, io, color, samples_per_pixel);
            }
        }
    } else {
        // ===== SINGLE-THREADED PATH (original) =====
        std.debug.print("Single-threaded mode\n", .{});

        // Random number generator
        var prng = std.Random.DefaultPrng.init(42);
        const rng = prng.random();

        // Render
        var j: u32 = 0;
        while (j < camera.image_height) : (j += 1) {
            std.debug.print("\rScanlines remaining: {d} ", .{camera.image_height - j});

            var i: u32 = 0;
            while (i < camera.image_width) : (i += 1) {
                var pixel_color = Color{ 0, 0, 0 };

                // Take multiple samples per pixel
                var sample: u32 = 0;
                while (sample < samples_per_pixel) : (sample += 1) {
                    const ray = camera.getRay(i, j, rng);
                    pixel_color = add(pixel_color, rayColor(render_ctx, ray, max_depth, rng));
                }

                try writeColor(file, io, pixel_color, samples_per_pixel);
            }
        }
    }

    std.debug.print("\rRendered in {f}\n", .{timer.lap()});
}

// ============================================================================
// Tests - Caustic Photon Mapping
// ============================================================================

test "projection map samples stay inside the cell they belong to" {
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();

    const theta_f = @as(f64, ProjectionMap.theta_cells);
    const phi_f = @as(f64, ProjectionMap.phi_cells);

    for (0..ProjectionMap.cell_count) |cell| {
        const dir = ProjectionMap.sampleCell(cell, rng);
        try std.testing.expectApproxEqAbs(1.0, length(dir), 1e-9);

        const i = @as(f64, @floatFromInt(cell / ProjectionMap.phi_cells));
        const j = @as(f64, @floatFromInt(cell % ProjectionMap.phi_cells));

        // Polar angle: cells are uniform in cos(theta), which is just dir.y.
        try std.testing.expect(dir[1] <= 1.0 - 2.0 * i / theta_f + 1e-12);
        try std.testing.expect(dir[1] >= 1.0 - 2.0 * (i + 1.0) / theta_f - 1e-12);

        // Azimuth: atan2 returns (-pi, pi], so shift it into [0, 2pi).
        var phi = std.math.atan2(dir[2], dir[0]);
        if (phi < 0.0) phi += 2.0 * std.math.pi;
        try std.testing.expect(phi >= 2.0 * std.math.pi * j / phi_f - 1e-9);
        try std.testing.expect(phi <= 2.0 * std.math.pi * (j + 1.0) / phi_f + 1e-9);
    }
}

test "projection map only marks directions that reach specular geometry" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();

    var diffuse_only = [_]Primitive{
        Primitive{ .sphere = Sphere.init(Point3{ 0, -1000, 0 }, 1000, Material.lambertian(Color{ 0.5, 0.5, 0.5 })) },
    };
    const world_diffuse = try BVH.init(allocator, &diffuse_only);
    defer world_diffuse.deinit();

    const no_targets = try ProjectionMap.init(allocator, Point3{ 0, 5, 0 }, world_diffuse, rng);
    defer no_targets.deinit();
    try std.testing.expectEqual(@as(usize, 0), no_targets.cells.len);

    var with_glass = [_]Primitive{
        Primitive{ .sphere = Sphere.init(Point3{ 0, -1000, 0 }, 1000, Material.lambertian(Color{ 0.5, 0.5, 0.5 })) },
        Primitive{ .sphere = Sphere.init(Point3{ 0, 2, 0 }, 1.0, Material.dielectric(1.5)) },
    };
    const world_glass = try BVH.init(allocator, &with_glass);
    defer world_glass.deinit();

    const targets = try ProjectionMap.init(allocator, Point3{ 0, 5, 0 }, world_glass, rng);
    defer targets.deinit();

    // The sphere is straight below the light and covers a small part of the
    // sphere of directions, so emission should be aimed at a small subset.
    try std.testing.expect(targets.cells.len > 0);
    try std.testing.expect(targets.sphereFraction() < 0.25);
}

test "photon map only stores photons that arrived via a specular bounce" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();

    const ground = Primitive{ .sphere = Sphere.init(Point3{ 0, -1000, 0 }, 1000, Material.lambertian(Color{ 0.5, 0.5, 0.5 })) };
    const straight_down = Ray.init(Point3{ 0, 5, 0 }, Vec3{ 0, -1, 0 });
    const power = Color{ 1, 1, 1 };

    var photons: [64]Photon = undefined;

    // L -> D only: the path tracer already accounts for this light, so the
    // caustic map must stay empty.
    var diffuse_only = [_]Primitive{ground};
    const world_diffuse = try BVH.init(allocator, &diffuse_only);
    defer world_diffuse.deinit();

    var diffuse_count: usize = 0;
    for (0..16) |_| {
        tracePhoton(straight_down, world_diffuse, photon_max_bounces, power, 0, &photons, &diffuse_count, photons.len, rng);
    }
    try std.testing.expectEqual(@as(usize, 0), diffuse_count);

    // L -> S -> D: the same ray now refracts through glass before landing on
    // the ground, which is exactly what a caustic map should hold.
    var with_glass = [_]Primitive{
        ground,
        Primitive{ .sphere = Sphere.init(Point3{ 0, 2, 0 }, 1.0, Material.dielectric(1.5)) },
    };
    const world_glass = try BVH.init(allocator, &with_glass);
    defer world_glass.deinit();

    var caustic_count: usize = 0;
    for (0..16) |_| {
        tracePhoton(straight_down, world_glass, photon_max_bounces, power, 0, &photons, &caustic_count, photons.len, rng);
    }
    try std.testing.expect(caustic_count > 0);

    // Dielectrics do not absorb, so the stored photons carry the full power.
    for (photons[0..caustic_count]) |photon| {
        try std.testing.expectEqual(power, photon.power);
    }
}

test "radiance estimate applies the lambertian brdf" {
    const allocator = std.testing.allocator;

    var map = try PhotonMap.init(allocator, 4, Point3{ -1, -1, -1 }, Point3{ 1, 1, 1 }, 4);
    defer map.deinit();

    const power = Color{ 0.5, 0.25, 0.125 };
    map.photons[0] = Photon{ .position = Point3{ 0, 0, 0 }, .direction = Vec3{ 0, -1, 0 }, .power = power };
    try map.buildGrid(1);

    const radius = 0.2;
    const albedo = Color{ 0.8, 0.4, 0.2 };
    const estimate = map.estimateRadiance(Point3{ 0, 0, 0 }, Vec3{ 0, 1, 0 }, albedo, radius);

    // Flux over the gather disk, turned into radiance by the BRDF (albedo / pi).
    const expected = div(mulVec(power, albedo), std.math.pi * radius * radius * std.math.pi);
    inline for (0..3) |channel| {
        try std.testing.expectApproxEqRel(expected[channel], estimate[channel], 1e-12);
    }

    // The estimate is tinted by the surface: a grey surface reflects less of the
    // red channel than a red one does.
    const grey = map.estimateRadiance(Point3{ 0, 0, 0 }, Vec3{ 0, 1, 0 }, Color{ 0.5, 0.5, 0.5 }, radius);
    const red = map.estimateRadiance(Point3{ 0, 0, 0 }, Vec3{ 0, 1, 0 }, Color{ 1.0, 0.1, 0.1 }, radius);
    try std.testing.expect(red[0] > grey[0]);
    try std.testing.expect(red[2] < grey[2]);

    // A photon arriving from behind the surface is not part of the estimate.
    const backside = map.estimateRadiance(Point3{ 0, 0, 0 }, Vec3{ 0, -1, 0 }, albedo, radius);
    try std.testing.expectEqual(Color{ 0, 0, 0 }, backside);
}

test "aiming emission redistributes the light power instead of adding energy" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();

    var primitives = [_]Primitive{
        Primitive{ .sphere = Sphere.init(Point3{ 0, -1000, 0 }, 1000, Material.lambertian(Color{ 0.5, 0.5, 0.5 })) },
        Primitive{ .sphere = Sphere.init(Point3{ 0, 2, 0 }, 1.0, Material.dielectric(1.5)) },
    };
    const world = try BVH.init(allocator, &primitives);
    defer world.deinit();

    const light = Light.pointLight(Point3{ 0, 5, 0 }, Color{ 1, 1, 1 }, 1000.0);

    const projection = try ProjectionMap.init(allocator, light.position, world, rng);
    defer projection.deinit();

    var map = try PhotonMap.init(allocator, 4096, Point3{ -15, -5, -15 }, Point3{ 15, 15, 15 }, 32);
    defer map.deinit();

    const emitted: u32 = 4096;
    const stored = emitPhotonsFromLight(light, world, &map, emitted, projection, rng);
    try std.testing.expect(stored > 0);

    // Every photon stands for the light's power over the aimed solid angle, so
    // the flux stored can never exceed what the light emits into it.
    var flux: f64 = 0;
    for (map.photons[0..stored]) |photon| flux += photon.power[0];
    try std.testing.expect(flux <= light.power * projection.sphereFraction() + 1e-9);
}

// ============================================================================
// Tests - Scenes, Lights and Placement
// ============================================================================

test "the scene registry answers only for names it knows" {
    try std.testing.expect(findScene("no-such-scene") == null);

    // A typo here would break every run that does not pass --scene.
    const fallback = findScene(default_scene_name);
    try std.testing.expect(fallback != null);
    try std.testing.expectEqualStrings(default_scene_name, fallback.?.name);

    for (&scenes) |scene| {
        try std.testing.expect(scene.name.len > 0);
        try std.testing.expect(scene.description.len > 0);

        const found = findScene(scene.name);
        try std.testing.expect(found != null);
        try std.testing.expectEqualStrings(scene.name, found.?.name);
    }

    // Names have to be unique, or findScene quietly returns the first of them
    // and one scene becomes unreachable from the command line.
    for (&scenes, 0..) |scene, i| {
        for (scenes[i + 1 ..]) |other| {
            try std.testing.expect(!std.mem.eql(u8, scene.name, other.name));
        }
    }
}

test "every fetched model a scene asks for is in the manifest" {
    // Embedded rather than read at run time: this is a question about the
    // repository, not about what happens to be on this machine, and a scene
    // pointing at a model no one can fetch should fail the build.
    const manifest = @embedFile("models_manifest");

    const fetched = [_][]const u8{ dragon_model_path, bunny_model_path, spot_model_path };

    for (fetched) |path| {
        // The manifest names files; the scenes name paths under models/.
        const prefix = "models/";
        try std.testing.expect(std.mem.startsWith(u8, path, prefix));
        const filename = path[prefix.len..];

        var listed = false;
        var lines = std.mem.splitScalar(u8, manifest, '\n');
        while (lines.next()) |line| {
            if (line.len == 0 or line[0] == '#') continue;
            var fields = std.mem.splitScalar(u8, line, '\t');
            const name = fields.next() orelse continue;
            if (std.mem.eql(u8, std.mem.trim(u8, name, " \r"), filename)) {
                // A url and a checksum, or `make models` cannot act on it.
                const url = fields.next() orelse return error.ManifestEntryMissingUrl;
                const sum = fields.next() orelse return error.ManifestEntryMissingChecksum;
                try std.testing.expect(url.len > 0);
                try std.testing.expectEqual(@as(usize, 64), std.mem.trim(u8, sum, " \r").len);
                listed = true;
                break;
            }
        }

        if (!listed) {
            std.debug.print("model '{s}' is used by a scene but not listed in models/manifest.tsv\n", .{filename});
            return error.ModelNotInManifest;
        }
    }
}

test "a light emits from one face and scatters nothing" {
    const material = Material.diffuseLight(Color{ 1.0, 0.5, 0.25 }, 4.0);

    try std.testing.expectEqual(Color{ 4.0, 2.0, 1.0 }, material.emitted(true));
    try std.testing.expectEqual(Color{ 0, 0, 0 }, material.emitted(false));

    var prng = std.Random.DefaultPrng.init(1);
    var scattered: Ray = undefined;
    var attenuation: Color = undefined;
    const rec = HitRecord{
        .point = Point3{ 0, 0, 0 },
        .normal = Vec3{ 0, 1, 0 },
        .material = material,
        .t = 1.0,
        .front_face = true,
    };
    try std.testing.expect(!material.scatter(
        Ray.init(Point3{ 0, 1, 0 }, Vec3{ 0, -1, 0 }),
        rec,
        &attenuation,
        &scattered,
        prng.random(),
    ));
}

test "an area light is sampled from its front face and through nothing solid" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();

    // Facing down: edge_u x edge_v points along -y.
    const light = Light.quadLight(
        Point3{ -1, 3, -1 },
        Vec3{ 2, 0, 0 },
        Vec3{ 0, 0, 2 },
        Color{ 10, 10, 10 },
    );
    const albedo = Color{ 1, 1, 1 };

    var nothing = [_]Primitive{};
    const empty_world = try BVH.init(allocator, &nothing);
    defer empty_world.deinit();

    // Below it, facing up: lit.
    const lit = light.sampleDirect(empty_world, Point3{ 0, 0, 0 }, Vec3{ 0, 1, 0 }, albedo, rng);
    try std.testing.expect(lit[0] > 0.0);

    // Above it, facing down. The lamp's back is dark, and it is the near-zero
    // distance to a point just above the panel that makes this worth checking:
    // the geometry term divides by the square of it.
    const behind = light.sampleDirect(empty_world, Point3{ 0, 3.001, 0 }, Vec3{ 0, -1, 0 }, albedo, rng);
    try std.testing.expectEqual(Color{ 0, 0, 0 }, behind);

    // Facing away from the light entirely.
    const turned_away = light.sampleDirect(empty_world, Point3{ 0, 0, 0 }, Vec3{ 0, -1, 0 }, albedo, rng);
    try std.testing.expectEqual(Color{ 0, 0, 0 }, turned_away);

    // With something in the way.
    var blocker = [_]Primitive{
        Primitive{ .sphere = Sphere.init(Point3{ 0, 1.5, 0 }, 0.5, Material.lambertian(Color{ 0.5, 0.5, 0.5 })) },
    };
    const blocked_world = try BVH.init(allocator, &blocker);
    defer blocked_world.deinit();

    const blocked = light.sampleDirect(blocked_world, Point3{ 0, 0, 0 }, Vec3{ 0, 1, 0 }, albedo, rng);
    try std.testing.expectEqual(Color{ 0, 0, 0 }, blocked);
}

test "emission stops being counted once a path has touched a diffuse surface" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();

    // A mirror on the floor with a lamp above it, so a ray aimed straight down
    // reflects straight back up into the light. Metal with no fuzz makes the
    // whole path deterministic.
    var scene = SceneData.init(allocator, CameraSpec{
        .lookfrom = Point3{ 0, 1, 0 },
        .lookat = Point3{ 0, 0, 0 },
        .vfov = 40.0,
    });
    defer scene.deinit();

    try scene.addQuad(
        Point3{ -2, 0, -2 },
        Vec3{ 4, 0, 0 },
        Vec3{ 0, 0, 4 },
        Material.metal(Color{ 1, 1, 1 }, 0.0),
    );
    try scene.addQuad(
        Point3{ -2, 3, -2 },
        Vec3{ 4, 0, 0 },
        Vec3{ 0, 0, 4 },
        Material.diffuseLight(Color{ 1, 1, 1 }, 6.0),
    );

    const world = try BVH.init(allocator, scene.primitives.items);
    defer world.deinit();

    var map = try PhotonMap.init(allocator, 1, Point3{ -4, -1, -4 }, Point3{ 4, 4, 4 }, 4);
    defer map.deinit();

    const down = Ray.init(Point3{ 0, 1, 0 }, Vec3{ 0, -1, 0 });
    const with_map = RenderContext{
        .world = world,
        .background = .{ .solid = Color{ 0, 0, 0 } },
        .photon_map = &map,
    };

    // Straight from the camera: the lamp's reflection is visible, and nothing
    // else accounts for it.
    const seen = rayColorInner(with_map, down, 8, rng, true);
    try std.testing.expectApproxEqAbs(6.0, seen[0], 1e-9);

    // The same path, but reached after a diffuse bounce. Direct sampling
    // covered the lamp at that surface and the caustic map covers it arriving
    // via the mirror, so counting it here would be the third time.
    const already_accounted = rayColorInner(with_map, down, 8, rng, false);
    try std.testing.expectEqual(Color{ 0, 0, 0 }, already_accounted);

    // Without a caustic map nothing else covers specular paths, so the
    // renderer falls back to counting them: plain path tracing.
    const path_traced = RenderContext{
        .world = world,
        .background = .{ .solid = Color{ 0, 0, 0 } },
        .photon_map = null,
    };
    const counted = rayColorInner(path_traced, down, 8, rng, false);
    try std.testing.expectApproxEqAbs(6.0, counted[0], 1e-9);
}

test "the photon grid ignores the ground sphere it stands on" {
    const ground = Primitive{ .sphere = Sphere.init(Point3{ 0, -1000, 0 }, 1000, Material.lambertian(Color{ 0.5, 0.5, 0.5 })) };
    const glass = Primitive{ .sphere = Sphere.init(Point3{ 0, 1, 0 }, 1.0, Material.dielectric(1.5)) };

    var open_scene = [_]Primitive{ ground, glass };
    const open_box = AABB.fromBoxes(ground.boundingBox(), glass.boundingBox());
    const open = derivePhotonBounds(&open_scene, open_box);

    // The scene is 2000 units across. Covering it would put every cell about
    // 15 units wide against a gather radius of 0.2.
    inline for (0..3) |axis| {
        try std.testing.expect(open.max[axis] - open.min[axis] <= min_photon_extent + 1e-9);
    }

    // The glass has to be inside it, or its caustic never lands anywhere.
    try std.testing.expect(open.min[1] <= 0.0 and open.max[1] >= 2.0);
    try std.testing.expect(open.min[0] <= -1.0 and open.max[0] >= 1.0);

    // A scene small enough to cover entirely should be covered entirely, walls
    // and all: caustics in a closed room land well above the floor.
    var room = [_]Primitive{
        Primitive{ .sphere = Sphere.init(Point3{ 0, 1, 0 }, 1.0, Material.dielectric(1.5)) },
        Primitive{ .triangle = Triangle.init(Point3{ -3, 0, -3 }, Point3{ 3, 0, -3 }, Point3{ 3, 6, -3 }, Material.lambertian(Color{ 0.7, 0.7, 0.7 })) },
    };
    var room_box = AABB.empty;
    for (room) |prim| room_box = AABB.fromBoxes(room_box, prim.boundingBox());

    const closed = derivePhotonBounds(&room, room_box);
    inline for (0..3) |axis| {
        try std.testing.expectApproxEqAbs(room_box.axis(axis).min, closed.min[axis], 1e-9);
        try std.testing.expectApproxEqAbs(room_box.axis(axis).max, closed.max[axis], 1e-9);
    }
}

test "a fitted mesh lands where it was asked to" {
    const allocator = std.testing.allocator;

    const triangles = try allocator.alloc(Triangle, 1);
    triangles[0] = Triangle.init(
        Point3{ 100, 50, 20 },
        Point3{ 110, 50, 20 },
        Point3{ 100, 54, 20 },
        Material.lambertian(Color{ 0.5, 0.5, 0.5 }),
    );
    var mesh = Mesh{ .triangles = triangles, .allocator = allocator };
    defer mesh.deinit();

    // Longest side is 10, so fitting it to 5 halves the mesh.
    mesh.fitTo(Point3{ 1, 2, 3 }, 5.0);

    var box = mesh.bounds();
    try std.testing.expectApproxEqAbs(5.0, box.x.size(), 1e-9);
    try std.testing.expectApproxEqAbs(2.0, box.y.size(), 1e-9);
    try std.testing.expectApproxEqAbs(1.0, (box.x.min + box.x.max) / 2.0, 1e-9);
    try std.testing.expectApproxEqAbs(2.0, (box.y.min + box.y.max) / 2.0, 1e-9);
    try std.testing.expectApproxEqAbs(3.0, (box.z.min + box.z.max) / 2.0, 1e-9);

    mesh.placeOnGround(0.0);
    box = mesh.bounds();
    try std.testing.expectApproxEqAbs(0.0, box.y.min, 1e-9);
    try std.testing.expectApproxEqAbs(2.0, box.y.max, 1e-9);

    // Sitting it on the floor must not have resized or slid it sideways.
    try std.testing.expectApproxEqAbs(5.0, box.x.size(), 1e-9);
    try std.testing.expectApproxEqAbs(1.0, (box.x.min + box.x.max) / 2.0, 1e-9);
}
