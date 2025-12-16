const std = @import("std");
const build_options = @import("build_options");

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
};

const Material = struct {
    material_type: MaterialType,
    albedo: Color,
    fuzz: f64, // Only used for metal
    refraction_index: f64, // Only used for dielectric

    pub fn lambertian(albedo: Color) Material {
        return Material{
            .material_type = .lambertian,
            .albedo = albedo,
            .fuzz = 0.0,
            .refraction_index = 0.0,
        };
    }

    pub fn metal(albedo: Color, fuzz: f64) Material {
        return Material{
            .material_type = .metal,
            .albedo = albedo,
            .fuzz = if (fuzz < 1.0) fuzz else 1.0,
            .refraction_index = 0.0,
        };
    }

    pub fn dielectric(refraction_index: f64) Material {
        return Material{
            .material_type = .dielectric,
            .albedo = Color{ 1.0, 1.0, 1.0 },
            .fuzz = 0.0,
            .refraction_index = refraction_index,
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
        }
    }
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
            return a_center[ctx.axis_idx] < b_center[ctx.axis_idx];
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
// OBJ File Parser
// ============================================================================

const OBJParseError = error{
    InvalidFormat,
    MissingData,
    InvalidIndex,
} || std.mem.Allocator.Error || std.fs.File.OpenError || std.fs.File.ReadError;

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

fn parseOBJ(allocator: std.mem.Allocator, filepath: []const u8) !OBJData {
    const file = try std.fs.cwd().openFile(filepath, .{});
    defer file.close();

    // Read entire file (reasonable for small-medium meshes up to 50MB)
    const max_file_size = 50 * 1024 * 1024;
    const contents = try file.readToEndAlloc(allocator, max_file_size);
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
        filepath: []const u8,
        material: Material,
    ) !Mesh {
        const obj_data = try parseOBJ(allocator, filepath);
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

fn rayColor(ray: Ray, world: BVH, depth: i32, rng: std.Random) Color {
    // If we've exceeded the ray bounce limit, no more light is gathered
    if (depth <= 0) {
        return Color{ 0, 0, 0 };
    }

    var rec: HitRecord = undefined;

    if (world.hit(ray, Interval{ .min = 0.001, .max = std.math.inf(f64) }, &rec)) {
        var scattered: Ray = undefined;
        var attenuation: Color = undefined;

        if (rec.material.scatter(ray, rec, &attenuation, &scattered, rng)) {
            return mulVec(attenuation, rayColor(scattered, world, depth - 1, rng));
        }
        return Color{ 0, 0, 0 };
    }

    // Create a gradient background from white to blue
    const unit_direction = unitVector(ray.direction);
    const a = 0.5 * (unit_direction[1] + 1.0);

    // Linear interpolation: (1-a)*white + a*blue
    const white = Color{ 1.0, 1.0, 1.0 };
    const blue = Color{ 0.5, 0.7, 1.0 };

    return add(mul(white, 1.0 - a), mul(blue, a));
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
// Color Utilities
// ============================================================================

fn writeColor(file: std.fs.File, color: Color, samples_per_pixel: u32) !void {
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
    _ = try file.writeAll(line);
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
    mutex: std.Thread.Mutex,

    pub fn init(total_tiles: usize) Progress {
        return Progress{
            .completed_tiles = std.atomic.Value(usize).init(0),
            .total_tiles = total_tiles,
            .mutex = .{},
        };
    }

    pub fn increment(self: *Progress, thread_id: u32) void {
        const completed = self.completed_tiles.fetchAdd(1, .monotonic) + 1;

        // Print progress every 10 tiles to reduce output spam
        if (completed % 10 == 0 or completed == self.total_tiles) {
            self.mutex.lock();
            defer self.mutex.unlock();

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
    world: *const BVH,
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
                    pixel_color = add(pixel_color, rayColor(ray, ctx.world.*, ctx.max_depth, rng));
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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Random number generator for scene generation
    var scene_prng = std.Random.DefaultPrng.init(0);
    const scene_rng = scene_prng.random();

    // Build final scene with many random spheres
    var primitive_list = try std.ArrayList(Primitive).initCapacity(allocator, 500);
    defer primitive_list.deinit(allocator);

    // Ground
    try primitive_list.append(allocator, Primitive{
        .sphere = Sphere.init(
            Point3{ 0, -1000, 0 },
            1000,
            Material.lambertian(Color{ 0.5, 0.5, 0.5 }),
        ),
    });

    // Load test cube mesh
    const cube_mesh = try Mesh.fromOBJ(
        allocator,
        "models/test_cube.obj",
        Material.lambertian(Color{ 0.8, 0.3, 0.3 }), // Red-ish
    );
    defer cube_mesh.deinit();

    // Add cube triangles to scene
    for (cube_mesh.triangles) |tri| {
        try primitive_list.append(allocator, Primitive{ .triangle = tri });
    }

    // Load teapot mesh
    var teapot_mesh = try Mesh.fromOBJ(
        allocator,
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

    // Add teapot triangles to scene
    for (teapot_mesh.triangles) |tri| {
        try primitive_list.append(allocator, Primitive{ .triangle = tri });
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
                    try primitive_list.append(allocator, Primitive{ .sphere = Sphere.init(center, 0.2, Material.lambertian(albedo)) });
                } else if (choose_mat < 0.95) {
                    // Metal
                    const albedo = Color{
                        randomFloatRange(scene_rng, 0.5, 1.0),
                        randomFloatRange(scene_rng, 0.5, 1.0),
                        randomFloatRange(scene_rng, 0.5, 1.0),
                    };
                    const fuzz = randomFloatRange(scene_rng, 0.0, 0.5);
                    try primitive_list.append(allocator, Primitive{ .sphere = Sphere.init(center, 0.2, Material.metal(albedo, fuzz)) });
                } else {
                    // Glass
                    try primitive_list.append(allocator, Primitive{ .sphere = Sphere.init(center, 0.2, Material.dielectric(1.5)) });
                }
            }
        }
    }

    // Three large spheres
    try primitive_list.append(allocator, Primitive{ .sphere = Sphere.init(Point3{ 0, 1, 0 }, 1.0, Material.dielectric(1.5)) });
    try primitive_list.append(allocator, Primitive{ .sphere = Sphere.init(Point3{ -4, 1, 0 }, 1.0, Material.lambertian(Color{ 0.4, 0.2, 0.1 })) });
    try primitive_list.append(allocator, Primitive{ .sphere = Sphere.init(Point3{ 4, 1, 0 }, 1.0, Material.metal(Color{ 0.7, 0.6, 0.5 }, 0.0)) });

    // Build BVH for efficient ray-object intersection
    std.debug.print("Building BVH from {d} primitives...\n", .{primitive_list.items.len});
    const world = try BVH.init(allocator, primitive_list.items);
    defer world.deinit();
    std.debug.print("BVH built with {d} nodes\n", .{world.nodes.len});

    // Camera setup - quality controlled by build options
    // Preview: -Dwidth=400 -Dsamples=10 (fast, ~5-10 seconds)
    // Final: -Dwidth=1200 -Dsamples=500 (slow, ~minutes to hours)
    const camera = Camera.init(
        Point3{ 13, 2, 3 }, // lookfrom
        Point3{ 0, 0, 0 }, // lookat
        Vec3{ 0, 1, 0 }, // vup
        20.0, // vfov
        16.0 / 9.0, // aspect ratio
        build_options.image_width, // Configurable via -Dwidth=N
        0.6, // defocus angle
        10.0, // focus distance
    );

    const samples_per_pixel: u32 = build_options.samples_per_pixel; // Configurable via -Dsamples=N
    const max_depth: i32 = 50;

    // Configuration (build-time constants from build.zig)
    const use_multithreading = build_options.use_multithreading; // Set via -Dmultithreading=true/false
    const tile_size: u32 = 32; // 32x32 pixel tiles (only used if multithreading)

    // Create output file
    const file = try std.fs.cwd().createFile("image.ppm", .{});
    defer file.close();

    // Write PPM header
    var header_buf: [256]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf, "P3\n{d} {d}\n255\n", .{ camera.image_width, camera.image_height });
    _ = try file.writeAll(header);

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
        var progress = Progress.init(tiles.len);

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
                .world = &world,
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
                try writeColor(file, color, samples_per_pixel);
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
                    pixel_color = add(pixel_color, rayColor(ray, world, max_depth, rng));
                }

                try writeColor(file, pixel_color, samples_per_pixel);
            }
        }
    }

    std.debug.print("\rDone.                 \n", .{});
}
