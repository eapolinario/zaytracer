const std = @import("std");

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
        .x = randomFloat(rng),
        .y = randomFloat(rng),
        .z = randomFloat(rng),
    };
}

fn randomVec3Range(rng: std.Random, min: f64, max: f64) Vec3 {
    return Vec3{
        .x = randomFloatRange(rng, min, max),
        .y = randomFloatRange(rng, min, max),
        .z = randomFloatRange(rng, min, max),
    };
}

fn randomInUnitSphere(rng: std.Random) Vec3 {
    while (true) {
        const p = randomVec3Range(rng, -1.0, 1.0);
        if (p.lengthSquared() < 1.0) {
            return p;
        }
    }
}

fn randomUnitVector(rng: std.Random) Vec3 {
    return randomInUnitSphere(rng).unitVector();
}

// ============================================================================
// Vec3 - 3D Vector/Point/Color
// ============================================================================

const Vec3 = struct {
    x: f64,
    y: f64,
    z: f64,

    pub fn init(x: f64, y: f64, z: f64) Vec3 {
        return Vec3{ .x = x, .y = y, .z = z };
    }

    pub inline fn add(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x + other.x,
            .y = self.y + other.y,
            .z = self.z + other.z,
        };
    }

    pub inline fn sub(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x - other.x,
            .y = self.y - other.y,
            .z = self.z - other.z,
        };
    }

    pub inline fn mul(self: Vec3, scalar: f64) Vec3 {
        return Vec3{
            .x = self.x * scalar,
            .y = self.y * scalar,
            .z = self.z * scalar,
        };
    }

    pub inline fn mulVec(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x * other.x,
            .y = self.y * other.y,
            .z = self.z * other.z,
        };
    }

    pub inline fn div(self: Vec3, scalar: f64) Vec3 {
        return self.mul(1.0 / scalar);
    }

    pub inline fn dot(self: Vec3, other: Vec3) f64 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    pub inline fn cross(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.y * other.z - self.z * other.y,
            .y = self.z * other.x - self.x * other.z,
            .z = self.x * other.y - self.y * other.x,
        };
    }

    pub inline fn lengthSquared(self: Vec3) f64 {
        return self.x * self.x + self.y * self.y + self.z * self.z;
    }

    pub inline fn length(self: Vec3) f64 {
        return @sqrt(self.lengthSquared());
    }

    pub inline fn unitVector(self: Vec3) Vec3 {
        return self.div(self.length());
    }

    pub inline fn neg(self: Vec3) Vec3 {
        return Vec3{
            .x = -self.x,
            .y = -self.y,
            .z = -self.z,
        };
    }

    pub inline fn nearZero(self: Vec3) bool {
        const s = 1e-8;
        return (@abs(self.x) < s) and (@abs(self.y) < s) and (@abs(self.z) < s);
    }
};

// Type aliases for semantic clarity
const Point3 = Vec3;
const Color = Vec3;

// ============================================================================
// Vector Utilities
// ============================================================================

fn reflect(v: Vec3, n: Vec3) Vec3 {
    return v.sub(n.mul(2.0 * v.dot(n)));
}

// ============================================================================
// Material
// ============================================================================

const MaterialType = enum {
    lambertian,
    metal,
};

const Material = struct {
    material_type: MaterialType,
    albedo: Color,
    fuzz: f64, // Only used for metal

    pub fn lambertian(albedo: Color) Material {
        return Material{
            .material_type = .lambertian,
            .albedo = albedo,
            .fuzz = 0.0,
        };
    }

    pub fn metal(albedo: Color, fuzz: f64) Material {
        return Material{
            .material_type = .metal,
            .albedo = albedo,
            .fuzz = if (fuzz < 1.0) fuzz else 1.0,
        };
    }

    pub fn scatter(self: Material, ray_in: Ray, rec: HitRecord, attenuation: *Color, scattered: *Ray, rng: std.Random) bool {
        switch (self.material_type) {
            .lambertian => {
                var scatter_direction = rec.normal.add(randomUnitVector(rng));

                // Catch degenerate scatter direction
                if (scatter_direction.nearZero()) {
                    scatter_direction = rec.normal;
                }

                scattered.* = Ray.init(rec.point, scatter_direction);
                attenuation.* = self.albedo;
                return true;
            },
            .metal => {
                const reflected = reflect(ray_in.direction.unitVector(), rec.normal);
                scattered.* = Ray.init(rec.point, reflected.add(randomInUnitSphere(rng).mul(self.fuzz)));
                attenuation.* = self.albedo;
                return scattered.direction.dot(rec.normal) > 0;
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
        return self.origin.add(self.direction.mul(t));
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
        self.front_face = ray.direction.dot(outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else outward_normal.neg();
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

    pub fn hit(self: Sphere, ray: Ray, ray_t: Interval, rec: *HitRecord) bool {
        const oc = self.center.sub(ray.origin);
        const a = ray.direction.lengthSquared();
        const h = ray.direction.dot(oc);
        const c = oc.lengthSquared() - self.radius * self.radius;
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
        const outward_normal = rec.point.sub(self.center).div(self.radius);
        rec.setFaceNormal(ray, outward_normal);

        return true;
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
// Ray Color (Background)
// ============================================================================

fn rayColor(ray: Ray, world: HittableList, depth: i32, rng: std.Random) Color {
    // If we've exceeded the ray bounce limit, no more light is gathered
    if (depth <= 0) {
        return Color{ .x = 0, .y = 0, .z = 0 };
    }

    var rec: HitRecord = undefined;

    if (world.hit(ray, Interval{ .min = 0.001, .max = std.math.inf(f64) }, &rec)) {
        var scattered: Ray = undefined;
        var attenuation: Color = undefined;

        if (rec.material.scatter(ray, rec, &attenuation, &scattered, rng)) {
            return attenuation.mulVec(rayColor(scattered, world, depth - 1, rng));
        }
        return Color{ .x = 0, .y = 0, .z = 0 };
    }

    // Create a gradient background from white to blue
    const unit_direction = ray.direction.unitVector();
    const a = 0.5 * (unit_direction.y + 1.0);

    // Linear interpolation: (1-a)*white + a*blue
    const white = Color{ .x = 1.0, .y = 1.0, .z = 1.0 };
    const blue = Color{ .x = 0.5, .y = 0.7, .z = 1.0 };

    return white.mul(1.0 - a).add(blue.mul(a));
}

// ============================================================================
// Color Utilities
// ============================================================================

fn writeColor(file: std.fs.File, color: Color, samples_per_pixel: u32) !void {
    var r = color.x;
    var g = color.y;
    var b = color.z;

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
// Main
// ============================================================================

pub fn main() !void {
    // World - create a scene with different materials
    const material_ground = Material.lambertian(Color{ .x = 0.8, .y = 0.8, .z = 0.0 });
    const material_center = Material.lambertian(Color{ .x = 0.1, .y = 0.2, .z = 0.5 });
    const material_left = Material.metal(Color{ .x = 0.8, .y = 0.8, .z = 0.8 }, 0.3);
    const material_right = Material.metal(Color{ .x = 0.8, .y = 0.6, .z = 0.2 }, 1.0);

    const spheres = [_]Sphere{
        Sphere.init(Point3{ .x = 0, .y = -100.5, .z = -1 }, 100, material_ground), // Ground
        Sphere.init(Point3{ .x = 0, .y = 0, .z = -1.2 }, 0.5, material_center),     // Center
        Sphere.init(Point3{ .x = -1.0, .y = 0, .z = -1 }, 0.5, material_left),      // Left
        Sphere.init(Point3{ .x = 1.0, .y = 0, .z = -1 }, 0.5, material_right),      // Right
    };
    const world = HittableList.init(&spheres);

    // Image dimensions
    const aspect_ratio: f64 = 16.0 / 9.0;
    const image_width: u32 = 400;
    const image_height: u32 = @max(1, @as(u32, @intFromFloat(@as(f64, @floatFromInt(image_width)) / aspect_ratio)));
    const samples_per_pixel: u32 = 100;
    const max_depth: i32 = 50;

    // Random number generator
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Camera
    const focal_length: f64 = 1.0;
    const viewport_height: f64 = 2.0;
    const viewport_width: f64 = viewport_height * (@as(f64, @floatFromInt(image_width)) / @as(f64, @floatFromInt(image_height)));
    const camera_center = Point3{ .x = 0, .y = 0, .z = 0 };

    // Calculate the vectors across the horizontal and down the vertical viewport edges
    const viewport_u = Vec3{ .x = viewport_width, .y = 0, .z = 0 };
    const viewport_v = Vec3{ .x = 0, .y = -viewport_height, .z = 0 };

    // Calculate the horizontal and vertical delta vectors from pixel to pixel
    const pixel_delta_u = viewport_u.div(@floatFromInt(image_width));
    const pixel_delta_v = viewport_v.div(@floatFromInt(image_height));

    // Calculate the location of the upper left pixel
    const viewport_upper_left = camera_center
        .sub(Vec3{ .x = 0, .y = 0, .z = focal_length })
        .sub(viewport_u.div(2.0))
        .sub(viewport_v.div(2.0));

    const pixel00_loc = viewport_upper_left.add(pixel_delta_u.add(pixel_delta_v).mul(0.5));

    // Create output file
    const file = try std.fs.cwd().createFile("image.ppm", .{});
    defer file.close();

    // Write PPM header
    var header_buf: [256]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf, "P3\n{d} {d}\n255\n", .{ image_width, image_height });
    _ = try file.writeAll(header);

    // Render
    var j: u32 = 0;
    while (j < image_height) : (j += 1) {
        std.debug.print("\rScanlines remaining: {d} ", .{image_height - j});

        var i: u32 = 0;
        while (i < image_width) : (i += 1) {
            var pixel_color = Color{ .x = 0, .y = 0, .z = 0 };

            // Take multiple samples per pixel
            var sample: u32 = 0;
            while (sample < samples_per_pixel) : (sample += 1) {
                const offset_u = randomFloat(rng) - 0.5;
                const offset_v = randomFloat(rng) - 0.5;

                const pixel_sample = pixel00_loc
                    .add(pixel_delta_u.mul(@as(f64, @floatFromInt(i)) + offset_u))
                    .add(pixel_delta_v.mul(@as(f64, @floatFromInt(j)) + offset_v));

                const ray_direction = pixel_sample.sub(camera_center);
                const ray = Ray.init(camera_center, ray_direction);

                pixel_color = pixel_color.add(rayColor(ray, world, max_depth, rng));
            }

            try writeColor(file, pixel_color, samples_per_pixel);
        }
    }

    std.debug.print("\rDone.                 \n", .{});
}
