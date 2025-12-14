const std = @import("std");

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

fn randomInUnitDisk(rng: std.Random) Vec3 {
    while (true) {
        const p = Vec3{
            .x = randomFloatRange(rng, -1.0, 1.0),
            .y = randomFloatRange(rng, -1.0, 1.0),
            .z = 0.0,
        };
        if (p.lengthSquared() < 1.0) {
            return p;
        }
    }
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

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f64) Vec3 {
    const cos_theta = @min(uv.neg().dot(n), 1.0);
    const r_out_perp = uv.add(n.mul(cos_theta)).mul(etai_over_etat);
    const r_out_parallel = n.mul(-@sqrt(@abs(1.0 - r_out_perp.lengthSquared())));
    return r_out_perp.add(r_out_parallel);
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
            .albedo = Color{ .x = 1.0, .y = 1.0, .z = 1.0 },
            .fuzz = 0.0,
            .refraction_index = refraction_index,
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
            .dielectric => {
                attenuation.* = Color{ .x = 1.0, .y = 1.0, .z = 1.0 };
                const ri = if (rec.front_face) (1.0 / self.refraction_index) else self.refraction_index;

                const unit_direction = ray_in.direction.unitVector();
                const cos_theta = @min(unit_direction.neg().dot(rec.normal), 1.0);
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
        const w = lookfrom.sub(lookat).unitVector();
        const u = vup.cross(w).unitVector();
        const v = w.cross(u);

        const center = lookfrom;

        // Calculate the vectors across the horizontal and down the vertical viewport edges
        const viewport_u = u.mul(viewport_width);
        const viewport_v = v.neg().mul(viewport_height);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel
        const pixel_delta_u = viewport_u.div(@floatFromInt(image_width));
        const pixel_delta_v = viewport_v.div(@floatFromInt(image_height));

        // Calculate the location of the upper left pixel
        const viewport_upper_left = center
            .sub(w.mul(focus_dist))
            .sub(viewport_u.div(2.0))
            .sub(viewport_v.div(2.0));

        const pixel00_loc = viewport_upper_left.add(pixel_delta_u.add(pixel_delta_v).mul(0.5));

        // Calculate the camera defocus disk basis vectors
        const defocus_radius = focus_dist * @tan(degreesToRadians(defocus_angle / 2.0));
        const defocus_disk_u = u.mul(defocus_radius);
        const defocus_disk_v = v.mul(defocus_radius);

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

        const pixel_sample = self.pixel00_loc
            .add(self.pixel_delta_u.mul(@as(f64, @floatFromInt(i)) + offset_u))
            .add(self.pixel_delta_v.mul(@as(f64, @floatFromInt(j)) + offset_v));

        const ray_origin = if (self.defocus_angle <= 0) self.center else self.defocusDiskSample(rng);
        const ray_direction = pixel_sample.sub(ray_origin);
        return Ray.init(ray_origin, ray_direction);
    }

    fn defocusDiskSample(self: Camera, rng: std.Random) Point3 {
        const p = randomInUnitDisk(rng);
        return self.center.add(self.defocus_disk_u.mul(p.x)).add(self.defocus_disk_v.mul(p.y));
    }
};

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
        @memset(pixels, Color{ .x = 0, .y = 0, .z = 0 });
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
    world: *const HittableList,
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
                var pixel_color = Color{ .x = 0, .y = 0, .z = 0 };

                // Multiple samples per pixel
                var sample: u32 = 0;
                while (sample < ctx.samples_per_pixel) : (sample += 1) {
                    const ray = ctx.camera.getRay(x, y, rng);
                    pixel_color = pixel_color.add(rayColor(ray, ctx.world.*, ctx.max_depth, rng));
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
    var sphere_list = try std.ArrayList(Sphere).initCapacity(allocator, 500);
    defer sphere_list.deinit(allocator);

    // Ground
    try sphere_list.append(allocator, Sphere.init(
        Point3{ .x = 0, .y = -1000, .z = 0 },
        1000,
        Material.lambertian(Color{ .x = 0.5, .y = 0.5, .z = 0.5 }),
    ));

    // Random small spheres
    var a: i32 = -11;
    while (a < 11) : (a += 1) {
        var b: i32 = -11;
        while (b < 11) : (b += 1) {
            const choose_mat = randomFloat(scene_rng);
            const center = Point3{
                .x = @as(f64, @floatFromInt(a)) + 0.9 * randomFloat(scene_rng),
                .y = 0.2,
                .z = @as(f64, @floatFromInt(b)) + 0.9 * randomFloat(scene_rng),
            };

            if (center.sub(Point3{ .x = 4, .y = 0.2, .z = 0 }).length() > 0.9) {
                if (choose_mat < 0.8) {
                    // Diffuse
                    const albedo = Color{
                        .x = randomFloat(scene_rng) * randomFloat(scene_rng),
                        .y = randomFloat(scene_rng) * randomFloat(scene_rng),
                        .z = randomFloat(scene_rng) * randomFloat(scene_rng),
                    };
                    try sphere_list.append(allocator, Sphere.init(center, 0.2, Material.lambertian(albedo)));
                } else if (choose_mat < 0.95) {
                    // Metal
                    const albedo = Color{
                        .x = randomFloatRange(scene_rng, 0.5, 1.0),
                        .y = randomFloatRange(scene_rng, 0.5, 1.0),
                        .z = randomFloatRange(scene_rng, 0.5, 1.0),
                    };
                    const fuzz = randomFloatRange(scene_rng, 0.0, 0.5);
                    try sphere_list.append(allocator, Sphere.init(center, 0.2, Material.metal(albedo, fuzz)));
                } else {
                    // Glass
                    try sphere_list.append(allocator, Sphere.init(center, 0.2, Material.dielectric(1.5)));
                }
            }
        }
    }

    // Three large spheres
    try sphere_list.append(allocator, Sphere.init(Point3{ .x = 0, .y = 1, .z = 0 }, 1.0, Material.dielectric(1.5)));
    try sphere_list.append(allocator, Sphere.init(Point3{ .x = -4, .y = 1, .z = 0 }, 1.0, Material.lambertian(Color{ .x = 0.4, .y = 0.2, .z = 0.1 })));
    try sphere_list.append(allocator, Sphere.init(Point3{ .x = 4, .y = 1, .z = 0 }, 1.0, Material.metal(Color{ .x = 0.7, .y = 0.6, .z = 0.5 }, 0.0)));

    const world = HittableList.init(sphere_list.items);

    // Camera setup - use lower resolution/samples for reasonable render time
    // Final scene settings: width=1200, samples=500 (takes hours to render)
    const camera = Camera.init(
        Point3{ .x = 13, .y = 2, .z = 3 }, // lookfrom
        Point3{ .x = 0, .y = 0, .z = 0 }, // lookat
        Vec3{ .x = 0, .y = 1, .z = 0 }, // vup
        20.0, // vfov
        16.0 / 9.0, // aspect ratio
        1200, // image width (use 1200 for final)
        0.6, // defocus angle
        10.0, // focus distance
    );

    const samples_per_pixel: u32 = 100; // Use 500 for final quality
    const max_depth: i32 = 50;

    // Configuration
    const use_multithreading = true; // Toggle between single/multi-threaded
    const tile_size: u32 = 32; // 32x32 pixel tiles (only used if multithreading)

    // Create output file
    const file = try std.fs.cwd().createFile("image.ppm", .{});
    defer file.close();

    // Write PPM header
    var header_buf: [256]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf, "P3\n{d} {d}\n255\n", .{ camera.image_width, camera.image_height });
    _ = try file.writeAll(header);

    if (use_multithreading) {
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
                var pixel_color = Color{ .x = 0, .y = 0, .z = 0 };

                // Take multiple samples per pixel
                var sample: u32 = 0;
                while (sample < samples_per_pixel) : (sample += 1) {
                    const ray = camera.getRay(i, j, rng);
                    pixel_color = pixel_color.add(rayColor(ray, world, max_depth, rng));
                }

                try writeColor(file, pixel_color, samples_per_pixel);
            }
        }
    }

    std.debug.print("\rDone.                 \n", .{});
}
