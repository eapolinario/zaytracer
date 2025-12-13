const std = @import("std");

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
// Ray Color (Background)
// ============================================================================

fn rayColor(ray: Ray) Color {
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

fn writeColor(writer: anytype, color: Color) !void {
    // Convert to 0-255 range
    const ir = @as(u8, @intFromFloat(255.999 * color.x));
    const ig = @as(u8, @intFromFloat(255.999 * color.y));
    const ib = @as(u8, @intFromFloat(255.999 * color.z));

    try writer.print("{d} {d} {d}\n", .{ ir, ig, ib });
}

// ============================================================================
// Main
// ============================================================================

pub fn main() !void {
    // Image dimensions
    const aspect_ratio: f64 = 16.0 / 9.0;
    const image_width: u32 = 400;
    const image_height: u32 = @max(1, @as(u32, @intFromFloat(@as(f64, @floatFromInt(image_width)) / aspect_ratio)));

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

    const writer = file.writer();

    // Write PPM header
    try writer.print("P3\n{d} {d}\n255\n", .{ image_width, image_height });

    // Render
    var j: u32 = 0;
    while (j < image_height) : (j += 1) {
        std.debug.print("\rScanlines remaining: {d} ", .{image_height - j});

        var i: u32 = 0;
        while (i < image_width) : (i += 1) {
            const pixel_center = pixel00_loc
                .add(pixel_delta_u.mul(@floatFromInt(i)))
                .add(pixel_delta_v.mul(@floatFromInt(j)));

            const ray_direction = pixel_center.sub(camera_center);
            const ray = Ray.init(camera_center, ray_direction);

            const pixel_color = rayColor(ray);
            try writeColor(writer, pixel_color);
        }
    }

    std.debug.print("\rDone.                 \n", .{});
}
