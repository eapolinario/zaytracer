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
    const image_width: u32 = 256;
    const image_height: u32 = 256;

    // Create output file
    const file = try std.fs.cwd().createFile("image.ppm", .{});
    defer file.close();

    const writer = file.writer();

    // Write PPM header
    try writer.print("P3\n{d} {d}\n255\n", .{ image_width, image_height });

    // Write pixel data
    var j: u32 = 0;
    while (j < image_height) : (j += 1) {
        // Progress indicator
        std.debug.print("\rScanlines remaining: {d} ", .{image_height - j});

        var i: u32 = 0;
        while (i < image_width) : (i += 1) {
            // Calculate RGB values (0.0 to 1.0)
            const r = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(image_width - 1));
            const g = @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(image_height - 1));
            const b = 0.0;

            const pixel_color = Color{ .x = r, .y = g, .z = b };
            try writeColor(writer, pixel_color);
        }
    }

    std.debug.print("\rDone.                 \n", .{});
}
