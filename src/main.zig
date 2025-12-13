const std = @import("std");

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

            // Convert to 0-255 range
            const ir = @as(u8, @intFromFloat(255.999 * r));
            const ig = @as(u8, @intFromFloat(255.999 * g));
            const ib = @as(u8, @intFromFloat(255.999 * b));

            try writer.print("{d} {d} {d}\n", .{ ir, ig, ib });
        }
    }

    std.debug.print("\rDone.                 \n", .{});
}
