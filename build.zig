const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build option for multithreading (default: true)
    const multithreading = b.option(bool, "multithreading", "Enable multithreaded rendering (default: true)") orelse true;

    // Build options for image quality (for preview vs final renders)
    const image_width = b.option(u32, "width", "Image width in pixels (default: 1200)") orelse 1200;
    const samples_per_pixel = b.option(u32, "samples", "Samples per pixel for antialiasing (default: 100)") orelse 100;

    const options = b.addOptions();
    options.addOption(bool, "use_multithreading", multithreading);
    options.addOption(u32, "image_width", image_width);
    options.addOption(u32, "samples_per_pixel", samples_per_pixel);

    const exe_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe_module.addImport("build_options", options.createModule());

    const exe = b.addExecutable(.{
        .name = "zaytracer",
        .root_module = exe_module,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the raytracer");
    run_step.dependOn(&run_cmd.step);

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_module.addImport("build_options", options.createModule());

    const exe_unit_tests = b.addTest(.{
        .name = "zaytracer-test",
        .root_module = test_module,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
