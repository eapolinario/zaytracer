# Zaytracer

A raytracer implementation in Zig, following the "Ray Tracing in One Weekend" book series by Peter Shirley.

## About

This project implements a complete raytracer from scratch, building up features chapter by chapter:
- PPM image output
- Vector mathematics (Vec3)
- Ray-sphere intersection
- Surface normals and materials (Lambertian, metal, dielectric)
- Antialiasing with multisampling
- Positionable camera with depth of field
- Final scene with hundreds of spheres

## Development Environment

### Using Nix Flakes (Recommended)

If you have Nix with flakes enabled:

```bash
nix develop
```

This will provide a shell with Zig and ZLS (Zig Language Server) available.

### Manual Installation

Alternatively, install Zig 0.13.0 or later from [ziglang.org](https://ziglang.org/download/).

## Building

```bash
zig build
```

## Running

```bash
zig build run
```

This will generate an `image.ppm` file in the current directory.

## Viewing the Output

The raytracer outputs PPM format images. You can view them with:
- GIMP
- ImageMagick: `display image.ppm`
- Convert to PNG: `convert image.ppm image.png`

## Implementation Notes

- Uses full Zig idioms (error handling, tagged unions, comptime)
- Monolithic `main.zig` structure following the book's progression
- Optimized with ReleaseFast for performance

## References

- [Ray Tracing in One Weekend](https://raytracing.github.io/)
- Book by Peter Shirley, Trevor David Black, Steve Hollasch

## License

This implementation follows the public domain approach of the original book series.
