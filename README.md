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

Alternatively, install Zig 0.16.0 from [ziglang.org](https://ziglang.org/download/).

## Building

```bash
zig build
```

## Running

```bash
zig build run
```

This will generate an `image.ppm` file in the current directory.

## Build Options

All options are compile-time and passed with `-D`:

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `-Dwidth` | integer | `1200` | Image width in pixels |
| `-Dsamples` | integer | `100` | Samples per pixel (antialiasing) |
| `-Dmultithreading` | `true`/`false` | `true` | Tile-based multithreaded rendering |
| `-Dio` | `threaded`, `single_threaded`, `evented` | `threaded` | `std.Io` backend for file I/O and locking (not render parallelism) |

```bash
# Fast preview
zig build run -Doptimize=ReleaseFast -Dwidth=400 -Dsamples=10

# Single-threaded, deterministic output
zig build run -Doptimize=ReleaseFast -Dmultithreading=false

# Select the Io implementation
zig build run -Dio=single_threaded
```

### About `-Dio`

`std.Io` is an interface, so the backing implementation is selectable:

- **`threaded`** (default) — `std.Io.Threaded` with a thread pool. Supports
  `Io.async`/`Io.concurrent`.
- **`single_threaded`** — `std.Io.Threaded.init_single_threaded`. No concurrency
  support, but blocking file I/O and the futex-based `Io.Mutex` still work, so
  it remains correct even with `-Dmultithreading=true`.
- **`evented`** — `std.Io.Evented` (io_uring on Linux, kqueue on BSD, Dispatch
  on Darwin). **Does not compile on Zig 0.16.0**: `std.Io.Uring` lets
  `error.ReadOnlyFileSystem` escape `dirOpenDir` and `dirRealPathFile`, whose
  declared error sets (`Dir.OpenError`, `Dir.RealPathFileError`) do not include
  it. This is an upstream standard library bug, tracked as
  [ziglang/zig#32023](https://codeberg.org/ziglang/zig/issues/32023) (duplicate
  of [#31828](https://codeberg.org/ziglang/zig/issues/31828)) and fixed on
  master by [PR #31764](https://codeberg.org/ziglang/zig/pulls/31764), merged
  after 0.16.0 was tagged. Selecting it on 0.16.0 produces an explicit error
  naming the upstream issue rather than confusing standard library errors.
  Note that Zig's issue tracker lives on Codeberg, not GitHub.

The Io implementation does not affect rendered output. With
`-Dmultithreading=false` (deterministic), `threaded` and `single_threaded`
produce byte-identical images.

### `-Dio` vs `-Dmultithreading`

These are independent knobs, and the names invite confusion:

- **`-Dmultithreading`** controls **render parallelism**. The renderer spawns
  workers directly with `std.Thread.spawn` and feeds them tiles from a shared
  queue.
- **`-Dio`** controls **which `std.Io` backend** services file I/O and
  `Io.Mutex`. It does not start, stop, or size the render thread pool.

In particular, `-Dio=single_threaded` does *not* make rendering
single-threaded. With `-Dmultithreading=true` the renderer still uses every
core; only I/O and locking go through the non-concurrent backend, which stays
correct because `Io.Mutex` reaches the backend solely on the contended path
and the underlying futex helpers are static.

All four combinations are therefore valid. The renderer does not currently use
`Io.async`/`Io.concurrent`; if it were ported to them, the Io implementation
would decide how render tasks execute and `-Dio` would subsume
`-Dmultithreading`.

## Viewing the Output

The raytracer writes `image.ppm` (plain-text PPM / Netpbm `P3`).

Note that many desktop image viewers cannot decode PPM at all. In particular
`imv` — the default handler for `image/x-portable-pixmap` on most Wayland
desktops — has no PNM backend, so it opens a window that decodes nothing and
looks like a completely black image. The PPM file itself is fine; the viewer
just cannot read it.

Options:
- `make png` — convert the render to `image.png` without opening anything
- `make view` — convert to PNG and open it in your default viewer
- `make run view` — render and then view, as two explicit steps
- `make view IMAGE=image-release` — view an already-rendered image (`image`,
  `image-debug` or `image-release`)
- Convert manually: `magick image.ppm image.png` (or `pnmtopng image.ppm > image.png`)
- Use a viewer with PPM support: GIMP, or ImageMagick's `display image.ppm`

Rendering and viewing are deliberately separate: the raytracer only ever writes
a `.ppm`, and `png`/`view` only act on a `.ppm` that already exists. Conversion
is a normal make rule, so re-viewing an unchanged render does not reconvert it.

## Implementation Notes

- Uses full Zig idioms (error handling, tagged unions, comptime)
- Monolithic `main.zig` structure following the book's progression
- Optimized with ReleaseFast for performance

## References

- [Ray Tracing in One Weekend](https://raytracing.github.io/)
- Book by Peter Shirley, Trevor David Black, Steve Hollasch

## License

This implementation follows the public domain approach of the original book series.
