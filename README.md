# Zaytracer

A raytracer implementation in Zig, following the "Ray Tracing in One Weekend" book series by Peter Shirley.

## About

This project implements a complete raytracer from scratch, building up features chapter by chapter:
- PPM image output
- Vector mathematics (Vec3)
- Ray-sphere intersection
- Surface normals and materials (Lambertian, metal, dielectric, emissive)
- Antialiasing with multisampling
- Positionable camera with depth of field
- Triangle meshes loaded from OBJ, accelerated with a BVH
- Area lights sampled directly, and caustics from a photon map
- Several scenes, selected at run time

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

## Scenes

The scene is chosen at run time, so switching between them costs no rebuild:

```bash
zig build run -- --scene=cornell-box
./zig-out/bin/zaytracer --list-scenes
make preview SCENE=cornell-box
```

| Scene | Needs | Description |
| --- | --- | --- |
| `cover` (default) | committed models | The book's cover: random spheres, a diamond teapot and a cube, under a sky |
| `cornell-box` | nothing | The closed Cornell box, lit only by a panel set into its ceiling, with a glass and a metal sphere |
| `glass-dragon` | `make models` | The XYZ RGB Asian Dragon in glass, 249,882 triangles, lit by a panel against a dark sky |
| `glass-bunny` | `make models` | The Stanford bunny in glass, 69,451 triangles: the same studio, quick enough to iterate on |
| `spot` | `make models` | Spot the cow under a daylit sky, with a glass sphere for company |

Without `--scene` the cover scene is rendered, which is also what CI renders,
so the default must never depend on a fetched model.

`glass-dragon` and `glass-bunny` are the same studio — one model on a floor
under a panel, against a near-black sky — pointed at different models. The
camera and the lamp are placed from the model's own fitted bounds, so aiming it
at something else needs no new constants. Reach for the bunny when changing
anything about caustics: it builds its BVH in 0.76s against the dragon's 3.6s
and still throws a proper caustic.

Two things a scene decides for itself, beyond geometry:

- **Its background.** `cornell-box` is only closed because its background is
  black; with the sky gradient, daylight would pour in through the open wall
  the camera looks through.
- **Its photon budget.** How much of the emitted light comes back as stored
  caustic photons varies by nearly a factor of five, so one global budget does
  not fit: about 41% for the closed Cornell box, 32% for the bunny and for
  Spot's single glass sphere, and roughly 7% for the open cover scene. Setting
  it too low is not silent — emission stops early and the renderer says the
  caustics are missing the light it never emitted.

## Models

`models/` holds two small models outright: `test_cube.obj` and `teapot.obj`.
Anything larger is fetched rather than committed:

```bash
make models          # fetch what is missing, verifying each sha256
make models-verify   # re-check what is already there
```

`models/manifest.tsv` lists a filename, a URL and the sha256 of the file's
bytes. Downloads land in a `.part` file and are only moved into place once the
checksum matches, so a failed or corrupted fetch never leaves behind something
that looks like a model. Sources are pinned to a commit, not a branch.

| Model | Triangles | Used by |
| --- | --- | --- |
| `test_cube.obj`, `teapot.obj` | 12, 6,320 | `cover` (committed) |
| `xyzrgb_dragon.obj` | 249,882 | `glass-dragon` |
| `stanford-bunny.obj` | 69,451 | `glass-bunny` |
| `spot.obj` | 5,856 | `spot` |

A test checks that every model a scene asks for is listed in the manifest, so a
scene pointing at something nobody can fetch fails the build rather than the
render. The manifest is embedded at compile time for this, which is why
`build.zig` hands it to the test module.

Note that this sha256 is of the file itself, which is **not** the SHA the
GitHub API reports for a blob — that one is `sha1("blob <len>\0" + content)`.
Comparing against the wrong one fails every time.

Fetched models are gitignored. The Stanford-derived ones (the dragon and the
bunny) are for research and non-commercial use with attribution to the
[Stanford Computer Graphics Laboratory](https://graphics.stanford.edu/data/3Dscanrep/),
which is a good reason to link to them rather than redistribute them. They are
served from
[alecjacobson/common-3d-test-models](https://github.com/alecjacobson/common-3d-test-models),
which carries the classic test models already converted to OBJ.

## Build Options

Image size, sample count, multithreading and the `std.Io` backend are
compile-time and passed with `-D`. The scene is not: it is an argument, so
`--scene` needs no rebuild.

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

# A different scene, at preview quality
zig build run -Doptimize=ReleaseFast -Dwidth=400 -Dsamples=10 -- --scene=glass-dragon
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

### Lighting

There are two ways light reaches the camera, and they must not overlap:

- **Direct sampling.** At every diffuse hit, each area light is sampled and a
  shadow ray is fired at it. Waiting for a bounce to stumble onto a panel that
  covers 9% of a ceiling is almost pure noise. Point lights are not sampled
  this way: they exist only to drive the caustic map, and always have.
- **Emissive geometry.** A `diffuse_light` material emits and scatters nothing.
  It emits from one face only — the one its geometry's normal points out of —
  because a lamp hung a few centimetres below a ceiling would otherwise light
  the gap above itself across almost no distance, and direct sampling divides
  by the square of that distance.

Paths end by Russian roulette rather than by running out of depth. After four
bounces a path survives with a probability equal to how much of its energy it
still carries, and survivors are divided by that probability, which leaves the
average untouched while cutting the work. It matters most where paths never
escape: the closed Cornell box renders in half the time, and a converged render
of it agrees with the old one to within 0.2% in every region, which is the
difference between an unbiased shortcut and simply rendering less.

The depth limit remains as a ceiling, since a ray inside glass can keep
reflecting internally, and roulette caps the survival probability below 1 so
that even a path losing nothing has a way out.

Emission is counted only while a path has been specular the whole way from the
camera: looking at the lamp, at its reflection, or at it through glass. Once a
path touches a diffuse surface, direct sampling has covered that light and the
caustic map covers it arriving through anything specular further on, so
counting it again would be both too bright and violently noisy — finding a
small lamp by chance through a specular bounce is exactly the rare,
high-energy sample that shows up as a white speck. Without a photon map
nothing covers specular paths, so there emission is counted and the renderer
degrades to plain path tracing.

### Caustics

Caustics come from a photon map built before rendering, on top of the path
tracer rather than inside it:

- Photons are emitted from a single light and stored where they land on a
  diffuse surface *after* at least one specular bounce (an `LS+D` path). A
  photon that reaches a diffuse surface directly is direct light rather than a
  caustic, and is dropped: storing it made this a global photon map whose
  energy was added on top of the path traced result. A photon that strikes a
  lamp is absorbed, since the emission already accounts for it.
- Emission is aimed with a projection map — a coarse grid of directions around
  the light that probes which ones reach specular geometry. Photon power is
  scaled by the fraction of the sphere those directions cover, so aiming
  redistributes the light's power without adding energy.
- Area lights emit their photons from a single point at their centre. This is
  an approximation an area light does not really deserve, but it keeps the
  projection map and the emission pass unchanged.
- Gathering is a fixed-radius density estimate scaled by the Lambertian BRDF
  (`albedo / pi`), so a caustic takes on the colour of the surface it lands on.

The grid the photons are looked up in is sized from the scene rather than
hardcoded. Using the scene's own bounding box is not enough: the cover scene's
ground is a sphere of radius 1000, so that box spans 2000 units and each of the
128 cells per axis would be about 15 across, against a gather radius of 0.2.
Since photons are only stored after a specular bounce, the box is anchored on
the specular geometry, grown to at least 32 units to catch the splash around
it, and clipped to the scene — unless the whole scene already fits, in which
case it is covered entirely, because a closed room throws caustics onto its
walls too. Photons landing outside the box are reported rather than silently
dropped.

The tuning constants (`photons_emitted`, `caustic_gather_radius`,
`photon_grid_size`, `min_photon_extent`, ...) sit together above `PhotonMap` in
`src/main.zig`, and a scene can override the budget. The photon pass is
deterministic; multithreaded rendering is not, so use `-Dmultithreading=false`
when comparing two renders.

### Cost

Stage timings at 500x281 and 80 samples on 16 threads, which is what the
`Scene built` / `BVH built` / `caustic pass` / `Rendered in` lines report:

| Scene | Primitives | Scene build | BVH build | Caustics | Render |
| --- | --- | --- | --- | --- | --- |
| `cover` | 6,817 | 3.7ms | 21.0ms | 705.9ms | 2.9s |
| `spot` | 5,859 | 3.9ms | 29.6ms | 756.0ms | 6.8s |
| `cornell-box` | 20 | 42.9us | 21.0us | 447.0ms | 1m14s |
| `glass-bunny` | 69,455 | 33.1ms | 771.9ms | 9.3s | 35.7s |
| `glass-dragon` | 249,886 | 175.9ms | 3.6s | 12.6s | 10.2s |

Parsing 11.3 MB of OBJ and generating smooth normals for a quarter of a million
triangles costs 176ms, which is nothing. The BVH build is the one to watch:
3.6s, single-threaded, and the reason the dragon takes longer to set up than to
render. The closed Cornell box is still the slowest render off just 20
primitives, because no path escapes it, though Russian roulette has roughly
halved what that costs.

Triangle count is not what makes a render slow, though. The bunny has a quarter
of the dragon's triangles and takes three times as long, because what costs is
how much glass a ray has to fight its way through: the bunny is a solid lump of
it, where the dragon is mostly thin limbs with background between them. Rays
that miss are cheap.

## References

- [Ray Tracing in One Weekend](https://raytracing.github.io/)
- Book by Peter Shirley, Trevor David Black, Steve Hollasch

## License

This implementation follows the public domain approach of the original book series.
