# Ray Tracing in One Weekend - Zig Implementation Plan

## Project Overview
Implement a complete raytracer in Zig following "Ray Tracing in One Weekend" by Peter Shirley, using full Zig idioms while maintaining the book's chapter-by-chapter progression in a monolithic `main.zig` file.

## Implementation Strategy
- **Language**: Zig (using modern Zig idioms)
- **Structure**: Monolithic `main.zig` that grows with each chapter
- **Version Control**: Git commit after each completed chapter
- **Style**: Full Zig idioms (comptime, tagged unions, error handling, slices)

## Phase 0: Project Setup ✅

### Files to Create:
- `build.zig` - Zig build system configuration
- `build.zig.zon` - Package manifest
- `.gitignore` - Ignore zig-cache/, zig-out/, and output images
- `README.md` - Project documentation
- `src/main.zig` - Main raytracer implementation (initially empty)

### Build Configuration:
- Target: native
- Optimize: ReleaseFast (for performance)
- Standard library: use @import("std")

**Git Checkpoint**: Commit "Initial project setup" ✅

---

## Chapter 1: Output an Image ✅

### Goal: Generate a PPM format image file

### Implementation Details:
1. Create PPM image writer using Zig's `std.io.Writer`
2. Use nested loops to generate pixel data
3. Implement proper error handling for file I/O
4. Use `std.ArrayList(u8)` for buffering output

### Key Zig Idioms:
- Use `try` for error propagation instead of C++ exceptions
- Use `defer` for resource cleanup
- Use `std.fs.cwd().createFile()` for file creation
- Use `std.fmt.format()` or `writer.print()` for formatted output

### Output:
- File: `image.ppm` (256x256 pixels)
- Format: P3 (ASCII PPM)
- Content: Simple gradient (red-to-white horizontally, green vertically)

**Git Checkpoint**: Commit "Chapter 1: Output an image" ✅

---

## Chapter 2: The Vec3 Class ✅

### Goal: Implement a 3D vector type for geometry and colors

### Implementation Details:
1. Create `Vec3` struct with x, y, z fields (using `f64`)
2. Implement vector operations as methods
3. Create type aliases: `Point3` and `Color` for semantic clarity
4. Use operator overloading where Zig allows (through functions)

### Key Zig Idioms:
- Use `const Vec3 = struct { ... }` for type definition
- Use methods with `self` parameter
- Create utility functions for operations (add, sub, mul, div, dot, cross)
- Use `inline` for performance-critical small functions
- Consider using `@Vector(3, f64)` for SIMD optimization (optional)

### Vec3 Operations to Implement:
- Addition: `add(self: Vec3, other: Vec3) Vec3`
- Subtraction: `sub(self: Vec3, other: Vec3) Vec3`
- Scalar multiplication: `mul(self: Vec3, scalar: f64) Vec3`
- Element-wise multiplication: `mulVec(self: Vec3, other: Vec3) Vec3`
- Division: `div(self: Vec3, scalar: f64) Vec3`
- Dot product: `dot(self: Vec3, other: Vec3) f64`
- Cross product: `cross(self: Vec3, other: Vec3) Vec3`
- Length: `length(self: Vec3) f64`
- Length squared: `lengthSquared(self: Vec3) f64`
- Unit vector: `unitVector(self: Vec3) Vec3`
- Negation: `neg(self: Vec3) Vec3`

### Color Utility Functions:
- `writeColor(writer: anytype, color: Color) !void` - Write color to output
- Handle color scaling and gamma correction (added later)

**Git Checkpoint**: Commit "Chapter 2: Vec3 class implementation" ✅

---

## Chapter 3: Rays, Camera, and Background ✅

### Goal: Create ray structure and render a gradient background

### Implementation Details:
1. Implement `Ray` struct with origin and direction
2. Create `ray.at(t)` function to get point along ray
3. Implement simple camera model (fixed position, looking down -Z axis)
4. Create `rayColor()` function for background gradient

### Key Zig Idioms:
- Use struct for Ray: `{ origin: Point3, direction: Vec3 }`
- Use pure functions where possible
- Avoid mutable state

### Camera Setup:
- Aspect ratio: 16:9
- Image width: 400 pixels
- Viewport height: 2.0
- Focal length: 1.0

### Background Color:
- Linear interpolation from white (top) to blue (bottom)
- Based on ray's Y coordinate

**Git Checkpoint**: Commit "Chapter 3: Rays and camera basics" ✅

---

## Chapter 4: Adding a Sphere

### Goal: Implement ray-sphere intersection and render first 3D object

### Implementation Details:
1. Implement sphere intersection test using discriminant
2. Create `hitSphere()` function
3. Render sphere at center (0, 0, -1) with radius 0.5
4. Color sphere based on hit (red) or miss (background)

### Key Zig Idioms:
- Use optional types `?f64` to represent hit/miss
- Use pattern matching for handling optionals
- Keep mathematical clarity in intersection code

### Sphere Intersection Math:
- Quadratic equation: `t² b·b + 2t b·(A-C) + (A-C)·(A-C) - r² = 0`
- Discriminant: `b² - 4ac`
- Return closest t value if discriminant ≥ 0

### Integration:
- Update `rayColor()` to check sphere intersection first
- If hit: return red color
- If miss: return gradient background

**Git Checkpoint**: Commit "Chapter 4: Sphere intersection"

---

## Chapter 5: Surface Normals and Multiple Objects

### Goal: Implement proper surface normals and create hittable object abstraction

### Implementation Details:
1. Calculate surface normals for sphere
2. Create `HitRecord` struct to store hit information
3. Create `Hittable` interface using Zig's patterns
4. Implement `HittableList` for multiple objects
5. Handle front-face vs back-face determination

### Key Zig Idioms:
- Use tagged union or interface pattern for polymorphism
- Consider using `std.ArrayList(Hittable)` for object list
- Use structs with function pointers or tagged unions for dispatch
- Implement `Interval` type for t_min/t_max ranges

### HitRecord Structure:
```zig
const HitRecord = struct {
    point: Point3,
    normal: Vec3,
    t: f64,
    front_face: bool,

    pub fn setFaceNormal(self: *HitRecord, ray: Ray, outward_normal: Vec3) void {
        self.front_face = ray.direction.dot(outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else outward_normal.neg();
    }
};
```

### Hittable Interface:
- Use tagged union or vtable pattern
- `hit(ray: Ray, t_min: f64, t_max: f64, rec: *HitRecord) bool`

### Interval Type:
```zig
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
```

### Color Update:
- Visualize normals as RGB colors: `(normal + 1) * 0.5`

**Git Checkpoint**: Commit "Chapter 5: Surface normals and hittable abstraction"

---

## Chapter 6: Antialiasing

### Goal: Implement multisampling antialiasing for smoother images

### Implementation Details:
1. Add random number generation using Zig's std.rand
2. Create camera class to manage rays
3. Implement samples-per-pixel parameter
4. Add random ray perturbation for each sample
5. Average sample colors

### Key Zig Idioms:
- Use `std.rand.DefaultPrng` for RNG
- Use `std.rand.Random` interface
- Pass RNG as parameter (don't use global state)
- Use `comptime` for constants where appropriate

### Random Number Utilities:
```zig
// Random float in [0, 1)
fn randomFloat(rng: std.rand.Random) f64 {
    return rng.float(f64);
}

// Random float in [min, max)
fn randomFloatRange(rng: std.rand.Random, min: f64, max: f64) f64 {
    return min + (max - min) * randomFloat(rng);
}
```

### Parameters:
- Samples per pixel: 100
- Image width: 400
- Aspect ratio: 16:9

**Git Checkpoint**: Commit "Chapter 6: Antialiasing with multisampling"

---

## Chapter 7: Diffuse Materials

### Goal: Implement realistic diffuse (matte) material

### Implementation Details:
1. Implement recursive ray bouncing
2. Add maximum recursion depth
3. Create random unit vector generation
4. Implement Lambertian reflection

### Key Zig Idioms:
- Use recursion with explicit depth parameter
- Use `inline` for potential tail-call optimization
- Handle depth limit explicitly

### Random Vector Functions:
```zig
fn randomVec3(rng: std.rand.Random) Vec3 {
    return Vec3{
        .x = randomFloat(rng),
        .y = randomFloat(rng),
        .z = randomFloat(rng),
    };
}

fn randomVec3Range(rng: std.rand.Random, min: f64, max: f64) Vec3 {
    return Vec3{
        .x = randomFloatRange(rng, min, max),
        .y = randomFloatRange(rng, min, max),
        .z = randomFloatRange(rng, min, max),
    };
}

fn randomInUnitSphere(rng: std.rand.Random) Vec3 {
    while (true) {
        const p = randomVec3Range(rng, -1.0, 1.0);
        if (p.lengthSquared() < 1.0) return p;
    }
}

fn randomUnitVector(rng: std.rand.Random) Vec3 {
    return randomInUnitSphere(rng).unitVector();
}
```

### Parameters:
- Max depth: 50

### Shadow Acne Fix:
- Use t_min = 0.001 instead of 0.0 to avoid self-intersection

**Git Checkpoint**: Commit "Chapter 7: Diffuse materials"

---

## Chapter 8: Metal Materials

### Goal: Add metal (reflective) material type

### Implementation Details:
1. Create `Material` abstraction using tagged union
2. Implement Lambertian and Metal materials
3. Add material to HitRecord
4. Implement reflection calculation
5. Add fuzziness parameter for imperfect reflections

### Key Zig Idioms:
- Use tagged union for Material polymorphism
- Use switch statements for material dispatch
- Store material index/reference in HitRecord

### Reflection Function:
```zig
fn reflect(v: Vec3, n: Vec3) Vec3 {
    return v.sub(n.mul(2.0 * v.dot(n)));
}
```

**Git Checkpoint**: Commit "Chapter 8: Metal materials"

---

## Chapter 9: Dielectrics (Glass)

### Goal: Add transparent dielectric material (glass, water, diamond)

### Implementation Details:
1. Add Dielectric material type
2. Implement Snell's law for refraction
3. Handle total internal reflection
4. Implement Schlick approximation for reflectance
5. Use random choice between reflection and refraction

### Key Zig Idioms:
- Add new variant to Material tagged union
- Use precise floating-point math
- Handle edge cases (total internal reflection)

### Refraction Function:
```zig
fn refract(uv: Vec3, n: Vec3, etai_over_etat: f64) Vec3 {
    const cos_theta = @min(uv.neg().dot(n), 1.0);
    const r_out_perp = uv.add(n.mul(cos_theta)).mul(etai_over_etat);
    const r_out_parallel = n.mul(-@sqrt(@abs(1.0 - r_out_perp.lengthSquared())));
    return r_out_perp.add(r_out_parallel);
}
```

### Schlick Approximation:
```zig
fn reflectance(cosine: f64, refraction_index: f64) f64 {
    var r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * std.math.pow(f64, 1.0 - cosine, 5);
}
```

**Git Checkpoint**: Commit "Chapter 9: Dielectric materials (glass)"

---

## Chapter 10: Positionable Camera

### Goal: Add camera positioning and orientation control

### Implementation Details:
1. Add camera position (look-from point)
2. Add look-at point
3. Add up vector
4. Implement field-of-view control
5. Calculate camera basis vectors (u, v, w)

### Key Zig Idioms:
- Use pure initialization functions
- Avoid mutable camera state where possible
- Use clear parameter names

### Utility Functions:
```zig
fn degreesToRadians(degrees: f64) f64 {
    return degrees * std.math.pi / 180.0;
}
```

**Git Checkpoint**: Commit "Chapter 10: Positionable camera"

---

## Chapter 11: Defocus Blur (Depth of Field)

### Goal: Add depth-of-field effect by simulating camera lens

### Implementation Details:
1. Add defocus_angle (aperture size) parameter
2. Add focus_distance parameter
3. Implement defocus disk for ray origin randomization
4. Generate rays from random points on defocus disk

### Random Disk Sampling:
```zig
fn randomInUnitDisk(rng: std.rand.Random) Vec3 {
    while (true) {
        const p = Vec3{
            .x = randomFloatRange(rng, -1.0, 1.0),
            .y = randomFloatRange(rng, -1.0, 1.0),
            .z = 0.0,
        };
        if (p.lengthSquared() < 1.0) return p;
    }
}
```

**Git Checkpoint**: Commit "Chapter 11: Defocus blur (depth of field)"

---

## Chapter 12: Final Scene

### Goal: Create the book's final cover scene with many spheres

### Implementation Details:
1. Create random scene generation function
2. Add ground sphere (large radius)
3. Add three large spheres (diffuse, metal, glass)
4. Add grid of random small spheres
5. Position camera for dramatic view

### Final Camera Settings:
- Image width: 1200
- Aspect ratio: 16:9
- Samples per pixel: 500
- Max depth: 50
- vfov: 20 degrees
- lookfrom: (13, 2, 3)
- lookat: (0, 0, 0)
- vup: (0, 1, 0)
- defocus_angle: 0.6
- focus_dist: 10.0

**Git Checkpoint**: Commit "Chapter 12: Final scene (book cover)"

---

## Additional Enhancements (Optional)

### Performance Optimizations:
1. Add multithreading using `std.Thread`
2. Use `@Vector(3, f64)` for SIMD Vec3 operations
3. Add BVH (Bounding Volume Hierarchy) for faster intersection tests
4. Implement row-based parallelization

### Quality Improvements:
1. Add gamma correction (currently missing)
2. Implement better random number distribution
3. Add progress bar using stderr
4. Support different output formats (PNG via external library)

### Code Quality:
1. Add comprehensive unit tests
2. Add benchmark suite
3. Add documentation comments
4. Consider splitting into modules (if monolithic becomes unwieldy)

---

## Progress Tracking

- [x] Phase 0: Project Setup
- [x] Chapter 1: Output an Image
- [x] Chapter 2: Vec3 Class
- [x] Chapter 3: Rays, Camera, and Background
- [x] Chapter 4: Adding a Sphere
- [x] Chapter 5: Surface Normals and Multiple Objects
- [x] Chapter 6: Antialiasing
- [x] Chapter 7: Diffuse Materials
- [x] Chapter 8: Metal Materials
- [x] Chapter 9: Dielectrics (Glass)
- [x] Chapter 10: Positionable Camera
- [x] Chapter 11: Defocus Blur (Depth of Field)
- [x] Chapter 12: Final Scene
