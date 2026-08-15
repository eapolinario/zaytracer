.PHONY: help build build-debug build-release build-preview \
        run run-debug run-release run-both preview scenes \
        bench bench-debug bench-release bench-both clean test png view \
        models models-verify

# Default target
.DEFAULT_GOAL := help

# Never leave a half-written file behind when a recipe fails
.DELETE_ON_ERROR:

# Multithreading flag (can be overridden: make build MULTITHREAD=false)
MULTITHREAD ?= true

# std.Io implementation: threaded, single_threaded, evented
IO ?= threaded

# Preview quality settings (fast iteration)
PREVIEW_WIDTH ?= 400
PREVIEW_SAMPLES ?= 10

# Which render the png/view targets act on: image, image-debug or image-release
IMAGE ?= image

# Scene to render. Empty means the binary's default (the book cover scene).
# List them with: make scenes
SCENE ?=
SCENE_ARG = $(if $(SCENE),--scene=$(SCENE),)

# Models that are fetched rather than committed
MODELS_DIR := models
MANIFEST := $(MODELS_DIR)/manifest.tsv
SHA256 := $(shell command -v sha256sum >/dev/null 2>&1 && echo sha256sum || echo "shasum -a 256")

# Help message
help:
	@echo "Zaytracer - Available targets:"
	@echo ""
	@echo "Build targets:"
	@echo "  make build              - Build in default mode (Debug, multithreaded)"
	@echo "  make build-debug        - Build in Debug mode (with safety checks)"
	@echo "  make build-release      - Build in ReleaseFast mode (optimized)"
	@echo "  make build-preview      - Build for fast preview (400x225, 10 samples)"
	@echo ""
	@echo "Run targets:"
	@echo "  make run                - Build and run in default mode"
	@echo "  make run-debug          - Build and run in Debug mode"
	@echo "  make run-release        - Build and run in ReleaseFast mode"
	@echo "  make run-both           - Run both modes and save separate images"
	@echo "  make preview            - FAST preview render (400x225, 10 samples) ⚡"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  make bench              - Compare Debug vs ReleaseFast performance"
	@echo "  make bench-debug        - Benchmark Debug build with timing"
	@echo "  make bench-release      - Benchmark ReleaseFast build with timing"
	@echo "  make bench-both         - Benchmark both and save separate images"
	@echo ""
	@echo "Utility targets:"
	@echo "  make clean              - Clean build artifacts and images"
	@echo "  make test               - Run unit tests"
	@echo "  make scenes             - List the available scenes"
	@echo "  make models             - Fetch the models in models/manifest.tsv (sha256 verified)"
	@echo "  make models-verify      - Re-check fetched models against the manifest"
	@echo "  make png                - Convert an existing render to PNG"
	@echo "  make view               - Convert to PNG and open in the image viewer"
	@echo "  make help               - Show this help message"
	@echo ""
	@echo "Build modes:"
	@echo "  Debug       - Safety checks enabled, no optimizations (slower, safer)"
	@echo "  ReleaseFast - Full optimizations, no safety checks (faster, production)"
	@echo ""
	@echo "Build options:"
	@echo "  MULTITHREAD=true/false  - Enable/disable multithreading (default: true)"
	@echo "  IO=threaded|single_threaded|evented - std.Io implementation (default: threaded)"
	@echo "  IMAGE=image|image-debug|image-release - render used by png/view (default: image)"
	@echo "  SCENE=<name>            - Scene to render (default: cover). See 'make scenes'"
	@echo "  Examples:"
	@echo "    make build MULTITHREAD=false       # Single-threaded debug build"
	@echo "    make run-release MULTITHREAD=false # Single-threaded release run"
	@echo "    make preview IO=single_threaded    # Preview using the single-threaded Io"
	@echo "    make preview SCENE=cornell-box     # Preview a different scene"
	@echo "    make run view                      # Render, then open the result"
	@echo "    make view IMAGE=image-release      # View an already-rendered image"

# Build targets
build: build-debug

build-debug:
	@echo "Building in Debug mode (multithreaded=$(MULTITHREAD))..."
	zig build -Dmultithreading=$(MULTITHREAD) -Dio=$(IO)

build-release:
	@echo "Building in ReleaseFast mode (multithreaded=$(MULTITHREAD))..."
	zig build -Doptimize=ReleaseFast -Dmultithreading=$(MULTITHREAD) -Dio=$(IO)

build-preview:
	@echo "Building PREVIEW mode ($(PREVIEW_WIDTH)px, $(PREVIEW_SAMPLES) samples)..."
	zig build -Doptimize=ReleaseFast -Dmultithreading=$(MULTITHREAD) -Dwidth=$(PREVIEW_WIDTH) -Dsamples=$(PREVIEW_SAMPLES) -Dio=$(IO)

# Run targets
run: run-debug

run-debug: build-debug
	@echo "Running Debug build..."
	./zig-out/bin/zaytracer $(SCENE_ARG)

run-release: build-release
	@echo "Running ReleaseFast build..."
	./zig-out/bin/zaytracer $(SCENE_ARG)

preview: build-preview
	@echo "========================================="
	@echo "   PREVIEW MODE (Fast Iteration)"
	@echo "   Resolution: $(PREVIEW_WIDTH)x$$(echo "$(PREVIEW_WIDTH) / 16 * 9" | bc)"
	@echo "   Samples: $(PREVIEW_SAMPLES)"
	@echo "========================================="
	@echo ""
	./zig-out/bin/zaytracer $(SCENE_ARG)
	@echo ""
	@echo "✓ Preview complete! Output: image.ppm"
	@echo "  For final quality: make run-release"

run-both: build-debug build-release
	@echo "========================================="
	@echo "Running both Debug and Release versions"
	@echo "========================================="
	@echo ""
	@echo "Running Debug build..."
	@rm -f image.ppm image-debug.ppm
	./zig-out/bin/zaytracer $(SCENE_ARG)
	@mv image.ppm image-debug.ppm
	@echo "✓ Debug output saved to: image-debug.ppm"
	@echo ""
	@echo "Running ReleaseFast build..."
	@rm -f image.ppm image-release.ppm
	./zig-out/bin/zaytracer $(SCENE_ARG)
	@mv image.ppm image-release.ppm
	@echo "✓ Release output saved to: image-release.ppm"
	@echo ""
	@echo "========================================="
	@echo "Both renders complete!"
	@echo "  Debug:   image-debug.ppm"
	@echo "  Release: image-release.ppm"
	@echo "========================================="

# Benchmark targets (with timing)
bench-debug: build-debug
	@echo "=== Benchmarking Debug build ==="
	@rm -f image.ppm
	@bash -c 'time ./zig-out/bin/zaytracer $(SCENE_ARG)'
	@echo ""

bench-release: build-release
	@echo "=== Benchmarking ReleaseFast build ==="
	@rm -f image.ppm
	@bash -c 'time ./zig-out/bin/zaytracer $(SCENE_ARG)'
	@echo ""

bench:
	@echo "========================================="
	@echo "Performance Comparison: Debug vs Release"
	@echo "========================================="
	@echo ""
	@$(MAKE) bench-debug
	@echo "========================================="
	@echo ""
	@$(MAKE) bench-release
	@echo "========================================="
	@echo "Benchmark complete!"
	@echo ""
	@echo "Compare the 'real' times above to see the speedup."
	@echo "ReleaseFast should be 2-3x faster than Debug."

bench-both: build-debug build-release
	@echo "========================================="
	@echo "Performance Comparison: Debug vs Release"
	@echo "       (with saved images)"
	@echo "========================================="
	@echo ""
	@echo "=== Benchmarking Debug build ==="
	@rm -f image.ppm image-debug.ppm
	@bash -c 'time ./zig-out/bin/zaytracer $(SCENE_ARG)'
	@mv image.ppm image-debug.ppm
	@echo "✓ Debug output saved to: image-debug.ppm"
	@echo ""
	@echo "========================================="
	@echo ""
	@echo "=== Benchmarking ReleaseFast build ==="
	@rm -f image.ppm image-release.ppm
	@bash -c 'time ./zig-out/bin/zaytracer $(SCENE_ARG)'
	@mv image.ppm image-release.ppm
	@echo "✓ Release output saved to: image-release.ppm"
	@echo ""
	@echo "========================================="
	@echo "Benchmark complete!"
	@echo ""
	@echo "Compare the 'real' times above to see the speedup."
	@echo "ReleaseFast should be 2-3x faster than Debug."
	@echo ""
	@echo "Output files:"
	@echo "  Debug:   image-debug.ppm"
	@echo "  Release: image-release.ppm"
	@echo "========================================="

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf zig-cache zig-out .zig-cache
	rm -f image.ppm image-debug.ppm image-release.ppm
	rm -f image.png image-debug.png image-release.png
	@echo "Clean complete!"

# Run tests
test:
	zig build test

# Fetch the models listed in models/manifest.tsv that are not already present,
# and verify each against its sha256 before putting it in place. Models large
# enough to be worth fetching are too large to commit, and the Stanford-derived
# ones are not ours to redistribute anyway.
models:
	@set -e; \
	tab=$$(printf '\t'); \
	while IFS="$$tab" read -r name url sum rest; do \
		case "$$name" in ''|\#*) continue ;; esac; \
		dest="$(MODELS_DIR)/$$name"; \
		if [ -f "$$dest" ]; then echo "have   $$name"; continue; fi; \
		if [ -z "$$url" ] || [ -z "$$sum" ]; then \
			echo "Error: $(MANIFEST): entry '$$name' is missing a url or a sha256" >&2; exit 1; \
		fi; \
		echo "fetch  $$name"; \
		curl -fL --progress-bar -o "$$dest.part" "$$url"; \
		actual=$$($(SHA256) "$$dest.part" | cut -d' ' -f1); \
		if [ "$$actual" != "$$sum" ]; then \
			rm -f "$$dest.part"; \
			echo "Error: $$name does not match its checksum" >&2; \
			echo "  expected $$sum" >&2; \
			echo "  actual   $$actual" >&2; \
			exit 1; \
		fi; \
		mv "$$dest.part" "$$dest"; \
		echo "ok     $$name"; \
	done < $(MANIFEST)

# Re-check the models already on disk against the manifest.
models-verify:
	@set -e; \
	tab=$$(printf '\t'); \
	missing=0; bad=0; \
	while IFS="$$tab" read -r name url sum rest; do \
		case "$$name" in ''|\#*) continue ;; esac; \
		dest="$(MODELS_DIR)/$$name"; \
		if [ ! -f "$$dest" ]; then echo "absent $$name"; missing=$$((missing+1)); continue; fi; \
		actual=$$($(SHA256) "$$dest" | cut -d' ' -f1); \
		if [ "$$actual" = "$$sum" ]; then echo "ok     $$name"; \
		else echo "BAD    $$name ($$actual)"; bad=$$((bad+1)); fi; \
	done < $(MANIFEST); \
	if [ $$bad -gt 0 ]; then echo "$$bad model(s) failed verification" >&2; exit 1; fi; \
	if [ $$missing -gt 0 ]; then echo "$$missing model(s) not fetched; run 'make models'"; fi

# List the scenes the binary knows about
scenes: build-debug
	@./zig-out/bin/zaytracer --list-scenes

# View image targets
#
# Rendering and viewing are separate steps: the binary always writes a .ppm,
# and the rules below turn an existing .ppm into something a desktop viewer can
# actually open. Chain them when you want both: make run view

# Most desktop image viewers cannot decode PPM. In particular imv, the default
# handler for image/x-portable-pixmap on Wayland, has no PNM backend: it opens a
# window, decodes nothing, and the render looks completely black. PNG works
# everywhere. Make skips this when the .png is already newer than the .ppm.
%.png: %.ppm
	@if command -v magick > /dev/null 2>&1; then \
		magick "$<" "$@"; \
	elif command -v convert > /dev/null 2>&1; then \
		convert "$<" "$@"; \
	elif command -v pnmtopng > /dev/null 2>&1; then \
		pnmtopng "$<" > "$@"; \
	else \
		echo "Error: no PPM->PNG converter found. Install imagemagick or netpbm."; \
		exit 1; \
	fi
	@echo "✓ Converted $< -> $@"

# Friendlier than make's default "No rule to make target" when nothing has been
# rendered yet.
image.ppm image-debug.ppm image-release.ppm:
	@echo "Error: $@ not found. Render it first with 'make run', 'make preview',"
	@echo "       'make run-both' or 'make bench-both'."
	@exit 1

png: $(IMAGE).png

view: $(IMAGE).png
	@if command -v xdg-open > /dev/null 2>&1; then \
		xdg-open "$<"; \
	elif command -v feh > /dev/null 2>&1; then \
		feh "$<"; \
	elif command -v eog > /dev/null 2>&1; then \
		eog "$<"; \
	elif command -v display > /dev/null 2>&1; then \
		display "$<"; \
	else \
		echo "No image viewer found. Image saved to $<"; \
	fi
