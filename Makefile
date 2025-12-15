.PHONY: help build build-debug build-release \
        run run-debug run-release run-both \
        bench bench-debug bench-release bench-both clean test view view-debug view-release

# Default target
.DEFAULT_GOAL := help

# Multithreading flag (can be overridden: make build MULTITHREAD=false)
MULTITHREAD ?= true

# Help message
help:
	@echo "Zaytracer - Available targets:"
	@echo ""
	@echo "Build targets:"
	@echo "  make build              - Build in default mode (Debug, multithreaded)"
	@echo "  make build-debug        - Build in Debug mode (with safety checks)"
	@echo "  make build-release      - Build in ReleaseFast mode (optimized)"
	@echo ""
	@echo "Run targets:"
	@echo "  make run                - Build and run in default mode"
	@echo "  make run-debug          - Build and run in Debug mode"
	@echo "  make run-release        - Build and run in ReleaseFast mode"
	@echo "  make run-both           - Run both modes and save separate images"
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
	@echo "  make view               - Run and view the output image"
	@echo "  make view-debug         - View debug output image"
	@echo "  make view-release       - View release output image"
	@echo "  make help               - Show this help message"
	@echo ""
	@echo "Build modes:"
	@echo "  Debug       - Safety checks enabled, no optimizations (slower, safer)"
	@echo "  ReleaseFast - Full optimizations, no safety checks (faster, production)"
	@echo ""
	@echo "Build options:"
	@echo "  MULTITHREAD=true/false  - Enable/disable multithreading (default: true)"
	@echo "  Examples:"
	@echo "    make build MULTITHREAD=false       # Single-threaded debug build"
	@echo "    make run-release MULTITHREAD=false # Single-threaded release run"

# Build targets
build: build-debug

build-debug:
	@echo "Building in Debug mode (multithreaded=$(MULTITHREAD))..."
	zig build -Dmultithreading=$(MULTITHREAD)

build-release:
	@echo "Building in ReleaseFast mode (multithreaded=$(MULTITHREAD))..."
	zig build -Doptimize=ReleaseFast -Dmultithreading=$(MULTITHREAD)

# Run targets
run: run-debug

run-debug: build-debug
	@echo "Running Debug build..."
	./zig-out/bin/zaytracer

run-release: build-release
	@echo "Running ReleaseFast build..."
	./zig-out/bin/zaytracer

run-both: build-debug build-release
	@echo "========================================="
	@echo "Running both Debug and Release versions"
	@echo "========================================="
	@echo ""
	@echo "Running Debug build..."
	@rm -f image.ppm image-debug.ppm
	./zig-out/bin/zaytracer
	@mv image.ppm image-debug.ppm
	@echo "✓ Debug output saved to: image-debug.ppm"
	@echo ""
	@echo "Running ReleaseFast build..."
	@rm -f image.ppm image-release.ppm
	./zig-out/bin/zaytracer
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
	@bash -c 'time ./zig-out/bin/zaytracer'
	@echo ""

bench-release: build-release
	@echo "=== Benchmarking ReleaseFast build ==="
	@rm -f image.ppm
	@bash -c 'time ./zig-out/bin/zaytracer'
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
	@bash -c 'time ./zig-out/bin/zaytracer'
	@mv image.ppm image-debug.ppm
	@echo "✓ Debug output saved to: image-debug.ppm"
	@echo ""
	@echo "========================================="
	@echo ""
	@echo "=== Benchmarking ReleaseFast build ==="
	@rm -f image.ppm image-release.ppm
	@bash -c 'time ./zig-out/bin/zaytracer'
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
	@echo "Clean complete!"

# Run tests
test:
	zig build test

# View image targets
view: run
	@if command -v xdg-open > /dev/null 2>&1; then \
		xdg-open image.ppm; \
	elif command -v feh > /dev/null 2>&1; then \
		feh image.ppm; \
	elif command -v eog > /dev/null 2>&1; then \
		eog image.ppm; \
	elif command -v display > /dev/null 2>&1; then \
		display image.ppm; \
	else \
		echo "No image viewer found. Image saved to image.ppm"; \
	fi

view-debug:
	@if [ ! -f image-debug.ppm ]; then \
		echo "Error: image-debug.ppm not found. Run 'make run-both' or 'make bench-both' first."; \
		exit 1; \
	fi
	@if command -v xdg-open > /dev/null 2>&1; then \
		xdg-open image-debug.ppm; \
	elif command -v feh > /dev/null 2>&1; then \
		feh image-debug.ppm; \
	elif command -v eog > /dev/null 2>&1; then \
		eog image-debug.ppm; \
	elif command -v display > /dev/null 2>&1; then \
		display image-debug.ppm; \
	else \
		echo "No image viewer found. Image at: image-debug.ppm"; \
	fi

view-release:
	@if [ ! -f image-release.ppm ]; then \
		echo "Error: image-release.ppm not found. Run 'make run-both' or 'make bench-both' first."; \
		exit 1; \
	fi
	@if command -v xdg-open > /dev/null 2>&1; then \
		xdg-open image-release.ppm; \
	elif command -v feh > /dev/null 2>&1; then \
		feh image-release.ppm; \
	elif command -v eog > /dev/null 2>&1; then \
		eog image-release.ppm; \
	else \
		echo "No image viewer found. Image at: image-release.ppm"; \
	fi
