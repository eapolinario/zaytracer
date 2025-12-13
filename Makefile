.PHONY: help build run clean test view

# Default target
help:
	@echo "Zaytracer - Available targets:"
	@echo "  make build       - Build the raytracer"
	@echo "  make run         - Build and run the raytracer"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make test        - Run tests"
	@echo "  make view        - Run and view the output image"
	@echo "  make help        - Show this help message"

# Build the project
build:
	zig build

# Build and run
run:
	zig build run

# Clean build artifacts
clean:
	rm -rf zig-cache zig-out .zig-cache
	rm -f image.ppm

# Run tests
test:
	zig build test

# Build, run, and view the output
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
