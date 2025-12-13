{
  description = "Zaytracer - Ray Tracing in One Weekend (Zig implementation)";

  inputs = {
    # Use latest nixpkgs with Zig 0.15.x
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            zig
            zls  # Zig Language Server (optional, for IDE support)
          ];

          shellHook = ''
            echo "Zaytracer development environment"
            echo "Zig version: $(zig version)"
            echo ""
            echo "Available commands:"
            echo "  zig build       - Build the project"
            echo "  zig build run   - Build and run the raytracer"
            echo "  zig build test  - Run tests"
          '';
        };
      }
    );
}
