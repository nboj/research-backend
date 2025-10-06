{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };
  inputs.rust-overlay.url = "github:oxalica/rust-overlay";

  outputs =
    { nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        python = pkgs.python313;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.bashInteractive
            python
            python.pkgs.virtualenv
            python.pkgs.pip
            pkgs.clang           # <- use clang, not gcc
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.rust-bin.stable.latest.default
          ];
          shellHook = ''
            unset CC CXX
            export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH
            source ./venv/bin/activate
          '';
        };
      }
    );

}
