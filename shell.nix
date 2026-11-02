# Development shell for NixOS, where PyPI wheels (numpy, pyarrow) can't find libstdc++
# and PyPI's prebuilt ruff binary can't run. Usage: `nix-shell`, then `uv sync --extra eval`.
# Other Linux distributions don't need this file.
{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [ pkgs.uv pkgs.ruff pkgs.python312 ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
  UV_PYTHON = "${pkgs.python312}/bin/python3.12";
  shellHook = ''
    echo "kengine dev shell: run ruff as plain 'ruff' (the PyPI binary doesn't run on NixOS)"
  '';
}
