# gsc_pgo

Native C++ pose-graph-optimization (PGO) for DimOS loop closure — GTSAM iSAM2 +
PCL ICP, with Scan-Context descriptors and AprilTag landmark factors.

Extracted from the DimOS nav stack so the C++ and its nix flake can be built and
pinned independently of the Python tree.

## Build

```sh
nix build .#default        # -> result/bin/pgo
```

Outputs `pgo` (the live module binary) and `pgo_landmark_test` (a no-LCM lockstep
unit test driving `SimplePGO` directly).

## Layout

- `main.cpp` — LCM I/O loop, wires odometry/lidar/tags into the optimizer
- `simple_pgo.{cpp,h}` — iSAM2 pose graph + ICP loop closure
- `scan_context.{cpp,h}` — Scan-Context place descriptors / loop candidate search
- `commons.{cpp,h}`, `point_cloud_utils.hpp` — shared helpers
- `dimos_native_module.hpp` — DimOS NativeModule LCM glue
- `msgs/*.hpp` — C++ wire helpers mirroring `dimos.navigation.jnav.msgs.*`
  (`Graph3D`, `GraphDelta3D`, `Landmark`); keep `encode()` in sync with the
  Python decoders.

## Consumed by

`dimos/navigation/jnav/components/loop_closure/gsc_pgo/module.py`, via
`nix build "github:jeff-hykin/gsc_pgo#default"`.
