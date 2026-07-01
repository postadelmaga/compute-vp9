# compute-vp9

**VP9 hardware-accelerated decode via Vulkan compute shaders + VA-API backend**

For GPUs that lack a fixed-function VP9 decoder (Intel Skylake/HD 500 series,
NVIDIA Maxwell Gen 1), this project offloads VP9 decoding to the GPU's
programmable shader units, exposing the result via VA-API so browsers like
Chrome and Firefox can use it transparently.

## Motivation

Intel HD 520 (Skylake) and NVIDIA 940MX (Maxwell Gen 1) have no VP9 fixed-function
hardware decoder. YouTube defaults to VP9, causing high CPU load. Existing solutions
(h264ify) work around the problem by forcing H.264; this project attacks the root
cause by implementing VP9 decode on the GPU's compute units.

## Architecture

```
Chrome / Firefox
      │  VA-API
      ▼
compute_vp9_drv_video.so   ← VA-API driver plugin
      │
      ▼
libcompute_vp9.so
  ├── vp9_bitstream.c       CPU: entropy decode (sequential by spec)
  ├── vp9_tile.c            CPU: tile partitioning + MB parsing
  └── Vulkan compute backend
        ├── vp9_idct        GPU: inverse transform (4×4 … 32×32)
        ├── vp9_mc          GPU: motion compensation (inter prediction)
        ├── vp9_intra_pred  GPU: intra prediction modes
        └── vp9_loopfilter  GPU: deblocking filter
```

> **Note**: VP9 entropy coding is inherently sequential and cannot be GPU-parallelised.
> Only the transform/prediction/filter stages (which dominate decode time) run on GPU.

## Requirements

- Linux (Wayland or X11)
- Vulkan 1.1+ capable GPU
- `libva` ≥ 2.0
- CMake ≥ 3.20, Ninja
- `glslc` (Vulkan SDK or `shaderc` package)

## Build

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_VULKAN=ON \
  -DENABLE_VAAPI=ON
cmake --build build --parallel
sudo cmake --install build
```

## Usage

After installing, set the VA-API driver:

```bash
export LIBVA_DRIVER_NAME=compute_vp9
vainfo  # should show VAProfileVP9Profile0: VAEntrypointVLD
```

Then launch Chrome with VA-API enabled:

```bash
google-chrome-stable --enable-features=VaapiVideoDecoder
```

## Project Status

| Component | Status |
|---|---|
| Bitstream parser (frame header) | ✅ Implemented |
| VP9 4×4 IDCT shader | ✅ Implemented |
| VP9 8/16/32 IDCT shaders | 🚧 TODO |
| Motion compensation shader | ✅ Skeleton |
| Intra prediction shader | ✅ Skeleton (DC/V/H/TM) |
| Loop filter shader | ✅ Skeleton |
| Vulkan backend init | 🚧 TODO |
| VA-API entrypoint | ✅ Skeleton |
| Entropy decoder (CPU) | 🚧 TODO |
| Tile/MB parser | 🚧 TODO |
| Full decode pipeline | 🚧 TODO |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome — this is a
complex project and there is plenty of low-hanging fruit.

## References

- [VP9 Bitstream Specification](https://www.webmproject.org/vp9/)
- [libva VA-API](https://github.com/intel/libva)
- [Vulkan Compute Guide](https://vkguide.dev/docs/gpudriven/compute_shaders/)
- [VP9 decode complexity analysis](https://arxiv.org/abs/1602.02200)

## License

Apache 2.0 — see [LICENSE](LICENSE).
