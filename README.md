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
      │  VA-API (incl. vaExportSurfaceHandle → NV12 DMA-BUF, zero-copy)
      ▼
compute_vp9_drv_video.so   ← VA-API driver plugin
      │
      ▼
libcompute_vp9.so
  ├── vp9_bitstream.c       CPU: entropy decode (sequential by spec)
  ├── vp9_tile.c            CPU: tile partitioning + MB parsing
  └── Vulkan compute backend (3 frames in flight)
        ├── vp9_idct        GPU: inverse transform (4×4 … 32×32)
        ├── vp9_mc          GPU: motion compensation (inter prediction)
        ├── vp9_intra_pred  GPU: intra prediction modes
        └── vp9_loopfilter  GPU: deblocking filter (2D dispatch)
```

> **Note**: VP9 entropy coding is inherently sequential and cannot be GPU-parallelised.
> Only the transform/prediction/filter stages (which dominate decode time) run on GPU.

### Pipelined decode

Decoding is pipelined with up to **3 frames in flight**: the CPU entropy-decodes
frame N+1/N+2 while the GPU reconstructs frame N. Each pipeline slot has its own
persistent command buffer, fence, staging and output buffers; delivery latency is
at most 2 frames. `cvp9_get_frame()` is non-blocking (`CVP9_ERR_AGAIN` while the
oldest frame is still on the GPU); `cvp9_get_frame_sync()` blocks.

### Cross-GPU shared memory (DMA-BUF)

When `VK_EXT_external_memory_dma_buf` is available, decoded frames live in
DMA-BUF-exportable memory (dedicated allocation, HOST_VISIBLE|HOST_CACHED
preferred), so a *second* GPU — e.g. the iGPU driving the display — can import
the frame over PCIe with no CPU bounce:

- `cvp9_get_frame_dmabuf()` exports the decoded frame as an I420 DMA-BUF fd.
- The VA-API driver backs every render-target surface with an NV12 DMA-BUF and
  implements `vaExportSurfaceHandle` (DRM PRIME 2, linear modifier), which is
  the zero-copy import path used by Chrome's `VaapiVideoDecoder`.

## Performance

Synthetic decode benchmark (`tests/benchmark.c`, 100 frames, minimal
coefficient load — measures pipeline/backend overhead, not full bitstream
complexity). Laptop: i5-6200U + Intel HD 520 + NVIDIA 940MX:

| Backend / GPU | 1280×720 | 1920×1080 | Max frame latency |
|---|---|---|---|
| CPU fallback (1 thread) | 30 FPS | 13 FPS | — |
| Vulkan, Intel HD 520 | **237 FPS** | **87 FPS** | 16 ms @720p |
| Vulkan, NVIDIA 940MX | — | **140 FPS** | 17 ms @1080p |

Impact of the frame pipelining + 2D loop filter dispatch (vs the previous
serial decode→wait→copy backend, same hardware):

| GPU | Before | After | Speedup |
|---|---|---|---|
| Intel HD 520 @720p | 142 FPS | 237 FPS | **+67%** |
| Intel HD 520 @1080p | 63 FPS | 87 FPS | **+38%** |
| NVIDIA 940MX @1080p | 60 FPS | 140 FPS | **+131%** |

The dGPU gains the most: PCIe readback no longer stalls the pipeline. Numbers
are upper bounds on backend throughput — real-stream decode is dominated by
the CPU entropy stage until multi-threaded tile parsing lands.

## Requirements

- Linux (Wayland or X11)
- Vulkan 1.1+ capable GPU
- `libva` ≥ 2.0
- CMake ≥ 3.20, Ninja
- `glslc` (Vulkan SDK or `shaderc` package)

## Build & install

```bash
./install.sh              # Release build + install to system dirs (sudo)
./install.sh --uninstall  # remove
```

The installer places files where the system expects them:

| File | Location |
|---|---|
| `libcompute_vp9.so` | `/usr/lib` |
| `compute_vp9_drv_video.so` | libva driverdir (e.g. `/usr/lib/dri`) |
| SPIR-V shaders | `/usr/share/compute-vp9/shaders` |
| headers | `/usr/include/compute_vp9` |

Manual build (development):

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_VULKAN=ON \
  -DENABLE_VAAPI=ON
cmake --build build --parallel
```

## Usage

After installing, verify the driver is picked up:

```bash
LIBVA_DRIVER_NAME=compute_vp9 vainfo
# should show VAProfileVP9Profile0/2 : VAEntrypointVLD
```

Then launch Chrome with VA-API enabled:

```bash
LIBVA_DRIVER_NAME=compute_vp9 google-chrome-stable \
    --enable-features=VaapiVideoDecoder,VaapiVideoDecodeLinuxGL \
    --disable-features=UseChromeOSDirectVideoDecoder
```

Select the decode GPU with `CVP9_GPU_VENDOR=intel|nvidia|amd`.

## Project Status

| Component | Status |
|---|---|
| Bitstream parser (frame header) | ✅ Implemented (simplified dialect — real streams not yet parsed) |
| VP9 4×4/8×8/16×16/32×32 IDCT shaders | ✅ Implemented (1D fast butterfly) |
| Motion compensation shader | ✅ 8-tap subpel (16 phases), LAST/GOLDEN/ALTREF, ref pool refresh — luma only |
| Intra prediction shader | ✅ Skeleton (DC/V/H/TM) |
| Loop filter shader | ✅ Skeleton (2D dispatch) |
| Vulkan backend init | ✅ Implemented |
| Pipelined decode (3 frames in flight) | ✅ Implemented |
| DMA-BUF export (cross-GPU zero-copy) | ✅ Implemented |
| GPU NV12 pack direct into VA surfaces | ✅ Implemented (no CPU copies to the client) |
| Dequantization (spec qindex tables) | ✅ Implemented |
| VA-API entrypoint | ✅ Implemented (Chrome-compatible: `vaQuerySurfaceAttributes`, `vaCreateSurfaces2`, `vaExportSurfaceHandle`) |
| Entropy decoder (CPU) | 🚧 Partial |
| Tile/MB parser | 🚧 Partial (multi-threaded across tile columns, O(1) neighbor lookup) |
| Chrome integration | ✅ Driver accepted: "Video Decode: Hardware accelerated", VP9 profile0/2 listed in chrome://gpu |
| Full decode pipeline (correct output) | 🚧 TODO — real bitstreams are rejected by the parser (not yet spec-conformant), Chrome falls back to software |

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
