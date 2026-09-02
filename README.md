# WebGPU K-means Palette Finder

Extract a color palette from an image using the k-means clustering algorithm, with every iteration running on the GPU via WebGPU compute shaders.

## Run

```bash
npm install
npm run dev
```

Open the printed URL in a **WebGPU-capable browser** (recent Chrome for example). Load an image, pick the number of colors (K) and a max iteration count, then click **Run**. You'll see the palette update live, plus a quantized version of the image rendered on the GPU.

## How it works

The palette is clustered **perceptually**: in CIELAB (so distance ≈ how different
two colours *look*) over a colour histogram weighted by `√count` (so a small vivid
accent isn't outvoted by a big flat region). The whole clustering loop runs on the
GPU. See [docs/perceptual-palette.md](docs/perceptual-palette.md) for the why.

K-means alternates two steps until the iteration budget runs out:

1. **Assign** — every colour picks its nearest centroid (squared distance in Lab).
2. **Update** — each centroid becomes the `√count`-weighted mean of its colours.

### GPU mapping

| Pass | Shader | Parallelism | Notes |
|------|--------|-------------|-------|
| Histogram | `histogram.wgsl` | one thread / pixel | Atomic-bins pixels into 2¹⁵ bins (5 bits/channel); keeps sum RGB + count per bin. |
| Bin prep | `binprep.wgsl` | one thread / bin | Average RGB → Lab; weight = `√count`. |
| Assign + accumulate | `bassign.wgsl` | one thread / bin | Nearest centroid in Lab; scatter-adds the weighted Lab sums into per-cluster accumulators. |
| Finalize | `bfinalize.wgsl` | one thread / centroid | `newCentroid = Σ(w·lab) / Σw`; re-seeds empty clusters; also emits centroid RGB + count. |
| Quantize | `quantize.wgsl` | one thread / pixel | Assigns each pixel in Lab and writes its centroid colour into a storage texture. |
| Blit | `blit.wgsl` | fullscreen triangle | Draws the storage texture to the canvas. |

WGSL atomics are integer-only, so the assign pass accumulates the (fractional)
weighted Lab sums in **fixed point** — scale by a constant, truncate, and let the
scale cancel in the finalize divide. `a`/`b` are biased by +128 to stay
non-negative. Cluster counts ride in the centroid's `.w` so the UI can show each
colour's proportion.

The k-means++ **seed** is computed on the CPU (it's sequential and cheap over the
small histogram); everything O(pixels) — histogram, the iteration loop, and the
quantized preview — stays on the GPU. The loop is encoded and submitted **once**,
with no per-iteration readback: only the seed's histogram and the final palette
cross back to the CPU.

Images are downscaled so the longest side is ≤ 512 px before clustering — a
palette rarely needs full resolution, and it keeps the histogram pass fast.

### Files

```
src/
  gpu/device.ts       WebGPU adapter/device init
  color.ts            sRGB <-> CIELAB (CPU, for seeding)
  image.ts            file -> downscaled packed-u32 pixel buffer
  palette-finder.ts   PaletteFinder (pipelines + GPU loop) + CPU k-means++ seed
  main.ts             UI wiring
  shaders/lab.wgsl    sRGB <-> CIELAB, shared (prepended into the passes that need it)
  shaders/*.wgsl      the GPU passes
```
