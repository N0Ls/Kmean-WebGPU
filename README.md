# WebGPU K-means Palette Finder

Extract a color palette from an image using the k-means clustering algorithm, with every iteration running on the GPU via WebGPU compute shaders.

## Run

```bash
npm install
npm run dev
```

Open the printed URL in a **WebGPU-capable browser** (recent Chrome for example). Load an image, pick the number of colors (K) and a max iteration count, then click **Run**. You'll see the palette update live, plus a quantized version of the image rendered on the GPU.

## How it works

K-means alternates two steps until the centroids stop moving:

1. **Assign** — every pixel picks its nearest centroid (by squared RGB distance).
2. **Update** — each centroid becomes the mean color of its assigned pixels.

### GPU mapping

| Pass | Shader | Parallelism | Notes |
|------|--------|-------------|-------|
| Assign + accumulate | `assign.wgsl` | one thread / pixel | Finds nearest centroid, scatter-adds color into per-cluster accumulators with **integer atomics** (`atomicAdd`). |
| Finalize | `finalize.wgsl` | one thread / centroid | `newCentroid = sum / count`; re-seeds empty clusters. |
| Quantize | `quantize.wgsl` | one thread / pixel | Writes each pixel's centroid color into a storage texture (the render target). |
| Blit | `blit.wgsl` | fullscreen triangle | Draws the storage texture to the canvas. |

The accumulators are `atomic<u32>` because WGSL atomics are integer-only. Since
RGB channels are 0–255, the integer sums are exact. Cluster counts are stashed
in the centroid's `.w` so the UI can show each color's proportion.

`kmeansPlusPlusInit` (in `kmeans.ts`) seeds centroids on the CPU with k-means++
so the GPU loop converges to a better palette than random seeding would.

Images are downscaled so the longest side is ≤ 512 px before clustering — a
palette rarely needs full resolution, and this keeps each pass fast.

### Files

```
src/
  gpu/device.ts     WebGPU adapter/device init
  image.ts          file -> downscaled packed-u32 pixel buffer
  kmeans.ts         PaletteFinder (pipelines + iteration loop) + k-means++
  main.ts           UI wiring
  shaders/*.wgsl    the four GPU passes
```
