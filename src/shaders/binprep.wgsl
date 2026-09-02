// Pass 1b: turn the raw histogram into clustering-ready bins. One thread per bin:
// average RGB = sum/count, convert to Lab (perceptual distance), and set the
// frequency-de-weighting weight = sqrt(count) so a small-but-vivid colour can still
// earn a centroid. Empty bins are flagged with weight 0 and skipped downstream.
// (rgbToLab is prepended from lab.wgsl at compile time.)

struct Params { k: u32, n: u32, width: u32, height: u32, numBins: u32 };

@group(0) @binding(0) var<storage, read>       hist:     array<u32>;        // numBins*4: sumR,sumG,sumB,count
@group(0) @binding(1) var<storage, read_write> binLab:   array<vec4<f32>>;  // L, a, b, weight(=sqrt(count), 0 if empty)
@group(0) @binding(2) var<storage, read_write> binCount: array<f32>;        // raw pixel count (for palette proportions)
@group(0) @binding(3) var<uniform>             params:   Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numBins) { return; }

  let base = i * 4u;
  let count = hist[base + 3u];
  if (count == 0u) {
    binLab[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    binCount[i] = 0.0;
    return;
  }

  let inv = 1.0 / f32(count);
  let r = f32(hist[base + 0u]) * inv;
  let g = f32(hist[base + 1u]) * inv;
  let b = f32(hist[base + 2u]) * inv;

  let lab = rgbToLab(r, g, b);
  binLab[i] = vec4<f32>(lab, sqrt(f32(count))); // weight = count^0.5
  binCount[i] = f32(count);
}
