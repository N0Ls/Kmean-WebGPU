// Lloyd update. One thread per centre: divide the weighted Lab sums by the weight
// sum (the fixed-point scale S cancels), giving the new centre in Lab. We also emit
// the centre in RGB with the raw pixel count in .w, so the same buffer doubles as
// the palette read back after the loop. Empty clusters re-seed to a live bin so they
// get another chance to capture colours (mirrors the CPU path's behaviour).
// (labToRgb is prepended from lab.wgsl at compile time.)

struct Params { k: u32, n: u32, width: u32, height: u32, numBins: u32 };

@group(0) @binding(0) var<storage, read_write> centers:    array<vec4<f32>>; // Lab, updated in place
@group(0) @binding(1) var<storage, read>       accum:      array<u32>;       // k*8, written atomically last pass
@group(0) @binding(2) var<storage, read_write> centersRgb: array<vec4<f32>>; // r,g,b,count (palette output)
@group(0) @binding(3) var<storage, read>       binLab:     array<vec4<f32>>; // for re-seeding empty clusters
@group(0) @binding(4) var<uniform>             params:     Params;

const BIAS: f32 = 128.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= params.k) { return; }

  let base = c * 8u;
  let wSum = accum[base + 3u];

  if (wSum > 0u) {
    let invW = 1.0 / f32(wSum);
    let L = f32(accum[base + 0u]) * invW;
    let a = f32(accum[base + 1u]) * invW - BIAS;
    let b = f32(accum[base + 2u]) * invW - BIAS;
    centers[c] = vec4<f32>(L, a, b, 0.0);
    centersRgb[c] = vec4<f32>(labToRgb(L, a, b), f32(accum[base + 4u]));
  } else {
    // Empty cluster: hash to a starting bin, then linear-probe for a live one.
    var idx = (c * 2654435761u) % params.numBins; // Knuth multiplicative hash
    for (var t: u32 = 0u; t < params.numBins; t = t + 1u) {
      if (binLab[idx].w > 0.0) { break; }
      idx = (idx + 1u) % params.numBins;
    }
    let lab = binLab[idx].xyz;
    centers[c] = vec4<f32>(lab, 0.0);
    centersRgb[c] = vec4<f32>(labToRgb(lab.x, lab.y, lab.z), 0.0);
  }
}
