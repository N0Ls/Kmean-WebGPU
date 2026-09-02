// Lloyd assign + accumulate, over bins, in Lab. One thread per bin finds its
// nearest centre, then scatter-adds the weighted Lab sums into per-cluster
// accumulators. WGSL atomics are integer-only and the sums are fractional, so we
// accumulate in FIXED POINT: multiply by S and truncate. The scale cancels in the
// finalize divide (mean = Sum(w*x*S) / Sum(w*S)), so S only trades range for
// precision. `a`/`b` are biased by +128 to stay non-negative for the u32 atomics.
//
// Range check (why it doesn't overflow a u32): weight = sqrt(count), and the worst
// case for Sum(weight) over all bins is an evenly spread histogram, ~sqrt(numBins*n).
// The largest accumulator is Sum(w*(a+128)*S) <= Sum(w)*256*S. With numBins=32768,
// S=64 that stays < 2^32 up to ~1e6 pixels (comfortably past the 512px downscale cap).

struct Params { k: u32, n: u32, width: u32, height: u32, numBins: u32 };

@group(0) @binding(0) var<storage, read>       binLab:   array<vec4<f32>>; // L,a,b,weight
@group(0) @binding(1) var<storage, read>       binCount: array<f32>;       // raw pixel count
@group(0) @binding(2) var<storage, read>       centers:  array<vec4<f32>>; // L,a,b,_ (Lab)
@group(0) @binding(3) var<storage, read_write> accum:    array<atomic<u32>>; // k*8: wL,wA,wB,wSum,countSum, +pad
@group(0) @binding(4) var<uniform>             params:   Params;

const S: f32 = 64.0;      // fixed-point scale
const BIAS: f32 = 128.0;  // keeps biased a/b non-negative

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numBins) { return; }

  let bl = binLab[i];
  let w = bl.w;
  if (w <= 0.0) { return; } // empty bin
  let lab = bl.xyz;

  var best: u32 = 0u;
  var bestDist: f32 = 3.0e38;
  for (var c: u32 = 0u; c < params.k; c = c + 1u) {
    let d = lab - centers[c].xyz;
    let dist = dot(d, d);
    if (dist < bestDist) { bestDist = dist; best = c; }
  }

  let base = best * 8u;
  atomicAdd(&accum[base + 0u], u32(w * lab.x * S));
  atomicAdd(&accum[base + 1u], u32(w * (lab.y + BIAS) * S));
  atomicAdd(&accum[base + 2u], u32(w * (lab.z + BIAS) * S));
  atomicAdd(&accum[base + 3u], u32(w * S));
  atomicAdd(&accum[base + 4u], u32(binCount[i])); // raw count, for prominence
}
