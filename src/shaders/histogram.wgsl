// Pass 1a: build a colour histogram on the GPU. One thread per pixel scatter-adds
// into a fixed 32768-bin table (5 bits/channel) using integer atomics. Each bin
// keeps the running sum of the real RGB values plus a pixel count, so bin-prep can
// recover the *average* colour of the pixels that fell into it (not the bin centre).
// Integer atomics are exact here: R/G/B are 0-255 and counts are integers.

struct Params { k: u32, n: u32, width: u32, height: u32, numBins: u32 };

@group(0) @binding(0) var<storage, read>       pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> hist:   array<atomic<u32>>; // numBins*4: sumR,sumG,sumB,count
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }

  let p = pixels[i];
  let r = p & 0xffu;
  let g = (p >> 8u) & 0xffu;
  let b = (p >> 16u) & 0xffu;

  let key = ((r >> 3u) << 10u) | ((g >> 3u) << 5u) | (b >> 3u);
  let base = key * 4u;
  atomicAdd(&hist[base + 0u], r);
  atomicAdd(&hist[base + 1u], g);
  atomicAdd(&hist[base + 2u], b);
  atomicAdd(&hist[base + 3u], 1u);
}
