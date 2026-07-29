struct Params { k: u32, n: u32, width: u32, height: u32 };

@group(0) @binding(0) var<storage, read_write> centroids: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accum:     array<atomic<u32>>;
@group(0) @binding(2) var<storage, read>       pixels:    array<u32>;
@group(0) @binding(3) var<uniform>             params:    Params;

fn unpack(p: u32) -> vec3<f32> {
  return vec3<f32>(
    f32(p & 0xffu),
    f32((p >> 8u) & 0xffu),
    f32((p >> 16u) & 0xffu),
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= params.k) { return; }

  let base = k * 4u;
  let count = atomicLoad(&accum[base + 3u]);

  if (count > 0u) {
    let inv = 1.0 / f32(count);
    let r = f32(atomicLoad(&accum[base + 0u])) * inv;
    let g = f32(atomicLoad(&accum[base + 1u])) * inv;
    let b = f32(atomicLoad(&accum[base + 2u])) * inv;
    // Store the population count in .w so the UI can show cluster proportions.
    centroids[k] = vec4<f32>(r, g, b, f32(count));
  } else {
    let idx = (k * 2654435761u) % params.n;   // Knuth multiplicative hash
    centroids[k] = vec4<f32>(unpack(pixels[idx]), 0.0);
  }
}
