struct Params { k: u32, n: u32, width: u32, height: u32 };

@group(0) @binding(0) var<storage, read>        pixels:      array<u32>;
@group(0) @binding(1) var<storage, read>        centroids:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write>  accum:       array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write>  assignments: array<u32>;
@group(0) @binding(4) var<uniform>              params:      Params;

fn unpack(p: u32) -> vec3<f32> {
  return vec3<f32>(
    f32(p & 0xffu),
    f32((p >> 8u) & 0xffu),
    f32((p >> 16u) & 0xffu),
  );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }

  let color = unpack(pixels[i]);

  var best: u32 = 0u;
  var bestDist: f32 = 3.0e38;
  for (var k: u32 = 0u; k < params.k; k = k + 1u) {
    let d = color - centroids[k].xyz;
    let dist = dot(d, d);
    if (dist < bestDist) {
      bestDist = dist;
      best = k;
    }
  }

  assignments[i] = best;

  let base = best * 4u;
  atomicAdd(&accum[base + 0u], u32(color.r));
  atomicAdd(&accum[base + 1u], u32(color.g));
  atomicAdd(&accum[base + 2u], u32(color.b));
  atomicAdd(&accum[base + 3u], 1u);
}
