struct Params { k: u32, n: u32, width: u32, height: u32 };
// This texture is the "render target" that the blit pass then draws to screen.

@group(0) @binding(0) var<storage, read> assignments: array<u32>;
@group(0) @binding(1) var<storage, read> centroids:   array<vec4<f32>>;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let i = gid.y * params.width + gid.x;
  let color = centroids[assignments[i]].xyz / 255.0;
  textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(color, 1.0));
}
