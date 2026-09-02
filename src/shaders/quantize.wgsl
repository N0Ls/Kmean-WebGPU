// Quantise the image against the final palette, one thread per pixel. Assignment is
// done in Lab (perceptual) to match how the palette itself was clustered, then the
// pixel is written in its centre's RGB into the storage texture the blit pass draws.
// (rgbToLab is prepended from lab.wgsl at compile time.)

struct Params { k: u32, n: u32, width: u32, height: u32, numBins: u32 };

@group(0) @binding(0) var<storage, read> pixels:     array<u32>;
@group(0) @binding(1) var<storage, read> centersLab: array<vec4<f32>>; // L,a,b,_
@group(0) @binding(2) var<storage, read> centersRgb: array<vec4<f32>>; // r,g,b,count (0-255)
@group(0) @binding(3) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;

  let p = pixels[idx];
  let lab = rgbToLab(f32(p & 0xffu), f32((p >> 8u) & 0xffu), f32((p >> 16u) & 0xffu));

  var best: u32 = 0u;
  var bestDist: f32 = 3.0e38;
  for (var c: u32 = 0u; c < params.k; c = c + 1u) {
    let d = lab - centersLab[c].xyz;
    let dist = dot(d, d);
    if (dist < bestDist) { bestDist = dist; best = c; }
  }

  let color = centersRgb[best].xyz / 255.0;
  textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(color, 1.0));
}
