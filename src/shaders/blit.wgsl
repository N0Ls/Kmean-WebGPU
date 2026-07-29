@group(0) @binding(0) var tex:  texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0)       uv:  vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var out: VOut;
  let clip = p[vi];
  out.pos = vec4<f32>(clip, 0.0, 1.0);
  // Map clip space to texture UVs, flipping Y so the image is upright.
  out.uv = vec2<f32>(clip.x * 0.5 + 0.5, 1.0 - (clip.y * 0.5 + 0.5));
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  return textureSample(tex, samp, in.uv);
}
