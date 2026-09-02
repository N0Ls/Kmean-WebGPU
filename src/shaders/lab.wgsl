// sRGB <-> CIELAB (D65), ported from src/color.ts so the GPU can cluster in the
// same perceptual space. Inputs/outputs are 0-255 per channel. This file is not
// an entry point; it is prepended to the shaders that need Lab (see withLab()).

const Xn = 0.95047;
const Yn = 1.0;
const Zn = 1.08883;
const LAB_EPS = 0.00885645167904; // 216/24389
const KAPPA = 903.2962962963;     // 24389/27

fn srgbToLinear(c: f32) -> f32 {
  let v = c / 255.0;
  if (v <= 0.04045) { return v / 12.92; }
  return pow((v + 0.055) / 1.055, 2.4);
}

fn labFwd(t: f32) -> f32 {
  if (t > LAB_EPS) { return pow(t, 1.0 / 3.0); }
  return (KAPPA * t + 16.0) / 116.0;
}

fn rgbToLab(r: f32, g: f32, b: f32) -> vec3<f32> {
  let rl = srgbToLinear(r);
  let gl = srgbToLinear(g);
  let bl = srgbToLinear(b);

  let x = (rl * 0.4124 + gl * 0.3576 + bl * 0.1805) / Xn;
  let y = (rl * 0.2126 + gl * 0.7152 + bl * 0.0722) / Yn;
  let z = (rl * 0.0193 + gl * 0.1192 + bl * 0.9505) / Zn;

  let fx = labFwd(x);
  let fy = labFwd(y);
  let fz = labFwd(z);

  return vec3<f32>(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}

fn labInv(t: f32) -> f32 {
  let t3 = t * t * t;
  if (t3 > LAB_EPS) { return t3; }
  return (116.0 * t - 16.0) / KAPPA;
}

fn linearToSrgb(v: f32) -> f32 {
  var c: f32;
  if (v <= 0.0031308) { c = v * 12.92; } else { c = 1.055 * pow(max(v, 0.0), 1.0 / 2.4) - 0.055; }
  return clamp(c * 255.0, 0.0, 255.0);
}

fn labToRgb(L: f32, a: f32, b: f32) -> vec3<f32> {
  let fy = (L + 16.0) / 116.0;
  let fx = fy + a / 500.0;
  let fz = fy - b / 200.0;

  let x = labInv(fx) * Xn;
  let y = labInv(fy) * Yn;
  let z = labInv(fz) * Zn;

  let rl = x * 3.2406 + y * (-1.5372) + z * (-0.4986);
  let gl = x * (-0.9689) + y * 1.8758 + z * 0.0415;
  let bl = x * 0.0557 + y * (-0.2040) + z * 1.0570;

  return vec3<f32>(linearToSrgb(rl), linearToSrgb(gl), linearToSrgb(bl));
}
