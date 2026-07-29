// sRGB <-> CIELAB (D65) conversion. Clustering in Lab makes Euclidean distance
// approximate perceptual difference, so distinct hues stay apart and genuine
// near-shades collapse together. Inputs/outputs are 0-255 per channel.

const Xn = 0.95047;
const Yn = 1.0;
const Zn = 1.08883;
const EPS = 216 / 24389; // 0.008856
const KAPPA = 24389 / 27; // 903.3

function srgbToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

function linearToSrgb(v: number): number {
  const c = v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(c * 255)));
}

function fwd(t: number): number {
  return t > EPS ? Math.cbrt(t) : (KAPPA * t + 16) / 116;
}

function inv(t: number): number {
  const t3 = t * t * t;
  return t3 > EPS ? t3 : (116 * t - 16) / KAPPA;
}

export function rgbToLab(r: number, g: number, b: number): [number, number, number] {
  const rl = srgbToLinear(r);
  const gl = srgbToLinear(g);
  const bl = srgbToLinear(b);

  const x = (rl * 0.4124 + gl * 0.3576 + bl * 0.1805) / Xn;
  const y = (rl * 0.2126 + gl * 0.7152 + bl * 0.0722) / Yn;
  const z = (rl * 0.0193 + gl * 0.1192 + bl * 0.9505) / Zn;

  const fx = fwd(x);
  const fy = fwd(y);
  const fz = fwd(z);

  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

export function labToRgb(L: number, a: number, b: number): [number, number, number] {
  const fy = (L + 16) / 116;
  const fx = fy + a / 500;
  const fz = fy - b / 200;

  const x = inv(fx) * Xn;
  const y = inv(fy) * Yn;
  const z = inv(fz) * Zn;

  const rl = x * 3.2406 + y * -1.5372 + z * -0.4986;
  const gl = x * -0.9689 + y * 1.8758 + z * 0.0415;
  const bl = x * 0.0557 + y * -0.204 + z * 1.057;

  return [linearToSrgb(rl), linearToSrgb(gl), linearToSrgb(bl)];
}
