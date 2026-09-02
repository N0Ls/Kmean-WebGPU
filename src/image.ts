export interface LoadedImage {
  width: number;
  height: number;
  pixels: Uint32Array<ArrayBuffer>; // RGBA packed into a single u32 per pixel
}

// Decode + downscale the file (longest side capped at maxDim) into one packed
// u32 per pixel, ready for the GPU. No point clustering at full res for a palette.
export async function loadImageToPixels(
  file: File,
  maxDim = 4096,
): Promise<LoadedImage> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));

  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not create a 2D context for decoding.");
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();

  const rgba = ctx.getImageData(0, 0, width, height).data;
  const n = width * height;
  const pixels = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    pixels[i] =
      (rgba[o] | (rgba[o + 1] << 8) | (rgba[o + 2] << 16) | (rgba[o + 3] << 24)) >>> 0;
  }
  return { width, height, pixels };
}
