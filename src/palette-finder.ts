import type { LoadedImage } from "./image";
import { rgbToLab, labToRgb } from "./color";
import blitWGSL from "./shaders/blit.wgsl?raw";
import assignWGSL from "./shaders/assign.wgsl?raw";
import finalizeWGSL from "./shaders/finalize.wgsl?raw";
import quantizeWGSL from "./shaders/quantize.wgsl?raw";

const WORKGROUP = 256;

export interface KMeansOptions {
  k: number;
  maxIterations: number;
  epsilon?: number; // Stop early when no centroid moves more than this (in 0-255 units).
  onIteration?: (iteration: number, centroids: Float32Array) => void;
}

export interface KMeansResult {
  centroids: Float32Array; // k*4: r,g,b,count
  iterations: number;
}

export class PaletteFinder {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private canvasFormat: GPUTextureFormat;

  // Compiled and configured form of the compute shaders 
  private assignPipeline!: GPUComputePipeline;
  private finalizePipeline!: GPUComputePipeline;
  private quantizePipeline!: GPUComputePipeline;

  // Compiled and configured form of the blit shader
  private blitPipeline!: GPURenderPipeline;

  private sampler!: GPUSampler;
  private built = false;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    const ctx = canvas.getContext("webgpu");
    if (!ctx) throw new Error("Could not get a WebGPU canvas context.");
    this.context = ctx;
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.canvasFormat, alphaMode: "opaque" });
  }

  private build() {
    // Build the pipelines and bind groups once, lazily on first run
    if (this.built) return;
    const d = this.device;

    // Define a helper to compile the compute shaders
    const compute = (code: string) =>
      d.createComputePipeline({
        layout: "auto",
        compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
      });

    // Compile and configure the compute shaders
    this.assignPipeline = compute(assignWGSL);
    this.finalizePipeline = compute(finalizeWGSL);
    this.quantizePipeline = compute(quantizeWGSL);

    // Compile and configure the blit shader
    const blitModule = d.createShaderModule({ code: blitWGSL });
    this.blitPipeline = d.createRenderPipeline({
      layout: "auto",
      vertex: { module: blitModule, entryPoint: "vs" },
      fragment: {
        module: blitModule,
        entryPoint: "fs",
        targets: [{ format: this.canvasFormat }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.sampler = d.createSampler({ magFilter: "nearest", minFilter: "nearest" });
    this.built = true;
  }

  /**
   * Clear the output canvas to black. Call when a new image is loaded so the
   * previously quantized frame doesn't linger on the WebGPU canvas (resizing the
   * canvas alone doesn't reliably clear a configured context).
   */
  clearCanvas() {
    const d = this.device;
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.end();
    d.queue.submit([enc.finish()]);
  }

  async run(image: LoadedImage, initialCentroids: Float32Array<ArrayBuffer>, opts: KMeansOptions,
  ): Promise<KMeansResult> {
    // Build first once
    this.build();

    // Get the GPU device and image data
    const d = this.device;
    const { k, maxIterations } = opts;
    const epsilon = opts.epsilon ?? 0.5;
    const { width, height, pixels } = image;
    const nbPixels = width * height;

    // ---- Buffers ---------------------
    const pixelBuf = d.createBuffer({
      size: pixels.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(pixelBuf, 0, pixels);

    const centroidBuf = d.createBuffer({
      size: k * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(centroidBuf, 0, initialCentroids);

    const accumBuf = d.createBuffer({
      size: k * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const zeros = new Uint32Array(k * 4);

    const assignBuf = d.createBuffer({
      size: nbPixels * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    const paramsBuf = d.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(paramsBuf, 0, new Uint32Array([k, nbPixels, width, height]));

    const readbackBuf = d.createBuffer({
      size: k * 4 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const outputTexture = d.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    // ---- Bind groups -------------------
    const assignBind = d.createBindGroup({
      layout: this.assignPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pixelBuf } },
        { binding: 1, resource: { buffer: centroidBuf } },
        { binding: 2, resource: { buffer: accumBuf } },
        { binding: 3, resource: { buffer: assignBuf } },
        { binding: 4, resource: { buffer: paramsBuf } },
      ],
    });

    const finalizeBind = d.createBindGroup({
      layout: this.finalizePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: centroidBuf } },
        { binding: 1, resource: { buffer: accumBuf } },
        { binding: 2, resource: { buffer: pixelBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
    });

    const quantizeBind = d.createBindGroup({
      layout: this.quantizePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: assignBuf } },
        { binding: 1, resource: { buffer: centroidBuf } },
        { binding: 2, resource: outputTexture.createView() },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
    });

    const blitBind = d.createBindGroup({
      layout: this.blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: outputTexture.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });

    // ---- Iteration loop -----------------------------
    const assignGroups = Math.ceil(nbPixels / WORKGROUP);
    const finalizeGroups = Math.ceil(k / 64);
    let prev = initialCentroids.slice();
    let iterationsRun = 0;

    for (let iter = 0; iter < maxIterations; iter++) {
      d.queue.writeBuffer(accumBuf, 0, zeros);

      const enc = d.createCommandEncoder();
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.assignPipeline);
        pass.setBindGroup(0, assignBind);
        pass.dispatchWorkgroups(assignGroups);
        pass.end();
      }
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.finalizePipeline);
        pass.setBindGroup(0, finalizeBind);
        pass.dispatchWorkgroups(finalizeGroups);
        pass.end();
      }
      enc.copyBufferToBuffer(centroidBuf, 0, readbackBuf, 0, k * 4 * 4);
      d.queue.submit([enc.finish()]);

      await readbackBuf.mapAsync(GPUMapMode.READ);
      const centroids = new Float32Array(readbackBuf.getMappedRange().slice(0));
      readbackBuf.unmap();

      iterationsRun = iter + 1;
      opts.onIteration?.(iterationsRun, centroids);

      // Convergence: max centroid movement across RGB.
      let moved = 0;
      for (let i = 0; i < k; i++) {
        for (let c = 0; c < 3; c++) {
          moved = Math.max(moved, Math.abs(centroids[i * 4 + c] - prev[i * 4 + c]));
        }
      }
      prev = centroids;
      if (moved < epsilon) break;
    }

    // ---- Final assign + quantize + blit to canvas -------
    // On the last iteration, we need to re-run the assign and quantize passes so that the output image matches the final centroids.
    // We then blit the quantized image to the canvas.
    // Quantize and blit are only done once, after the loop, to avoid unnecessary GPU work on every iteration.
    {
      const enc = d.createCommandEncoder();
      // Re-assign with the final centroids so the quantized image matches them.
      d.queue.writeBuffer(accumBuf, 0, zeros);
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.assignPipeline);
        pass.setBindGroup(0, assignBind);
        pass.dispatchWorkgroups(assignGroups);
        pass.end();
      }
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.quantizePipeline);
        pass.setBindGroup(0, quantizeBind);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
      }
      {
        const pass = enc.beginRenderPass({
          colorAttachments: [
            {
              view: this.context.getCurrentTexture().createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
            },
          ],
        });
        pass.setPipeline(this.blitPipeline);
        pass.setBindGroup(0, blitBind);
        pass.draw(3);
        pass.end();
      }
      d.queue.submit([enc.finish()]);
    }

    // Free per-run GPU resources
    pixelBuf.destroy();
    centroidBuf.destroy();
    accumBuf.destroy();
    assignBuf.destroy();
    paramsBuf.destroy();
    readbackBuf.destroy();
    outputTexture.destroy();

    return { centroids: prev, iterations: iterationsRun };
  }
}

// Seeded PRNG for deterministic k-means++ initialization. Returns a float in [0,1).
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * k-means++ init on the CPU. Spreads the seeds apart so the GPU loop lands on a
 * better palette than plain random seeding. Deterministic for a given seed
 * (same image + k + seed -> same palette); bump the seed to try another.
 * Returns k*4 floats: r,g,b,0.
 */
export function kmeansPlusPlusInit(
  pixels: Uint32Array,
  k: number,
  seed = 0x9e3779b9,
): Float32Array<ArrayBuffer> {
  const n = pixels.length;
  const centroids = new Float32Array(k * 4);
  const rng = mulberry32(seed);

  const first = pixels[Math.floor(rng() * n)];
  centroids[0] = first & 255;
  centroids[1] = (first >> 8) & 255;
  centroids[2] = (first >> 16) & 255;

  const dist2 = new Float32Array(n).fill(Infinity);

  for (let c = 1; c < k; c++) {
    const cx = centroids[(c - 1) * 4];
    const cy = centroids[(c - 1) * 4 + 1];
    const cz = centroids[(c - 1) * 4 + 2];

    let sum = 0;
    for (let i = 0; i < n; i++) {
      const p = pixels[i];
      const dr = (p & 255) - cx;
      const dg = ((p >> 8) & 255) - cy;
      const db = ((p >> 16) & 255) - cz;
      const d = dr * dr + dg * dg + db * db;
      if (d < dist2[i]) dist2[i] = d;
      sum += dist2[i];
    }

    let target = rng() * sum;
    let idx = 0;
    for (; idx < n - 1; idx++) {
      target -= dist2[idx];
      if (target <= 0) break;
    }

    const p = pixels[idx];
    centroids[c * 4] = p & 255;
    centroids[c * 4 + 1] = (p >> 8) & 255;
    centroids[c * 4 + 2] = (p >> 16) & 255;
  }

  return centroids;
}

// ---- Perceptual, frequency-de-weighted palette (CIELAB + sqrt(count)) --------

// Pull the accent colours out that plain k-means drowns under the dominant hue.
// Two changes vs the GPU path: (1) cluster in CIELAB so distance is perceptual,
// (2) cluster the unique-colour histogram weighted by sqrt(count) instead of one
// vote per pixel, so a small-but-vivid region can still earn its own centroid.
const WEIGHT_EXP = 0.5; // 0 = every distinct colour equal, 1 = area-proportional

interface Histogram {
  n: number;
  lab: Float32Array; // n*3
  weight: Float32Array; // n, = count^WEIGHT_EXP
  count: Float32Array; // n, raw pixel count (for palette proportions)
}

// Quantise to 5 bits/channel so near-identical colours merge and the bin count
// stays bounded; each bin keeps the average of the pixels that fell into it.
function buildHistogram(pixels: Uint32Array): Histogram {
  const bins = new Map<number, number[]>(); // key -> [sumR, sumG, sumB, count]
  for (let i = 0; i < pixels.length; i++) {
    const p = pixels[i];
    const r = p & 255;
    const g = (p >> 8) & 255;
    const b = (p >> 16) & 255;
    const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
    let e = bins.get(key);
    if (!e) {
      e = [0, 0, 0, 0];
      bins.set(key, e);
    }
    e[0] += r;
    e[1] += g;
    e[2] += b;
    e[3]++;
  }

  const n = bins.size;
  const lab = new Float32Array(n * 3);
  const weight = new Float32Array(n);
  const count = new Float32Array(n);
  let i = 0;
  for (const e of bins.values()) {
    const c = e[3];
    const [L, a, b] = rgbToLab(e[0] / c, e[1] / c, e[2] / c);
    lab[i * 3] = L;
    lab[i * 3 + 1] = a;
    lab[i * 3 + 2] = b;
    weight[i] = Math.pow(c, WEIGHT_EXP);
    count[i] = c;
    i++;
  }
  return { n, lab, weight, count };
}

function pickWeighted(weight: Float32Array, n: number, rng: () => number): number {
  let total = 0;
  for (let i = 0; i < n; i++) total += weight[i];
  let target = rng() * total;
  for (let i = 0; i < n - 1; i++) {
    target -= weight[i];
    if (target <= 0) return i;
  }
  return n - 1;
}

/**
 * Perceptual palette finder (CPU). Clusters a sqrt(count)-weighted colour
 * histogram in CIELAB, then returns centroids as k*4 floats (r,g,b,count) with
 * the raw pixel count in .w so the palette can still sort by prominence. Same
 * shape as kmeansPlusPlusInit's output, so the GPU can render the quantised
 * image straight from it (run with maxIterations: 0).
 */
export function findPalettePerceptual(
  pixels: Uint32Array,
  k: number,
  seed = 0x9e3779b9,
  maxIterations = 30,
): Float32Array<ArrayBuffer> {
  const { n, lab, weight, count } = buildHistogram(pixels);
  const rng = mulberry32(seed);
  const centers = new Float32Array(k * 3);

  // Fewer distinct colours than clusters: just hand each bin back as a centroid.
  if (n <= k) {
    const out = new Float32Array(k * 4);
    for (let c = 0; c < k; c++) {
      const i = Math.min(c, n - 1);
      const [r, g, b] = labToRgb(lab[i * 3], lab[i * 3 + 1], lab[i * 3 + 2]);
      out[c * 4] = r;
      out[c * 4 + 1] = g;
      out[c * 4 + 2] = b;
      out[c * 4 + 3] = c < n ? count[i] : 0;
    }
    return out;
  }

  // ---- Weighted k-means++ seeding (in Lab) ----
  const dist2 = new Float32Array(n).fill(Infinity);
  const first = pickWeighted(weight, n, rng);
  centers[0] = lab[first * 3];
  centers[1] = lab[first * 3 + 1];
  centers[2] = lab[first * 3 + 2];

  for (let c = 1; c < k; c++) {
    const cx = centers[(c - 1) * 3];
    const cy = centers[(c - 1) * 3 + 1];
    const cz = centers[(c - 1) * 3 + 2];
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const dl = lab[i * 3] - cx;
      const da = lab[i * 3 + 1] - cy;
      const db = lab[i * 3 + 2] - cz;
      const d = dl * dl + da * da + db * db;
      if (d < dist2[i]) dist2[i] = d;
      sum += weight[i] * dist2[i];
    }
    let target = rng() * sum;
    let idx = 0;
    for (; idx < n - 1; idx++) {
      target -= weight[idx] * dist2[idx];
      if (target <= 0) break;
    }
    centers[c * 3] = lab[idx * 3];
    centers[c * 3 + 1] = lab[idx * 3 + 1];
    centers[c * 3 + 2] = lab[idx * 3 + 2];
  }

  // ---- Weighted Lloyd iterations ----
  const sumL = new Float64Array(k);
  const sumA = new Float64Array(k);
  const sumB = new Float64Array(k);
  const sumW = new Float64Array(k);
  const sumCount = new Float64Array(k);

  for (let iter = 0; iter < maxIterations; iter++) {
    sumL.fill(0);
    sumA.fill(0);
    sumB.fill(0);
    sumW.fill(0);
    sumCount.fill(0);

    for (let i = 0; i < n; i++) {
      const li = lab[i * 3];
      const ai = lab[i * 3 + 1];
      const bi = lab[i * 3 + 2];
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const dl = li - centers[c * 3];
        const da = ai - centers[c * 3 + 1];
        const db = bi - centers[c * 3 + 2];
        const d = dl * dl + da * da + db * db;
        if (d < bestD) {
          bestD = d;
          best = c;
        }
      }
      const wi = weight[i];
      sumL[best] += wi * li;
      sumA[best] += wi * ai;
      sumB[best] += wi * bi;
      sumW[best] += wi;
      sumCount[best] += count[i];
    }

    let moved = 0;
    for (let c = 0; c < k; c++) {
      if (sumW[c] > 0) {
        const nl = sumL[c] / sumW[c];
        const na = sumA[c] / sumW[c];
        const nb = sumB[c] / sumW[c];
        moved = Math.max(
          moved,
          Math.abs(nl - centers[c * 3]),
          Math.abs(na - centers[c * 3 + 1]),
          Math.abs(nb - centers[c * 3 + 2]),
        );
        centers[c * 3] = nl;
        centers[c * 3 + 1] = na;
        centers[c * 3 + 2] = nb;
      } else {
        // Empty cluster: re-seed to a random weighted bin so it can grab colours.
        const idx = pickWeighted(weight, n, rng);
        centers[c * 3] = lab[idx * 3];
        centers[c * 3 + 1] = lab[idx * 3 + 1];
        centers[c * 3 + 2] = lab[idx * 3 + 2];
        moved = Infinity;
      }
    }
    if (moved < 0.5) break; // Lab units; ~sub-perceptual movement
  }

  // Convert back to RGB, carry the raw pixel count in .w for prominence sorting.
  const out = new Float32Array(k * 4);
  for (let c = 0; c < k; c++) {
    const [r, g, b] = labToRgb(centers[c * 3], centers[c * 3 + 1], centers[c * 3 + 2]);
    out[c * 4] = r;
    out[c * 4 + 1] = g;
    out[c * 4 + 2] = b;
    out[c * 4 + 3] = sumCount[c];
  }
  return out;
}
